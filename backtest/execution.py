"""
backtest/execution.py
-----------------------

Turns a Strategy's Orders into Fills, based on an ExecutionModel.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta

from backtest.context import Cancel, MarketContext, MarketStatus, Order, RunnerState, Side

TRADED_VOLUME_SIDES = 2  # Betfair double counts traded volume for both sides of a matched bet
_EPS = 1e-9


@dataclass
class Fill:
    order: Order
    price: float
    stake: float
    removed_at_fill: frozenset[int]
    publish_time: datetime


@dataclass
class Unmatched:
    """Stake that will never match. reason: killed / cancelled / lapsed / rejected."""
    order: Order
    stake: float
    reason: str


@dataclass
class RestingOrder:
    order: Order
    remaining: float
    queue_ahead: float  # AUD ahead of us at our price (inf = price not visible yet)


class ExecutionModel(ABC):
    """Decides if/how Orders get matched. Each method returns the Fills/Unmatched it caused."""

    @abstractmethod
    def reset(self) -> None:
        """Called at the start of each market."""

    @abstractmethod
    def on_snapshot(self, ctx: MarketContext) -> list[Fill | Unmatched]:
        """Called with every snapshot, before the Strategy sees it."""

    @abstractmethod
    def submit(self, order: Order, ctx: MarketContext) -> list[Fill | Unmatched]:
        """Called with each new Order."""

    @abstractmethod
    def cancel(self, cancel: Cancel, ctx: MarketContext) -> list[Fill | Unmatched]:
        """Called with each Cancel."""

    @abstractmethod
    def close_market(self, ctx: MarketContext) -> list[Fill | Unmatched]:
        """Called after the last snapshot. Lapses anything still open."""


class QueueExecutionModel(ExecutionModel):
    """
    Orders take what they can from the visible ladder. With keep_unmatched, the rest waits at
    its price and is filled by trades at that price once the money ahead of it has traded.

    Orders and cancels reach the market `latency` seconds after they're sent. Everything 
    unmatched lapses when the market suspends.
    """

    def __init__(self, latency: float = 0.0, ladder_depth: int = 3):
        self.latency = latency
        self.ladder_depth = ladder_depth  # must match LADDER_DEPTH in streamloaders.py
        self.reset()

    def reset(self) -> None:
        self._resting: dict[int, RestingOrder] = {}
        self._pending: list[tuple[datetime, Order | Cancel]] = []
        self._taken: dict[tuple[int, Side, float], float] = {}  # size we took this snapshot

    def on_snapshot(self, ctx: MarketContext) -> list[Fill | Unmatched]:
        self._taken.clear()
        if ctx.status is not MarketStatus.OPEN:
            return self._lapse_all()

        events: list[Fill | Unmatched] = []
        for ro in list(self._resting.values()):
            events += self._update(ro, ctx)
        while self._pending and self._pending[0][0] <= ctx.publish_time:
            _, action = self._pending.pop(0)
            events += self._apply(action, ctx)
        return events

    def submit(self, order: Order, ctx: MarketContext) -> list[Fill | Unmatched]:
        if order.stake is None:
            raise ValueError(f"Order for selection {order.selection_id} has no stake.")
        if order.keep_unmatched and order.price is None:
            raise ValueError("An order with keep_unmatched needs a price.")
        if order.stake <= 0:
            raise ValueError(f"Order for selection {order.selection_id} has stake {order.stake}.")
        if order.price is not None and not _valid_price(order.price):
            raise ValueError(f"{order.price} is not a valid Betfair price.")

        return self._send(order, ctx)

    def cancel(self, cancel: Cancel, ctx: MarketContext) -> list[Fill | Unmatched]:
        return self._send(cancel, ctx)

    def close_market(self, ctx: MarketContext) -> list[Fill | Unmatched]:
        return self._lapse_all()

    def _send(self, action: Order | Cancel, ctx: MarketContext) -> list[Fill | Unmatched]:
        if ctx.status is not MarketStatus.OPEN:
            return [Unmatched(action, action.stake, "rejected")] if isinstance(action, Order) else []
        if self.latency <= 0:
            return self._apply(action, ctx)
        self._pending.append((ctx.publish_time + timedelta(seconds=self.latency), action))
        return []

    def _apply(self, action: Order | Cancel, ctx: MarketContext) -> list[Fill | Unmatched]:
        """An order or cancel reaching the market."""
        if isinstance(action, Cancel):
            ro = self._resting.pop(action.order_id, None)
            return [Unmatched(ro.order, ro.remaining, "cancelled")] if ro else []

        order = action
        runner = ctx.runner(order.selection_id)
        if runner.removed:
            return [Unmatched(order, order.stake, "rejected")]

        events, remaining = self._take(order, order.stake, ctx, own_price=False)
        if remaining > _EPS:
            if order.keep_unmatched:
                queue = self._visible_queue(runner, order.side, order.price)
                self._resting[order.order_id] = RestingOrder(order, remaining, queue)
            else:
                events.append(Unmatched(order, remaining, "killed"))
        return events

    def _update(self, ro: RestingOrder, ctx: MarketContext) -> list[Fill | Unmatched]:
        """Move a resting order through its queue using this snapshot's book and trades."""
        order, price = ro.order, ro.order.price
        runner = ctx.runner(order.selection_id)
        if runner.removed:
            del self._resting[order.order_id]
            return [Unmatched(order, ro.remaining, "lapsed")]

        # The other side now offers our price or better, so it would have matched us
        events, ro.remaining = self._take(order, ro.remaining, ctx, own_price=True)

        # Trades at our price use up the queue ahead; trades at worse prices went through us
        at_price = through = 0.0
        for traded_price, size in runner.traded:
            if _same(traded_price, price):
                at_price += size / TRADED_VOLUME_SIDES
            elif _ahead(price, traded_price, order.side):
                through += size / TRADED_VOLUME_SIDES
        ro.queue_ahead -= at_price
        amount = min(ro.remaining, max(0.0, -ro.queue_ahead) + through)
        if amount > _EPS:
            events.append(self._fill(order, price, amount, ctx))
            ro.remaining -= amount

        # Everyone ahead of us is in the visible size, so cancels can only shrink the queue to it
        ro.queue_ahead = max(0.0, min(ro.queue_ahead, self._visible_queue(runner, order.side, price)))

        if ro.remaining <= _EPS:
            del self._resting[order.order_id]
        return events

    def _take(self, order: Order, stake: float, ctx: MarketContext, own_price: bool) -> tuple[list, float]:
        """Match `stake` against the opposite ladder. Returns (fills, stake left)."""
        runner = ctx.runner(order.selection_id)
        ladder = runner.atb if order.side is Side.BACK else runner.atl
        fills = []
        for level_price, size in ladder:
            if stake <= _EPS:
                break
            if order.price is not None:
                if order.side is Side.BACK and level_price < order.price - _EPS:
                    break
                if order.side is Side.LAY and level_price > order.price + _EPS:
                    break
            key = (order.selection_id, order.side, level_price)
            available = size - self._taken.get(key, 0.0)
            if not available > _EPS:
                continue
            amount = min(stake, available)
            self._taken[key] = self._taken.get(key, 0.0) + amount
            fills.append(self._fill(order, order.price if own_price else level_price, amount, ctx))
            stake -= amount
        return fills, stake

    def _visible_queue(self, runner: RunnerState, side: Side, price: float) -> float:
        """AUD waiting at `price` on our side of the book (inf if deeper than the visible ladder)."""
        # A resting BACK waits with the backers' money, which shows as available to LAY (and vice versa)
        ladder = runner.atl if side is Side.BACK else runner.atb
        for level_price, size in ladder:
            if _same(level_price, price):
                return size
            if _ahead(price, level_price, side):
                return 0.0
        return math.inf if len(ladder) >= self.ladder_depth else 0.0

    def _lapse_all(self) -> list[Fill | Unmatched]:
        events: list[Fill | Unmatched] = [Unmatched(ro.order, ro.remaining, "lapsed") for ro in self._resting.values()]
        events += [Unmatched(a, a.stake, "lapsed") for _, a in self._pending if isinstance(a, Order)]
        self._resting.clear()
        self._pending.clear()
        return events

    @staticmethod
    def _fill(order: Order, price: float, stake: float, ctx: MarketContext) -> Fill:
        removed = frozenset(r.selection_id for r in ctx.runners if r.removed)
        return Fill(order=order, price=price, stake=stake, removed_at_fill=removed, publish_time=ctx.publish_time)


def _same(a: float, b: float) -> bool:
    return abs(a - b) < 1e-6


def _ahead(a: float, b: float, side: Side) -> bool:
    """For a resting order on `side`, True if price a is nearer the front of the queue than b."""
    return a < b if side is Side.BACK else a > b

_TICK_BANDS = [(1.01, 2, 0.01), (2, 3, 0.02), (3, 4, 0.05), (4, 6, 0.1), (6, 10, 0.2),
               (10, 20, 0.5), (20, 30, 1), (30, 50, 2), (50, 100, 5), (100, 1000, 10)]


def _valid_price(price: float) -> bool:
    """True if `price` is on Betfair's price ladder (1.01 to 1000)."""
    if price < 1.01 - 1e-9 or price > 1000 + 1e-9:
        return False
    for low, high, step in _TICK_BANDS:
        if price <= high + 1e-9:
            steps = (price - low) / step
            return abs(steps - round(steps)) < 1e-6
    return False
