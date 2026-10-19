"""
strategies/naivequoter.py
-------------------

"""
from __future__ import annotations

import bisect
from typing import Optional

from backtest.context import Cancel, MarketContext, MarketStatus, Order, RaceResult, RunnerState, Side
from backtest.execution import _TICK_BANDS, Fill, Unmatched
from backtest.strategy import Strategy


_EPS = 1e-9
INF = float("inf")

# ---------------------------------------------------------------------------
# Betfair price ladder
# ---------------------------------------------------------------------------

def _build_ladder() -> list[float]:
    """Builds the Betfair price ladder as a list of floats."""
    prices = []
    for low, high, step in _TICK_BANDS:
        for i in range(round((high - low) / step)):
            prices.append(round(low + i * step, 2))
    prices.append(1000.0)
    return prices

# The Betfair price ladder is a list of floats representing valid odds
# For example, you can't place an order at $2.35; the closest valid odds are $2.34 or $2.36
LADDER = _build_ladder()

def odds_tick(odds: float, n: int) -> Optional[float]:
    """Move `n` ticks along the ladder from a ladder price (`n` > 0 = longer odds, 
    `n` < 0 = shorter odds, otherwise None if invalid)."""
    i = bisect.bisect_left(LADDER, odds - 1e-9) + n
    return LADDER[i] if 0 <= i < len(LADDER) else None

def _ticks_apart(a: float, b: float) -> int:
    """Number of ladder ticks between two ladder prices."""
    return abs(bisect.bisect_left(LADDER, a - 1e-9) - bisect.bisect_left(LADDER, b - 1e-9))

# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class NaiveStrategy(Strategy):
    def __init__(
        self,
        quote_units: float = 1.0,           # payout units per quote ($ won if the dog wins)
        q_max: float = 5.0,                 # inventory cap, payout units
        start_seconds: float = 350.0,       # start quoting this many seconds before the off
        greenup_seconds: float = 30.0,      # T_e: stop quoting and green up this many seconds before the off
        greenup_slippage_ticks: int = 3,    # how many ticks through the best price a green-up may take
        greenup_min_units: float = 0.5,     # ignore |q| below this when greening up
        min_odds: float = 1.2,              # only quote runners whose fair odds are in this range
        max_odds: float = 6.0,
    ):
        self.quote_units = quote_units
        self.q_max = q_max
        self.start_seconds = start_seconds
        self.greenup_seconds = greenup_seconds
        self.greenup_slippage_ticks = greenup_slippage_ticks
        self.greenup_min_units = greenup_min_units
        self.min_odds = min_odds
        self.max_odds = max_odds

        self.fill_log: list[dict] = []      # every fill across the run, for analysis afterwards
        self.market_log: list[dict] = []    # one row per runner per market

    # ---------------------------------------------------------------------------
    # Engine callbacks
    # ---------------------------------------------------------------------------

    def on_market_open(self, ctx: MarketContext) -> None:
        self.q: dict[int, float] = {}                         # selection_id -> inventory
        self.n_fills: dict[int, int] = {}
        self._quotes: dict[tuple[int, Side], Order] = {}      # our current resting quote per slot
        self._open: dict[int, list] = {}                      # id(order) -> [order, remaining stake]

    def on_snapshot(self, ctx: MarketContext) -> list[Order | Cancel]:
        if ctx.status is not MarketStatus.OPEN:
            return []
        ttj = ctx.seconds_to_off
        if ttj > self.start_seconds:
            return []
        if ttj <= self.greenup_seconds:
            return self._cancel_all() + self._green_up(ctx)

        fair = self._fair_values(ctx)
        actions: list[Order | Cancel] = []
        for r in ctx.runners:
            S = fair.get(r.selection_id)
            if S is None or not (1 / self.max_odds <= S <= 1 / self.min_odds):
                actions += self._cancel_runner(r.selection_id)
                continue
            bid_odds, ask_odds = self._quote_odds(r, S, ttj)
            q = self.q.get(r.selection_id, 0.0)

            # Inventory cap: never post a quote whose full fill would take |q| past q_max
            back_units = min(self.quote_units, self.q_max - q)
            lay_units = min(self.quote_units, self.q_max + q)

            actions += self._requote(ctx, r.selection_id, Side.BACK, bid_odds, back_units)
            actions += self._requote(ctx, r.selection_id, Side.LAY, ask_odds, lay_units)
        return actions

    def on_fill(self, fill: Fill) -> None:
        order = fill.order
        sign = 1.0 if order.side is Side.BACK else -1.0
        sel = order.selection_id
        self.q[sel] = self.q.get(sel, 0.0) + sign * fill.stake * fill.price
        self.n_fills[sel] = self.n_fills.get(sel, 0) + 1

        entry = self._open.get(id(order))
        if entry is not None:
            entry[1] -= fill.stake
            if entry[1] <= _EPS:
                self._forget(order)

        self.fill_log.append({
            "market_id": order.market_id,
            "selection_id": sel,
            "publish_time": fill.publish_time,
            "side": order.side.value,
            "tag": order.tag,
            "price": fill.price,
            "stake": fill.stake,
            "q_after": self.q[sel],
        })

    def on_unmatched(self, unmatched: Unmatched) -> None:
        self._forget(unmatched.order)

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        for sel, q in self.q.items():
            self.market_log.append({
                "market_id": ctx.market_id,
                "selection_id": sel,
                "won": sel == result.winner_selection_id,
                "final_q": q,
                "fills": self.n_fills.get(sel, 0),
            })

    # ---------------------------------------------------------------------------
    # Fair value and quotes
    # ---------------------------------------------------------------------------

    def _fair_values(self, ctx: MarketContext) -> dict[int, float]:
        """Mid implied probability per runner that has both sides of the book."""
        fair: dict[int, float] = {}
        for r in ctx.runners:
            if r.removed or not r.atb or not r.atl:
                continue
            ask_p, bid_p = 1.0 / r.atb[0][0], 1.0 / r.atl[0][0]
            fair[r.selection_id] = 0.5 * (bid_p + ask_p)
        return fair

    def _quote_odds(self, r: RunnerState, S: float, ttj: float) -> tuple[Optional[float], Optional[float]]:
        """(back odds, lay odds): join the best price on each side of the book."""
        best_back_odds = r.atb[0][0]   # best odds anyone can back at right now
        best_lay_odds = r.atl[0][0]    # best odds anyone can lay at right now

        # Our BACK waits with the backers' money, which sits at the best lay price (and vice versa)
        return best_lay_odds, best_back_odds

    # ---------------------------------------------------------------------------
    # Order management
    # ---------------------------------------------------------------------------

    def _requote(self, ctx: MarketContext, sel: int, side: Side,
                 odds: Optional[float], units: float) -> list[Order | Cancel]:
        """
        Keep one resting quote per (runner, side) at `odds`, with at most `units` of payout
        live on that side. Orders we've cancelled still count until the cancel is confirmed,
        because with latency they can still fill - otherwise q can overshoot q_max.
        """
        current = self._quotes.get((sel, side))
        if odds is None or units <= _EPS:
            return self._cancel_slot(sel, side)
        live = self._live_units(sel, side)   # includes `current` and any pending cancels
        if current is not None and _ticks_apart(current.price, odds) == 0:
            if live <= units + 1e-6:
                return []   # already resting at the right price: keep our queue position
            return self._cancel_slot(sel, side)   # would breach the cap if it all filled

        actions = self._cancel_slot(sel, side)
        stake = round((units - live) / odds, 2)  # `current` stays live until its cancel lands
        if stake <= 0:
            return actions   # post the new quote once the old one is confirmed gone
        order = Order(ctx.market_id, sel, side, price=odds, stake=stake,
                      keep_unmatched=True, tag="quote")
        self._quotes[(sel, side)] = order
        self._open[id(order)] = [order, stake]
        actions.append(order)
        return actions

    def _live_units(self, sel: int, side: Side) -> float:
        """Payout units still able to fill on this runner and side (resting + pending cancel)."""
        return sum(rem * o.price for o, rem in self._open.values()
                   if o.selection_id == sel and o.side is side and o.tag == "quote")

    def _cancel_slot(self, sel: int, side: Side) -> list[Cancel]:
        order = self._quotes.pop((sel, side), None)
        if order is None or order.order_id is None or id(order) not in self._open:
            return []
        return [Cancel(order.order_id)]

    def _cancel_runner(self, sel: int) -> list[Cancel]:
        return self._cancel_slot(sel, Side.BACK) + self._cancel_slot(sel, Side.LAY)

    def _cancel_all(self) -> list[Cancel]:
        actions: list[Cancel] = []
        for sel, side in list(self._quotes):
            actions += self._cancel_slot(sel, side)
        return actions

    def _forget(self, order: Order) -> None:
        self._open.pop(id(order), None)
        slot = (order.selection_id, order.side)
        if self._quotes.get(slot) is order:
            del self._quotes[slot]

    def _green_up(self, ctx: MarketContext) -> list[Order]:
        """Take liquidity to flatten q on every runner (fill-or-kill, retried each snapshot)."""
        ticks = self.greenup_slippage_ticks
        pending = {o.selection_id for o, _ in self._open.values() if o.tag == "greenup"}
        orders: list[Order] = []
        for r in ctx.runners:
            q = self.q.get(r.selection_id, 0.0)
            if r.removed or r.selection_id in pending or abs(q) < self.greenup_min_units:
                continue
            if q > 0 and r.atl:
                # Long: LAY into the backers' money. Worse for a layer = higher odds.
                best = r.atl[0][0]
                side, limit = Side.LAY, odds_tick(best, +ticks) or best
            elif q < 0 and r.atb:
                # Short: BACK into the layers' money. Worse for a backer = lower odds.
                best = r.atb[0][0]
                side, limit = Side.BACK, odds_tick(best, -ticks) or best
            else:
                continue
            stake = round(abs(q) / best, 2)
            if stake <= 0:
                continue
            order = Order(ctx.market_id, r.selection_id, side, price=limit, stake=stake,
                          keep_unmatched=False, tag="greenup")
            self._open[id(order)] = [order, stake]
            orders.append(order)
        return orders