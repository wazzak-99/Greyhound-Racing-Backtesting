"""
strategies/queue_test.py
------------------------
Test file for QueueExecutionModel. Once per race, 60s before the off, joins the backers'
queue at the best lay price on the favourite and prints what happened to the order.
"""
from __future__ import annotations

from backtest.context import MarketContext, MarketStatus, Order, RaceResult, Side
from backtest.execution import Fill, Unmatched
from backtest.strategy import Strategy


class QueueTestStrategy(Strategy):
    def __init__(self, seconds_before_off: float = 60.0):
        self.seconds_before_off = seconds_before_off

    def on_market_open(self, ctx: MarketContext) -> None:
        self.sent = False
        self.price = None
        self.matched = 0.0
        self.remainder = "never sent"

    def on_snapshot(self, ctx: MarketContext) -> list[Order]:
        if self.sent or ctx.status is not MarketStatus.OPEN or ctx.seconds_to_off > self.seconds_before_off:
            return []
        runners = [r for r in ctx.runners if not r.removed and r.atb and r.atl]
        if not runners:
            return []
        fav = min(runners, key=lambda r: r.atb[0][0])
        self.sent = True
        self.price = fav.atl[0][0]
        self.remainder = "none (fully matched)"
        # Backing at the best lay price joins the backers' queue instead of taking
        return [Order(ctx.market_id, fav.selection_id, Side.BACK, price=self.price, keep_unmatched=True)]

    def on_fill(self, fill: Fill) -> None:
        self.matched += fill.stake

    def on_unmatched(self, unmatched: Unmatched) -> None:
        self.remainder = f"{unmatched.stake:.2f} {unmatched.reason}"

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        print(f"{ctx.market_id}  back @ {self.price}  matched {self.matched:.2f}  unmatched: {self.remainder}")
