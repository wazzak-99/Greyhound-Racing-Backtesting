"""
strategies/template.py
----------------------
Copy this file to start a new strategy. The engine calls:

    on_market_open(ctx)          once, with the first snapshot of each race
    on_snapshot(ctx) -> [Order]  for every snapshot (tick) of the race
    on_market_close(ctx, result) once, after the race is settled

A strategy only ever sees a MarketContext (see backtest/context.py), so it
cannot look ahead. It is NOT told whether its orders filled.

The example logic below (back the favourite once, inside 60s of the jump) is
only there to show the mechanics. Delete it and put your own logic in.
"""
from __future__ import annotations

from backtest.context import MarketContext, MarketStatus, Order, RaceResult, Side
from backtest.strategy import Strategy


class TemplateStrategy(Strategy):
    def __init__(self, seconds_before_off: float = 60.0):
        self.seconds_before_off = seconds_before_off

    def on_market_open(self, ctx: MarketContext) -> None:
        # Per-race state lives here and is reset every race.
        self.done = False

    def on_snapshot(self, ctx: MarketContext) -> list[Order]:
        if self.done or ctx.status is not MarketStatus.OPEN:
            return []
        if ctx.seconds_to_off > self.seconds_before_off:
            return []

        # pull_price_size(Side.BACK) = best price you can BACK at right now (price, size)
        # pull_price_size(Side.LAY)  = best price you can LAY at right now
        quotes = []
        for r in ctx.runners:
            if r.removed:
                continue
            best_back = r.pull_price_size(Side.BACK)
            if best_back is not None:
                quotes.append((best_back[0], r.selection_id))
        if not quotes:
            return []

        price, selection_id = min(quotes)  # shortest price = favourite
        self.done = True
        return [
            Order(
                market_id=ctx.market_id,
                selection_id=selection_id,
                side=Side.BACK,
                price=price,   # limit price; None = take whatever is best
                stake=None,    # None = let the Staking model decide
            )
        ]

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        pass
