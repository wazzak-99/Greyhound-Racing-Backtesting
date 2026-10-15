"""
backtest/strategy.py
--------------------

Includes the base class for building a Strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backtest.context import Cancel, MarketContext, Order, RaceResult
from backtest.execution import Fill, Unmatched


class Strategy(ABC):
    """
    To build a strategy, subclass this and override `on_snapshot`.

    Optionally, override `on_market_open`, `on_market_close`, `on_fill` and `on_unmatched`.
    """

    @abstractmethod
    def on_snapshot(self, ctx: MarketContext) -> list[Order | Cancel]:
        """Called for every snapshot of a race. Return the Orders (and Cancels) to send, if any."""
        ...

    def on_market_open(self, ctx: MarketContext) -> None:
        """Called once with the first snapshot of a race. Optional method."""
        pass

    def on_fill(self, fill: Fill) -> None:
        """Called when some or all of an order matches. Optional method."""
        pass

    def on_unmatched(self, unmatched: Unmatched) -> None:
        """Called when stake will never match (killed, cancelled, lapsed or rejected). Optional method."""
        pass

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        """Called once, after the race is settles. Optional method for logging/cleanup."""
        pass
