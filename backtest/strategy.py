"""
backtest/strategy.py
--------------------

Includes the base class for building a Strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backtest.context import MarketContext, Order, RaceResult


class Strategy(ABC):
    """
    To build a strategy, subclass this and override `on_snapshot`.

    Optionally, override `on_market_open` and `on_market_close` to implement logic
    on market open/close.
    """
    
    @abstractmethod
    def on_snapshot(self, ctx: MarketContext) -> list[Order]:
        """Called for every snapshot of a race. Return the Orders to place, if any."""
        ...

    def on_market_open(self, ctx: MarketContext) -> None:
        """Called once with the first snapshot of a race. Optional method."""
        pass

    def on_market_close(self, ctx: MarketContext, result: RaceResult) -> None:
        """Called once, after the race is settles. Optional method for logging/cleanup."""
        pass