"""
backtest/staking.py
---------------------
Provides logic for staking. 
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from backtest.context import MarketContext, Order


class Staking(ABC):
    @abstractmethod
    def size(self, order: Order, ctx: MarketContext, bankroll: float) -> float:
        """Return the stake to use for this order."""
        ...


class FlatStake(Staking):
    def __init__(self, stake: float):
        self.stake = stake

    def size(self, order: Order, ctx: MarketContext, bankroll: float) -> float:
        return self.stake

class KellyStake(Staking):
    ...

