"""
backtest/execution.py
-----------------------

Turns a Strategy's Orders into Fills, based on an ExecutionModel.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.context import MarketContext, MarketStatus, Order, Side


@dataclass
class Fill:
    order: Order
    price: float
    stake: float
    removed_at_fill: frozenset[int] 


class ExecutionModel(ABC):
    """Decides if/how an Order gets matched against a MarketContext."""

    @abstractmethod
    def fill(self, order: Order, ctx: MarketContext) -> Optional[Fill]:
        """Return a Fill if the order matches now, else None."""
        ...


class TickExecutionModel(ExecutionModel):
    """
    Matches an Order against the top (only) of the ladder.

    Only fills if the market is OPEN.
    """

    def fill(self, order: Order, ctx: MarketContext) -> Optional[Fill]:
        if ctx.status is not MarketStatus.OPEN:
            return None

        if order.stake is None:
            raise ValueError(f"Order for selection {order.selection_id} has no stake.")

        runner = ctx.runner(order.selection_id)
        top = runner.pull_price_size(order.side, depth=1)
        if top is None:
            return None

        available_price, available_size = top

        if order.price is not None:
            if order.side is Side.BACK and available_price < order.price:
                return None
            if order.side is Side.LAY and available_price > order.price:
                return None

        stake = order.stake
        if pd.notna(available_size):
            stake = min(stake, available_size)

        removed_at_fill = frozenset(r.selection_id for r in ctx.runners if r.removed)
        return Fill(order=order, price=available_price, stake=stake, removed_at_fill=removed_at_fill)
