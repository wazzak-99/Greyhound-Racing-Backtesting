"""
backtest/context.py

A Strategy will only ever read from a MarketContext object, which will include
market and runner shapshots for a single tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

class Side(str, Enum):
    BACK = "BACK"
    LAY = "LAY"

class MarketStatus(str, Enum):
    OPEN = "OPEN"
    SUSPENDED = "SUSPENDED"
    CLOSED = "CLOSED"

class MarketType(str, Enum):
    WIN = "WIN"
    PLACE = "PLACE"
    MATCH_BET = "MATCH_BET"

@dataclass
class RunnerState:
    """A runner's state at a single tick within a market."""
    selection_id: int
    selection_name: str
    trap: Optional[int] = None
    removed: bool = False

    # MUST BE SORTED
    atb: list[tuple[float, float]] = field(default_factory=list) # (price, size) available to back
    atl: list[tuple[float, float]] = field(default_factory=list) # (price, size) available to lay

    sp_near: Optional[float] = None
    sp_far: Optional[float] = None

    last_price_traded: Optional[float] = None
    total_matched: Optional[float] = None

    adjustment_factor: Optional[float] = None

    def pull_price_size(self, side: Side, depth: int = 1) -> Optional[tuple[float, float]]:
        """Gets (price, size) at the given ladder depth (1 is nearest). Assumes list is sorted."""
        ladder = self.atb if side is Side.BACK else self.atl
        idx = depth - 1
        if idx < 0 or idx >= len(ladder):
            return None
        return ladder[idx]

@dataclass
class MarketContext:
    """
    The market's state at a single tick, handed to a Strategy. Everything
    in this class is what is available to a strategy.
    """
    market_id: str
    venue: Optional[str]
    race_no: Optional[int]
    market_time: datetime
    publish_time: datetime
    runners: list[RunnerState]
    status: MarketStatus 
    market_type: MarketType

    @property
    def seconds_to_off(self) -> float:
        return (self.market_time - self.publish_time).total_seconds()
    
    def runner(self, selection_id: int) -> RunnerState:
        for r in self.runners:
            if r.selection_id == selection_id:
                return r
        raise KeyError(f"ERROR: selection_id {selection_id} not in market {self.market_id}!")
    
@dataclass
class RaceResult:
    """
    Settled outcome of a race. Only ever passed to Strategy.on_market_close.
    """
    market_id: str
    winner_selection_id: int
    bsp: dict[int, float] = field(default_factory=dict) # selection_id -> BSP

@dataclass
class Order:
    """
    Engine and ExecutionModel will decide if/how an Order gets filled.
    """
    market_id: str
    selection_id: int
    side: Side
    price: Optional[float] = None # None = take best available at market
    stake: Optional[float] = None # None = let a Staking model decide
    