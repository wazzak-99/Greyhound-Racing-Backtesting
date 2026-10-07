"""
backtest/datasources.py
-----------------------
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import pandas as pd
import glob
import re

from backtest.context import MarketContext, MarketStatus, MarketType, RaceResult, RunnerState
from data.streamloaders import load_stream, race_key, read_market_type

def _ladder_columns(df: pd.DataFrame, side: str) -> list[tuple[str, str]]:
    """(price_col, size_col) for each ladder level in the CSV: back_price_1, back_price_2, ..."""
    cols = []
    level = 1
    while f"{side}_price_{level}" in df.columns:
        cols.append((f"{side}_price_{level}", f"{side}_size_{level}"))
        level += 1
    return cols


def _ladder(row, cols: list[tuple[str, str]]) -> list[tuple[float, float]]:
    """Build a (price, size) ladder from one CSV row, best first, stopping at the first empty level."""
    ladder = []
    for price_col, size_col in cols:
        price = getattr(row, price_col)
        if pd.isna(price):
            break
        ladder.append((price, getattr(row, size_col)))
    return ladder


@dataclass
class MarketReplay:
    """
    Holds all tick snapshots of a race in chronological order.
    """
    contexts: list[MarketContext]
    result: RaceResult


class MarketSource(ABC):
    """
    Replays a set of historical races as Market replay objects.
    """
    
    @abstractmethod
    def markets(self) -> Iterator[MarketReplay]:
        ...

class StreamMarketSource(MarketSource):
    """
    Wraps the per-market .parquet files written by scripts/build_dataset.py.

    Yields one MarketContext per tick/conflation window per race.
    """

    def __init__(
            self, 
            pattern: str, 
            race_no_lookup: dict[str, int], 
            market_type_filter: set[MarketType] | None = None,
        ):
        self.pattern = pattern
        self.race_no_lookup = race_no_lookup or {}
        self.market_type_filter = market_type_filter

    def markets(self) -> Iterator[MarketReplay]:
        for f in sorted(glob.glob(self.pattern)):
            if self.market_type_filter is not None:
                if read_market_type(f) not in self.market_type_filter:
                    continue
            df = load_stream(f)
            replay = self._build_replay(df)
            if replay is not None:
                yield replay

    def _build_replay(self, df: pd.DataFrame) -> MarketReplay:
        market_id = str(df["market_id"].iloc[-1])
        market_type = str(df["market_type"].iloc[-1])
        key = race_key(df["event_id"].iloc[-1], df["market_time"].iloc[-1])
        race_no = self.race_no_lookup.get(key)
        contexts: list[MarketContext] = []
        back_cols = _ladder_columns(df, "back")
        lay_cols = _ladder_columns(df, "lay")

        for publish_time, snapshot in df.groupby("publish_time", sort=False):
            runners = [
                RunnerState(
                    selection_id=int(r.selection_id),
                    selection_name=r.selection_name,
                    removed=(r.runner_status == "REMOVED"),
                    atb=_ladder(r, back_cols),
                    atl=_ladder(r, lay_cols),
                    last_price_traded=r.last_price_traded if pd.notna(r.last_price_traded) else None,
                    total_matched=r.total_runner_matched if pd.notna(r.total_runner_matched) else None,
                    adjustment_factor=r.adjustment_factor if pd.notna(r.adjustment_factor) else None,
                )
                for r in snapshot.itertuples()
            ]

            contexts.append(MarketContext(
                market_id=market_id,
                venue=snapshot["venue"].iloc[0],
                race_no=race_no,
                market_time=snapshot["market_time"].iloc[0],
                publish_time=publish_time,
                runners=runners,
                status=MarketStatus(snapshot["status"].iloc[0]),
                market_type=MarketType(market_type)
            ))

        final = df.groupby("selection_id").last()
        winner = final.loc[final["runner_status"] == "WINNER"]

        if winner.empty:
            print(f"WARNING: No winner found for market {market_id}! Skipping...")
            return None

        result = RaceResult(
            market_id=market_id,
            winner_selection_id=int(winner.index[0]),
            bsp=(
                {int(sel_id): row.sp for sel_id, row in final.iterrows() if pd.notna(row.sp)}
                if "sp" in final.columns else {}
            ),
        )

        return MarketReplay(contexts=contexts, result=result)