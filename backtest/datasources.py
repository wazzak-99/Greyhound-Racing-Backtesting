"""
backtest/datasources.py
-----------------------
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import glob

import numpy as np
import pandas as pd

from backtest.context import MarketContext, MarketStatus, MarketType, RaceResult, RunnerState
from data.streamloaders import load_stream, race_key, read_market_type

def _ladder_columns(df: pd.DataFrame, side: str) -> list[tuple[str, str]]:
    """(price_col, size_col) for each ladder level in the file: back_price_1, back_price_2, ..."""
    cols = []
    level = 1
    while f"{side}_price_{level}" in df.columns:
        cols.append((f"{side}_price_{level}", f"{side}_size_{level}"))
        level += 1
    return cols


def _optional(series: pd.Series) -> list:
    """Column as a plain Python list, with missing values as None."""
    return series.astype(object).where(series.notna(), None).tolist()


def _ladders(df: pd.DataFrame, cols: list[tuple[str, str]]) -> list[list[tuple[float, float]]]:
    """Returns (price, size) ladder for every row, best first, stopping at the first empty level."""
    levels = [(df[p].tolist(), df[z].tolist(), df[p].isna().to_numpy()) for p, z in cols]
    out = []
    for i in range(len(df)):
        ladder = []
        for prices, sizes, missing in levels:
            if missing[i]:
                break
            ladder.append((prices[i], sizes[i]))
        out.append(ladder)
    return out


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

        # Group rows by update (publish_time)
        codes, update_times = pd.factorize(df["publish_time"])
        order = np.argsort(codes, kind="stable")
        df = df.iloc[order].reset_index(drop=True)
        bounds = np.flatnonzero(np.diff(codes[order])) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(df)]))

        # Convert each column to a plain Python list once (much faster than reading row by row).
        selection_id = df["selection_id"].astype(int).tolist()
        selection_name = df["selection_name"].tolist()
        removed = (df["runner_status"] == "REMOVED").tolist()
        atb = _ladders(df, _ladder_columns(df, "back"))
        atl = _ladders(df, _ladder_columns(df, "lay"))
        last_price_traded = _optional(df["last_price_traded"])
        total_matched = _optional(df["total_runner_matched"])
        adjustment_factor = _optional(df["adjustment_factor"])
        venue = df["venue"].tolist()
        market_time = df["market_time"].tolist()
        status = df["status"].tolist()
        market_type = MarketType(market_type)

        for publish_time, start, end in zip(update_times, starts, ends):
            runners = [
                RunnerState(
                    selection_id=selection_id[i],
                    selection_name=selection_name[i],
                    removed=removed[i],
                    atb=atb[i],
                    atl=atl[i],
                    last_price_traded=last_price_traded[i],
                    total_matched=total_matched[i],
                    adjustment_factor=adjustment_factor[i],
                )
                for i in range(start, end)
            ]

            contexts.append(MarketContext(
                market_id=market_id,
                venue=venue[start],
                race_no=race_no,
                market_time=market_time[start],
                publish_time=publish_time,
                runners=runners,
                status=MarketStatus(status[start]),
                market_type=market_type,
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