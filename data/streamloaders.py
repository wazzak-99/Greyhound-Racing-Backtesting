"""
data/streamloaders.py
----------------------
Load Betfair historical market-stream (.bz2) files into tabular data.

`create_historical_generator_stream` opens `file_path` as plain text, so
each .bz2 file is decompressed to a temp file before streaming.
"""
from __future__ import annotations

import bz2
import csv
import glob
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Optional

import betfairlightweight
import pandas as pd
from betfairlightweight import StreamListener
from betfairlightweight.resources.bettingresources import MarketBook, RunnerBook
from betfairlightweight.resources.streamingresources import MarketDefinitionRunner
from currency_converter import CurrencyConverter

# ---------------------------------------------------------------------------
# Column extractors
# ---------------------------------------------------------------------------

ColumnExtractor = Callable[[MarketBook, RunnerBook, Optional[MarketDefinitionRunner]], object]

# All stream files are in GBP, we use publish time as the date for conversion.
_CURRENCY_CONVERTER = CurrencyConverter(fallback_on_wrong_date=True, fallback_on_missing_rate=True)

@lru_cache(maxsize=None)
def _gbp_aud_rate(day) -> float:
    """GBP -> AUD rate for one day. Cached, since the rate only changes daily."""
    return _CURRENCY_CONVERTER.convert(1.0, "GBP", "AUD", date=day)


def _gbp_to_aud(amount: Optional[float], market_book: MarketBook) -> Optional[float]:
    if amount is None:
        return None
    return round(amount * _gbp_aud_rate(market_book.publish_time.date()), 2)


def _level(ladder, i: int):
    """Level i of a ladder (0 = best price), or None if it isn't that deep."""
    return ladder[i] if len(ladder) > i else None


COLUMN_EXTRACTORS: dict[str, ColumnExtractor] = {
    "publish_time": lambda mb, r, rd: mb.publish_time,
    "market_id": lambda mb, r, rd: mb.market_id,
    "event_id": lambda mb, r, rd: mb.market_definition.event_id,
    "market_time": lambda mb, r, rd: mb.market_definition.market_time,
    "market_type": lambda mb, r, rd: mb.market_definition.market_type,
    "market_name": lambda mb, r, rd: mb.market_definition.name,
    "venue": lambda mb, r, rd: mb.market_definition.venue,
    "status": lambda mb, r, rd: mb.status,
    "inplay": lambda mb, r, rd: mb.inplay,
    "selection_id": lambda mb, r, rd: r.selection_id,
    "selection_name": lambda mb, r, rd: rd.name if rd else None,
    "last_price_traded": lambda mb, r, rd: r.last_price_traded,
    "total_matched": lambda mb, r, rd: _gbp_to_aud(mb.total_matched, mb),
    "total_runner_matched": lambda mb, r, rd: _gbp_to_aud(r.total_matched, mb),
    "adjustment_factor": lambda mb, r, rd: rd.adjustment_factor if rd else None,
    # ACTIVE / REMOVED / WINNER / LOSER (the backtester uses this to find the winner)
    "runner_status": lambda mb, r, rd: rd.status if rd else r.status,
}

# Order book ladder columns, numbered from 1 (best price) up to LADDER_DEPTH:
# back_price_1, back_size_1, lay_price_1, lay_size_1, back_price_2, ...
LADDER_DEPTH = 3


def _ladder_extractors(depth: int) -> dict[str, ColumnExtractor]:
    out: dict[str, ColumnExtractor] = {}
    for level in range(1, depth + 1):
        i = level - 1
        for side, attr in (("back", "available_to_back"), ("lay", "available_to_lay")):
            out[f"{side}_price_{level}"] = (
                lambda mb, r, rd, a=attr, i=i: lvl.price if (lvl := _level(getattr(r.ex, a), i)) else None
            )
            out[f"{side}_size_{level}"] = (
                lambda mb, r, rd, a=attr, i=i: _gbp_to_aud(lvl.size, mb) if (lvl := _level(getattr(r.ex, a), i)) else None
            )
    return out


COLUMN_EXTRACTORS.update(_ladder_extractors(LADDER_DEPTH))

# Every column above, in order. This is what build_dataset.py writes.
ALL_COLUMNS = list(COLUMN_EXTRACTORS)

# Columns converted from GBP to AUD (rounded to cents).
_SIZE_COLUMNS = [c for c in ALL_COLUMNS if c in ("total_matched", "total_runner_matched") or "_size_" in c]

DEFAULT_COLUMNS = [
    "publish_time",
    "market_type",
    "market_name",
    "venue",
    "status",
    "selection_name",   
    "last_price_traded",
    "total_matched",
    "back_price_1",
    "back_size_1",
    "lay_price_1",
    "lay_size_1",
]


class BetfairStreamLoader:
    """Streams Betfair historical .bz2 market-stream files into rows/DataFrames.

    ## Notes:
        - Size amounts for `total_matched`, `total_runner_matched` and all \
        `back_size_N` / `lay_size_N` columns have been converted from GBP to AUD. The amounts given in \
        the data are in GBP. Convert it manually if you wish to add extra columns.
        - Look into this module to find all possible column extractors since I'm too \
        lazy to write them all here (you can define your own with the `extra_columns` \
        parameter.)

    Args:
        columns (Iterable[str], optional): Output columns, in order. Must be keys \
            of `COLUMN_EXTRACTORS` (or `extra_columns`). Defaults to `DEFAULT_COLUMNS`.
        conflate_seconds (float | None, optional): Minimum gap, in seconds, between \
            two logged snapshots of the same market. `None` logs every update \
            with no conflation. Defaults to None.
        status_filter (Iterable[str] | None, optional): Only log rows where \
            `market_book.status` is in this set, e.g. `{"OPEN"}`. Defaults to None.
        selection_ids (Iterable[int] | None, optional): Restrict output to these \
            selection_ids only. Defaults to None (all runners).
        extra_columns (dict[str, ColumnExtractor] | None, optional): Extra/override \
            column extractors, merged over `COLUMN_EXTRACTORS`.
        market_type_filter (Iterable[str] | None, optional): Only load markets of \
            these types, e.g. `["WIN", "PLACE"]`. Other files return an empty \
            DataFrame and are abandoned early. Defaults to None (all types).

    Example:
        >>> loader = BetfairStreamLoader(
        ...     columns=["publish_time", "selection_id", "selection_name", "total_matched"],
        ...     conflate_seconds=5,
        ...     status_filter={"OPEN"},
        ... )
        >>> df = loader.load("data/raw/market_stream/.../1.249249319.bz2")
    """

    def __init__(
        self,
        columns: Iterable[str] = DEFAULT_COLUMNS,
        conflate_seconds: Optional[float] = None,
        status_filter: Optional[Iterable[str]] = None,
        selection_ids: Optional[Iterable[int]] = None,
        extra_columns: Optional[dict[str, ColumnExtractor]] = None,
        market_type_filter: Optional[Iterable[str]] = None,
    ) -> None:
        self.extractors = {**COLUMN_EXTRACTORS, **(extra_columns or {})}
        # The fast row builder only knows the built-in columns.
        self._use_fast_path = not extra_columns

        columns = list(columns)
        unknown = set(columns) - set(self.extractors)
        if unknown:
            raise ValueError(
                f"Unknown column(s): {sorted(unknown)}. Available: {sorted(self.extractors)}"
            )
        self.columns = columns

        self.conflate_seconds = conflate_seconds
        self.status_filter = set(status_filter) if status_filter else None
        self.selection_ids = set(selection_ids) if selection_ids else None
        self.market_type_filter = set(market_type_filter) if market_type_filter else None

    # ---------------------------------------------------------------------------
    # Filtering
    # ---------------------------------------------------------------------------
    def _passes_filters(self, market_book: MarketBook) -> bool:
        if self.status_filter and market_book.status not in self.status_filter:
            return False
        return True

    def _passes_conflation(self, market_book: MarketBook, last_logged: dict[str, float]) -> bool:
        if self.conflate_seconds is None:
            return True
        market_id = market_book.market_id
        pt = market_book.publish_time_epoch / 1000.0
        last = last_logged.get(market_id)
        if last is not None and (pt - last) < self.conflate_seconds:
            return False
        last_logged[market_id] = pt
        return True

    def _extract_row(
        self, market_book: MarketBook, runner: RunnerBook, runner_def: Optional[MarketDefinitionRunner]
    ) -> dict:
        return {col: self.extractors[col](market_book, runner, runner_def) for col in self.columns}

    # ---------------------------------------------------------------------------
    # Streaming
    # ---------------------------------------------------------------------------

    def _iter_runner_books(self, bz2_path: str | Path):
        """Yield (market_book, runner, runner_def) for every runner on every logged update."""
        # Historical files need no login, but the client still requires an app key, so pass a dummy one.
        trading = betfairlightweight.APIClient("username", "password", app_key="historical")
        listener = StreamListener(
            max_latency=None,
            cumulative_runner_tv=True,
            calculate_market_tv=True,
            update_clk=False
        )
        last_logged: dict[str, float] = {}

        with bz2.open(bz2_path, "rt", encoding="utf-8") as src:
            fd, decompressed_path = tempfile.mkstemp(suffix=".txt")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as dst:
                    dst.write(src.read())

                stream = trading.streaming.create_historical_generator_stream(
                    file_path=decompressed_path,
                    listener=listener,
                )
                gen = stream.get_generator()

                for market_books in gen():
                    for market_book in market_books:
                        # A file holds one market, so if its type is wrong, stop reading it.
                        if (
                            self.market_type_filter
                            and market_book.market_definition.market_type not in self.market_type_filter
                        ):
                            return
                        if not self._passes_filters(market_book):
                            continue
                        if not self._passes_conflation(market_book, last_logged):
                            continue

                        runners_dict = {
                            r.selection_id: r
                            for r in market_book.market_definition.runners
                        }

                        for runner in market_book.runners:
                            if self.selection_ids and runner.selection_id not in self.selection_ids:
                                continue
                            yield market_book, runner, runners_dict.get(runner.selection_id)
            finally:
                os.remove(decompressed_path)

    def _iter_rows(self, bz2_path: str | Path):
        """Rows as dicts, built column by column from the extractors (supports extra_columns)."""
        for market_book, runner, runner_def in self._iter_runner_books(bz2_path):
            yield self._extract_row(market_book, runner, runner_def)

    def _iter_fast_rows(self, bz2_path: str | Path):
        """Rows as tuples in ALL_COLUMNS order. Same values as _iter_rows, but much faster:
        market-level fields and times are worked out once per update instead of once per
        cell, and sizes are left unrounded (load() rounds them all at once)."""
        last_book = None
        for mb, r, rd in self._iter_runner_books(bz2_path):
            if mb is not last_book:
                last_book = mb
                md = mb.market_definition
                rate = _gbp_aud_rate(mb.publish_time.date())
                total_matched = mb.total_matched * rate if mb.total_matched is not None else None
                head = (
                    mb.publish_time_epoch,
                    mb.market_id,
                    md.event_id,
                    md.market_time,
                    md.market_type,
                    md.name,
                    md.venue,
                    mb.status,
                    mb.inplay,
                )

            ladder = []
            backs, lays = r.ex.available_to_back, r.ex.available_to_lay
            for i in range(LADDER_DEPTH):
                if len(backs) > i:
                    ladder += (backs[i].price, backs[i].size * rate)
                else:
                    ladder += (None, None)
                if len(lays) > i:
                    ladder += (lays[i].price, lays[i].size * rate)
                else:
                    ladder += (None, None)

            yield head + (
                r.selection_id,
                rd.name if rd else None,
                r.last_price_traded,
                total_matched,
                r.total_matched * rate if r.total_matched is not None else None,
                rd.adjustment_factor if rd else None,
                rd.status if rd else r.status,
                *ladder,
            )

    # ---------------------------------------------------------------------------
    # Public API 
    # ---------------------------------------------------------------------------

    def load(self, bz2_path: str | Path) -> pd.DataFrame:
        """Load a single .bz2 market-stream file into a DataFrame."""
        if not self._use_fast_path:
            return pd.DataFrame(list(self._iter_rows(bz2_path)), columns=self.columns)

        df = pd.DataFrame(list(self._iter_fast_rows(bz2_path)), columns=ALL_COLUMNS)
        df["publish_time"] = pd.to_datetime(df["publish_time"], unit="ms", utc=True)
        df["market_time"] = pd.to_datetime(df["market_time"], utc=True)
        df[_SIZE_COLUMNS] = df[_SIZE_COLUMNS].round(2)
        return df[self.columns]

    def load_to_csv(self, bz2_path: str | Path, out_path: str | Path) -> None:
        """Stream a .bz2 market-stream file straight to CSV without holding it all in memory."""
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            for row in self._iter_rows(bz2_path):
                writer.writerow(row)

    def load_glob(self, pattern: str) -> pd.DataFrame:
        """Load and concatenate all .bz2 files matching a glob pattern.

        Raises:
            FileNotFoundError: No files matched the glob pattern.
        """
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matched: {pattern}")
        frames = [self.load(f) for f in files]
        return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Helpers used by build_dataset.py and backtest/datasources.py
# ---------------------------------------------------------------------------

def race_key(event_id, market_time) -> str:
    """Identify a race by meeting (event_id) + scheduled start time.

    The WIN, PLACE and MATCH_BET markets for one race share this key. Works on
    both the loader's output (datetime) and a re-read CSV (string).
    """
    return f"{event_id}_{pd.Timestamp(market_time).strftime('%Y-%m-%dT%H:%M:%S')}"


def load_stream(path: str | Path) -> pd.DataFrame:
    """Read a per-market file written by build_dataset.py (.parquet, or an older .csv)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)

    df = pd.read_csv(
        path,
        dtype={"market_id": str, "event_id": str, "selection_id": "int64"},
    )
    df["publish_time"] = pd.to_datetime(df["publish_time"], format="ISO8601")
    df["market_time"] = pd.to_datetime(df["market_time"], format="ISO8601")
    return df


def read_market_type(path: str | Path) -> str | None:
    """Market type (WIN / PLACE / MATCH_BET) of a per-market file, without loading all of it."""
    path = Path(path)
    if path.suffix == ".parquet":
        col = pd.read_parquet(path, columns=["market_type"])["market_type"]
    else:
        col = pd.read_csv(path, usecols=["market_type"], nrows=1)["market_type"]
    return None if col.empty else str(col.iloc[0])