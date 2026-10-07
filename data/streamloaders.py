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

def _gbp_to_aud(amount: Optional[float], market_book: MarketBook) -> Optional[float]:
    if amount is None:
        return None
    return round(_CURRENCY_CONVERTER.convert(amount, "GBP", "AUD", date=market_book.publish_time.date()), 2)


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
    "best_back_price": lambda mb, r, rd: r.ex.available_to_back[0].price if r.ex.available_to_back else None,
    "best_back_size": lambda mb, r, rd: _gbp_to_aud(
        r.ex.available_to_back[0].size if r.ex.available_to_back else None, mb
    ),
    "best_lay_price": lambda mb, r, rd: r.ex.available_to_lay[0].price if r.ex.available_to_lay else None,
    "best_lay_size": lambda mb, r, rd: _gbp_to_aud(
        r.ex.available_to_lay[0].size if r.ex.available_to_lay else None, mb
    ),
    "adjustment_factor": lambda mb, r, rd: rd.adjustment_factor if rd else None,
}

DEFAULT_COLUMNS = [
    "publish_time",
    "market_type",
    "market_name",
    "venue",
    "status",
    "selection_name",   
    "last_price_traded",
    "total_matched",
    "best_back_price",
    "best_back_size",
    "best_lay_price",
    "best_lay_size",
]


class BetfairStreamLoader:
    """Streams Betfair historical .bz2 market-stream files into rows/DataFrames.

    ## Notes:
        - Size amounts for `total_matched`, `total_runner_matched`, `best_back_size` \
        and `best_lay_size` have been converted from GBP to AUD. The amounts given in \
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
    ) -> None:
        self.extractors = {**COLUMN_EXTRACTORS, **(extra_columns or {})}

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

    def _iter_rows(self, bz2_path: str | Path):
        trading = betfairlightweight.APIClient("username", "password")
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
                            runner_def = runners_dict.get(runner.selection_id)
                            yield self._extract_row(market_book, runner, runner_def)
            finally:
                os.remove(decompressed_path)

    # ---------------------------------------------------------------------------
    # Public API 
    # ---------------------------------------------------------------------------

    def load(self, bz2_path: str | Path) -> pd.DataFrame:
        """Load a single .bz2 market-stream file into a DataFrame."""
        rows = list(self._iter_rows(bz2_path))
        return pd.DataFrame(rows, columns=self.columns)

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
