"""
scripts/build_dataset.py
------------------------
Convert raw Betfair PRO stream files (.bz2) into one Parquet file per market,
which is the format the backtester reads. Also writes race_no_lookup.json.

Files are processed in parallel, one per CPU core. If the run is stopped
part-way, just run it again: markets that already have a file are skipped.

Run from the project root:
    python scripts/build_dataset.py

Input  (raw):       data/raw/market_stream/<...>/PRO/<year>/<month>/<day>/<event_id>/<market_id>.bz2
Output (processed): data/processed/market_stream/<month>/<day>/<event_id>/<market_id>.parquet
                    data/processed/market_stream/race_no_lookup.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.streamloaders import ALL_COLUMNS, BetfairStreamLoader, race_key  # noqa: E402

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# The folder that directly contains the month folders (Oct/, Nov/, ...)
SOURCE_ROOT = ROOT / "data/raw/market_stream/2025_10_ProAUSGreyhounds/PRO/2025"
DEST_ROOT = ROOT / "data/processed/market_stream"
MARKET_TYPES = ["WIN", "PLACE", "MATCH_BET"]

# Which raw files to process, relative to SOURCE_ROOT: <month>/<day>/<event_id>/<market_id>.bz2
# Use "Oct/1/*/*.bz2" to test on a single day first.
PATTERN = "*/*/*/*.bz2"

# Number of files processed at once (default: one per CPU core).
# Lower it if your computer is too slow to use while this runs.
WORKERS = os.cpu_count() or 1

# Skip markets that already have a file (lets you resume an interrupted run).
# Set to False if you change the loader and want to rebuild everything.
SKIP_EXISTING = True

RACE_NO_RE = re.compile(r"R(\d+)")  # WIN market names look like "R1 330m Mdn"


# ---------------------------------------------------------------------------
# Worker: runs in a separate process, handles one raw file
# ---------------------------------------------------------------------------

_loader: Optional[BetfairStreamLoader] = None


def _init_worker() -> None:
    global _loader
    _loader = BetfairStreamLoader(columns=ALL_COLUMNS, market_type_filter=MARKET_TYPES)


def _race_info(frame: pd.DataFrame) -> tuple[str, str, Optional[int]]:
    """(race key, market type, race number) from a market's rows."""
    last = frame.iloc[-1]
    key = race_key(last["event_id"], last["market_time"])
    race_no = None
    if last["market_type"] == "WIN":
        match = RACE_NO_RE.search(str(last["market_name"]))
        if match:
            race_no = int(match.group(1))
    return key, last["market_type"], race_no


def process_file(src: Path) -> Optional[tuple[str, str, str, Optional[int]]]:
    """Convert one .bz2 file to Parquet. Returns (file name, race key, market type, race number),
    or None if the market was filtered out or empty."""
    dest = DEST_ROOT / src.relative_to(SOURCE_ROOT).with_suffix(".parquet")

    if SKIP_EXISTING and dest.exists():
        frame = pd.read_parquet(dest, columns=["event_id", "market_time", "market_type", "market_name"])
    else:
        frame = _loader.load(src)
        if len(frame) == 0:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename, so a stopped run never leaves a half-written file behind.
        tmp = dest.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, dest)

    if len(frame) == 0:
        return None
    return (src.name, *_race_info(frame))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    files = sorted(SOURCE_ROOT.glob(PATTERN))
    if not files:
        raise FileNotFoundError(f"No .bz2 files matching {PATTERN} under {SOURCE_ROOT}")
    print(f"Found {len(files):,} raw market files. Using {WORKERS} workers.")

    race_no_lookup: dict[str, int] = {}
    non_win: list[tuple[str, str]] = []
    start = time.time()

    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as pool:
        for i, result in enumerate(pool.map(process_file, files, chunksize=8), 1):
            if result is not None:
                name, key, market_type, race_no = result
                if race_no is not None:
                    race_no_lookup[key] = race_no
                elif market_type != "WIN":
                    non_win.append((name, key))

            if i % 200 == 0 or i == len(files):
                elapsed = time.time() - start
                remaining = elapsed / i * (len(files) - i)
                print(f"  {i:,} / {len(files):,}  |  {elapsed / 60:.0f} min elapsed, ~{remaining / 60:.0f} min left")

    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    with open(DEST_ROOT / "race_no_lookup.json", "w") as fp:
        json.dump(race_no_lookup, fp)

    unmatched = [(name, key) for name, key in non_win if key not in race_no_lookup]
    print(f"Done in {(time.time() - start) / 60:.1f} min. {len(race_no_lookup):,} races in lookup, "
          f"{len(unmatched)} non-WIN markets with no race number.")
    for name, key in unmatched[:20]:
        print(f"  no race number: {name} (key={key})")


if __name__ == "__main__":  # required for multiprocessing on Windows
    main()