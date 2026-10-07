"""
scripts/build_dataset.py
------------------------
Convert raw Betfair PRO stream files (.bz2) into one CSV per market, which is
the format the backtester reads. Also writes race_no_lookup.json.

Run from the project root:
    python scripts/build_dataset.py

Input  (raw):       data/raw/market_stream/<...>/PRO/<year>/<month>/<day>/<event_id>/<market_id>.bz2
Output (processed): data/processed/market_stream/<month>/<day>/<event_id>/<market_id>.csv
                    data/processed/market_stream/race_no_lookup.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.streamloaders import ALL_COLUMNS, BetfairStreamLoader, race_key  # noqa: E402

# ---------------------------------------------------------------------------
# Settings: change SOURCE_ROOT if you download a different month/year
# ---------------------------------------------------------------------------

# The folder that directly contains the month folders (Oct/, Nov/, ...)
SOURCE_ROOT = ROOT / "data/raw/market_stream/2025_10_ProAUSGreyhounds/PRO/2025"
DEST_ROOT = ROOT / "data/processed/market_stream"
MARKET_TYPES = ["WIN", "PLACE", "MATCH_BET"]

RACE_NO_RE = re.compile(r"R(\d+)")  # WIN market names look like "R1 330m Mdn"


def main() -> None:
    loader = BetfairStreamLoader(columns=ALL_COLUMNS, market_type_filter=MARKET_TYPES)

    files = sorted(SOURCE_ROOT.glob("*/*/*/*.bz2"))  # <month>/<day>/<event_id>/<market_id>.bz2
    if not files:
        raise FileNotFoundError(f"No .bz2 files found under {SOURCE_ROOT}")
    print(f"Found {len(files):,} raw market files")

    race_no_lookup: dict[str, int] = {}
    non_win_keys: list[tuple[Path, str]] = []

    for i, src in enumerate(files, 1):
        frame = loader.load(src)
        if len(frame) == 0:
            continue

        dest = DEST_ROOT / src.relative_to(SOURCE_ROOT).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(dest, index=False)

        # Race number only appears in WIN market names, so we look it up by race
        # (event_id + start time) and reuse it for that race's PLACE / MATCH_BET markets.
        key = race_key(frame["event_id"].iloc[-1], frame["market_time"].iloc[-1])
        if frame["market_type"].iloc[-1] == "WIN":
            match = RACE_NO_RE.search(frame["market_name"].iloc[-1])
            if match:
                race_no_lookup[key] = int(match.group(1))
        else:
            non_win_keys.append((src, key))

        if i % 500 == 0:
            print(f"  processed {i:,} / {len(files):,}")

    with open(DEST_ROOT / "race_no_lookup.json", "w") as fp:
        json.dump(race_no_lookup, fp)

    unmatched = [(src, key) for src, key in non_win_keys if key not in race_no_lookup]
    print(f"Done. {len(race_no_lookup):,} races in lookup, {len(unmatched)} non-WIN markets with no race number.")
    for src, key in unmatched[:20]:
        print(f"  no race number: {src.name} (key={key})")


if __name__ == "__main__":
    main()
