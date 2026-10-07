"""
data/marketvisualiser.py
------------------------

Step through a Betfair market-stream CSV (as produced by streamloaders.py) tick by tick.

Usage:
    `python marketvisualiser.py path/to/market.csv [--max-columns 3]`

## Notes:
    - `publish_time` in the market CSV must be in chronological (non-decreasing) order
"""
from __future__ import annotations

import argparse
import bisect
import os
import msvcrt

import pandas as pd

# ---------------------------------------------------------------------------
# Constants and output formatting
# ---------------------------------------------------------------------------

GRAY = "\033[90m"
WHITE = "\033[97m"
PURPLE = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"
GOLD = "\033[38;5;220m"
RESET = "\033[0m"

FIELD_WIDTH = 8
NAME_WIDTH = 20
MAX_DEPTH = 3


def read_key() -> str:
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):
        ch2 = msvcrt.getch()
        return {b"M": "right", b"K": "left"}.get(ch2, "") 
    if ch == b"\r":
        return "right"
    if ch == b" ":
        return "seek"
    if ch in (b"q", b"Q", b"\x1b"):
        return "quit"
    return ""


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")

# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def depth_columns(depth: int) -> tuple[str, str, str, str]:
    return (
        f"back_price_{depth}",
        f"back_size_{depth}",
        f"lay_price_{depth}",
        f"lay_size_{depth}",
    )


def detect_max_depth(columns, max_columns: int = MAX_DEPTH) -> int:
    columns = set(columns)
    depth = 1
    for d in range(2, max_columns + 1):
        if any(c in columns for c in depth_columns(d)):
            depth = d
    return depth


def load_ticks(csv_path: str, max_columns: int = MAX_DEPTH) -> tuple[list[pd.DataFrame], list[pd.Timestamp], int]:
    """Loads a CSV and returns lists of each tick and important values."""
    df = pd.read_csv(csv_path)
    df["publish_time"] = pd.to_datetime(df["publish_time"], utc=True, format="ISO8601")
    if "market_time" in df.columns:
        df["market_time"] = pd.to_datetime(df["market_time"], utc=True, format="ISO8601")
    df = df.sort_values("publish_time")

    max_depth = detect_max_depth(df.columns, max_columns)

    # Get a list of DataFrames, each DataFrame has all the rows with the same publish_time
    # in chronological order
    tick_dfs = [g for _, g in df.groupby("publish_time", sort=False)]

    # Get the list of `publish_time`s in chronological order
    tick_times = [pd.Timestamp(t["publish_time"].iat[0]) for t in tick_dfs]
    
    return tick_dfs, tick_times, max_depth


def apply_tick(state: dict, tick_df: pd.DataFrame, max_depth: int) -> None:
    """Mutate `state` in place with the rows of a single tick (one publish_time)."""
    for _, r in tick_df.iterrows():
        depths = {}
        for d in range(1, max_depth + 1):
            bp_col, bs_col, lp_col, ls_col = depth_columns(d)
            depths[d] = (
                r[bp_col] if bp_col in r else float("nan"),
                r[bs_col] if bs_col in r else float("nan"),
                r[lp_col] if lp_col in r else float("nan"),
                r[ls_col] if ls_col in r else float("nan"),
            )
        state[r["selection_name"]] = depths


def state_at(tick_dfs: list[pd.DataFrame], idx: int, max_depth: int) -> dict:
    """Build and return the `state` dictionary at the `idx`-th tick"""
    state: dict = {} # selection_name: {1: depth_columns(1), ...}
    apply_tick(state, tick_dfs[idx], max_depth)
    return state


def apply_runner_matched_tick(runner_matched: dict, tick_df: pd.DataFrame) -> None:
    """Mutate `runner_matched` in place with the total_runner_matched values of a 
    single tick, if that column exists in the CSV."""
    for _, r in tick_df.iterrows():
        trm = r["total_runner_matched"]
        if pd.notna(trm):
            runner_matched[r["selection_name"]] = trm


def runner_matched_at(tick_dfs: list[pd.DataFrame], idx: int) -> dict:
    """Build and return the `runner_matched` dictionary at the `idx`-th tick."""
    runner_matched: dict = {} # selection_name: total_runner_matched
    apply_runner_matched_tick(runner_matched, tick_dfs[idx])
    return runner_matched

# ---------------------------------------------------------------------------
# Helper functions for rendering
# ---------------------------------------------------------------------------

def trades_since(
    ticks: list[pd.DataFrame],
    prev_idx: int,
    idx: int,
    prev_state: dict,
    prev_runner_matched: dict,
) -> dict:
    """Return {selection_name: 'back' | 'lay' | 'both'} for runners that traded between the
    last-displayed tick and the tick about to be displayed.
    """
    # not moving forward
    if idx <= prev_idx:
        return {}

    trades: dict = {}

    # traverse through all rows between `prev_idx` and `idx`
    # `running` tells us when `total_runner_matched` increases
    running = dict(prev_runner_matched)
    for i in range(prev_idx + 1, idx + 1):
        for _, r in ticks[i].iterrows():
            name = r["selection_name"]
            trm = r["total_runner_matched"]
            if pd.isna(trm):
                continue
            prev_trm = running.get(name)
            if prev_trm is not None and trm > prev_trm:
                price = r["last_price_traded"]
                prev_back, _, prev_lay, _ = prev_state.get(name, {}).get(
                    1, (None, None, None, None)
                )
                if pd.isna(price) or pd.isna(prev_back) or pd.isna(prev_lay):
                    continue
                
                # back side got matched
                if price <= prev_back:
                    if trades.get(name) == "lay":
                        trades[name] = "both"
                    else:
                        trades[name] = "back"

                # lay side got matched
                elif price >= prev_lay:
                    if trades.get(name) == "back":
                        trades[name] = "both"
                    else:
                        trades[name] = "lay"
            running[name] = trm
    return trades


def seek_index(tick_times: list[pd.Timestamp], query: str) -> int | None: 
    """Resolve a user-typed query to a tick index.
    """
    query = query.strip()
    if not query:
        return None

    # user wishes to seek by index
    if query.isdigit():
        n = int(query)
        if 1 <= n <= len(tick_times):
            return n - 1
        return None

    ts0 = tick_times[0]

    # user gave an explicit date (and optionally a time)
    if "-" in query or "/" in query:
        try:
            target = pd.Timestamp(query)
        except (ValueError, TypeError):
            return None
        target = target.tz_localize(ts0.tzinfo) if target.tzinfo is None else target.tz_convert(ts0.tzinfo)
        j = bisect.bisect_left(tick_times, target)
        return min(j, len(tick_times) - 1)

    # user gave just a time (no date)
    # markets can span multiple days, scan each row in chronological order
    # and return the first one where a tick actually exists at or right after that time
    try:
        target_time = pd.Timestamp(f"1900-01-01 {query}").time()
    except (ValueError, TypeError):
        return None

    dates = []
    last_date = None
    for t in tick_times:
        d = t.date()
        if d != last_date:
            dates.append(d)
            last_date = d

    for d in dates:
        candidate = pd.Timestamp.combine(d, target_time).tz_localize(ts0.tzinfo)
        j = bisect.bisect_left(tick_times, candidate)
        if j < len(tick_times) and tick_times[j].date() == d:
            return j

    return None


def format_to_scheduled_off(delta: pd.Timedelta) -> str:
    """Format to_scheduled_off"""
    seconds = int(delta.total_seconds())
    if seconds < 0:
        return f"T+{-seconds}"
    if seconds < 240:
        return f"T-{seconds}"
    minutes = seconds // 60
    if minutes < 120:
        return f"T-{minutes}m"
    hours = seconds // 3600
    return f"T-{hours}h"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def cell(value, color: str) -> str:
    s = f"{value:>{FIELD_WIDTH}.2f}" if pd.notna(value) else " " * FIELD_WIDTH
    return f"{color}{s}{RESET}"

def render(
    ticks: list[pd.DataFrame],
    tick_times: list[pd.Timestamp],
    idx: int,
    state: dict,
    max_depth: int,
    trades: dict | None = None,
) -> None:
    trades = trades or {}
    clear_screen()
    ts = tick_times[idx]
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    ts_str = ts.isoformat(sep=" ", timespec="milliseconds")
    print(f"{GRAY}{ts_str}\t(UTC){RESET}")

    info_fields = [
        ticks[idx][col].iat[0]
        for col in ("market_name", "venue", "status")
        if col in ticks[idx].columns
    ]
    if "market_time" in ticks[idx].columns:
        market_time = ticks[idx]["market_time"].iat[0]
        if pd.notna(market_time):
            info_fields.append(format_to_scheduled_off(market_time - tick_times[idx]))
    if info_fields:
        print(f"{GRAY}" + "  |  ".join(str(f) for f in info_fields) + f"{RESET}")

    print(f"{GRAY}{'=' * 80}{RESET}")
    print()
    total_matched = ticks[idx]["total_matched"].iat[0]
    matched_s = f"{total_matched:,.2f}" if pd.notna(total_matched) else "0.00"
    print(f"{PURPLE}Matched: {matched_s}{RESET}")

    for name, depths in state.items():
        side = trades.get(name)
        gold_back = side == "both" or side == "back"
        gold_lay = side == "both" or side == "lay"

        # farthest depth on the outside, best (depth 1) in the middle
        back_price_cells = [
            cell(depths[d][0], GOLD if (gold_back) else BLUE)
            for d in range(max_depth, 0, -1)
        ]
        lay_price_cells = [
            cell(depths[d][2], GOLD if (gold_lay) else RED)
            for d in range(1, max_depth + 1)
        ]
        print(
            f"{RESET}{name[:NAME_WIDTH]:<{NAME_WIDTH}s} "
            + " ".join(back_price_cells)
            + " "
            + " ".join(lay_price_cells)
        )

        back_size_cells = [
            cell(depths[d][1], GOLD if (gold_back) else WHITE)
            for d in range(max_depth, 0, -1)
        ]
        lay_size_cells = [
            cell(depths[d][3], GOLD if (gold_lay) else WHITE)
            for d in range(1, max_depth + 1)
        ]
        print(
            f"{'':{NAME_WIDTH}s} "
            + " ".join(back_size_cells)
            + " "
            + " ".join(lay_size_cells)
        )
    print()
    print(f"{GRAY}{'=' * 80}{RESET}")
    print(f"{GRAY}[{idx + 1}/{len(ticks)}]  [←] prev   [→] next   [SPACE] seek   [q] quit{RESET}")

NECESSARY_COLUMNS = [
    "publish_time",
    "selection_name",
    "last_price_traded",
    "total_matched",
    "total_runner_matched",
    "back_price_1",
    "back_size_1",
    "lay_price_1",
    "lay_size_1",
]

def validate_csv(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    cols_not_in = []
    for col in NECESSARY_COLUMNS:
        if col not in df.columns:
            cols_not_in.append(col)
    return cols_not_in

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a market CSV file")
    parser.add_argument(
        "--max-columns",
        type=int,
        choices=range(1, MAX_DEPTH + 1),
        default=MAX_DEPTH,
        help=f"Maximum depth of book to display, 1–{MAX_DEPTH} (default: {MAX_DEPTH})",
    )
    args = parser.parse_args()

    if (cols_not_in := validate_csv(args.csv_path)):
        print(f"Columns not found in CSV: {cols_not_in}")

    ticks, tick_times, max_depth = load_ticks(args.csv_path, args.max_columns)
    if not ticks:
        print("No rows found in CSV.")
        return

    idx = 0
    state = state_at(ticks, idx, max_depth)
    runner_matched = runner_matched_at(ticks, idx)
    state_cache: list[dict | None] = [None] * len(ticks)
    runner_matched_cache: list[dict | None] = [None] * len(ticks)
    state_cache[idx] = dict(state)
    runner_matched_cache[idx] = dict(runner_matched)
    render(ticks, tick_times, idx, state, max_depth)

    while True:
        key = read_key()
        if key == "quit":
            break

        if key == "right" and idx + 1 < len(ticks):
            trades = trades_since(ticks, idx, idx + 1, state, runner_matched)
            idx += 1
            apply_tick(state, ticks[idx], max_depth)
            apply_runner_matched_tick(runner_matched, ticks[idx])
            state_cache[idx] = dict(state)
            runner_matched_cache[idx] = dict(runner_matched)
            render(ticks, tick_times, idx, state, max_depth, trades)

        elif key == "left" and idx > 0:
            idx -= 1
            if state_cache[idx] is not None:
                state = dict(state_cache[idx])
                runner_matched = dict(runner_matched_cache[idx])
            else:
                state = state_at(ticks, idx, max_depth)
                runner_matched = runner_matched_at(ticks, idx)
                state_cache[idx] = dict(state)
                runner_matched_cache[idx] = dict(runner_matched)
            render(ticks, tick_times, idx, state, max_depth)

        elif key == "seek":
            print(f"{GRAY}Seek to (tick # or [YYYY/MM/DD] HH:MM:SS[.mmm]): {RESET}", end="", flush=True)
            query = input()
            new_idx = seek_index(tick_times, query)
            if new_idx is None or new_idx == idx:
                render(ticks, tick_times, idx, state, max_depth)
            elif new_idx > idx:
                trades = trades_since(ticks, idx, new_idx, state, runner_matched)
                for i in range(idx + 1, new_idx + 1):
                    apply_tick(state, ticks[i], max_depth)
                    apply_runner_matched_tick(runner_matched, ticks[i])
                    state_cache[i] = dict(state)
                    runner_matched_cache[i] = dict(runner_matched)
                idx = new_idx
                render(ticks, tick_times, idx, state, max_depth, trades)
            else:
                idx = new_idx
                if state_cache[idx] is not None:
                    state = dict(state_cache[idx])
                    runner_matched = dict(runner_matched_cache[idx])
                else:
                    state = state_at(ticks, idx, max_depth)
                    runner_matched = runner_matched_at(ticks, idx)
                    state_cache[idx] = dict(state)
                    runner_matched_cache[idx] = dict(runner_matched)
                render(ticks, tick_times, idx, state, max_depth)


if __name__ == "__main__":
    main()