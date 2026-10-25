"""
scripts/pnl_breakdown.py
------------------------
For every runner we traded, the gross PnL is split into parts that add up exactly to the
backtest's own settlement:

1. spread capture: how much we made from spread from getting filled;
2. adverse selection: how much the price moved after we got filled;
3. liquidation cost: at T, how much we lost crossing the spread in order to flatten our position;
4. settlement: any inventory still open at the jump, settled against the race result, 
    plus payout adjustments from Betfair reduction factors when another runner was scratched. 

Run from the project root:

    python scripts/pnl_breakdown.py --fills research/results/ASGMStrategy/fills.csv --data Oct
    python scripts/pnl_breakdown.py --fills research/results/as_baseline/fills.csv --data Oct \
        --out research/results/as_baseline/breakdown.csv

"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/market_stream"
COLUMNS = ["publish_time", "selection_id", "back_price_1", "lay_price_1", "status", "runner_status",
           "adjustment_factor"]
PARTS = ["spread", "liquidation", "adverse", "settlement"]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill-level PnL decomposition of a backtest.")
    p.add_argument("--fills", required=True, help="fills.csv from run_backtest.py")
    p.add_argument("--data", nargs="+", required=True,
                   help="folders or glob patterns of the market files, relative to data/processed/market_stream")
    p.add_argument("--out", default=None, help="where to save the per-race breakdown (CSV)")
    return p.parse_args()


def find_files(patterns: list[str]) -> dict[str, str]:
    """market_id -> parquet path, for every file matching the folders or patterns."""
    files = {}
    for pat in patterns:
        path = Path(pat) if Path(pat).is_absolute() else PROCESSED / pat
        found = glob.glob(str(path / "**" / "*.parquet"), recursive=True) if path.is_dir() \
            else glob.glob(str(path), recursive=True)
        files.update({Path(f).stem: f for f in found})
    return files


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def _runner_breakdown(book: pd.DataFrame, fills: pd.DataFrame, last_open: int, won: bool,
                      removals: list[tuple[int, float]]) -> dict[str, float]:
    """Decompose one runner's gross PnL. `book` holds this runner's mid at every update;
    `removals` lists (time removed, reduction factor %) for every other runner withdrawn."""
    T = book["t"].to_numpy()
    M = book["mid"].to_numpy()
    N = int(np.searchsorted(T, last_open, side="right")) - 1   # last update before the off

    # Which update each fill happened on, and the mid just before it
    k = np.maximum(np.searchsorted(T, fills["t"].to_numpy(), side="right") - 1, 0)
    sign = np.where(fills["side"].to_numpy() == "BACK", 1.0, -1.0)
    price = fills["price"].to_numpy()
    stake = fills["stake"].to_numpy()
    units = stake * price
    edge = sign * units * (M[np.maximum(k - 1, 0)] - 1.0 / price)
    is_quote = fills["tag"].to_numpy() == "quote"

    # Inventory after each update, and the mid change into each update
    dq = np.zeros(len(T))
    np.add.at(dq, k, sign * units)
    q = np.cumsum(dq)
    dm = np.concatenate([[0.0], np.diff(M)])
    s = slice(1, N + 1)

    # Reduction factors: a runner withdrawn after the bet was matched shortens its price,
    # p' = 1 + (p - 1) * (1 - factor / 100) per withdrawal. It only changes the payout if we win.
    scratch = 0.0
    if won and removals:
        adj = price.copy()
        for t_removed, factor in removals:
            later = fills["t"].to_numpy() < t_removed
            adj = np.where(later, 1 + (adj - 1) * (1 - factor / 100.0), adj)
        scratch = float((sign * stake * (adj - price)).sum())

    return {
        "spread": float(edge[is_quote].sum()),
        "liquidation": float(edge[~is_quote].sum()),
        "adverse": float((q[s] * dm[s]).sum()),        # every move against held inventory, fill to jump
        "settlement": float(q[N] * ((1.0 if won else 0.0) - M[N])) + scratch,   # incl. reduction factors
    }


def market_breakdown(path: str, fills: pd.DataFrame) -> dict[str, float]:
    """Sum the runner breakdowns for one market."""
    d = pd.read_parquet(path, columns=COLUMNS)
    d["t"] = d["publish_time"].astype("datetime64[ns, UTC]").astype("int64")   # nanoseconds
    d["mid"] = 0.5 * (1.0 / d["back_price_1"] + 1.0 / d["lay_price_1"])
    last_open = d.loc[d["status"] == "OPEN", "t"].max()
    winners = set(d.loc[d["runner_status"] == "WINNER", "selection_id"])

    # Withdrawn runners: when each was first shown as removed, and its reduction factor
    gone = d[d["runner_status"] == "REMOVED"].sort_values("t").groupby("selection_id").first()
    removed_at = {int(sel): (int(r["t"]), float(r["adjustment_factor"]) if pd.notna(r["adjustment_factor"]) else 0.0)
                  for sel, r in gone.iterrows()}

    out = dict.fromkeys(PARTS, 0.0)
    for sel, f in fills.groupby("selection_id"):
        if int(sel) in removed_at:
            continue   # bets on a withdrawn runner are void
        book = d.loc[d["selection_id"] == sel, ["t", "mid"]].drop_duplicates("t", keep="last")
        book["mid"] = book["mid"].ffill().bfill()
        if book.empty or book["mid"].isna().all():
            continue
        others = [v for s_, v in removed_at.items() if s_ != int(sel)]
        parts = _runner_breakdown(book.reset_index(drop=True), f.sort_values("t"), last_open,
                                  sel in winners, others)
        for p in PARTS:
            out[p] += parts[p]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    fills = pd.read_csv(args.fills, dtype={"market_id": str})
    fills["t"] = (pd.to_datetime(fills["publish_time"], format="ISO8601", utc=True)
                  .astype("datetime64[ns, UTC]").astype("int64"))                 # nanoseconds
    files = find_files(args.data)

    rows, missing = [], 0
    groups = list(fills.groupby("market_id"))
    for i, (mid, f) in enumerate(groups, 1):
        if mid not in files:
            missing += 1
            continue
        rows.append({"market_id": mid, **market_breakdown(files[mid], f)})
        if i % 250 == 0 or i == len(groups):
            print(f"[{i:>5}/{len(groups)}] markets decomposed", flush=True)
    if missing:
        print(f"{missing} markets had no matching market file and were skipped")

    b = pd.DataFrame(rows)
    b["total"] = b[PARTS].sum(axis=1)
    n = len(b)

    print("\n=========== GROSS PnL BREAKDOWN ===========")
    print(f"markets             {n}")
    print(f"{'':20}{'total':>10}{'per market':>14}")
    labels = {"spread": "spread capture", "liquidation": "liquidation cost", "adverse": "adverse selection",
              "settlement": "settlement", "total": "gross PnL"}
    for p in PARTS + ["total"]:
        print(f"{labels[p]:20}{b[p].sum():>+10.2f}{b[p].mean():>+14.4f}")
    if b["spread"].sum() > 0:
        print(f"\nadverse selection = {-b['adverse'].sum() / b['spread'].sum():.0%} of spread capture")
    print("===========================================")

    if args.out:
        b.to_csv(args.out, index=False)
        print(f"Saved per-market breakdown to {args.out}")


if __name__ == "__main__":
    sys.exit(main())