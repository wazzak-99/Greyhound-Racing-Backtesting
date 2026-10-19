"""
scripts/pnl_breakdown.py
------------------------
Fill-level decomposition of a backtest's gross profit and loss (before commission).

For every runner we traded, the gross PnL is split into five parts that add up exactly:

    spread capture   what each passive (quote) fill earned against the mid just before it
    green-up cost    the same for green-up fills (crossing the spread to exit, usually negative)
    adverse move     mid moves against inventory picked up in the last `--horizon` seconds
    inventory drift  mid moves on older inventory
    settlement       inventory still open at the off, settled against the result

Prices are implied probabilities, inventory is in payout units ($ won if the runner wins).

Run from the project root:

    python scripts/pnl_breakdown.py --fills research/results/ASGMStrategy/fills.csv --data Oct
    python scripts/pnl_breakdown.py --fills research/results/as_baseline/fills.csv --data Oct \
        --markets research/results/as_baseline/markets.csv --out research/results/as_baseline/breakdown.csv

--markets is optional: if given, the decomposition is checked against the backtest's gross PnL.
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
COLUMNS = ["publish_time", "selection_id", "back_price_1", "lay_price_1", "status", "runner_status"]
PARTS = ["spread", "greenup", "adverse", "drift", "settlement"]


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fill-level PnL decomposition of a backtest.")
    p.add_argument("--fills", required=True, help="fills.csv from run_backtest.py")
    p.add_argument("--data", nargs="+", required=True,
                   help="folders or glob patterns of the market files, relative to data/processed/market_stream")
    p.add_argument("--markets", default=None, help="markets.csv from the same run, to check the totals")
    p.add_argument("--horizon", type=float, default=10.0,
                   help="seconds after a fill during which mid moves count as adverse selection")
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
                      horizon_ns: int) -> dict[str, float]:
    """Decompose one runner's gross PnL. `book` holds this runner's mid at every update."""
    T = book["t"].to_numpy()
    M = book["mid"].to_numpy()
    N = int(np.searchsorted(T, last_open, side="right")) - 1   # last update before the off

    # Which update each fill happened on, and the mid just before it
    k = np.searchsorted(T, fills["t"].to_numpy(), side="right") - 1
    sign = np.where(fills["side"].to_numpy() == "BACK", 1.0, -1.0)
    units = fills["stake"].to_numpy() * fills["price"].to_numpy()
    edge = sign * units * (M[np.maximum(k - 1, 0)] - 1.0 / fills["price"].to_numpy())
    is_quote = fills["tag"].to_numpy() == "quote"

    # Inventory after each update, and the mid change into each update
    dq = np.zeros(len(T))
    np.add.at(dq, k, sign * units)
    q = np.cumsum(dq)
    q_before = np.concatenate([[0.0], q[:-1]])
    dm = np.concatenate([[0.0], np.diff(M)])

    # Inventory picked up within `horizon` of each update counts as "recent"
    j = np.searchsorted(T, T - horizon_ns, side="right") - 1
    q_old = np.where(j >= 0, q[np.maximum(j, 0)], 0.0)
    recent = q_before - q_old

    s = slice(1, N + 1)
    return {
        "spread": float(edge[is_quote].sum()),
        "greenup": float(edge[~is_quote].sum()),
        "adverse": float((dq[s] * dm[s]).sum() + (recent[s] * dm[s]).sum()),
        "drift": float(((q_before[s] - recent[s]) * dm[s]).sum()),
        "settlement": float(q[N] * ((1.0 if won else 0.0) - M[N])),
    }


def market_breakdown(path: str, fills: pd.DataFrame, horizon: float) -> dict[str, float]:
    """Sum the runner breakdowns for one market."""
    d = pd.read_parquet(path, columns=COLUMNS)
    d["t"] = d["publish_time"].astype("datetime64[ns, UTC]").astype("int64")   # nanoseconds
    d["mid"] = 0.5 * (1.0 / d["back_price_1"] + 1.0 / d["lay_price_1"])
    last_open = d.loc[d["status"] == "OPEN", "t"].max()
    winners = set(d.loc[d["runner_status"] == "WINNER", "selection_id"])

    out = dict.fromkeys(PARTS, 0.0)
    for sel, f in fills.groupby("selection_id"):
        book = d.loc[d["selection_id"] == sel, ["t", "mid"]].drop_duplicates("t", keep="last")
        book["mid"] = book["mid"].ffill().bfill()
        if book.empty or book["mid"].isna().all():
            continue
        parts = _runner_breakdown(book.reset_index(drop=True), f.sort_values("t"), last_open,
                                  sel in winners, int(horizon * 1e9))
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
        rows.append({"market_id": mid, **market_breakdown(files[mid], f, args.horizon)})
        if i % 250 == 0 or i == len(groups):
            print(f"[{i:>5}/{len(groups)}] markets decomposed", flush=True)
    if missing:
        print(f"{missing} markets had no matching market file and were skipped")

    b = pd.DataFrame(rows)
    b["total"] = b[PARTS].sum(axis=1)
    n = len(b)

    print("\n=========== GROSS PnL BREAKDOWN ===========")
    print(f"markets             {n}   (adverse horizon {args.horizon:g}s)")
    print(f"{'':20}{'total':>10}{'per market':>14}")
    labels = {"spread": "spread capture", "greenup": "green-up cost", "adverse": "adverse move",
              "drift": "inventory drift", "settlement": "settlement", "total": "gross PnL"}
    for p in PARTS + ["total"]:
        print(f"{labels[p]:20}{b[p].sum():>+10.2f}{b[p].mean():>+14.4f}")
    if b["spread"].sum() > 0:
        print(f"\nadverse move = {-b['adverse'].sum() / b['spread'].sum():.0%} of spread capture")

    if args.markets:
        m = pd.read_csv(args.markets, dtype={"market_id": str}).set_index("market_id")
        common = b.set_index("market_id").join(m["gross_pnl"], how="inner")
        resid = common["total"] - common["gross_pnl"]
        print(f"check vs backtest gross PnL: {common['gross_pnl'].sum():+.2f}  "
              f"(difference {resid.sum():+.2f}; from scratchings, voids and rounding)")
    print("===========================================")

    if args.out:
        b.to_csv(args.out, index=False)
        print(f"Saved per-market breakdown to {args.out}")


if __name__ == "__main__":
    sys.exit(main())