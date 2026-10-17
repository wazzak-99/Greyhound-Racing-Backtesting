"""
scripts/run_backtest.py
-----------------------
Run any strategy over the processed stream data, prints progress as it goes, 
saves the results and prints summary statistics once completed.

Run from the project root (after build_dataset.py):

    python scripts/run_backtest.py --data Jan --strategy strategies/asgmv1.py --class ASGMStrategy
    python scripts/run_backtest.py --data Feb Mar --strategy strategies/asgmv1.py --class ASGMStrategy
    python scripts/run_backtest.py --data "Feb/1/*/*.parquet" --strategy strategies.asgmv1 --class ASGMStrategy
    python scripts/run_backtest.py --data Feb --strategy strategies/asgmv1.py --class ASGMStrategy \
        --params gamma=2.5 adverse_scale=0.5 --out research/results/gamma_2.5

Required:
    --data        one or more folders or glob patterns, relative to data/processed/market_stream
                  (a folder means every .parquet file inside it, at any depth)
    --strategy    the strategy file or module, e.g. strategies/asmodel.py or strategies.asmodel
    --class       the strategy class inside that file, e.g. ASGMStrategy

Optional:
    --params      constructor arguments as key=value, e.g. gamma=2.5 adverse_keep_reducing=False
    --commission  Betfair commission rate                                            default: 0.04
    --latency     order latency in seconds                                           default: 0.2
    --out         folder for the CSVs              default: research/results/<class name>
    --print-every print a progress line every this many files                        default: 25

Outputs (in --out): bets.csv, markets.csv, fills.csv, inventory.csv and run_config.json
(the exact settings used, so every result can be reproduced).
"""
from __future__ import annotations

import argparse
import ast
import glob
import importlib
import importlib.util
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from backtest.context import MarketType  # noqa: E402
from backtest.datasources import MarketReplay, StreamMarketSource  # noqa: E402
from backtest.engine import Engine  # noqa: E402
from backtest.execution import QueueExecutionModel  # noqa: E402
from backtest.ledger import Ledger  # noqa: E402
from backtest.staking import FlatStake  # noqa: E402
from data.streamloaders import load_stream, read_market_type  # noqa: E402

PROCESSED = ROOT / "data/processed/market_stream"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest a strategy on Betfair stream data.")
    p.add_argument("--data", nargs="+", required=True,
                   help="folders or glob patterns, relative to data/processed/market_stream")
    p.add_argument("--strategy", required=True,
                   help="strategy file (e.g. strategies/asmodel.py) or module (e.g. strategies.asmodel)")
    p.add_argument("--class", dest="class_name", required=True,
                   help="name of the strategy class in that file")
    p.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE",
                   help="constructor arguments, e.g. gamma=2.5 adverse_scale=0.5")
    p.add_argument("--commission", type=float, default=0.04)
    p.add_argument("--latency", type=float, default=0.2)
    p.add_argument("--out", default=None, help="output folder (default research/results/<class>)")
    p.add_argument("--print-every", type=int, default=25)
    return p.parse_args()


def parse_params(items: list[str]) -> dict:
    """['gamma=2.5', 'adverse_keep_reducing=False'] -> {'gamma': 2.5, 'adverse_keep_reducing': False}"""
    params = {}
    for item in items:
        if "=" not in item:
            sys.exit(f"--params expects key=value, got '{item}'")
        key, raw = item.split("=", 1)
        try:
            params[key] = ast.literal_eval(raw)   # numbers, True/False/None, lists, tuples
        except (ValueError, SyntaxError):
            params[key] = raw                     # anything else stays a string, e.g. mode=greenup
    return params


def load_strategy_class(strategy: str, class_name: str):
    """Import `class_name` from a file path (strategies/asmodel.py) or a module (strategies.asmodel)."""
    if strategy.endswith(".py"):
        path = Path(strategy)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            sys.exit(f"Strategy file not found: {path}")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(strategy)
    if not hasattr(module, class_name):
        classes = [n for n in dir(module) if isinstance(getattr(module, n), type)]
        sys.exit(f"No class '{class_name}' in {strategy}. Classes found: {', '.join(classes)}")
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def find_files(patterns: list[str]) -> list[str]:
    """Expand folders and glob patterns into a de-duplicated list of .parquet files."""
    files: set[str] = set()
    for pat in patterns:
        path = Path(pat) if Path(pat).is_absolute() else PROCESSED / pat
        if path.is_dir():
            files.update(glob.glob(str(path / "**" / "*.parquet"), recursive=True))
        else:
            files.update(glob.glob(str(path), recursive=True))
    return list(files)


def _natural_key(path: str):
    """Sort 'Jan/2/...' before 'Jan/10/...' (plain string sorting would not)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path)]


def _start_time(path: str):
    """Scheduled race start (market_time) from the file's first row, read without loading the file."""
    try:
        table = pq.ParquetFile(path).read_row_group(0, columns=["market_time"])
        return pd.Timestamp(table.column(0)[0].as_py())
    except Exception:
        return None


def sort_chronologically(files: list[str]) -> list[str]:
    """Order files by scheduled race start. Files whose start time can't be read go last, in path order."""
    print(f"Sorting {len(files)} files by race start time...", flush=True)
    timed, untimed = [], []
    for f in files:
        t = _start_time(f)
        (timed if t is not None else untimed).append((t, f))
    timed.sort(key=lambda tf: (tf[0], _natural_key(tf[1])))
    untimed.sort(key=lambda tf: _natural_key(tf[1]))
    if untimed:
        print(f"  {len(untimed)} files had no readable start time - run last, in folder order")
    return [f for _, f in timed] + [f for _, f in untimed]


# ---------------------------------------------------------------------------
# Data source with progress printing
# ---------------------------------------------------------------------------

class ProgressSource(StreamMarketSource):
    """StreamMarketSource over a fixed, ordered file list that prints progress as it goes."""

    def __init__(self, files: list[str], *args, ledger: Ledger, strategy, print_every: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.files = files
        self.ledger = ledger
        self.strategy = strategy
        self.print_every = print_every
        self.done = 0
        self.races = 0
        self.start = time.time()

    def markets(self) -> Iterator[MarketReplay]:
        total = len(self.files)
        print(f"{total} files to scan\n")
        for f in self.files:
            self.done += 1
            if self.market_type_filter is None or read_market_type(f) in self.market_type_filter:
                replay = self._build_replay(load_stream(f))
                if replay is not None:
                    self.races += 1
                    yield replay
            if self.done % self.print_every == 0 or self.done == total:
                self._print_progress(total)

    def _print_progress(self, total: int) -> None:
        elapsed = time.time() - self.start
        eta = elapsed / self.done * (total - self.done)
        net = sum(m.net_pnl for m in self.ledger.markets)
        traded = len(self.ledger.markets)
        fills = len(getattr(self.strategy, "fill_log", []))
        print(f"[{self.done:>6}/{total}] {100 * self.done / total:5.1f}%  "
              f"elapsed {_fmt(elapsed)}  ETA {_fmt(eta)}  |  "
              f"races {self.races}  traded {traded}  fills {fills}  "
              f"net PnL {net:+.2f}", flush=True)


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(ledger: Ledger, strategy, commission: float) -> None:
    m = ledger.markets_frame()
    if m.empty:
        print("No markets were traded.")
        return
    bets = ledger.bets_frame()
    fills = pd.DataFrame(getattr(strategy, "fill_log", []))
    x, g = m["net_pnl"], m["gross_pnl"]
    n = len(x)
    sd = x.std(ddof=1)
    t = x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    cum = x.cumsum()   # markets are in chronological order, so this is a real drawdown
    print("\n================ SUMMARY ================")
    print(f"markets traded      {n}")
    print(f"gross PnL           {g.sum():+.2f}")
    print(f"commission          {m['commission'].sum():.2f}  (rate {commission:.0%})")
    print(f"net PnL             {x.sum():+.2f}")
    print(f"mean / median       {x.mean():+.4f} / {x.median():+.4f} per market")
    print(f"std dev             {sd:.4f} per market")
    print(f"t-stat (mean = 0)   {t:+.2f}")
    try:
        from scipy import stats
        print(f"p-value (t-test)    {stats.ttest_1samp(x, 0).pvalue:.4g}")
        nz = x[x != 0]
        if len(nz) > 10:
            print(f"Wilcoxon p          {stats.wilcoxon(nz).pvalue:.4g}")
    except ImportError:
        pass
    print(f"markets profitable  {(x > 0).mean():.0%}")
    print(f"worst / best market {x.min():+.2f} / {x.max():+.2f}")
    print(f"max drawdown        {(cum.cummax() - cum).max():.2f}")
    staked = bets["stake"].sum()
    if staked > 0:
        print(f"staked              {staked:.2f}   ROI {x.sum() / staked:+.4%}")
    if not fills.empty:
        if "tag" in fills:
            print(f"fills               {len(fills)}  {fills['tag'].value_counts().to_dict()}")
        if "q_after" in fills:
            print(f"max |q|             {fills['q_after'].abs().max():.2f}")
    print("=========================================")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    params = parse_params(args.params)
    strategy_cls = load_strategy_class(args.strategy, args.class_name)

    files = find_files(args.data)
    if not files:
        sys.exit(f"No .parquet files found for {args.data} under {PROCESSED}")
    files = sort_chronologically(files)

    lookup_path = PROCESSED / "race_no_lookup.json"
    race_no_lookup = json.loads(lookup_path.read_text()) if lookup_path.exists() else {}

    print(f"Strategy  {args.class_name} from {args.strategy}  {params or ''}")
    print(f"Data      {' '.join(args.data)}   commission {args.commission:.0%}   latency {args.latency}s")
    strategy = strategy_cls(**params)
    ledger = Ledger(commission_rate=args.commission)
    source = ProgressSource(files, pattern=str(PROCESSED), race_no_lookup=race_no_lookup,
                            market_type_filter={MarketType.WIN}, ledger=ledger, strategy=strategy,
                            print_every=args.print_every)
    engine = Engine(source, strategy, QueueExecutionModel(latency=args.latency, ladder_depth=3),
                    ledger, FlatStake(1.0), starting_bankroll=1000.0)

    started = datetime.now()
    try:
        engine.run()
        finished = True
    except KeyboardInterrupt:
        print("\nStopped early - saving the markets finished so far.")
        finished = False

    out = Path(args.out) if args.out else ROOT / "research" / "results" / args.class_name
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ledger.bets_frame().to_csv(out / "bets.csv", index=False)
    ledger.markets_frame().to_csv(out / "markets.csv", index=False)
    pd.DataFrame(getattr(strategy, "fill_log", [])).to_csv(out / "fills.csv", index=False)
    pd.DataFrame(getattr(strategy, "market_log", [])).to_csv(out / "inventory.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "strategy": args.strategy, "class": args.class_name, "params": params,
        "data": args.data, "files": len(files), "files_done": source.done,
        "finished": finished, "commission": args.commission, "latency": args.latency,
        "started": started.isoformat(timespec="seconds"),
    }, indent=2, default=str))

    summarise(ledger, strategy, args.commission)
    print(f"Saved bets, markets, fills, inventory and run_config to {out}")


if __name__ == "__main__":
    main()