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
    --window      only replay the last this many seconds before the scheduled off    default: 400
                  (must be longer than the strategy's start time; 0 = replay everything)
    --workers     worker processes (0 = all cores, 1 = one process with live progress) default: 0

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
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed 
from datetime import datetime
from functools import lru_cache, partial
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
    p.add_argument("--workers", type=int, default=0,
                   help="number of worker processes to use (0 = all cores, 1 = disable parallelism)")
    p.add_argument("--window", type=float, default=400.0,
                   help="seconds before the scheduled off to replay (0 = everything); must exceed the "
                        "strategy's start time")
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


@lru_cache(maxsize=None)
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


def trim_to_window(df: pd.DataFrame, window: float) -> pd.DataFrame:
    """Drop updates more than `window` seconds before the scheduled off. Nothing a strategy does
    before its start time is affected, but the market is much quicker to replay. Runners removed
    earlier are still marked removed in the rows that remain, so settlement is unchanged."""
    if not window or window <= 0 or df.empty:
        return df
    # Earliest scheduled time: if a race is put back, the strategy may already have started quoting
    # against the original time, so keep everything from `window` seconds before that
    cutoff = df["market_time"].min() - pd.Timedelta(seconds=window)
    return df[df["publish_time"] >= cutoff]


class _SingleReplaySource:
    """Minimal MarketSource for a single replay so one file can be evaluated in a worker process."""

    def __init__(self, replay):
        self.replay = replay

    def markets(self):
        yield self.replay


def _run_market_file(path: str, strategy_ref: str, class_name: str, params: dict,
                     commission: float, latency: float,
                     race_no_lookup: dict[str, int], window: float = 0.0) -> dict:
    """Backtest one parquet file in a separate process and return aggregated results."""
    empty = {"bets": [], "markets": [], "fills": [], "inventory": [], "files_done": 0, "start": None}
    if read_market_type(path) != MarketType.WIN.value:
        return empty   # WIN markets only (same as the single-process path)
    strategy_cls = load_strategy_class(strategy_ref, class_name)
    strategy = strategy_cls(**params)
    df = trim_to_window(load_stream(path), window)
    if df.empty:
        return empty
    replay = StreamMarketSource(pattern=path, race_no_lookup=race_no_lookup,
                               market_type_filter={MarketType.WIN})._build_replay(df)
    if replay is None:
        return empty

    ledger = Ledger(commission_rate=commission)
    engine = Engine(_SingleReplaySource(replay), strategy,
                    QueueExecutionModel(latency=latency, ladder_depth=3),
                    ledger, FlatStake(1.0), starting_bankroll=1000.0)
    engine.run()

    return {
        "bets": [vars(b) for b in ledger.bets],
        "markets": [vars(m) for m in ledger.markets],
        "fills": list(getattr(strategy, "fill_log", [])),
        "inventory": list(getattr(strategy, "market_log", [])),
        "files_done": 1,
        "start": df["market_time"].iloc[-1],   # used to put the results in race order afterwards
    }


def _workers_for(args) -> int:
    if args.workers is None or args.workers <= 0:
        return os.cpu_count() or 1
    return args.workers


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

    def __init__(self, files: list[str], *args, ledger: Ledger, strategy, print_every: int,
                 window: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
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
                df = trim_to_window(load_stream(f), self.window)
                replay = self._build_replay(df) if not df.empty else None
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

def summarise(ledger: Ledger | None, strategy, commission: float,
              bets: pd.DataFrame | None = None,
              markets: pd.DataFrame | None = None,
              fills: pd.DataFrame | None = None) -> str:
    summary_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg)
        summary_lines.append(str(msg))

    if markets is None or bets is None:
        if ledger is None:
            raise ValueError("Either a ledger or markets/bets frames are required.")
        markets = ledger.markets_frame()
        bets = ledger.bets_frame()
    if fills is None:
        fills = pd.DataFrame(getattr(strategy, "fill_log", []))
    if markets.empty:
        log("No markets were traded.")
        return "\n".join(summary_lines)
    x, g = markets["net_pnl"], markets["gross_pnl"]
    n = len(x)
    sd = x.std(ddof=1)
    t = x.mean() / (sd / np.sqrt(n)) if n > 1 and sd > 0 else float("nan")
    cum = x.cumsum()   # markets are in chronological order, so this is a real drawdown
    log("\n================ SUMMARY ================")
    log(f"markets traded      {n}")
    log(f"gross PnL           {g.sum():+.2f}")
    log(f"commission          {markets['commission'].sum():.2f}  (rate {commission:.0%})")
    log(f"net PnL             {x.sum():+.2f}")
    log(f"mean / median       {x.mean():+.4f} / {x.median():+.4f} per market")
    log(f"std dev             {sd:.4f} per market")
    log(f"t-stat (mean = 0)   {t:+.2f}")
    try:
        from scipy import stats
        log(f"p-value (t-test)    {stats.ttest_1samp(x, 0).pvalue:.4g}")
        nz = x[x != 0]
        if len(nz) > 10:
            log(f"Wilcoxon p          {stats.wilcoxon(nz).pvalue:.4g}")
    except ImportError:
        pass
    log(f"markets profitable  {(x > 0).mean():.0%}")
    log(f"worst / best market {x.min():+.2f} / {x.max():+.2f}")
    log(f"max drawdown        {(cum.cummax() - cum).max():.2f}")
    staked = bets["stake"].sum()
    if staked > 0:
        log(f"staked              {staked:.2f}   ROI {x.sum() / staked:+.4%}")
    if not fills.empty:
        if "tag" in fills:
            log(f"fills               {len(fills)}  {fills['tag'].value_counts().to_dict()}")
        if "q_after" in fills:
            log(f"max |q|             {fills['q_after'].abs().max():.2f}")
    log("=========================================")
    return "\n".join(summary_lines)


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
    workers = _workers_for(args)
    if workers == 1 or len(files) == 1:
        files = sort_chronologically(files)   # parallel runs sort their results afterwards instead
    else:
        files.sort(key=_natural_key)

    lookup_path = PROCESSED / "race_no_lookup.json"
    race_no_lookup = json.loads(lookup_path.read_text()) if lookup_path.exists() else {}

    print(f"Strategy  {args.class_name} from {args.strategy}  {params or ''}")
    window_note = f"   window {args.window:g}s before the off" if args.window else "   full market replayed"
    print(f"Data      {' '.join(args.data)}   commission {args.commission:.0%}   latency {args.latency}s{window_note}")

    started = datetime.now()

    if workers == 1 or len(files) == 1:
        strategy = strategy_cls(**params)
        ledger = Ledger(commission_rate=args.commission)
        source = ProgressSource(files, pattern=str(PROCESSED), race_no_lookup=race_no_lookup,
                                market_type_filter={MarketType.WIN}, ledger=ledger, strategy=strategy,
                                print_every=args.print_every, window=args.window)
        engine = Engine(source, strategy, QueueExecutionModel(latency=args.latency, ladder_depth=3),
                        ledger, FlatStake(1.0), starting_bankroll=1000.0)

        try:
            engine.run()
            finished = True
        except KeyboardInterrupt:
            print("\nStopped early - saving the markets finished so far.")
            finished = False

        bets_df = ledger.bets_frame()
        markets_df = ledger.markets_frame()
        fills_df = pd.DataFrame(getattr(strategy, "fill_log", []))
        inventory_df = pd.DataFrame(getattr(strategy, "market_log", []))
        files_done = source.done
    else:
        print(f"Parallel mode: {workers} workers across {len(files)} files")
        task = partial(_run_market_file, strategy_ref=args.strategy, class_name=args.class_name,
                       params=params, commission=args.commission, latency=args.latency,
                       race_no_lookup=race_no_lookup, window=args.window)
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(task, f): f for f in files}
            for idx, fut in enumerate(as_completed(futures), 1):
                try:
                    results.append(fut.result())
                except Exception as e:
                    print(f"ERROR in {futures[fut]}: {e!r}  (skipped)", flush=True)
                if idx % 10 == 0 or idx == len(files):
                    print(f"[{idx:>6}/{len(files)}] files processed", flush=True)

        # Put races in order of scheduled start, so cumulative PnL and drawdown are chronological
        results.sort(key=lambda r: (r["start"] is None, r["start"] if r["start"] is not None else 0))
        bets_frames = [pd.DataFrame(r["bets"]) for r in results if r["bets"]]
        markets_frames = [pd.DataFrame(r["markets"]) for r in results if r["markets"]]
        fills_frames = [pd.DataFrame(r["fills"]) for r in results if r["fills"]]
        inventory_frames = [pd.DataFrame(r["inventory"]) for r in results if r["inventory"]]

        bets_df = pd.concat(bets_frames, ignore_index=True) if bets_frames else pd.DataFrame()
        markets_df = pd.concat(markets_frames, ignore_index=True) if markets_frames else pd.DataFrame()
        fills_df = pd.concat(fills_frames, ignore_index=True) if fills_frames else pd.DataFrame()
        inventory_df = pd.concat(inventory_frames, ignore_index=True) if inventory_frames else pd.DataFrame()
        files_done = sum(r["files_done"] for r in results)
        finished = True

    out = Path(args.out) if args.out else ROOT / "research" / "results" / args.class_name
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    bets_df.to_csv(out / "bets.csv", index=False)
    markets_df.to_csv(out / "markets.csv", index=False)
    fills_df.to_csv(out / "fills.csv", index=False)
    inventory_df.to_csv(out / "inventory.csv", index=False)
    (out / "run_config.json").write_text(json.dumps({
        "strategy": args.strategy, "class": args.class_name, "params": params,
        "data": args.data, "files": len(files), "files_done": files_done,
        "finished": finished, "commission": args.commission, "latency": args.latency,
        "workers": workers, "window": args.window, "started": started.isoformat(timespec="seconds"),
    }, indent=2, default=str))

    summary_text = summarise(None, strategy_cls(**params), args.commission, bets=bets_df, markets=markets_df, fills=fills_df)
    (out / "summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    print(f"Saved bets, markets, fills, inventory, run_config and summary.txt to {out}")


if __name__ == "__main__":
    main()