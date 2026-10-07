"""
scripts/run_backtest.py
-----------------------
Run one strategy over the processed stream data and print the results.

Run from the project root (after build_dataset.py):
    python scripts/run_backtest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backtest.context import MarketType  # noqa: E402
from backtest.datasources import StreamMarketSource  # noqa: E402
from backtest.engine import Engine  # noqa: E402
from backtest.execution import TickExecutionModel  # noqa: E402
from backtest.ledger import Ledger  # noqa: E402
from backtest.staking import FlatStake  # noqa: E402
from strategies.template import TemplateStrategy  # noqa: E402

PROCESSED = ROOT / "data/processed/market_stream"


def main() -> None:
    with open(PROCESSED / "race_no_lookup.json") as f:
        race_no_lookup = json.load(f)

    source = StreamMarketSource(
        pattern=str(PROCESSED / "*/*/*/*.csv"),   # e.g. "Oct/1/*/*.csv" for a single day
        race_no_lookup=race_no_lookup,
        market_type_filter={MarketType.WIN},      # WIN, PLACE and/or MATCH_BET
    )
    strategy = TemplateStrategy()
    execution = TickExecutionModel()
    ledger = Ledger(commission_rate=0.04)
    staking = FlatStake(stake=1.0)

    engine = Engine(source, strategy, execution, ledger, staking, starting_bankroll=1000.0)
    engine.run()

    print(ledger.summary())
    out = ROOT / "research" / "results"
    out.mkdir(parents=True, exist_ok=True)
    ledger.bets_frame().to_csv(out / "bets.csv", index=False)
    ledger.markets_frame().to_csv(out / "markets.csv", index=False)
    print(f"Saved bets and markets to {out}")


if __name__ == "__main__":
    main()
