"""
backtest/engine.py
--------------------
"""
from __future__ import annotations

from backtest.datasources import MarketReplay, MarketSource
from backtest.execution import ExecutionModel
from backtest.ledger import Ledger
from backtest.strategy import Strategy
from backtest.staking import Staking


class Engine:
    def __init__(
            self, 
            source: MarketSource,
            strategy: Strategy, 
            execution: ExecutionModel, 
            ledger: Ledger, 
            staking: Staking,
            starting_bankroll: float,
    ):
        self.source = source
        self.strategy = strategy
        self.execution = execution
        self.ledger = ledger
        self.staking = staking
        self.bankroll = starting_bankroll

    def run(self) -> None:
        for replay in self.source.markets():
            self._run_market(replay)

    def _run_market(self, replay: MarketReplay) -> None:
        contexts = replay.contexts
        if not contexts:
            return

        self.strategy.on_market_open(contexts[0])

        already_removed: set[int] = set()
        removed_sequence: list[tuple[int, float]] = []

        for ctx in contexts:
            for r in ctx.runners:
                if r.removed and r.selection_id not in already_removed:
                    already_removed.add(r.selection_id)
                    removed_sequence.append((r.selection_id, r.adjustment_factor or 0.0))

            for order in self.strategy.on_snapshot(ctx):
                if order.stake is None:
                    order.stake = self.staking.size(order, ctx, self.bankroll)
                fill = self.execution.fill(order, ctx)
                if fill is not None:
                    self.ledger.record_fill(fill)

        self.ledger.settle_market(replay.result, removed_sequence)
        self.strategy.on_market_close(contexts[-1], replay.result)
