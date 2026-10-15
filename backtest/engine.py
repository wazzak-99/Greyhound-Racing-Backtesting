"""
backtest/engine.py
--------------------
"""
from __future__ import annotations

from backtest.context import Cancel
from backtest.datasources import MarketReplay, MarketSource
from backtest.execution import ExecutionModel, Fill, Unmatched
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
        self._next_order_id = 1

    def run(self) -> None:
        for replay in self.source.markets():
            self._run_market(replay)

    def _run_market(self, replay: MarketReplay) -> None:
        contexts = replay.contexts
        if not contexts:
            return

        self.execution.reset()
        self.strategy.on_market_open(contexts[0])

        already_removed: set[int] = set()
        removed_sequence: list[tuple[int, float]] = []

        for ctx in contexts:
            for r in ctx.runners:
                if r.removed and r.selection_id not in already_removed:
                    already_removed.add(r.selection_id)
                    removed_sequence.append((r.selection_id, r.adjustment_factor or 0.0))

            # Orders already in the market get matched/lapsed before the strategy sees this snapshot
            self._report(self.execution.on_snapshot(ctx))

            for action in self.strategy.on_snapshot(ctx):
                if isinstance(action, Cancel):
                    self._report(self.execution.cancel(action, ctx))
                    continue
                action.order_id = self._next_order_id
                self._next_order_id += 1
                if action.stake is None:
                    action.stake = self.staking.size(action, ctx, self.bankroll)
                self._report(self.execution.submit(action, ctx))

        self._report(self.execution.close_market(contexts[-1]))
        self.ledger.settle_market(replay.result, removed_sequence)
        self.strategy.on_market_close(contexts[-1], replay.result)

    def _report(self, events: list[Fill | Unmatched]) -> None:
        for event in events:
            if isinstance(event, Fill):
                self.ledger.record_fill(event)
                self.strategy.on_fill(event)
            else:
                self.strategy.on_unmatched(event)
