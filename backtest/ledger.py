"""
backtest/ledger.py
--------------------
Settles matched Fills against a RaceResult, applying commission, and
accumulates a record of every bet and every market across a backtest run.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtest.context import RaceResult, Side
from backtest.execution import Fill


@dataclass
class SettledBet:
    """One matched bet, after its market has been settled. `gross_pnl` is before commission."""
    market_id: str
    selection_id: int
    side: Side
    price: float
    stake: float
    won: bool
    voided: bool
    gross_pnl: float


@dataclass
class MarketSettlement:
    """A market's aggregate result once all bets are settled"""
    market_id: str
    gross_pnl: float
    commission: float
    net_pnl: float


class Ledger:
    """
    Accumulates Fills per market via `record_fill()`, then settles them once the market's
    RaceResult is known via `settle_market`.
    """

    def __init__(self, commission_rate: float = 0.08):
        self.commission_rate =commission_rate
        self._open: dict[str, list[Fill]] = {}  # unsettled orders
        self.bets: list[SettledBet] = []        # settled orders
        self.markets: list[MarketSettlement] = []

    def record_fill(self, fill: Fill) -> None:
        """Add Fill to the Ledger's unsettled orders."""
        self._open.setdefault(fill.order.market_id, []).append(fill)

    def settle_market(self, result: RaceResult, removed_sequence: list[tuple[int, float]] | None = None) -> None:
        """Settle the market based on `result`. Settle all unsettled orders."""
        fills = self._open.pop(result.market_id, [])
        if not fills:
            return
        
        removed_sequence = removed_sequence or []
        removed_ids = {sel_id for sel_id, _ in removed_sequence}
        
        market_gross = 0.0
        for fill in fills:
            if fill.order.selection_id in removed_ids:
                # removed (scratched) before off, bet is voided
                self.bets.append(SettledBet(
                    market_id=result.market_id,
                    selection_id=fill.order.selection_id,
                    side=fill.order.side,
                    price=fill.price,
                    stake=fill.stake,
                    won=False,
                    voided=True,
                    gross_pnl=0.0,
                ))
                continue

            won = fill.order.selection_id == result.winner_selection_id
            adjusted_price = self._apply_adjustments(fill, removed_sequence)
            gross = self._gross_pnl_at_price(fill, won, adjusted_price)
            market_gross += gross
            self.bets.append(SettledBet(
                market_id=result.market_id,
                selection_id=fill.order.selection_id,
                side=fill.order.side,
                price=adjusted_price,
                stake=fill.stake,
                won=won,
                voided=False,
                gross_pnl=gross,
            ))

        commission = max(market_gross, 0.0) * self.commission_rate
        settlement = MarketSettlement(
            market_id=result.market_id,
            gross_pnl=market_gross,
            commission=commission,
            net_pnl=market_gross - commission,
        )
        self.markets.append(settlement)
        return settlement

    def _apply_adjustments(self, fill: Fill, removed_sequence: list[tuple[int, float]]) -> float:
        price = fill.price
        for selection_id, factor in removed_sequence:
            if selection_id in fill.removed_at_fill:
                continue  # already gone before this bet matched
            price = 1 + (price - 1) * (1 - factor / 100.0)
        return price

    def _gross_pnl_at_price(self, fill: Fill, won: bool, price: float) -> float:
        if fill.order.side is Side.BACK:
            return fill.stake * (price - 1) if won else -fill.stake
        return -fill.stake * (price - 1) if won else fill.stake

    # Summary    
    
    def bets_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(b) for b in self.bets])
    
    def markets_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(m) for m in self.markets])
    
    def summary(self) -> dict:
        markets = self.markets_frame()
        bets = self.bets_frame()
        if markets.empty:
            return {"markets": 0, "bets": 0, "staked": 0.0, "net_pnl": 0.0, "roi": None}
        staked = bets["stake"].sum()
        net_pnl = markets["net_pnl"].sum()
        return {
            "markets": len(markets),
            "bets": len(bets),
            "staked": staked,
            "net_pnl": net_pnl,
            "roi": (net_pnl / staked) if staked else None,
        }