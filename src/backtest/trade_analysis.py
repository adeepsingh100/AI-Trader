"""Step 8: per-trade analytics. Most fields (MFE/MAE/slippage/commission/
return%/exit-reason/confidence/opportunity-score/regime) already live on
portfolio_manager.ClosedTrade — this module adds the one thing that
doesn't (risk_reward) and converts a run's closed trades into the exact
dict shape backtest_trades expects."""

from __future__ import annotations

from src.backtest.portfolio_manager import ClosedTrade


def risk_reward(trade: ClosedTrade) -> float | None:
    """MFE/MAE-based proxy: how much favorable excursion was captured
    relative to adverse excursion experienced. None if the trade never
    moved against the position (mae_pct == 0) — undefined ratio, not a
    fabricated infinity."""
    if not trade.mae_pct:
        return None
    return trade.mfe_pct / trade.mae_pct


def to_row(trade: ClosedTrade) -> dict:
    return {
        "symbol": trade.symbol,
        "side": trade.side,
        "qty": trade.qty,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "entry_time": trade.entry_time.isoformat(),
        "exit_time": trade.exit_time.isoformat(),
        "holding_duration_seconds": int(trade.holding_duration_seconds),
        "mfe_pct": trade.mfe_pct,
        "mae_pct": trade.mae_pct,
        "slippage_cost": trade.slippage_cost,
        "commission": trade.fees,
        "pnl": trade.pnl,
        "return_pct": trade.return_pct,
        "risk_reward": risk_reward(trade),
        "exit_reason": trade.exit_reason,
        "confidence": trade.confidence,
        "opportunity_score": trade.opportunity_score,
        "market_regime": trade.market_regime,
    }


def to_rows(trades: list[ClosedTrade]) -> list[dict]:
    return [to_row(t) for t in trades]
