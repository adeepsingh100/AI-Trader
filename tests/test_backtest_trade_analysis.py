from datetime import datetime, timezone

from src.backtest.portfolio_manager import ClosedTrade
from src.backtest.trade_analysis import risk_reward, to_row, to_rows


def _trade(mfe=10.0, mae=5.0):
    return ClosedTrade(
        symbol="TESTINR",
        side="buy",
        qty=2.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 1, 1, 0, tzinfo=timezone.utc),
        pnl=18.0,
        fees=2.0,
        slippage_cost=0.5,
        exit_reason="ai_exit",
        confidence=80.0,
        opportunity_score=75.0,
        market_regime="weak_bull",
        mfe_pct=mfe,
        mae_pct=mae,
    )


def test_risk_reward_ratio_of_mfe_to_mae():
    assert risk_reward(_trade(mfe=10.0, mae=5.0)) == 2.0


def test_risk_reward_none_when_never_adverse():
    assert risk_reward(_trade(mfe=10.0, mae=0.0)) is None


def test_to_row_shape_matches_backtest_trades_columns():
    row = to_row(_trade())
    expected_keys = {
        "symbol", "side", "qty", "entry_price", "exit_price", "entry_time", "exit_time",
        "holding_duration_seconds", "mfe_pct", "mae_pct", "slippage_cost", "commission",
        "pnl", "return_pct", "risk_reward", "exit_reason", "confidence", "opportunity_score",
        "market_regime",
    }
    assert set(row.keys()) == expected_keys
    assert row["holding_duration_seconds"] == 3600
    assert row["commission"] == 2.0
    assert row["return_pct"] == 18.0 / (100.0 * 2.0) * 100


def test_to_rows_maps_every_trade():
    rows = to_rows([_trade(), _trade()])
    assert len(rows) == 2
