from datetime import datetime, timezone

import pytest

from src.backtest.performance_analyzer import (
    analyze,
    annual_returns,
    capital_utilization_pct,
    exposure_time_pct,
    gross_profit_loss,
    monthly_returns,
    omega_ratio,
    rolling_drawdown,
    rolling_sharpe,
    rolling_volatility,
    ulcer_index,
)
from src.backtest.portfolio_manager import ClosedTrade


def _trade(pnl, entry="2024-01-01T00:00:00+00:00", exit_="2024-01-02T00:00:00+00:00"):
    return ClosedTrade(
        symbol="TESTINR",
        side="buy",
        qty=1.0,
        entry_price=100.0,
        exit_price=100.0 + pnl,
        entry_time=datetime.fromisoformat(entry),
        exit_time=datetime.fromisoformat(exit_),
        pnl=pnl,
        fees=1.0,
        slippage_cost=0.1,
        exit_reason="ai_exit",
        confidence=None,
        opportunity_score=70.0,
        market_regime="strong_bull",
        mfe_pct=5.0,
        mae_pct=2.0,
    )


def test_gross_profit_loss_separates_wins_and_losses():
    trades = [_trade(10), _trade(-5), _trade(20)]
    result = gross_profit_loss(trades)
    assert result["gross_profit"] == 30
    assert result["gross_loss"] == -5
    assert result["largest_win"] == 20
    assert result["largest_loss"] == -5


def test_omega_ratio_none_when_no_losses():
    assert omega_ratio([0.01, 0.02, 0.03]) is None


def test_omega_ratio_ratio_of_gains_to_losses():
    result = omega_ratio([0.02, -0.01])
    assert result == 2.0


def test_ulcer_index_zero_for_flat_equity_curve():
    assert ulcer_index([1000, 1000, 1000]) == 0.0


def test_ulcer_index_positive_for_drawdown():
    assert ulcer_index([1000, 900, 950]) > 0


def test_exposure_time_pct_fraction_of_snapshots_with_open_positions():
    snapshots = [{"open_positions_count": 1}, {"open_positions_count": 0}, {"open_positions_count": 2}]
    assert exposure_time_pct(snapshots) == pytest.approx(100 * 2 / 3)


def test_capital_utilization_pct_averages_exposure():
    snapshots = [{"exposure_pct": 10}, {"exposure_pct": 20}]
    assert capital_utilization_pct(snapshots) == 15


def test_monthly_returns_groups_by_calendar_month():
    snapshots = [
        {"snapshot_time": datetime(2024, 1, 1, tzinfo=timezone.utc), "equity": 1000},
        {"snapshot_time": datetime(2024, 1, 31, tzinfo=timezone.utc), "equity": 1100},
        {"snapshot_time": datetime(2024, 2, 1, tzinfo=timezone.utc), "equity": 1100},
        {"snapshot_time": datetime(2024, 2, 28, tzinfo=timezone.utc), "equity": 1210},
    ]
    monthly = monthly_returns(snapshots)
    assert monthly["2024-01"] == 10.0
    assert monthly["2024-02"] == 10.0


def test_annual_returns_sums_monthly_pct():
    assert annual_returns({"2024-01": 5.0, "2024-02": 3.0, "2023-12": 1.0}) == {"2024": 8.0, "2023": 1.0}


def test_rolling_sharpe_none_before_window_filled():
    result = rolling_sharpe([0.01, 0.02, 0.03], window=3)
    assert result[0] is None
    assert result[1] is None
    assert result[2] is not None


def test_rolling_volatility_matches_stdev_at_window():
    from statistics import stdev

    returns = [0.01, 0.02, -0.01, 0.03]
    result = rolling_volatility(returns, window=2)
    assert result[1] == stdev(returns[0:2])


def test_rolling_drawdown_tracks_peak_to_trough():
    curve = [100, 110, 90, 120]
    dd = rolling_drawdown(curve)
    assert dd[0] == 0.0
    assert dd[1] == 0.0
    assert dd[2] == (110 - 90) / 110 * 100
    assert dd[3] == 0.0


def test_analyze_returns_full_metrics_bundle_with_recovery_factor():
    trades = [_trade(100), _trade(-50)]
    snapshots = [
        {"snapshot_time": datetime(2024, 1, 1, tzinfo=timezone.utc), "equity": 10000, "open_positions_count": 0, "exposure_pct": 0},
        {"snapshot_time": datetime(2024, 1, 2, tzinfo=timezone.utc), "equity": 10050, "open_positions_count": 1, "exposure_pct": 5},
    ]
    result = analyze(trades, snapshots, starting_capital=10000)

    assert result["trades_count"] == 2
    assert result["gross_profit"] == 100
    assert result["gross_loss"] == -50
    assert result["final_equity"] == 10050
    assert result["total_return_pct"] == 0.5
    assert "recovery_factor" in result
    assert "omega_ratio" in result
    assert "ulcer_index" in result
