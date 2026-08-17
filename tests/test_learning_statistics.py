import math
from unittest.mock import patch

import pytest

from src.learning.statistics import (
    _bucket_memberships,
    accuracy_rates,
    compute_bucket_statistics,
    streaks,
    z_test_two_means,
    z_test_two_proportions,
)


def _trade(pnl, opened="2026-01-01T00:00:00Z", closed="2026-01-01T01:00:00Z"):
    return {"pnl": pnl, "opened_at": opened, "closed_at": closed}


def test_sharpe_zero_on_symmetric_returns():
    trades = [_trade(100), _trade(-100), _trade(100), _trade(-100)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["sharpe_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_sortino_uses_downside_deviation_from_zero_not_stdev_of_losses():
    # asymmetric returns specifically chosen so the WRONG formula
    # (stdev of the losing subset alone) gives a different answer than
    # the correct one (deviation from zero over ALL trades) — this test
    # fails under the bug this module was built to avoid.
    trades = [_trade(50), _trade(-20), _trade(30), _trade(-40)]  # returns: .05, -.02, .03, -.04
    stats = compute_bucket_statistics(trades, capital_to_use=1000)

    returns = [0.05, -0.02, 0.03, -0.04]
    expected_downside_dev = math.sqrt((0**2 + 0.02**2 + 0**2 + 0.04**2) / 4)
    expected_sortino = (sum(returns) / 4) / expected_downside_dev

    assert stats["sortino_ratio"] == pytest.approx(expected_sortino)

    # the wrong formula (stdev of losses only, N-1) would give a
    # materially different number — assert we did NOT compute that
    wrong_stdev_of_losses = ((-0.02 - -0.03) ** 2 + (-0.04 - -0.03) ** 2) ** 0.5  # sample stdev of [-.02,-.04]
    wrong_sortino = (sum(returns) / 4) / wrong_stdev_of_losses
    assert stats["sortino_ratio"] != pytest.approx(wrong_sortino)


def test_sortino_none_when_no_downside_at_all():
    trades = [_trade(10), _trade(20), _trade(30)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["sortino_ratio"] is None


def test_calmar_percent_normalized_both_sides():
    # 3 trades: +500, -300, -100 on capital_to_use=1000
    # cumulative_pnl = 100 -> 10% ; max_drawdown walk: running 500(peak),
    # 200, 100 -> max_dd = 500-100=400 -> 40% of capital
    trades = [
        _trade(500, closed="2026-01-01T01:00:00Z"),
        _trade(-300, closed="2026-01-02T01:00:00Z"),
        _trade(-100, closed="2026-01-03T01:00:00Z"),
    ]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["max_drawdown_pct"] == pytest.approx(40.0)
    assert stats["calmar_ratio"] == pytest.approx(10.0 / 40.0)


def test_calmar_none_when_no_drawdown():
    trades = [_trade(10), _trade(20)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["calmar_ratio"] is None


def test_expectancy_matches_hand_computation():
    # 2 wins (100, 200), 1 loss (-50) -> win_rate=2/3, avg_win=150, avg_loss=-50
    trades = [_trade(100), _trade(200), _trade(-50)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    expected = (2 / 3) * 150 + (1 / 3) * -50
    assert stats["expectancy"] == pytest.approx(expected)


def test_profit_factor_matches_hand_computation():
    trades = [_trade(100), _trade(200), _trade(-50), _trade(-50)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["profit_factor"] == pytest.approx(300 / 100)


def test_profit_factor_none_when_no_losses():
    trades = [_trade(10), _trade(20)]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["profit_factor"] is None


def test_sharpe_none_with_fewer_than_two_trades():
    stats = compute_bucket_statistics([_trade(10)], capital_to_use=1000)
    assert stats["sharpe_ratio"] is None
    assert stats["trades_count"] == 1


def test_empty_bucket_returns_all_none_not_crash():
    stats = compute_bucket_statistics([], capital_to_use=1000)
    assert stats["trades_count"] == 0
    assert stats["win_rate"] is None
    assert stats["sharpe_ratio"] is None
    assert stats["calmar_ratio"] is None


def test_zero_capital_to_use_returns_all_none_not_divide_by_zero():
    stats = compute_bucket_statistics([_trade(10)], capital_to_use=0)
    assert stats["sharpe_ratio"] is None
    assert stats["win_rate"] is None


def test_avg_holding_time_seconds():
    trades = [
        _trade(10, opened="2026-01-01T00:00:00Z", closed="2026-01-01T01:00:00Z"),  # 3600s
        _trade(20, opened="2026-01-02T00:00:00Z", closed="2026-01-02T00:30:00Z"),  # 1800s
    ]
    stats = compute_bucket_statistics(trades, capital_to_use=1000)
    assert stats["avg_holding_time_seconds"] == pytest.approx(2700.0)


# --- z_test_two_proportions ---


def test_z_test_two_proportions_known_value():
    # p1=0.6 (30/50), p2=0.4 (20/50) -> hand-verified p-value
    assert z_test_two_proportions(30, 50, 20, 50) == pytest.approx(0.04550026389635842)


def test_z_test_two_proportions_identical_rates_high_p_value():
    assert z_test_two_proportions(25, 50, 25, 50) == pytest.approx(1.0)


def test_z_test_two_proportions_none_on_zero_sample():
    assert z_test_two_proportions(0, 0, 5, 10) is None
    assert z_test_two_proportions(5, 10, 0, 0) is None


def test_z_test_two_proportions_none_on_degenerate_variance():
    # both groups all-wins -> pooled variance is 0
    assert z_test_two_proportions(10, 10, 10, 10) is None


# --- z_test_two_means ---


def test_z_test_two_means_known_value():
    assert z_test_two_means(10, 2, 30, 8, 2, 30) == pytest.approx(0.00010751117672946897)


def test_z_test_two_means_identical_means_high_p_value():
    assert z_test_two_means(5, 1, 20, 5, 1, 20) == pytest.approx(1.0)


def test_z_test_two_means_none_when_fewer_than_two_samples():
    assert z_test_two_means(10, 2, 1, 8, 2, 30) is None
    assert z_test_two_means(10, 2, 30, 8, 2, 1) is None


def test_z_test_two_means_none_on_zero_variance():
    assert z_test_two_means(10, 0, 30, 8, 0, 30) is None


# --- streaks ---


def test_streaks_tracks_longest_and_current():
    trades = [
        _trade(100, closed="2026-01-01T00:00:00Z"),
        _trade(100, closed="2026-01-02T00:00:00Z"),
        _trade(-50, closed="2026-01-03T00:00:00Z"),
        _trade(100, closed="2026-01-04T00:00:00Z"),
        _trade(100, closed="2026-01-05T00:00:00Z"),
        _trade(100, closed="2026-01-06T00:00:00Z"),
    ]
    result = streaks(trades)
    assert result["longest_win_streak"] == 3
    assert result["longest_loss_streak"] == 1
    assert result["current_streak_type"] == "win"
    assert result["current_streak_length"] == 3


def test_streaks_current_streak_is_loss():
    trades = [
        _trade(100, closed="2026-01-01T00:00:00Z"),
        _trade(-50, closed="2026-01-02T00:00:00Z"),
        _trade(-50, closed="2026-01-03T00:00:00Z"),
    ]
    result = streaks(trades)
    assert result["current_streak_type"] == "loss"
    assert result["current_streak_length"] == 2


def test_streaks_empty_input():
    result = streaks([])
    assert result == {
        "longest_win_streak": 0,
        "longest_loss_streak": 0,
        "current_streak_type": None,
        "current_streak_length": 0,
    }


# --- exit_reason bucket dimension (Weakness Detection) ---


def test_bucket_memberships_includes_exit_reason_when_present():
    trade = {
        "symbol": "BTCINR",
        "version_id": 1,
        "closed_at": "2026-01-01T12:00:00Z",
        "exit_reason": "stop_loss",
    }
    memberships = _bucket_memberships(trade, opportunity_score=None, confidence=None)
    assert memberships["exit_reason"] == "stop_loss"


def test_bucket_memberships_omits_exit_reason_when_absent():
    trade = {"symbol": "BTCINR", "version_id": 1, "closed_at": "2026-01-01T12:00:00Z"}
    memberships = _bucket_memberships(trade, opportunity_score=None, confidence=None)
    assert "exit_reason" not in memberships


# --- accuracy_rates (Performance Analyzer) ---


def test_accuracy_rates_aggregates_over_trade_evaluations():
    rows = [
        {
            "confidence_was_accurate": True,
            "opportunity_score_was_accurate": False,
            "risk_assessment": "appropriate",
            "stop_loss_assessment": "too_tight",
            "target_assessment": "realistic",
        },
        {
            "confidence_was_accurate": True,
            "opportunity_score_was_accurate": True,
            "risk_assessment": "too_aggressive",
            "stop_loss_assessment": "appropriate",
            "target_assessment": None,
        },
    ]
    with patch("src.learning.statistics.models") as mock_models:
        mock_models.get_trade_evaluations.return_value = rows
        result = accuracy_rates([1, 2])

    assert result["confidence_accuracy_pct"] == pytest.approx(100.0)
    assert result["opportunity_score_accuracy_pct"] == pytest.approx(50.0)
    assert result["risk_accuracy_pct"] == pytest.approx(50.0)
    assert result["stop_loss_accuracy_pct"] == pytest.approx(50.0)
    assert result["target_accuracy_pct"] == pytest.approx(100.0)  # None row excluded, not counted as wrong


def test_accuracy_rates_none_when_no_rows():
    with patch("src.learning.statistics.models") as mock_models:
        mock_models.get_trade_evaluations.return_value = []
        result = accuracy_rates([])

    assert all(v is None for v in result.values())
