from unittest.mock import patch

import pytest

from src.learning.feature_importance import (
    compute_feature_importance,
    compute_subscore_correlation_weights,
    pearson_correlation,
    score_separation_p_value,
)


# --- pearson_correlation ---


def test_pearson_correlation_perfect_positive():
    assert pearson_correlation([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_pearson_correlation_perfect_negative():
    assert pearson_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_pearson_correlation_none_with_fewer_than_two_points():
    assert pearson_correlation([1], [1]) is None


def test_pearson_correlation_none_on_zero_variance():
    assert pearson_correlation([5, 5, 5], [1, 2, 3]) is None


# --- compute_feature_importance ---


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.PRIMARY_TIMEFRAME", "1h")
@patch("src.learning.feature_importance.models")
def test_compute_feature_importance_defaults_to_primary_timeframe(mock_models):
    mock_models.get_recently_closed_trades.return_value = [
        {"id": 1, "pnl": 100},
        {"id": 2, "pnl": -50},
    ]
    mock_models.get_entry_evaluation_for_trade.side_effect = [
        {"features": {"1h": {"rsi": 70}}},
        {"features": {"1h": {"rsi": 30}}},
    ]

    result = compute_feature_importance("paper")

    rsi_rows = [r for r in result if r["feature_name"] == "rsi"]
    assert len(rsi_rows) == 1
    assert rsi_rows[0]["timeframe"] == "1h"
    assert rsi_rows[0]["correlation_score"] == pytest.approx(1.0)
    mock_models.upsert_feature_importance.assert_any_call("paper", "rsi", pytest.approx(1.0), 2, "1h")


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.models")
def test_compute_feature_importance_multiple_timeframes_are_independent(mock_models):
    mock_models.get_recently_closed_trades.return_value = [
        {"id": 1, "pnl": 100},
        {"id": 2, "pnl": -50},
    ]
    mock_models.get_entry_evaluation_for_trade.side_effect = [
        {"features": {"1m": {"rsi": 70}, "1h": {"rsi": 20}}},
        {"features": {"1m": {"rsi": 30}, "1h": {"rsi": 80}}},
    ]

    result = compute_feature_importance("paper", timeframes=["1m", "1h"])

    timeframes_seen = {r["timeframe"] for r in result if r["feature_name"] == "rsi"}
    assert timeframes_seen == {"1m", "1h"}
    # only 2 trades fetched once, not once per timeframe
    assert mock_models.get_entry_evaluation_for_trade.call_count == 2


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
@patch("src.learning.feature_importance.models")
def test_compute_feature_importance_below_sample_size_returns_empty(mock_models):
    mock_models.get_recently_closed_trades.return_value = [{"id": 1, "pnl": 100}]
    assert compute_feature_importance("paper") == []
    mock_models.upsert_feature_importance.assert_not_called()


# --- compute_subscore_correlation_weights ---


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.models")
def test_compute_subscore_correlation_weights_below_sample_size_returns_none(mock_models):
    mock_models.get_recently_closed_trades.return_value = [{"id": 1, "pnl": 100}]
    assert compute_subscore_correlation_weights("paper") is None


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.models")
def test_compute_subscore_correlation_weights_normalizes_and_caches(mock_models):
    trades = [{"id": 1, "pnl": 100}, {"id": 2, "pnl": -50}]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_entry_evaluation_for_trade.side_effect = [
        {"trend_score": 80, "momentum_score": 40},
        {"trend_score": 20, "momentum_score": 60},
    ]

    weights = compute_subscore_correlation_weights("paper")

    assert weights["trend_score"] == pytest.approx(1.0)
    assert weights["momentum_score"] == pytest.approx(0.0)
    mock_models.upsert_feature_importance.assert_any_call("paper", "trend_score", pytest.approx(1.0), 2, "blended")


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.models")
def test_compute_subscore_correlation_weights_explicit_trades_skips_fetch(mock_models):
    trades = [{"id": 1, "pnl": 100}, {"id": 2, "pnl": -50}]
    mock_models.get_entry_evaluation_for_trade.side_effect = [{"trend_score": 80}, {"trend_score": 20}]

    compute_subscore_correlation_weights("paper", trades=trades, cache=False)

    mock_models.get_recently_closed_trades.assert_not_called()
    mock_models.upsert_feature_importance.assert_not_called()


@patch("src.learning.feature_importance.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.feature_importance.models")
def test_compute_subscore_correlation_weights_none_when_all_non_positive(mock_models):
    trades = [{"id": 1, "pnl": 100}, {"id": 2, "pnl": -50}]
    mock_models.get_entry_evaluation_for_trade.side_effect = [{"trend_score": 20}, {"trend_score": 80}]

    assert compute_subscore_correlation_weights("paper", trades=trades) is None


# --- score_separation_p_value ---


@patch("src.learning.feature_importance.models")
def test_score_separation_p_value_none_with_fewer_than_two_per_group(mock_models):
    mock_models.get_entry_evaluation_for_trade.return_value = {"trend_score": 80}
    assert score_separation_p_value([{"id": 1, "pnl": 100}], {"trend_score": 1.0}) is None


@patch("src.learning.feature_importance.models")
def test_score_separation_p_value_computes_group_means(mock_models):
    trades = [
        {"id": 1, "pnl": 100},
        {"id": 2, "pnl": 100},
        {"id": 3, "pnl": -50},
        {"id": 4, "pnl": -50},
    ]
    mock_models.get_entry_evaluation_for_trade.side_effect = [
        {"trend_score": 90},
        {"trend_score": 80},
        {"trend_score": 20},
        {"trend_score": 10},
    ]

    result = score_separation_p_value(trades, {"trend_score": 1.0})

    assert result["mean_win_score"] == pytest.approx(85.0)
    assert result["mean_loss_score"] == pytest.approx(15.0)
    assert result["n_win"] == 2
    assert result["n_loss"] == 2
    assert result["p_value"] is not None
