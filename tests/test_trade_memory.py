from unittest.mock import Mock, patch

import pytest

from src.learning.trade_memory import _distance, _feature_importance_weights, find_similar_trades


def _eval_row(trade_id, **scores):
    base = {
        "trade_id": trade_id,
        "trend_score": 80,
        "momentum_score": 70,
        "volume_score": 60,
        "volatility_score": 100,
        "risk_score": 90,
    }
    base.update(scores)
    return base


def _trade(trade_id, symbol="BTCINR", pnl=100, market_regime="strong_bull", status="closed"):
    return {
        "id": trade_id,
        "symbol": symbol,
        "pnl": pnl,
        "status": status,
        "market_regime": market_regime,
        "entry_price": 100,
        "opened_at": "2026-01-01T00:00:00Z",
        "closed_at": "2026-01-01T01:00:00Z",
    }


def test_distance_zero_for_identical_vectors():
    a = _eval_row(1)
    b = _eval_row(2)
    assert _distance(a, b) == 0.0


def test_distance_known_value():
    a = {"trend_score": 100, "momentum_score": None, "volume_score": None, "volatility_score": None, "risk_score": None}
    b = {"trend_score": 90, "momentum_score": None, "volume_score": None, "volatility_score": None, "risk_score": None}
    assert _distance(a, b) == pytest.approx(10.0)


def test_distance_none_when_no_overlapping_dimensions():
    a = {"trend_score": 100, "momentum_score": None, "volume_score": None, "volatility_score": None, "risk_score": None}
    b = {"trend_score": None, "momentum_score": 50, "volume_score": None, "volatility_score": None, "risk_score": None}
    assert _distance(a, b) is None


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_degrades_below_min_similar_trades(mock_models):
    mock_models.get_entry_evaluations_since.return_value = [_eval_row(1)]
    mock_models.get_trades_by_ids.return_value = [_trade(1)]

    with patch("src.learning.trade_memory.MIN_SIMILAR_TRADES", 5):
        result = find_similar_trades(_eval_row(999), market_regime=None, mode="paper")

    assert result["count"] == 0
    assert result["win_rate"] is None


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_degrades_when_nothing_within_max_distance(mock_models):
    # 5 candidates, but all wildly dissimilar
    far_candidates = [_eval_row(i, trend_score=0, momentum_score=0, volume_score=0, volatility_score=0, risk_score=0) for i in range(1, 6)]
    mock_models.get_entry_evaluations_since.return_value = far_candidates
    mock_models.get_trades_by_ids.return_value = [_trade(i) for i in range(1, 6)]

    with patch("src.learning.trade_memory.MIN_SIMILAR_TRADES", 3), patch(
        "src.learning.trade_memory.SIMILARITY_MAX_DISTANCE", 1.0
    ):
        # current candidate is the max-score vector, far from the all-zero candidates
        current = _eval_row(999, trend_score=100, momentum_score=100, volume_score=100, volatility_score=100, risk_score=100)
        result = find_similar_trades(current, market_regime=None, mode="paper")

    assert result["count"] == 0


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_returns_stats_when_enough_close_matches(mock_models):
    candidates = [_eval_row(i) for i in range(1, 6)]
    mock_models.get_entry_evaluations_since.return_value = candidates
    trades = [_trade(1, pnl=100), _trade(2, pnl=100), _trade(3, pnl=-50), _trade(4, pnl=100), _trade(5, pnl=-50)]
    mock_models.get_trades_by_ids.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}

    with patch("src.learning.trade_memory.MIN_SIMILAR_TRADES", 3):
        result = find_similar_trades(_eval_row(999), market_regime=None, mode="paper")

    assert result["count"] == 5
    assert result["win_rate"] == pytest.approx(3 / 5)


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_excludes_trades_with_unknown_outcome(mock_models):
    candidates = [_eval_row(1), _eval_row(2)]
    mock_models.get_entry_evaluations_since.return_value = candidates
    mock_models.get_trades_by_ids.return_value = [_trade(1, status="open", pnl=None), _trade(2, pnl=50)]

    with patch("src.learning.trade_memory.MIN_SIMILAR_TRADES", 1):
        result = find_similar_trades(_eval_row(999), market_regime=None, mode="paper")

    assert result["count"] == 1  # the still-open trade is excluded


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_filters_by_market_regime(mock_models):
    candidates = [_eval_row(1), _eval_row(2)]
    mock_models.get_entry_evaluations_since.return_value = candidates
    mock_models.get_trades_by_ids.return_value = [
        _trade(1, market_regime="strong_bull"),
        _trade(2, market_regime="strong_bear"),
    ]

    with patch("src.learning.trade_memory.MIN_SIMILAR_TRADES", 1):
        result = find_similar_trades(_eval_row(999), market_regime="strong_bull", mode="paper")

    assert result["count"] == 1


@patch("src.learning.trade_memory.models")
def test_find_similar_trades_empty_when_no_candidates(mock_models):
    mock_models.get_entry_evaluations_since.return_value = []
    result = find_similar_trades(_eval_row(999), market_regime=None, mode="paper")
    assert result == {
        "trades": [],
        "count": 0,
        "win_rate": None,
        "avg_profit_pct": None,
        "avg_loss_pct": None,
        "avg_holding_time_seconds": None,
    }


# --- _feature_importance_weights: reads the nightly-cached sub-score
# correlations (feature_importance table, timeframe="blended"), no longer
# a permanent no-op stub ---


@patch("src.learning.trade_memory.models")
def test_feature_importance_weights_none_when_nothing_cached(mock_models):
    mock_models.get_feature_importance.return_value = []
    assert _feature_importance_weights("paper") is None
    mock_models.get_feature_importance.assert_called_once_with(
        "paper", timeframe="blended", strategy_type="default"
    )


@patch("src.learning.trade_memory.models")
def test_feature_importance_weights_none_when_all_correlations_non_positive(mock_models):
    mock_models.get_feature_importance.return_value = [
        {"feature_name": "trend_score", "correlation_score": -0.2},
        {"feature_name": "momentum_score", "correlation_score": 0.0},
    ]
    assert _feature_importance_weights("paper") is None


@patch("src.learning.trade_memory.models")
def test_feature_importance_weights_normalizes_positive_correlations(mock_models):
    mock_models.get_feature_importance.return_value = [
        {"feature_name": "trend_score", "correlation_score": 0.4},
        {"feature_name": "momentum_score", "correlation_score": 0.2},
        {"feature_name": "volume_score", "correlation_score": -0.1},  # floored to 0
        {"feature_name": "rsi", "correlation_score": 0.9},  # not a sub-score key, ignored
    ]
    weights = _feature_importance_weights("paper")
    assert weights == pytest.approx({"trend_score": 2 / 3, "momentum_score": 1 / 3, "volume_score": 0.0})
