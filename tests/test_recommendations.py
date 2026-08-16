from unittest.mock import patch

import pytest

from src.learning.recommendations import (
    current_weights,
    generate_recommendations,
    generate_regime_recommendations,
    generate_symbol_recommendations,
    generate_weight_recommendations,
)


def _trade(trade_id, pnl, closed_at="2026-01-01T00:00:00Z", **overrides):
    base = {"id": trade_id, "pnl": pnl, "closed_at": closed_at, "opened_at": "2025-12-31T00:00:00Z", "symbol": "BTCINR"}
    base.update(overrides)
    return base


# --- generate_recommendations (threshold sweep) ---


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
@patch("src.learning.recommendations.models")
def test_generate_recommendations_below_sample_size_returns_empty(mock_models):
    mock_models.get_recently_closed_trades.return_value = [_trade(1, 100)]
    assert generate_recommendations("paper") == []


@patch("src.learning.recommendations.MIN_OPPORTUNITY_SCORE", 60)
@patch("src.learning.recommendations.OPPORTUNITY_SCORE_BUCKET_WIDTH", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_IMPROVEMENT_PCT", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.MIN_EXPECTANCY_DELTA", 1.0)
@patch("src.learning.recommendations.models")
def test_generate_recommendations_writes_recommendation_on_clear_improvement(mock_models):
    trades = [_trade(i, -100) for i in range(1, 5)] + [_trade(i, 200) for i in range(5, 9)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    scores = {**{i: 65 for i in range(1, 5)}, **{i: 85 for i in range(5, 9)}}
    mock_models.get_entry_evaluation_for_trade.side_effect = lambda tid: {"opportunity_score": scores[tid]}
    mock_models.get_latest_recommendation.return_value = None

    result = generate_recommendations("paper")

    assert result[0]["recommended_value"] == 80
    mock_models.insert_recommendation.assert_called_once()
    kwargs = mock_models.insert_recommendation.call_args.kwargs
    assert kwargs["metric_name"] == "MIN_OPPORTUNITY_SCORE"
    assert kwargs["recommended_value"] == 80
    assert kwargs["category"] == "threshold"


@patch("src.learning.recommendations.MIN_OPPORTUNITY_SCORE", 60)
@patch("src.learning.recommendations.OPPORTUNITY_SCORE_BUCKET_WIDTH", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_IMPROVEMENT_PCT", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.recommendations.MIN_EXPECTANCY_DELTA", 1.0)
@patch("src.learning.recommendations.models")
def test_generate_recommendations_blocked_by_absolute_expectancy_delta_floor(mock_models):
    # Relative improvement is 20% (clears RECOMMENDATION_MIN_IMPROVEMENT_PCT)
    # but the absolute expectancy delta is only 0.0025 — this is the
    # regression test for the bug a relative-only check would miss: a
    # near-zero baseline expectancy makes any tiny swing look huge in %.
    trades = [
        _trade(1, 1.0, score=65), _trade(2, -0.98, score=65),
        _trade(3, 1.0, score=85), _trade(4, -0.97, score=85),
    ]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    scores = {1: 65, 2: 65, 3: 85, 4: 85}
    mock_models.get_entry_evaluation_for_trade.side_effect = lambda tid: {"opportunity_score": scores[tid]}
    mock_models.get_latest_recommendation.return_value = None

    assert generate_recommendations("paper") == []
    mock_models.insert_recommendation.assert_not_called()


@patch("src.learning.recommendations.MIN_OPPORTUNITY_SCORE", 60)
@patch("src.learning.recommendations.OPPORTUNITY_SCORE_BUCKET_WIDTH", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_IMPROVEMENT_PCT", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.MIN_EXPECTANCY_DELTA", 1.0)
@patch("src.learning.recommendations.models")
def test_generate_recommendations_idempotent_when_not_materially_different(mock_models):
    trades = [_trade(i, -100) for i in range(1, 5)] + [_trade(i, 200) for i in range(5, 9)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    scores = {**{i: 65 for i in range(1, 5)}, **{i: 85 for i in range(5, 9)}}
    mock_models.get_entry_evaluation_for_trade.side_effect = lambda tid: {"opportunity_score": scores[tid]}
    # already on record at essentially the same value (80) -> not material
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 80}

    assert generate_recommendations("paper") == []
    mock_models.insert_recommendation.assert_not_called()


# --- generate_weight_recommendations ---


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_weight_recommendations_none_when_no_candidate_weights(mock_models, mock_weights):
    mock_weights.return_value = None
    assert generate_weight_recommendations("paper") == []
    mock_models.get_recently_closed_trades.assert_not_called()


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.recommendations.score_separation_p_value")
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_weight_recommendations_writes_one_row_per_subscore_sharing_batch_id(
    mock_models, mock_weights, mock_separation
):
    mock_models.get_recently_closed_trades.return_value = [_trade(1, 100), _trade(2, -50)]
    mock_models.get_latest_recommendation.return_value = None
    candidate = {"trend_score": 0.7, "momentum_score": 0.3}
    mock_weights.return_value = candidate

    def _separation(trades, weights):
        # candidate weights separate better (lower p) than current weights
        p = 0.01 if weights == candidate else 0.5
        return {"p_value": p, "mean_win_score": 80, "mean_loss_score": 20, "n_win": 5, "n_loss": 5}

    mock_separation.side_effect = _separation

    result = generate_weight_recommendations("paper")

    assert {r["metric_name"] for r in result} == {"OPPORTUNITY_WEIGHT_TREND", "OPPORTUNITY_WEIGHT_MOMENTUM"}
    batch_ids = {r["batch_id"] for r in result}
    assert len(batch_ids) == 1  # all co-generated rows share one batch
    assert mock_models.insert_recommendation.call_count == 2


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.recommendations.score_separation_p_value")
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_weight_recommendations_rejected_when_not_better_than_current(
    mock_models, mock_weights, mock_separation
):
    mock_models.get_recently_closed_trades.return_value = [_trade(1, 100), _trade(2, -50)]
    mock_weights.return_value = {"trend_score": 1.0}

    def _separation(trades, weights):
        # candidate and current separate equally well -> candidate p_value
        # is NOT strictly better, must be rejected
        return {"p_value": 0.01, "mean_win_score": 80, "mean_loss_score": 20, "n_win": 5, "n_loss": 5}

    mock_separation.side_effect = _separation

    assert generate_weight_recommendations("paper") == []
    mock_models.insert_recommendation.assert_not_called()


# --- generate_regime_recommendations: avoid-regime direction ---


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.z_test_two_proportions")
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_regime_recommendations_flags_worse_than_baseline_regime(
    mock_models, mock_weights, mock_z_test
):
    all_trades = [_trade(i, 100, market_regime="strong_bull") for i in range(1, 5)] + [
        _trade(i, -100, market_regime="sideways") for i in range(5, 9)
    ]
    mock_models.get_recently_closed_trades.return_value = all_trades
    mock_models.get_learning_statistics.return_value = [
        {"dimension_value": "strong_bull", "trades_count": 4, "win_rate": 1.0},
        {"dimension_value": "sideways", "trades_count": 4, "win_rate": 0.0},
    ]
    mock_models.get_latest_recommendation.return_value = None
    mock_z_test.return_value = 0.01  # significant
    mock_weights.return_value = None  # skip the weight-conditioning half

    result = generate_regime_recommendations("paper")

    avoid_rows = [r for r in result if r["metric_name"].startswith("avoid_regime:")]
    assert len(avoid_rows) == 1
    assert avoid_rows[0]["metric_name"] == "avoid_regime:sideways"  # the worse-performing regime
    assert avoid_rows[0]["recommended_value"] == 0.0


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_regime_recommendations_never_flags_a_regime_better_than_baseline(mock_models, mock_weights):
    all_trades = [_trade(i, 100, market_regime="strong_bull") for i in range(1, 9)]
    mock_models.get_recently_closed_trades.return_value = all_trades
    mock_models.get_learning_statistics.return_value = [
        {"dimension_value": "strong_bull", "trades_count": 8, "win_rate": 1.0},
    ]
    mock_weights.return_value = None

    result = generate_regime_recommendations("paper")

    assert not any(r["metric_name"].startswith("avoid_regime:") for r in result)


# --- generate_symbol_recommendations: avoid-symbol idempotency ---


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.z_test_two_proportions")
@patch("src.learning.recommendations.models")
def test_generate_symbol_recommendations_avoid_symbol_idempotent(mock_models, mock_z_test):
    all_trades = [_trade(i, 100, symbol="BTCINR") for i in range(1, 5)] + [
        _trade(i, -100, symbol="DOGEINR") for i in range(5, 9)
    ]
    mock_models.get_recently_closed_trades.return_value = all_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_learning_statistics.return_value = [
        {"dimension_value": "BTCINR", "trades_count": 4, "win_rate": 1.0},
        {"dimension_value": "DOGEINR", "trades_count": 4, "win_rate": 0.0},
    ]
    mock_z_test.return_value = 0.01
    # already flagged as avoid -> must not write again
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 0.0}
    mock_models.get_entry_evaluation_for_trade.return_value = None
    mock_models.get_confidence_calibration_for_evaluation.return_value = None

    result = generate_symbol_recommendations("paper")

    assert not any(r["metric_name"].startswith("avoid_symbol:") for r in result)
    mock_models.insert_recommendation.assert_not_called()


# --- current_weights ---


@patch("src.learning.recommendations.OPPORTUNITY_WEIGHT_TREND", 0.3)
@patch("src.learning.recommendations.OPPORTUNITY_WEIGHT_MOMENTUM", 0.25)
@patch("src.learning.recommendations.OPPORTUNITY_WEIGHT_VOLUME", 0.15)
@patch("src.learning.recommendations.OPPORTUNITY_WEIGHT_VOLATILITY", 0.15)
@patch("src.learning.recommendations.OPPORTUNITY_WEIGHT_RISK", 0.15)
def test_current_weights_matches_config():
    assert current_weights() == {
        "trend_score": 0.3,
        "momentum_score": 0.25,
        "volume_score": 0.15,
        "volatility_score": 0.15,
        "risk_score": 0.15,
    }
