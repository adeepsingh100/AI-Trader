from unittest.mock import patch

from src.learning.simulation import (
    _train_test_split,
    simulate_threshold_recommendation,
    simulate_weight_recommendation,
)


def _trade(trade_id, pnl, closed_at):
    return {"id": trade_id, "pnl": pnl, "closed_at": closed_at, "opened_at": "2025-12-01T00:00:00Z"}


# --- _train_test_split ---


def test_train_test_split_orders_by_closed_at_and_respects_pct():
    trades = [
        _trade(3, 10, "2026-01-03T00:00:00Z"),
        _trade(1, 10, "2026-01-01T00:00:00Z"),
        _trade(2, 10, "2026-01-02T00:00:00Z"),
        _trade(4, 10, "2026-01-04T00:00:00Z"),
    ]
    train, test = _train_test_split(trades, split_pct=0.5)
    assert [t["id"] for t in train] == [1, 2]
    assert [t["id"] for t in test] == [3, 4]


# --- simulate_weight_recommendation ---


@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_none_when_insufficient_trades(mock_models):
    mock_models.get_recently_closed_trades.return_value = [_trade(1, 10, "2026-01-01T00:00:00Z")]
    assert simulate_weight_recommendation("paper") is None
    mock_models.insert_strategy_simulation.assert_not_called()


@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.compute_subscore_correlation_weights")
@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_none_when_no_candidate_from_train_window(mock_models, mock_weights):
    trades = [_trade(i, 10, f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 5)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_weights.return_value = None
    assert simulate_weight_recommendation("paper") is None
    mock_models.insert_strategy_simulation.assert_not_called()


@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.score_separation_p_value")
@patch("src.learning.simulation.compute_subscore_correlation_weights")
@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_passes_and_creates_candidate_version(
    mock_models, mock_weights, mock_separation
):
    trades = [_trade(i, 10, f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 5)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_weights.return_value = {"trend_score": 1.0}

    def _separation(test_trades, weights):
        p = 0.01 if weights == {"trend_score": 1.0} else 0.5
        return {"p_value": p, "mean_win_score": 80, "mean_loss_score": 20, "n_win": 3, "n_loss": 3}

    mock_separation.side_effect = _separation
    mock_models.insert_strategy_simulation.return_value = {"id": 42, "passed": True}
    mock_models.get_latest_adaptive_strategy_version.return_value = None

    result = simulate_weight_recommendation("paper", batch_id="abc")

    assert result == {"id": 42, "passed": True}
    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is True
    assert kwargs["p_value"] == 0.01
    mock_models.insert_adaptive_strategy_version.assert_called_once()
    version_kwargs = mock_models.insert_adaptive_strategy_version.call_args.kwargs
    assert version_kwargs["version_number"] == 1
    assert version_kwargs["params_json"] == {"trend_score": 1.0}
    assert version_kwargs["source_simulation_id"] == 42


@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.score_separation_p_value")
@patch("src.learning.simulation.compute_subscore_correlation_weights")
@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_fails_writes_row_without_candidate(
    mock_models, mock_weights, mock_separation
):
    trades = [_trade(i, 10, f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 5)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_weights.return_value = {"trend_score": 1.0}
    # candidate p_value not significant
    mock_separation.return_value = {"p_value": 0.9, "mean_win_score": 50, "mean_loss_score": 50, "n_win": 3, "n_loss": 3}
    mock_models.insert_strategy_simulation.return_value = {"id": 42, "passed": False}

    result = simulate_weight_recommendation("paper")

    assert result == {"id": 42, "passed": False}
    mock_models.insert_adaptive_strategy_version.assert_not_called()


# --- simulate_threshold_recommendation ---


@patch("src.learning.simulation.models")
def test_simulate_threshold_recommendation_none_when_no_pending_recommendation(mock_models):
    mock_models.get_latest_recommendation.return_value = None
    assert simulate_threshold_recommendation("paper") is None


@patch("src.learning.simulation.models")
def test_simulate_threshold_recommendation_none_when_not_pending_status(mock_models):
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 70, "status": "dismissed"}
    assert simulate_threshold_recommendation("paper") is None


@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 100)
@patch("src.learning.simulation.models")
def test_simulate_threshold_recommendation_none_when_insufficient_trades(mock_models):
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 70, "status": "pending"}
    mock_models.get_recently_closed_trades.return_value = [_trade(1, 10, "2026-01-01T00:00:00Z")]
    assert simulate_threshold_recommendation("paper") is None
    mock_models.insert_strategy_simulation.assert_not_called()


@patch("src.learning.simulation.MIN_OPPORTUNITY_SCORE", 60)
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_threshold_recommendation_evaluates_only_test_window(mock_models):
    mock_models.get_latest_recommendation.return_value = {
        "recommended_value": 80, "status": "pending", "batch_id": None,
    }
    # 4 train trades (older), 4 test trades (newer) at an 0.7 split -> with
    # split_pct default 0.7 on 8 trades, train gets floor(8*0.7)=5, test=3;
    # use RECOMMENDATION_MIN_SAMPLE_SIZE=2 so both halves clear the floor
    trades = [_trade(i, 10, f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 9)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}

    # test window trades (the last 3, ids 6,7,8) score high enough for both
    # baseline (>=60) and candidate (>=80); pnl varies so stdev is nonzero
    scored = {6: (85, 50), 7: (85, -10), 8: (85, 30)}

    def _entry_eval(trade_id):
        if trade_id not in scored:
            return None
        score, _ = scored[trade_id]
        return {"opportunity_score": score}

    mock_models.get_entry_evaluation_for_trade.side_effect = _entry_eval
    mock_models.insert_strategy_simulation.return_value = {"id": 7, "passed": False}

    result = simulate_threshold_recommendation("paper")

    assert result is not None
    mock_models.insert_strategy_simulation.assert_called_once()
