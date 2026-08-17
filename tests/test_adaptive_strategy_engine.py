from unittest.mock import patch

from src.learning.adaptive_strategy_engine import AdaptiveStrategyEngine

_PATCHES = (
    "src.learning.adaptive_strategy_engine.models",
    "src.learning.adaptive_strategy_engine.compute_feature_importance",
    "src.learning.adaptive_strategy_engine.rejection_breakdown",
    "src.learning.adaptive_strategy_engine.identify_weaknesses",
    "src.learning.adaptive_strategy_engine.simulate_exit_params_recommendation",
    "src.learning.adaptive_strategy_engine.simulate_threshold_recommendation",
    "src.learning.adaptive_strategy_engine.simulate_weight_recommendation",
    "src.learning.adaptive_strategy_engine.generate_exit_params_recommendations",
    "src.learning.adaptive_strategy_engine.generate_recommendations",
    "src.learning.adaptive_strategy_engine.generate_symbol_recommendations",
    "src.learning.adaptive_strategy_engine.generate_regime_recommendations",
    "src.learning.adaptive_strategy_engine.generate_weight_recommendations",
)


@patch(_PATCHES[0])
@patch(_PATCHES[1])
@patch(_PATCHES[2])
@patch(_PATCHES[3])
@patch(_PATCHES[4])
@patch(_PATCHES[5])
@patch(_PATCHES[6])
@patch(_PATCHES[7])
@patch(_PATCHES[8])
@patch(_PATCHES[9])
@patch(_PATCHES[10])
@patch(_PATCHES[11])
def test_analyze_composes_all_generators_and_simulations(
    mock_weight_recs,
    mock_regime_recs,
    mock_symbol_recs,
    mock_threshold_recs,
    mock_exit_params_recs,
    mock_simulate_weight,
    mock_simulate_threshold,
    mock_simulate_exit_params,
    mock_weaknesses,
    mock_rejections,
    mock_feature_importance,
    mock_models,
):
    mock_weight_recs.return_value = [{"metric_name": "OPPORTUNITY_WEIGHT_TREND", "batch_id": "abc"}]
    mock_regime_recs.return_value = [{"metric_name": "avoid_regime:sideways"}]
    mock_symbol_recs.return_value = []
    mock_threshold_recs.return_value = [{"metric_name": "MIN_OPPORTUNITY_SCORE"}]
    mock_exit_params_recs.return_value = [{"metric_name": "stop_loss_pct"}]
    mock_simulate_weight.return_value = {"id": 1, "passed": True}
    mock_simulate_threshold.return_value = {"id": 2, "passed": False}
    mock_simulate_exit_params.return_value = [{"id": 3, "passed": True}]
    mock_weaknesses.return_value = {"worst_by_dimension": {}}
    mock_rejections.return_value = [{"reason": "block_max_positions", "count": 5, "pct_of_rejections": 100.0}]
    mock_feature_importance.return_value = [{"feature_name": "rsi", "timeframe": "1m"}]

    result = AdaptiveStrategyEngine().analyze(mode="paper")

    mock_weight_recs.assert_called_once_with("paper")
    mock_regime_recs.assert_called_once_with("paper")
    mock_symbol_recs.assert_called_once_with("paper")
    mock_threshold_recs.assert_called_once_with("paper", weakness_context=mock_weaknesses.return_value)
    mock_exit_params_recs.assert_called_once_with("paper")
    mock_simulate_weight.assert_called_once_with("paper", "abc")
    mock_simulate_threshold.assert_called_once_with("paper")
    mock_simulate_exit_params.assert_called_once_with("paper", symbol_to_pair=None)

    assert result["candidates_created"] == 2  # weight + exit-params simulations passed
    assert result["simulations"] == [
        {"id": 1, "passed": True}, {"id": 2, "passed": False}, {"id": 3, "passed": True},
    ]
    assert result["weaknesses"] == {"worst_by_dimension": {}}
    assert result["rejection_breakdown"][0]["reason"] == "block_max_positions"
    mock_models.log_agent_event.assert_called_once()


@patch(_PATCHES[0])
@patch(_PATCHES[1])
@patch(_PATCHES[2])
@patch(_PATCHES[3])
@patch(_PATCHES[4])
@patch(_PATCHES[5])
@patch(_PATCHES[6])
@patch(_PATCHES[7])
@patch(_PATCHES[8])
@patch(_PATCHES[9])
@patch(_PATCHES[10])
@patch(_PATCHES[11])
def test_analyze_skips_simulation_when_no_recommendations_generated(
    mock_weight_recs,
    mock_regime_recs,
    mock_symbol_recs,
    mock_threshold_recs,
    mock_exit_params_recs,
    mock_simulate_weight,
    mock_simulate_threshold,
    mock_simulate_exit_params,
    mock_weaknesses,
    mock_rejections,
    mock_feature_importance,
    mock_models,
):
    mock_weight_recs.return_value = []
    mock_regime_recs.return_value = []
    mock_symbol_recs.return_value = []
    mock_threshold_recs.return_value = []
    mock_exit_params_recs.return_value = []
    mock_weaknesses.return_value = {}
    mock_rejections.return_value = []
    mock_feature_importance.return_value = []

    result = AdaptiveStrategyEngine().analyze()

    mock_simulate_weight.assert_not_called()
    mock_simulate_threshold.assert_not_called()
    mock_simulate_exit_params.assert_not_called()
    assert result["simulations"] == []
    assert result["candidates_created"] == 0
