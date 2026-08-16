from unittest.mock import patch

from src.learning.adaptive_strategy_engine import AdaptiveStrategyEngine


@patch("src.learning.adaptive_strategy_engine.models")
@patch("src.learning.adaptive_strategy_engine.compute_feature_importance")
@patch("src.learning.adaptive_strategy_engine.simulate_threshold_recommendation")
@patch("src.learning.adaptive_strategy_engine.simulate_weight_recommendation")
@patch("src.learning.adaptive_strategy_engine.generate_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_symbol_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_regime_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_weight_recommendations")
def test_analyze_composes_all_generators_and_simulations(
    mock_weight_recs,
    mock_regime_recs,
    mock_symbol_recs,
    mock_threshold_recs,
    mock_simulate_weight,
    mock_simulate_threshold,
    mock_feature_importance,
    mock_models,
):
    mock_weight_recs.return_value = [{"metric_name": "OPPORTUNITY_WEIGHT_TREND", "batch_id": "abc"}]
    mock_regime_recs.return_value = [{"metric_name": "avoid_regime:sideways"}]
    mock_symbol_recs.return_value = []
    mock_threshold_recs.return_value = [{"metric_name": "MIN_OPPORTUNITY_SCORE"}]
    mock_simulate_weight.return_value = {"id": 1, "passed": True}
    mock_simulate_threshold.return_value = {"id": 2, "passed": False}
    mock_feature_importance.return_value = [{"feature_name": "rsi", "timeframe": "1m"}]

    result = AdaptiveStrategyEngine().analyze(mode="paper")

    mock_weight_recs.assert_called_once_with("paper")
    mock_regime_recs.assert_called_once_with("paper")
    mock_symbol_recs.assert_called_once_with("paper")
    mock_threshold_recs.assert_called_once_with("paper")
    mock_simulate_weight.assert_called_once_with("paper", "abc")
    mock_simulate_threshold.assert_called_once_with("paper")

    assert result["candidates_created"] == 1  # only the weight simulation passed
    assert result["simulations"] == [{"id": 1, "passed": True}, {"id": 2, "passed": False}]
    mock_models.log_agent_event.assert_called_once()


@patch("src.learning.adaptive_strategy_engine.models")
@patch("src.learning.adaptive_strategy_engine.compute_feature_importance")
@patch("src.learning.adaptive_strategy_engine.simulate_threshold_recommendation")
@patch("src.learning.adaptive_strategy_engine.simulate_weight_recommendation")
@patch("src.learning.adaptive_strategy_engine.generate_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_symbol_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_regime_recommendations")
@patch("src.learning.adaptive_strategy_engine.generate_weight_recommendations")
def test_analyze_skips_simulation_when_no_recommendations_generated(
    mock_weight_recs,
    mock_regime_recs,
    mock_symbol_recs,
    mock_threshold_recs,
    mock_simulate_weight,
    mock_simulate_threshold,
    mock_feature_importance,
    mock_models,
):
    mock_weight_recs.return_value = []
    mock_regime_recs.return_value = []
    mock_symbol_recs.return_value = []
    mock_threshold_recs.return_value = []
    mock_feature_importance.return_value = []

    result = AdaptiveStrategyEngine().analyze()

    mock_simulate_weight.assert_not_called()
    mock_simulate_threshold.assert_not_called()
    assert result["simulations"] == []
    assert result["candidates_created"] == 0
