from unittest.mock import patch

import pytest

from src.groq_client import AllModelsFailedError
from src.learning.learning_status import LearningStatus
from src.learning.recommendations import (
    _simulate_exit_pnl,
    current_weights,
    generate_ai_exit_params_recommendations,
    generate_exit_params_recommendations,
    generate_indicator_bucket_recommendations,
    generate_recommendations,
    generate_regime_recommendations,
    generate_symbol_recommendations,
    generate_weight_recommendations,
)


def _trade(trade_id, pnl, closed_at="2026-01-01T00:00:00Z", **overrides):
    base = {"id": trade_id, "pnl": pnl, "closed_at": closed_at, "opened_at": "2025-12-31T00:00:00Z", "symbol": "BTCINR"}
    base.update(overrides)
    return base


def _status(trades_collected=0, **overrides):
    """Explicit LearningStatus for every test below — passed straight into
    the status= param every generator now accepts, so no test here ever
    triggers a real compute_learning_status()/EvidenceEngine DB call.
    can_generate_hypotheses() etc. compare trades_collected against the
    real config defaults (100/250/500), so a high trades_collected reads
    as "ready" with zero constant-patching needed."""
    base = dict(
        stage="BOOTSTRAP", trades_collected=trades_collected, rejected_trades=0, winning_trades=0,
        losing_trades=0, evidence={}, evidence_readiness_pct=0.0, data_sufficiency_pct=0.0,
        recommendations_count=0, simulations_count=0, candidates_count=0, promotion_eligible=False,
        next_stage=None, trades_to_next_stage=0, evidence_gaps=[], current_activity="", reason="",
    )
    base.update(overrides)
    return LearningStatus(**base)


_READY = _status(trades_collected=999)
_NOT_READY = _status(trades_collected=0)


# --- generate_recommendations (threshold sweep) ---


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
@patch("src.learning.recommendations.models")
def test_generate_recommendations_below_sample_size_returns_empty(mock_models):
    assert generate_recommendations("paper", status=_NOT_READY) == []


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

    result = generate_recommendations("paper", status=_READY)

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

    assert generate_recommendations("paper", status=_READY) == []
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

    assert generate_recommendations("paper", status=_READY) == []
    mock_models.insert_recommendation.assert_not_called()


# --- generate_weight_recommendations ---


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_weight_recommendations_none_when_no_candidate_weights(mock_models, mock_weights):
    mock_weights.return_value = None
    assert generate_weight_recommendations("paper") == []
    mock_models.get_recently_closed_trades.assert_not_called()


@patch("src.learning.recommendations.compute_subscore_correlation_weights")
@patch("src.learning.recommendations.models")
def test_generate_weight_recommendations_stays_empty_below_hypothesis_stage(mock_models, mock_weights):
    """Evidence-Driven Learning Progression: a valid candidate weight set
    still doesn't get recommended while status.can_generate_hypotheses()
    is False — the LearningStatus gate is checked, not bypassed."""
    mock_weights.return_value = {"trend_score": 0.7}

    assert generate_weight_recommendations("paper", status=_NOT_READY) == []
    mock_models.insert_recommendation.assert_not_called()


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

    result = generate_weight_recommendations("paper", status=_READY)

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

    assert generate_weight_recommendations("paper", status=_READY) == []
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

    result = generate_regime_recommendations("paper", status=_READY)

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

    result = generate_regime_recommendations("paper", status=_READY)

    assert not any(r["metric_name"].startswith("avoid_regime:") for r in result)


# --- generate_indicator_bucket_recommendations: RSI/StochRSI/volatility
# evidence (Phases 7-9), reusing _avoid_bucket_recommendations verbatim ---


@patch("src.learning.recommendations.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.z_test_two_proportions")
@patch("src.learning.recommendations.models")
def test_generate_indicator_bucket_recommendations_flags_worse_than_baseline(mock_models, mock_z_test):
    all_trades = [_trade(i, 100) for i in range(1, 5)] + [_trade(i, -100) for i in range(5, 9)]
    mock_models.get_recently_closed_trades.return_value = all_trades
    # returned for all 3 dimension_type calls (rsi_bucket/stoch_rsi_bucket/
    # atr_volatility_bucket) — the mock doesn't discriminate by call args,
    # which is fine: it exercises the same generic logic 3 times.
    mock_models.get_learning_statistics.return_value = [
        {"dimension_value": "70-80", "trades_count": 4, "win_rate": 0.0},
    ]
    mock_models.get_latest_recommendation.return_value = None
    mock_z_test.return_value = 0.01  # significant

    result = generate_indicator_bucket_recommendations("paper", status=_READY)

    prefixes = {r["metric_name"].split(":")[0] for r in result}
    assert prefixes == {"avoid_rsi_bucket", "avoid_stoch_rsi_bucket", "avoid_volatility_bucket"}
    assert all(r["recommended_value"] == 0.0 for r in result)


@patch("src.learning.recommendations.models")
def test_generate_indicator_bucket_recommendations_not_ready_returns_empty(mock_models):
    assert generate_indicator_bucket_recommendations("paper", status=_NOT_READY) == []
    mock_models.get_recently_closed_trades.assert_not_called()


@patch("src.learning.recommendations.models")
def test_generate_indicator_bucket_recommendations_no_trades_returns_empty(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    assert generate_indicator_bucket_recommendations("paper", status=_READY) == []


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

    result = generate_symbol_recommendations("paper", status=_READY)

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


# --- _simulate_exit_pnl (exit-params candidate approximation) ---


def test_simulate_exit_pnl_stop_triggers_when_mae_exceeds_candidate():
    trade = {"entry_price": 100.0, "qty": 1.0, "mae_pct": 5.0, "mfe_pct": 0.5, "pnl": -30.0}
    assert _simulate_exit_pnl(trade, stop_loss_pct=0.02, take_profit_pct=None) == pytest.approx(-2.0)


def test_simulate_exit_pnl_stop_not_triggered_returns_actual_pnl():
    trade = {"entry_price": 100.0, "qty": 1.0, "mae_pct": 1.0, "mfe_pct": 0.5, "pnl": -5.0}
    assert _simulate_exit_pnl(trade, stop_loss_pct=0.02, take_profit_pct=None) == pytest.approx(-5.0)


def test_simulate_exit_pnl_target_triggers_when_mfe_exceeds_candidate():
    trade = {"entry_price": 100.0, "qty": 1.0, "mae_pct": 0.5, "mfe_pct": 6.0, "pnl": 20.0}
    assert _simulate_exit_pnl(trade, stop_loss_pct=None, take_profit_pct=0.03) == pytest.approx(3.0)


# --- generate_exit_params_recommendations ---


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
@patch("src.learning.recommendations.models")
def test_generate_exit_params_recommendations_below_sample_size_returns_empty(mock_models):
    assert generate_exit_params_recommendations("paper", status=_NOT_READY) == []


@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MIN_PCT", 0.01)
@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MAX_PCT", 0.05)
@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_STEP_PCT", 0.01)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_IMPROVEMENT_PCT", 10)
@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.MIN_EXPECTANCY_DELTA", 1.0)
@patch("src.learning.recommendations.models")
def test_generate_exit_params_recommendations_proposes_tighter_stop_on_clear_improvement(mock_models):
    # Every trade blew past a 1% adverse move before eventually closing at
    # a much bigger loss with no stop enforced — a tight stop clearly
    # would have helped, and none reached favorable territory (no
    # take_profit_pct candidate should fire).
    trades = [
        _trade(i, pnl=-30.0, entry_price=100.0, qty=1.0, mae_pct=5.0, mfe_pct=0.2)
        for i in range(1, 5)
    ]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_models.get_latest_recommendation.return_value = None

    result = generate_exit_params_recommendations("paper", status=_READY)

    assert result
    metric_names = {r["metric_name"] for r in result}
    assert "stop_loss_pct" in metric_names
    assert "take_profit_pct" not in metric_names  # mfe_pct never cleared any candidate
    calls = mock_models.insert_recommendation.call_args_list
    assert all(c.kwargs["category"] == "exit_params" for c in calls)


@patch("src.learning.recommendations.RECOMMENDATION_MIN_SAMPLE_SIZE", 4)
@patch("src.learning.recommendations.models")
def test_generate_exit_params_recommendations_empty_when_baseline_expectancy_none(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    assert generate_exit_params_recommendations("paper", status=_READY) == []


# --- generate_ai_exit_params_recommendations ---
#
# AI-assisted sibling: same idempotency/staging gates, but the candidate
# value comes from a mocked chat() call instead of a sweep. Never lets an
# LLM failure raise — the hourly job it feeds must keep going regardless.


def _ai_trades():
    return [_trade(i, pnl=-30.0, entry_price=100.0, qty=1.0, mae_pct=5.0, mfe_pct=0.2) for i in range(1, 5)]


@patch("src.learning.recommendations.chat")
@patch("src.learning.recommendations.models")
def test_generate_ai_exit_params_recommendations_below_stage_gate_returns_empty_without_calling_chat(
    mock_models, mock_chat
):
    assert generate_ai_exit_params_recommendations("paper", status=_NOT_READY) == []
    mock_chat.assert_not_called()


@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MIN_PCT", 0.005)
@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MAX_PCT", 0.10)
@patch("src.learning.recommendations.chat")
@patch("src.learning.recommendations.models")
def test_generate_ai_exit_params_recommendations_writes_recommendation_from_valid_proposal(
    mock_models, mock_chat
):
    mock_models.get_recently_closed_trades.return_value = _ai_trades()
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {"stop_loss_pct": 0.02}}
    mock_models.get_latest_recommendation.return_value = None
    mock_chat.return_value = (
        '{"stop_loss_pct": 0.03, "take_profit_pct": null, "rationale": "widen stop, MAE consistently exceeds 2%"}',
        ["usage-event"],
    )

    result = generate_ai_exit_params_recommendations("paper", status=_READY)

    assert result == [
        {
            "metric_name": "stop_loss_pct",
            "recommended_value": 0.03,
            "rationale": "AI-proposed: widen stop, MAE consistently exceeds 2%",
        }
    ]
    mock_models.insert_recommendation.assert_called_once()
    call_kwargs = mock_models.insert_recommendation.call_args.kwargs
    assert call_kwargs["metric_name"] == "stop_loss_pct"
    assert call_kwargs["recommended_value"] == 0.03
    assert call_kwargs["category"] == "exit_params"
    assert call_kwargs["evidence"]["ai_raw_response"]["stop_loss_pct"] == 0.03
    mock_models.log_model_usage.assert_called_once_with(["usage-event"])


@patch("src.learning.recommendations.chat")
@patch("src.learning.recommendations.models")
def test_generate_ai_exit_params_recommendations_all_models_failed_returns_empty_without_raising(
    mock_models, mock_chat
):
    mock_models.get_recently_closed_trades.return_value = _ai_trades()
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_chat.side_effect = AllModelsFailedError("all models failed", events=[])

    result = generate_ai_exit_params_recommendations("paper", status=_READY)

    assert result == []
    mock_models.insert_recommendation.assert_not_called()
    mock_models.log_agent_event.assert_called_once()
    assert mock_models.log_agent_event.call_args.args[0] == "ai_strategy_proposer"


@patch("src.learning.recommendations.chat")
@patch("src.learning.recommendations.models")
def test_generate_ai_exit_params_recommendations_unparseable_response_returns_empty(mock_models, mock_chat):
    mock_models.get_recently_closed_trades.return_value = _ai_trades()
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_chat.return_value = ("not json at all {{{", [])

    result = generate_ai_exit_params_recommendations("paper", status=_READY)

    assert result == []
    mock_models.insert_recommendation.assert_not_called()
    mock_models.log_agent_event.assert_called_once()


@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MIN_PCT", 0.005)
@patch("src.learning.recommendations.EXIT_PARAM_SWEEP_MAX_PCT", 0.10)
@patch("src.learning.recommendations.chat")
@patch("src.learning.recommendations.models")
def test_generate_ai_exit_params_recommendations_out_of_range_value_ignored(mock_models, mock_chat):
    mock_models.get_recently_closed_trades.return_value = _ai_trades()
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_models.get_latest_recommendation.return_value = None
    # way outside the valid sweep range — must be rejected, not blindly trusted
    mock_chat.return_value = ('{"stop_loss_pct": 5.0, "take_profit_pct": null, "rationale": "x"}', [])

    result = generate_ai_exit_params_recommendations("paper", status=_READY)

    assert result == []
    mock_models.insert_recommendation.assert_not_called()
