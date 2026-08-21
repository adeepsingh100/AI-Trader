from unittest.mock import patch

from src.learning.learning_status import LearningStatus
from src.learning.simulation import (
    _activate_exit_params_candidate,
    _train_test_split,
    simulate_exit_params_recommendation,
    simulate_threshold_recommendation,
    simulate_weight_recommendation,
)


def _status(trades_collected=0, **overrides):
    """Explicit LearningStatus threaded via status= into every call below
    — can_simulate()/can_create_candidate() compare trades_collected
    against the real config defaults (250/500), so no test here ever
    triggers a real compute_learning_status()/EvidenceEngine DB call."""
    base = dict(
        stage="BOOTSTRAP", trades_collected=trades_collected, rejected_trades=0, winning_trades=0,
        losing_trades=0, evidence={}, evidence_readiness_pct=0.0, data_sufficiency_pct=0.0,
        recommendations_count=0, simulations_count=0, candidates_count=0, promotion_eligible=False,
        next_stage=None, trades_to_next_stage=0, evidence_gaps=[], current_activity="", reason="",
    )
    base.update(overrides)
    return LearningStatus(**base)


_SIMULATE_READY = _status(trades_collected=999)  # can_simulate() and can_create_candidate() both True
_NOT_SIMULATE_READY = _status(trades_collected=0)
_SIMULATE_BUT_NOT_VALIDATE = _status(trades_collected=300)  # >=250 (can_simulate) but <500 (can_create_candidate)


def _trade(trade_id, pnl, closed_at):
    return {"id": trade_id, "pnl": pnl, "closed_at": closed_at, "opened_at": "2025-12-01T00:00:00Z"}


def _exit_trade(trade_id, pnl, closed_at, mae_pct=0.0, mfe_pct=0.0, entry_price=100.0, qty=1.0, symbol="BTCINR"):
    return {
        "id": trade_id,
        "pnl": pnl,
        "closed_at": closed_at,
        "opened_at": "2025-12-01T00:00:00Z",
        "entry_price": entry_price,
        "qty": qty,
        "mae_pct": mae_pct,
        "mfe_pct": mfe_pct,
        "symbol": symbol,
    }


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


@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_none_when_insufficient_trades(mock_models):
    assert simulate_weight_recommendation("paper", status=_NOT_SIMULATE_READY) is None
    mock_models.insert_strategy_simulation.assert_not_called()


@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.compute_subscore_correlation_weights")
@patch("src.learning.simulation.models")
def test_simulate_weight_recommendation_none_when_no_candidate_from_train_window(mock_models, mock_weights):
    trades = [_trade(i, 10, f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 5)]
    mock_models.get_recently_closed_trades.return_value = trades
    mock_weights.return_value = None
    assert simulate_weight_recommendation("paper", status=_SIMULATE_READY) is None
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

    result = simulate_weight_recommendation("paper", batch_id="abc", status=_SIMULATE_READY)

    assert result == {"id": 42, "passed": True}
    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is True
    assert kwargs["p_value"] == 0.01
    mock_models.insert_adaptive_strategy_version.assert_called_once()
    version_kwargs = mock_models.insert_adaptive_strategy_version.call_args.kwargs
    assert version_kwargs["version_number"] == 1
    assert version_kwargs["params_json"] == {"trend_score": 1.0}
    assert version_kwargs["source_simulation_id"] == 42
    # Scope boundary: weight candidates stay advisory, unlike exit-params
    # ones — no strategy_versions row auto-created for this candidate type.
    mock_models.insert_strategy_version.assert_not_called()


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

    result = simulate_weight_recommendation("paper", status=_SIMULATE_READY)

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


@patch("src.learning.simulation.models")
def test_simulate_threshold_recommendation_none_when_insufficient_trades(mock_models):
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 70, "status": "pending"}
    assert simulate_threshold_recommendation("paper", status=_NOT_SIMULATE_READY) is None
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

    result = simulate_threshold_recommendation("paper", status=_SIMULATE_READY)

    assert result is not None
    mock_models.insert_strategy_simulation.assert_called_once()


# --- simulate_exit_params_recommendation ---


@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_empty_when_insufficient_trades(mock_models):
    mock_models.get_latest_recommendation.return_value = {"recommended_value": 0.02, "status": "pending"}
    assert simulate_exit_params_recommendation("paper", status=_NOT_SIMULATE_READY) == []
    mock_models.insert_strategy_simulation.assert_not_called()


@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_skips_leg_with_no_pending_recommendation(mock_models):
    mock_models.get_latest_recommendation.return_value = None
    assert simulate_exit_params_recommendation("paper", status=_SIMULATE_READY) == []


@patch("src.learning.simulation.bootstrap_confidence_interval")
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_passes_and_creates_candidate(mock_models, mock_bootstrap):
    mock_models.get_latest_recommendation.side_effect = lambda mode, name: (
        {"recommended_value": 0.02, "status": "pending", "batch_id": None, "rationale": "tight stop hypothesis"}
        if name == "stop_loss_pct"
        else None
    )
    train_trades = [_exit_trade(i, pnl=10, closed_at=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 6)]
    test_trades = [
        _exit_trade(6, pnl=-30, closed_at="2026-01-06T00:00:00Z", mae_pct=5.0),
        _exit_trade(7, pnl=-40, closed_at="2026-01-07T00:00:00Z", mae_pct=5.0),
        _exit_trade(8, pnl=-35, closed_at="2026-01-08T00:00:00Z", mae_pct=5.0),
    ]
    mock_models.get_recently_closed_trades.return_value = train_trades + test_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {
        "id": 3, "version_number": 3, "prompt_text": "current prompt", "params_json": {"take_profit_pct": 0.05},
    }
    mock_models.get_latest_adaptive_strategy_version.return_value = None
    mock_models.insert_strategy_simulation.return_value = {"id": 9}
    mock_models.insert_adaptive_strategy_version.return_value = {
        "id": 42, "params_json": {"stop_loss_pct": 0.02}, "fitness_score": 70.0,
    }
    mock_models.insert_strategy_version.return_value = {"id": 100, "version_number": 4}
    mock_bootstrap.return_value = {
        "point_estimate": 0.01, "ci_low": 0.001, "ci_high": 0.02, "confidence_pct": 95.0, "iterations": 1000,
    }

    results = simulate_exit_params_recommendation("paper", status=_SIMULATE_READY)

    assert len(results) == 1
    mock_models.insert_strategy_simulation.assert_called_once()
    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is True
    assert "Observation:" in kwargs["research_note"]
    assert "Decision: Promoted" in kwargs["research_note"]
    assert kwargs["validation_detail"]["bootstrap_ci"]["ci_low"] == 0.001
    mock_models.insert_adaptive_strategy_version.assert_called_once()
    assert mock_models.insert_adaptive_strategy_version.call_args.kwargs["fitness_score"] is not None

    # Auto-activation: the candidate's one changed leg is merged onto the
    # current version's OTHER leg, not a bare replacement of params_json.
    mock_models.insert_strategy_version.assert_called_once()
    activate_kwargs = mock_models.insert_strategy_version.call_args.kwargs
    assert activate_kwargs["version_number"] == 4
    assert activate_kwargs["prompt_text"] == "current prompt"
    assert activate_kwargs["params_json"] == {"take_profit_pct": 0.05, "stop_loss_pct": 0.02}


@patch("src.learning.simulation.bootstrap_confidence_interval")
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_rejected_when_bootstrap_ci_crosses_zero(mock_models, mock_bootstrap):
    mock_models.get_latest_recommendation.side_effect = lambda mode, name: (
        {"recommended_value": 0.02, "status": "pending", "batch_id": None, "rationale": "tight stop hypothesis"}
        if name == "stop_loss_pct"
        else None
    )
    train_trades = [_exit_trade(i, pnl=10, closed_at=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 6)]
    test_trades = [
        _exit_trade(6, pnl=-30, closed_at="2026-01-06T00:00:00Z", mae_pct=5.0),
        _exit_trade(7, pnl=-40, closed_at="2026-01-07T00:00:00Z", mae_pct=5.0),
        _exit_trade(8, pnl=-35, closed_at="2026-01-08T00:00:00Z", mae_pct=5.0),
    ]
    mock_models.get_recently_closed_trades.return_value = train_trades + test_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_models.insert_strategy_simulation.return_value = {"id": 9}
    # CI lower bound crosses zero -> the z-test passed but the bootstrap
    # gate must still block promotion.
    mock_bootstrap.return_value = {
        "point_estimate": 0.01, "ci_low": -0.001, "ci_high": 0.03, "confidence_pct": 95.0, "iterations": 1000,
    }

    simulate_exit_params_recommendation("paper", status=_SIMULATE_READY)

    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is False
    mock_models.insert_adaptive_strategy_version.assert_not_called()


@patch("src.learning.simulation._has_historical_candles", return_value=False)
@patch("src.learning.simulation.bootstrap_confidence_interval")
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_skips_backtest_replay_without_candle_data(
    mock_models, mock_bootstrap, mock_has_candles
):
    mock_models.get_latest_recommendation.side_effect = lambda mode, name: (
        {"recommended_value": 0.02, "status": "pending", "batch_id": None, "rationale": "tight stop hypothesis"}
        if name == "stop_loss_pct"
        else None
    )
    train_trades = [_exit_trade(i, pnl=10, closed_at=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 6)]
    test_trades = [
        _exit_trade(6, pnl=-30, closed_at="2026-01-06T00:00:00Z", mae_pct=5.0),
        _exit_trade(7, pnl=-40, closed_at="2026-01-07T00:00:00Z", mae_pct=5.0),
        _exit_trade(8, pnl=-35, closed_at="2026-01-08T00:00:00Z", mae_pct=5.0),
    ]
    mock_models.get_recently_closed_trades.return_value = train_trades + test_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {
        "id": 3, "version_number": 3, "prompt_text": "current prompt", "params_json": {},
    }
    mock_models.get_latest_adaptive_strategy_version.return_value = None
    mock_models.insert_strategy_simulation.return_value = {"id": 9}
    mock_models.insert_adaptive_strategy_version.return_value = {
        "id": 42, "params_json": {"stop_loss_pct": 0.02}, "fitness_score": 70.0,
    }
    mock_models.insert_strategy_version.return_value = {"id": 100, "version_number": 4}
    mock_bootstrap.return_value = {
        "point_estimate": 0.01, "ci_low": 0.001, "ci_high": 0.02, "confidence_pct": 95.0, "iterations": 1000,
    }

    simulate_exit_params_recommendation("paper", symbol_to_pair={"BTCINR": "I-BTC_INR"}, status=_SIMULATE_READY)

    mock_has_candles.assert_called_once()
    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert "strategy_comparison" not in (kwargs["validation_detail"] or {})
    assert kwargs["passed"] is True


@patch("src.learning.simulation.bootstrap_confidence_interval")
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_defers_candidate_below_validation_stage(mock_models, mock_bootstrap):
    """Evidence-Driven Learning Progression: a simulation can pass its
    z-test + bootstrap CI at Stage 3 (status.can_simulate() True at 300
    trades) without yet clearing status.can_create_candidate() (False
    below 500) — the strategy_simulations row still honestly records
    passed=True, but no adaptive_strategy_versions candidate is created,
    and the research note explains why."""
    mock_models.get_latest_recommendation.side_effect = lambda mode, name: (
        {"recommended_value": 0.02, "status": "pending", "batch_id": None, "rationale": "tight stop hypothesis"}
        if name == "stop_loss_pct"
        else None
    )
    train_trades = [_exit_trade(i, pnl=10, closed_at=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 6)]
    test_trades = [
        _exit_trade(6, pnl=-30, closed_at="2026-01-06T00:00:00Z", mae_pct=5.0),
        _exit_trade(7, pnl=-40, closed_at="2026-01-07T00:00:00Z", mae_pct=5.0),
        _exit_trade(8, pnl=-35, closed_at="2026-01-08T00:00:00Z", mae_pct=5.0),
    ]
    mock_models.get_recently_closed_trades.return_value = train_trades + test_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_models.insert_strategy_simulation.return_value = {"id": 9}
    mock_bootstrap.return_value = {
        "point_estimate": 0.01, "ci_low": 0.001, "ci_high": 0.02, "confidence_pct": 95.0, "iterations": 1000,
    }

    simulate_exit_params_recommendation("paper", status=_SIMULATE_BUT_NOT_VALIDATE)

    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is True  # statistically genuine pass
    assert "Stage gate:" in kwargs["research_note"]
    assert "Decision: Rejected" in kwargs["research_note"]  # no candidate was actually created
    mock_models.insert_adaptive_strategy_version.assert_not_called()


@patch("src.learning.simulation._backtest_replay_gate")
@patch("src.learning.simulation._has_historical_candles", return_value=True)
@patch("src.learning.simulation.bootstrap_confidence_interval")
@patch("src.learning.simulation.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.learning.simulation.RECOMMENDATION_MIN_SAMPLE_SIZE", 2)
@patch("src.learning.simulation.models")
def test_simulate_exit_params_recommendation_backtest_replay_rejects_when_baseline_wins(
    mock_models, mock_bootstrap, mock_has_candles, mock_replay
):
    mock_models.get_latest_recommendation.side_effect = lambda mode, name: (
        {"recommended_value": 0.02, "status": "pending", "batch_id": None, "rationale": "tight stop hypothesis"}
        if name == "stop_loss_pct"
        else None
    )
    train_trades = [_exit_trade(i, pnl=10, closed_at=f"2026-01-{i:02d}T00:00:00Z") for i in range(1, 6)]
    test_trades = [
        _exit_trade(6, pnl=-30, closed_at="2026-01-06T00:00:00Z", mae_pct=5.0),
        _exit_trade(7, pnl=-40, closed_at="2026-01-07T00:00:00Z", mae_pct=5.0),
        _exit_trade(8, pnl=-35, closed_at="2026-01-08T00:00:00Z", mae_pct=5.0),
    ]
    mock_models.get_recently_closed_trades.return_value = train_trades + test_trades
    mock_models.get_capital_config.return_value = {"capital_to_use": 1000}
    mock_models.get_latest_version.return_value = {"params_json": {}}
    mock_models.insert_strategy_simulation.return_value = {"id": 9}
    mock_bootstrap.return_value = {
        "point_estimate": 0.01, "ci_low": 0.001, "ci_high": 0.02, "confidence_pct": 95.0, "iterations": 1000,
    }
    mock_replay.return_value = {"winner": "a", "promotion_recommended": False, "p_values": {}}

    simulate_exit_params_recommendation("paper", symbol_to_pair={"BTCINR": "I-BTC_INR"}, status=_SIMULATE_READY)

    kwargs = mock_models.insert_strategy_simulation.call_args.kwargs
    assert kwargs["passed"] is False
    assert kwargs["validation_detail"]["strategy_comparison"]["winner"] == "a"
    mock_models.insert_adaptive_strategy_version.assert_not_called()


# --- _activate_exit_params_candidate ---


@patch("src.learning.simulation.models")
def test_activate_exit_params_candidate_merges_onto_current_params(mock_models):
    mock_models.get_latest_version.return_value = {
        "id": 3, "version_number": 3, "prompt_text": "current prompt", "params_json": {"take_profit_pct": 0.05},
    }
    mock_models.insert_strategy_version.return_value = {"id": 100, "version_number": 4}
    candidate = {"id": 42, "params_json": {"stop_loss_pct": 0.02}, "fitness_score": 70.0}

    result = _activate_exit_params_candidate(candidate)

    assert result == {"id": 100, "version_number": 4}
    mock_models.insert_strategy_version.assert_called_once_with(
        version_number=4,
        prompt_text="current prompt",
        params_json={"take_profit_pct": 0.05, "stop_loss_pct": 0.02},
        notes="Auto-activated from adaptive_strategy_versions candidate 42 (fitness=70.0).",
    )
    mock_models.log_agent_event.assert_called_once()


@patch("src.learning.simulation.models")
def test_activate_exit_params_candidate_noop_when_no_current_version(mock_models):
    mock_models.get_latest_version.return_value = None
    candidate = {"id": 42, "params_json": {"stop_loss_pct": 0.02}, "fitness_score": 70.0}

    assert _activate_exit_params_candidate(candidate) is None
    mock_models.insert_strategy_version.assert_not_called()
