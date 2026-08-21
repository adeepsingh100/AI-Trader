import datetime
from unittest.mock import patch

from src.agents.evolution_agent import compute_metrics, promotion_ready, run_evolution
from src.learning.learning_status import LearningStatus
from src.learning.promotion_gate import PromotionDecision


def _trade(pnl, closed_at):
    return {"pnl": pnl, "closed_at": closed_at}


def _status(stage, trades_collected):
    return LearningStatus(
        stage=stage, trades_collected=trades_collected, rejected_trades=0, winning_trades=0,
        losing_trades=0, evidence={}, evidence_readiness_pct=0.0, data_sufficiency_pct=0.0,
        recommendations_count=0, simulations_count=0, candidates_count=0, promotion_eligible=False,
        next_stage=None, trades_to_next_stage=0, evidence_gaps=[], current_activity="", reason="",
    )


# --- compute_metrics ---


def test_compute_metrics_win_rate_and_averages():
    trades = [_trade(100, 1), _trade(-50, 2), _trade(200, 3), _trade(-25, 4)]
    metrics = compute_metrics(trades, capital_to_use=10000)

    assert metrics["trades_count"] == 4
    assert metrics["win_rate"] == 0.5
    assert metrics["avg_win"] == 150.0
    assert metrics["avg_loss"] == -37.5
    assert metrics["cumulative_pnl"] == 225


def test_compute_metrics_ignores_open_trades():
    trades = [_trade(100, 1), {"pnl": None, "closed_at": None}]
    metrics = compute_metrics(trades, capital_to_use=10000)
    assert metrics["trades_count"] == 1


def test_compute_metrics_empty_trades():
    metrics = compute_metrics([], capital_to_use=10000)
    assert metrics["win_rate"] == 0.0
    assert metrics["avg_win"] == 0.0
    assert metrics["avg_loss"] == 0.0
    assert metrics["cumulative_pnl"] == 0
    assert metrics["max_drawdown_pct"] == 0.0


def test_max_drawdown_uses_peak_to_trough_not_final_value():
    # runs up to +500 then down to +100: drawdown is the 400 pullback,
    # not (final - start) which would only be 100
    trades = [_trade(500, 1), _trade(-300, 2), _trade(-100, 3)]
    metrics = compute_metrics(trades, capital_to_use=1000)
    assert metrics["max_drawdown_pct"] == 40.0  # 400 / 1000 * 100


def test_max_drawdown_sorts_by_closed_at_not_input_order():
    trades = [_trade(-300, 2), _trade(500, 1), _trade(-100, 3)]  # out of order
    metrics = compute_metrics(trades, capital_to_use=1000)
    assert metrics["max_drawdown_pct"] == 40.0


def test_max_drawdown_zero_when_only_winners():
    trades = [_trade(100, 1), _trade(200, 2)]
    metrics = compute_metrics(trades, capital_to_use=1000)
    assert metrics["max_drawdown_pct"] == 0.0


# --- promotion_ready ---


def _version(days_ago, promoted=False, promotion_eligible=False):
    created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
    return {
        "id": 1,
        "created_at": created.isoformat(),
        "promoted_to_real": promoted,
        "promotion_eligible": promotion_eligible,
    }


def _metrics(**overrides):
    base = {"cumulative_pnl": 100.0, "max_drawdown_pct": 5.0}
    base.update(overrides)
    return base


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
def test_promotion_ready_true_when_all_criteria_met():
    assert promotion_ready(_version(days_ago=14), _metrics()) is True


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
def test_promotion_blocked_when_too_young():
    assert promotion_ready(_version(days_ago=13), _metrics()) is False


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
def test_promotion_blocked_when_pnl_negative():
    assert promotion_ready(_version(days_ago=14), _metrics(cumulative_pnl=-1)) is False


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
def test_promotion_blocked_when_drawdown_too_deep():
    assert promotion_ready(_version(days_ago=14), _metrics(max_drawdown_pct=16)) is False


# --- run_evolution ---
# The old promotion_eligible() 5-gate boolean is retired — run_evolution
# now delegates the actual decision to src/learning/promotion_gate.py::
# evaluate_promotion (see tests/test_promotion_gate.py for its own
# thorough coverage). These tests only prove run_evolution's WIRING:
# PROMOTE -> promote_version + eligible flag + audit row; REJECT/
# EXTEND_VALIDATION -> neither; already-promoted/real-mode -> the gate
# isn't even evaluated.
# No LLM call, no new strategy_versions row — a promotion-readiness
# monitor only. Trades are real (small, deterministic) so
# compute_bucket_statistics/compute_fitness_score/bootstrap_confidence_interval
# run for real rather than being mocked away, proving the wiring actually
# works end to end.

_CLEARLY_PROFITABLE_TRADES = [
    {"pnl": pnl, "closed_at": f"2026-01-{i + 1:02d}T00:00:00Z"}
    for i, pnl in enumerate([45, 50, 55, 48, 52, 60, 47, 53, 49, 51])
]


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.learning.promotion_gate.build_symbol_to_pair")
@patch("src.learning.promotion_gate.evaluate_promotion")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_promotes_on_PROMOTE_decision(mock_models, mock_evaluate, mock_build_pair, mock_learning_status):
    status = _status("HYPOTHESIS", 120)
    mock_learning_status.return_value = status
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES
    mock_models.get_latest_promoted_version.return_value = None
    mock_build_pair.return_value = None
    mock_evaluate.return_value = PromotionDecision("PROMOTE", 85.0, {"g": {"passed": True}}, ["all gates cleared"], {})

    result = run_evolution(mode="paper")

    assert result["promotion_eligible"] is True
    assert result["promoted"] is True
    assert result["promotion_decision"] == "PROMOTE"
    assert result["learning_status"] is status
    mock_models.set_strategy_version_promotion_eligible.assert_called_once_with(1, True)
    mock_models.promote_version.assert_called_once_with(1)
    mock_models.insert_promotion_audit.assert_called_once()
    audit_kwargs = mock_models.insert_promotion_audit.call_args.kwargs
    assert audit_kwargs["event_type"] == "promotion"
    assert audit_kwargs["decision"] == "PROMOTE"
    assert audit_kwargs["candidate_version_id"] == 1
    assert audit_kwargs["new_champion_id"] == 1


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.learning.promotion_gate.build_symbol_to_pair")
@patch("src.learning.promotion_gate.evaluate_promotion")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_does_not_promote_on_REJECT_decision(mock_models, mock_evaluate, mock_build_pair, mock_learning_status):
    mock_learning_status.return_value = _status("BOOTSTRAP", 10)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    # Starts eligible=True (e.g. a manual test fixture) so a REJECT
    # decision genuinely flips it to False, proving the write actually
    # fires rather than the flag simply staying at its default.
    version = _version(days_ago=2, promotion_eligible=True)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES
    mock_models.get_latest_promoted_version.return_value = None
    mock_build_pair.return_value = None
    mock_evaluate.return_value = PromotionDecision("REJECT", 20.0, {"g": {"passed": False}}, ["drawdown too deep"], {})

    result = run_evolution(mode="paper")

    assert result["promotion_eligible"] is False
    assert result["promotion_decision"] == "REJECT"
    assert result["promoted"] is False
    mock_models.set_strategy_version_promotion_eligible.assert_called_once_with(1, False)
    mock_models.promote_version.assert_not_called()
    assert mock_models.insert_promotion_audit.call_args.kwargs["event_type"] == "evaluation"


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.learning.promotion_gate.build_symbol_to_pair")
@patch("src.learning.promotion_gate.evaluate_promotion")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_extend_validation_does_not_promote(mock_models, mock_evaluate, mock_build_pair, mock_learning_status):
    mock_learning_status.return_value = _status("BOOTSTRAP", 10)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=2, promotion_eligible=False)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = []
    mock_models.get_latest_promoted_version.return_value = None
    mock_build_pair.return_value = None
    mock_evaluate.return_value = PromotionDecision("EXTEND_VALIDATION", None, {}, ["not enough paper trades"], {})

    run_evolution(mode="paper")

    # eligible stays False, matching the version's existing flag -> no write
    mock_models.set_strategy_version_promotion_eligible.assert_not_called()
    mock_models.promote_version.assert_not_called()
    assert mock_models.insert_promotion_audit.call_args.kwargs["event_type"] == "evaluation"


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.learning.promotion_gate.evaluate_promotion")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_never_evaluates_promotion_for_real_mode(mock_models, mock_evaluate, mock_learning_status):
    mock_learning_status.return_value = _status("HYPOTHESIS", 120)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20, promoted=True)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES

    result = run_evolution(mode="real")

    assert result["promotion_eligible"] is False
    assert result["promotion_decision"] is None
    assert result["promoted"] is False
    mock_models.promote_version.assert_not_called()
    mock_models.insert_promotion_audit.assert_not_called()
    mock_evaluate.assert_not_called()


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.learning.promotion_gate.evaluate_promotion")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_skips_gate_for_already_promoted_version(mock_models, mock_evaluate, mock_learning_status):
    mock_learning_status.return_value = _status("HYPOTHESIS", 120)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20, promoted=True)  # already promoted_to_real
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES

    result = run_evolution(mode="paper")

    assert result["promotion_decision"] is None
    mock_evaluate.assert_not_called()
    mock_models.insert_promotion_audit.assert_not_called()


# --- data retention purge (piggybacked on this already-hourly step) ---


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.agents.evolution_agent.OPERATIONAL_LOG_RETENTION_DAYS", 30)
@patch("src.agents.evolution_agent.LEARNING_HISTORY_WINDOW_DAYS", 180)
@patch("src.agents.evolution_agent.models")
def test_run_evolution_purges_old_data_with_correct_cutoffs(mock_models, mock_learning_status):
    mock_learning_status.return_value = _status("BOOTSTRAP", 3)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    # promoted=True so the promotion-gate branch (unrelated to this test,
    # would otherwise need its own mocking) is skipped entirely.
    mock_models.get_latest_version.return_value = _version(days_ago=1, promoted=True)
    mock_models.get_closed_trades.return_value = []
    mock_models.purge_old_data.return_value = {"opportunity_evaluations": 5, "agent_logs": 2}

    run_evolution(mode="paper")

    cutoffs = mock_models.purge_old_data.call_args[0][0]
    now = datetime.datetime.now(datetime.timezone.utc)
    assert set(cutoffs) == {
        "opportunity_evaluations", "confidence_calibration", "agent_logs",
        "model_usage", "system_metrics", "data_quality_log",
    }
    # learning-relevant tables reuse LEARNING_HISTORY_WINDOW_DAYS (180d);
    # pure operational logs use the shorter OPERATIONAL_LOG_RETENTION_DAYS (30d)
    assert abs((now - cutoffs["opportunity_evaluations"]).days - 180) <= 1
    assert abs((now - cutoffs["agent_logs"]).days - 30) <= 1
    mock_models.log_agent_event.assert_any_call(
        "evolution_agent", "info", "data retention purge: {'opportunity_evaluations': 5, 'agent_logs': 2}"
    )


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_purge_failure_fails_open(mock_models, mock_learning_status):
    # a purge error must never block the promotion-monitor result above it.
    mock_learning_status.return_value = _status("BOOTSTRAP", 3)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    mock_models.get_latest_version.return_value = _version(days_ago=1, promoted=True)
    mock_models.get_closed_trades.return_value = []
    mock_models.purge_old_data.side_effect = RuntimeError("supabase unavailable")

    result = run_evolution(mode="paper")

    assert result["promotion_eligible"] is False
