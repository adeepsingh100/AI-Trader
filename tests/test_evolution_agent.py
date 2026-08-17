import datetime
from unittest.mock import patch

from src.agents.evolution_agent import compute_metrics, promotion_eligible, promotion_ready, run_evolution
from src.learning.learning_status import LearningStatus


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


# --- promotion_eligible (Scientific Strategy Optimization Framework) ---


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
def test_promotion_eligible_false_when_promotion_ready_fails():
    trades = [_trade(100, "2026-01-01T00:00:00Z")]
    assert promotion_eligible(_version(days_ago=1), _metrics(), trades, fitness_score=90.0) is False


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
@patch("src.backtest.statistical_validation.bootstrap_confidence_interval")
def test_promotion_eligible_false_when_bootstrap_ci_crosses_zero(mock_bootstrap):
    mock_bootstrap.return_value = {"ci_low": -0.5, "ci_high": 50.0}
    trades = [_trade(100, "2026-01-01T00:00:00Z"), _trade(-90, "2026-01-02T00:00:00Z")]
    assert promotion_eligible(_version(days_ago=14), _metrics(), trades, fitness_score=90.0) is False


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
@patch("src.agents.evolution_agent.PROMOTION_MIN_FITNESS_SCORE", 60)
@patch("src.backtest.statistical_validation.bootstrap_confidence_interval")
def test_promotion_eligible_false_when_fitness_below_floor(mock_bootstrap):
    mock_bootstrap.return_value = {"ci_low": 5.0, "ci_high": 50.0}
    trades = [_trade(100, "2026-01-01T00:00:00Z"), _trade(90, "2026-01-02T00:00:00Z")]
    assert promotion_eligible(_version(days_ago=14), _metrics(), trades, fitness_score=40.0) is False


@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
@patch("src.agents.evolution_agent.PROMOTION_MIN_FITNESS_SCORE", 60)
@patch("src.backtest.statistical_validation.bootstrap_confidence_interval")
def test_promotion_eligible_true_when_everything_clears(mock_bootstrap):
    mock_bootstrap.return_value = {"ci_low": 5.0, "ci_high": 50.0}
    trades = [_trade(100, "2026-01-01T00:00:00Z"), _trade(90, "2026-01-02T00:00:00Z")]
    assert promotion_eligible(_version(days_ago=14), _metrics(), trades, fitness_score=90.0) is True


# --- run_evolution ---
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
@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.PROMOTION_MIN_CUMULATIVE_PNL", 0)
@patch("src.agents.evolution_agent.PROMOTION_MAX_DRAWDOWN_PCT", 15)
@patch("src.agents.evolution_agent.PROMOTION_MIN_FITNESS_SCORE", 60)
@patch("src.agents.evolution_agent.models")
def test_run_evolution_flags_promotion_eligible_when_criteria_clear(mock_models, mock_learning_status):
    status = _status("HYPOTHESIS", 120)
    mock_learning_status.return_value = status
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES

    result = run_evolution(mode="paper")

    assert result["promotion_eligible"] is True
    assert result["learning_status"] is status
    mock_models.set_strategy_version_promotion_eligible.assert_called_once_with(1, True)
    mock_models.insert_strategy_version.assert_not_called()
    mock_models.promote_version.assert_not_called()


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.agents.evolution_agent.PROMOTION_MIN_PAPER_DAYS", 14)
@patch("src.agents.evolution_agent.models")
def test_run_evolution_does_not_flag_eligible_when_too_young(mock_models, mock_learning_status):
    mock_learning_status.return_value = _status("BOOTSTRAP", 10)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    # Starts eligible=True (e.g. a manual test fixture) so the too-young
    # check genuinely flips it to False, proving the guard actually fires
    # rather than the flag simply staying at its already-False default.
    version = _version(days_ago=2, promotion_eligible=True)  # too young
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES

    result = run_evolution(mode="paper")

    assert result["promotion_eligible"] is False
    mock_models.set_strategy_version_promotion_eligible.assert_called_once_with(1, False)


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_skips_setting_flag_when_unchanged(mock_models, mock_learning_status):
    mock_learning_status.return_value = _status("BOOTSTRAP", 0)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=2, promotion_eligible=False)  # too young -> stays False
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = []

    run_evolution(mode="paper")

    mock_models.set_strategy_version_promotion_eligible.assert_not_called()


@patch("src.learning.learning_status.compute_learning_status")
@patch("src.agents.evolution_agent.models")
def test_run_evolution_never_eligible_for_real_mode(mock_models, mock_learning_status):
    mock_learning_status.return_value = _status("HYPOTHESIS", 120)
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20, promoted=True)
    version.update({"id": 1, "version_number": 3})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = _CLEARLY_PROFITABLE_TRADES

    result = run_evolution(mode="real")

    assert result["promotion_eligible"] is False
