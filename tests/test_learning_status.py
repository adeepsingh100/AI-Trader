from unittest.mock import patch

from src.learning.learning_status import compute_learning_status


def _closed_trades(n, wins):
    return [{"pnl": 10.0 if i < wins else -10.0} for i in range(n)]


def _status(mock_models, trades_count, wins=0, rejected=0, version=None):
    mock_models.get_recently_closed_trades.return_value = _closed_trades(trades_count, wins)
    mock_models.get_hold_evaluations_since.return_value = [{}] * rejected
    mock_models.get_recommendations.return_value = []
    mock_models.get_strategy_simulations.return_value = []
    mock_models.get_adaptive_strategy_versions.return_value = []
    mock_models.get_latest_version.return_value = version
    return compute_learning_status("paper")


@patch("src.learning.learning_status.models")
def test_stage_bootstrap_below_observation_boundary(mock_models):
    status = _status(mock_models, 24)
    assert status["stage"] == "BOOTSTRAP"
    assert status["next_stage"] == "OBSERVATION"
    assert status["trades_to_next_stage"] == 1


@patch("src.learning.learning_status.models")
def test_stage_observation_at_boundary(mock_models):
    status = _status(mock_models, 25)
    assert status["stage"] == "OBSERVATION"
    assert status["next_stage"] == "HYPOTHESIS"
    assert status["trades_to_next_stage"] == 75


@patch("src.learning.learning_status.models")
def test_stage_hypothesis_at_boundary(mock_models):
    status = _status(mock_models, 100)
    assert status["stage"] == "HYPOTHESIS"
    assert status["next_stage"] == "SIMULATION"
    assert status["trades_to_next_stage"] == 150


@patch("src.learning.learning_status.models")
def test_stage_simulation_at_boundary(mock_models):
    status = _status(mock_models, 250)
    assert status["stage"] == "SIMULATION"
    assert status["next_stage"] == "VALIDATION"
    assert status["trades_to_next_stage"] == 250


@patch("src.learning.learning_status.models")
def test_stage_validation_at_and_above_boundary(mock_models):
    status = _status(mock_models, 500)
    assert status["stage"] == "VALIDATION"
    assert status["next_stage"] is None
    assert status["trades_to_next_stage"] == 0

    status_above = _status(mock_models, 700)
    assert status_above["stage"] == "VALIDATION"
    assert status_above["data_sufficiency_pct"] == 100.0  # clamped, never over 100


@patch("src.learning.learning_status.models")
def test_fields_reflect_wins_losses_rejected_and_promotion(mock_models):
    status = _status(mock_models, 30, wins=12, rejected=88, version={"promotion_eligible": True})
    assert status["trades_collected"] == 30
    assert status["winning_trades"] == 12
    assert status["losing_trades"] == 18
    assert status["rejected_trades"] == 88
    assert status["promotion_eligible"] is True


@patch("src.learning.learning_status.models")
def test_promotion_eligible_false_when_no_version(mock_models):
    status = _status(mock_models, 10, version=None)
    assert status["promotion_eligible"] is False


@patch("src.learning.learning_status.models")
def test_current_activity_and_reason_are_nonempty_strings(mock_models):
    for n in (0, 25, 100, 250, 500):
        status = _status(mock_models, n)
        assert status["current_activity"]
        assert status["reason"]
