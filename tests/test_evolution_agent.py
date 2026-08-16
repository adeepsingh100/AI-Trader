import datetime
from unittest.mock import Mock, patch

from src.agents.evolution_agent import (
    compute_metrics,
    promotion_ready,
    propose_next_version,
    run_evolution,
)
from src.groq_client import AllModelsFailedError, ModelUsageEvent


def _trade(pnl, closed_at):
    return {"pnl": pnl, "closed_at": closed_at}


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


def _version(days_ago, promoted=False):
    created = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days_ago)
    return {
        "id": 1,
        "created_at": created.isoformat(),
        "promoted_to_real": promoted,
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


# --- propose_next_version ---


def test_propose_next_version_parses_json():
    with patch(
        "src.agents.evolution_agent.chat",
        return_value=('{"prompt_text": "new prompt", "params_json": {"x": 1}, "notes": "n"}', []),
    ):
        proposal, _events = propose_next_version({}, "old prompt", {})
    assert proposal["prompt_text"] == "new prompt"


def test_propose_next_version_falls_back_on_bad_json():
    with patch("src.agents.evolution_agent.chat", return_value=("garbage", [])):
        proposal, _events = propose_next_version({}, "old prompt", {"y": 2})
    assert proposal["prompt_text"] == "old prompt"
    assert proposal["params_json"] == {"y": 2}
    assert "unparseable" in proposal["notes"]


def test_propose_next_version_carries_forward_when_all_models_fail():
    # a total LLM outage must not crash the nightly evolution run — carry
    # the current prompt/params forward, same as the bad-JSON fallback
    failure_events = [ModelUsageEvent("model-a", None, 50, False)]
    with patch(
        "src.agents.evolution_agent.chat",
        side_effect=AllModelsFailedError("all models in chain failed: [...]", failure_events),
    ):
        proposal, returned_events = propose_next_version({}, "old prompt", {"y": 2})

    assert proposal["prompt_text"] == "old prompt"
    assert proposal["params_json"] == {"y": 2}
    assert "LLM call failed" in proposal["notes"]
    assert returned_events == failure_events


# --- run_evolution ---


@patch("src.learning.recommendations.generate_recommendations")
@patch("src.learning.feature_importance.compute_feature_importance")
@patch("src.agents.evolution_agent.models")
@patch("src.agents.evolution_agent.propose_next_version")
def test_run_evolution_inserts_next_version_and_promotes_when_ready(
    mock_propose, mock_models, mock_feature_importance, mock_recommendations
):
    mock_feature_importance.return_value = []
    mock_recommendations.return_value = []
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20)
    version.update({"id": 1, "version_number": 3, "prompt_text": "old", "params_json": {}})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = [_trade(100, 1)]
    mock_propose.return_value = (
        {"prompt_text": "improved", "params_json": {"a": 1}, "notes": "tweak"},
        [],
    )
    mock_models.insert_strategy_version.return_value = {"id": 2, "version_number": 4}

    result = run_evolution(mode="paper")

    mock_models.insert_strategy_version.assert_called_once_with(
        version_number=4, prompt_text="improved", params_json={"a": 1}, notes="tweak"
    )
    mock_models.promote_version.assert_called_once_with(1)
    assert result["promoted"] is True
    assert result["new_version"] == {"id": 2, "version_number": 4}


@patch("src.learning.recommendations.generate_recommendations")
@patch("src.learning.feature_importance.compute_feature_importance")
@patch("src.agents.evolution_agent.models")
@patch("src.agents.evolution_agent.propose_next_version")
def test_run_evolution_does_not_promote_when_criteria_unmet(
    mock_propose, mock_models, mock_feature_importance, mock_recommendations
):
    mock_feature_importance.return_value = []
    mock_recommendations.return_value = []
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=2)  # too young
    version.update({"id": 1, "version_number": 3, "prompt_text": "old", "params_json": {}})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = [_trade(100, 1)]
    mock_propose.return_value = ({"prompt_text": "improved", "params_json": {}, "notes": ""}, [])
    mock_models.insert_strategy_version.return_value = {"id": 2, "version_number": 4}

    result = run_evolution(mode="paper")

    mock_models.promote_version.assert_not_called()
    assert result["promoted"] is False


@patch("src.learning.recommendations.generate_recommendations")
@patch("src.learning.feature_importance.compute_feature_importance")
@patch("src.agents.evolution_agent.models")
@patch("src.agents.evolution_agent.propose_next_version")
def test_run_evolution_skips_promotion_check_for_real_mode(
    mock_propose, mock_models, mock_feature_importance, mock_recommendations
):
    mock_feature_importance.return_value = []
    mock_recommendations.return_value = []
    mock_models.get_capital_config.return_value = {"capital_to_use": 10000}
    version = _version(days_ago=20, promoted=True)
    version.update({"id": 1, "version_number": 3, "prompt_text": "old", "params_json": {}})
    mock_models.get_latest_version.return_value = version
    mock_models.get_closed_trades.return_value = []
    mock_propose.return_value = ({"prompt_text": "improved", "params_json": {}, "notes": ""}, [])
    mock_models.insert_strategy_version.return_value = {"id": 2, "version_number": 4}

    result = run_evolution(mode="real")

    mock_models.promote_version.assert_not_called()
    assert result["promoted"] is False
