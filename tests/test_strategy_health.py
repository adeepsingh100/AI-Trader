from unittest.mock import Mock, patch

import pytest

from src.db import models
from src.learning.strategy_health import (
    _drawdown_component,
    _profit_factor_component,
    _sharpe_component,
    _tier,
    _win_rate_component,
    compute_health_score,
    run_strategy_health,
)


def test_sharpe_component_scales_positively():
    assert _sharpe_component(2.0) > _sharpe_component(0.0) > _sharpe_component(-1.0)


def test_sharpe_component_none_passthrough():
    assert _sharpe_component(None) is None


def test_drawdown_component_lower_drawdown_scores_higher():
    assert _drawdown_component(1.0) > _drawdown_component(10.0)


def test_win_rate_component_scales_to_100():
    assert _win_rate_component(1.0) == 100.0
    assert _win_rate_component(0.0) == 0.0


def test_profit_factor_component_breakeven_is_midpoint():
    assert _profit_factor_component(1.0) == pytest.approx(100 / 3)


def test_tier_thresholds():
    assert _tier(None) == "unknown"
    assert _tier(90) == "excellent"
    assert _tier(70) == "good"
    assert _tier(50) == "warning"
    assert _tier(10) == "critical"


def test_compute_health_score_no_trades_returns_none_score(monkeypatch):
    monkeypatch.setattr(models, "get_closed_trades", lambda mode, version_id: [])
    result = compute_health_score("paper", {"id": 1}, capital_to_use=10000)
    assert result["health_score"] is None
    assert result["tier"] == "unknown"


def _mock_critical_setup(monkeypatch, champion_version_id=None, num_trades=30):
    """Shared setup: one active version with a clearly critical book (all
    losses, no stop-loss). champion_version_id=None means no real-mode
    champion exists at all — the common case, and the one the pre-
    existing suspend-only tests below want (isolated from rollback)."""
    critical_version = {"id": 5, "status": "active"}
    monkeypatch.setattr(models, "get_active_strategy_versions", lambda: [critical_version])
    monkeypatch.setattr(models, "get_capital_config", lambda mode: {"capital_to_use": 10000})
    closed_trades = [
        {"id": i, "pnl": -10, "closed_at": "2020-01-01T00:00:00+00:00", "opened_at": "2020-01-01T00:00:00+00:00"}
        for i in range(num_trades)
    ]
    monkeypatch.setattr(models, "get_closed_trades", lambda mode, version_id: closed_trades)
    monkeypatch.setattr(models, "insert_strategy_health_score", Mock())
    monkeypatch.setattr(
        models,
        "get_latest_promoted_version",
        lambda: ({"id": champion_version_id} if champion_version_id is not None else None),
    )
    return critical_version


def test_run_strategy_health_suspends_critical_version(monkeypatch):
    _mock_critical_setup(monkeypatch)
    suspend_mock = Mock()
    monkeypatch.setattr(models, "update_strategy_version_status", suspend_mock)

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", True):
        result = run_strategy_health("paper")

    suspend_mock.assert_called_once_with(5, "suspended")
    assert result["suspended"] == [5]
    assert result["rolled_back"] == []


def test_run_strategy_health_never_suspends_when_auto_suspend_disabled(monkeypatch):
    _mock_critical_setup(monkeypatch)
    suspend_mock = Mock()
    monkeypatch.setattr(models, "update_strategy_version_status", suspend_mock)

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", False):
        result = run_strategy_health("paper")

    suspend_mock.assert_not_called()
    assert result["suspended"] == []


# --- Automatic Rollback (Phase 20): suspending the CURRENT real-mode
# champion writes a promotion_audit 'rollback' row; suspending any other
# (non-champion) version never does. ---


def test_run_strategy_health_rollback_when_champion_suspended(monkeypatch):
    # champion_version_id=5 -> the version about to be suspended IS today's
    # real-mode champion.
    _mock_critical_setup(monkeypatch, champion_version_id=5)
    monkeypatch.setattr(models, "update_strategy_version_status", Mock())
    audit_mock = Mock()
    monkeypatch.setattr(models, "insert_promotion_audit", audit_mock)
    monkeypatch.setattr(models, "log_agent_event", Mock())
    # After suspension, get_latest_promoted_version() naturally falls back
    # to whatever's next (here: nothing else was ever promoted).
    monkeypatch.setattr(models, "get_latest_promoted_version", Mock(side_effect=[{"id": 5}, None]))

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", True):
        result = run_strategy_health("real")

    assert result["rolled_back"] == [5]
    audit_mock.assert_called_once()
    kwargs = audit_mock.call_args.kwargs
    assert kwargs["event_type"] == "rollback"
    assert kwargs["previous_champion_id"] == 5
    assert kwargs["new_champion_id"] is None
    assert kwargs["candidate_version_id"] == 5


def test_run_strategy_health_no_rollback_when_suspended_version_is_not_champion(monkeypatch):
    # champion_version_id=999 -> some OTHER version is champion; the
    # version being suspended (5) is an unrelated paper-only candidate.
    _mock_critical_setup(monkeypatch, champion_version_id=999)
    monkeypatch.setattr(models, "update_strategy_version_status", Mock())
    audit_mock = Mock()
    monkeypatch.setattr(models, "insert_promotion_audit", audit_mock)

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", True):
        result = run_strategy_health("paper")

    assert result["rolled_back"] == []
    audit_mock.assert_not_called()


def test_run_strategy_health_rollback_audit_failure_fails_open(monkeypatch):
    # A DB error while building the rollback audit (here: re-resolving the
    # new champion) must never crash the health run itself — suspension
    # already happened and is real regardless of whether the audit
    # write succeeds.
    _mock_critical_setup(monkeypatch, champion_version_id=5)
    monkeypatch.setattr(models, "update_strategy_version_status", Mock())
    monkeypatch.setattr(models, "insert_promotion_audit", Mock())
    monkeypatch.setattr(
        models, "get_latest_promoted_version", Mock(side_effect=[{"id": 5}, RuntimeError("supabase down")])
    )

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", True):
        result = run_strategy_health("real")

    assert result["suspended"] == [5]
    assert result["rolled_back"] == []  # the audit write never completed, so not counted as rolled back
