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


def test_run_strategy_health_suspends_critical_version(monkeypatch):
    critical_version = {"id": 5, "status": "active"}
    monkeypatch.setattr(models, "get_active_strategy_versions", lambda: [critical_version])
    monkeypatch.setattr(models, "get_capital_config", lambda mode: {"capital_to_use": 10000})

    # 30 closed trades, all losses with no stop-loss — a clearly critical
    # book (win_rate=0, drawdown=100%, worst possible on every component).
    closed_trades = [
        {"id": i, "pnl": -10, "closed_at": "2020-01-01T00:00:00+00:00", "opened_at": "2020-01-01T00:00:00+00:00"}
        for i in range(30)
    ]
    monkeypatch.setattr(models, "get_closed_trades", lambda mode, version_id: closed_trades)
    monkeypatch.setattr(models, "insert_strategy_health_score", Mock())
    suspend_mock = Mock()
    monkeypatch.setattr(models, "update_strategy_version_status", suspend_mock)

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", True):
        result = run_strategy_health("paper")

    suspend_mock.assert_called_once_with(5, "suspended")
    assert result["suspended"] == [5]


def test_run_strategy_health_never_suspends_when_auto_suspend_disabled(monkeypatch):
    critical_version = {"id": 5, "status": "active"}
    monkeypatch.setattr(models, "get_active_strategy_versions", lambda: [critical_version])
    monkeypatch.setattr(models, "get_capital_config", lambda mode: {"capital_to_use": 10000})
    closed_trades = [
        {"id": i, "pnl": -10, "closed_at": "2020-01-01T00:00:00+00:00", "opened_at": "2020-01-01T00:00:00+00:00"}
        for i in range(30)
    ]
    monkeypatch.setattr(models, "get_closed_trades", lambda mode, version_id: closed_trades)
    monkeypatch.setattr(models, "insert_strategy_health_score", Mock())
    suspend_mock = Mock()
    monkeypatch.setattr(models, "update_strategy_version_status", suspend_mock)

    with patch("src.learning.strategy_health.STRATEGY_HEALTH_AUTO_SUSPEND_ENABLED", False):
        result = run_strategy_health("paper")

    suspend_mock.assert_not_called()
    assert result["suspended"] == []
