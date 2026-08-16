from unittest.mock import Mock

from src.db import models
from src.monitoring.diagnostics import run_health_check


def _healthy_mocks(monkeypatch):
    monkeypatch.setattr(models, "get_client", lambda: Mock())
    monkeypatch.setattr(models, "get_entry_evaluations_since", lambda mode, since: [{"id": 1}])
    monkeypatch.setattr(
        models,
        "get_learning_statistics",
        lambda mode, dimension_type=None: [{"updated_at": "2999-01-01T00:00:00+00:00"}],
    )
    monkeypatch.setattr(models, "get_capital_config", lambda mode: {"mode": mode})
    monkeypatch.setattr(models, "get_open_trades", lambda mode: [{"qty": 1, "entry_price": 100}])
    monkeypatch.setattr(models, "get_recommendations", lambda mode: [])
    monkeypatch.setattr(models, "insert_system_metrics", Mock())


def test_run_health_check_all_healthy(monkeypatch):
    _healthy_mocks(monkeypatch)
    result = run_health_check("paper")
    assert result["overall_healthy"] is True
    assert all(c.get("healthy") is not False for c in result["checks"].values())


def test_run_health_check_flags_stale_market_feed(monkeypatch):
    _healthy_mocks(monkeypatch)
    monkeypatch.setattr(models, "get_entry_evaluations_since", lambda mode, since: [])
    result = run_health_check("paper")
    assert result["checks"]["market_feed"]["healthy"] is False
    assert result["overall_healthy"] is False


def test_run_health_check_flags_malformed_open_positions(monkeypatch):
    _healthy_mocks(monkeypatch)
    monkeypatch.setattr(models, "get_open_trades", lambda mode: [{"qty": 0, "entry_price": 100}])
    result = run_health_check("paper")
    assert result["checks"]["portfolio_engine"]["healthy"] is False
    assert result["overall_healthy"] is False


def test_run_health_check_db_error_marks_unhealthy_not_a_crash(monkeypatch):
    _healthy_mocks(monkeypatch)

    def _raise():
        raise RuntimeError("connection refused")

    monkeypatch.setattr(models, "get_client", _raise)
    result = run_health_check("paper")  # must not raise
    assert result["checks"]["database"]["healthy"] is False
    assert result["overall_healthy"] is False


def test_run_health_check_unconfigured_execution_engine_is_not_unhealthy(monkeypatch):
    # Real mode sitting unconfigured (no capital_config yet) is expected,
    # not an error — same "not ready yet" distinction orchestrator.py
    # already makes.
    _healthy_mocks(monkeypatch)
    monkeypatch.setattr(models, "get_capital_config", lambda mode: None)
    result = run_health_check("real")
    assert result["checks"]["execution_engine"]["healthy"] is None
    assert result["overall_healthy"] is True
