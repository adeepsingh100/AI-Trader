from unittest.mock import Mock

from src.db import models
from tests.conftest import _fluent_mock


# --- data_quality_log ---


def test_insert_data_quality_issues_noop_on_empty_list(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.insert_data_quality_issues([])

    client.table.assert_not_called()


def test_insert_data_quality_issues_batches_rows(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.insert_data_quality_issues([{"pair": "I-BTC_INR", "issue_type": "duplicate"}])

    client.table.assert_called_with("data_quality_log")
    rows = table.insert.call_args[0][0]
    assert rows[0]["pair"] == "I-BTC_INR"


def test_get_data_quality_log_filters_by_pair_and_source(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_data_quality_log(pair="I-BTC_INR", source="live")

    assert result == [{"id": 1}]
    table.eq.assert_any_call("pair", "I-BTC_INR")
    table.eq.assert_any_call("source", "live")


# --- drift_alerts ---


def test_insert_drift_alert_returns_inserted_row(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.insert_drift_alert("feature_engine", "feature_distribution:rsi", "warning", 0.05, 0.20)

    assert result == {"id": 1}
    inserted = table.insert.call_args[0][0]
    assert inserted["component"] == "feature_engine"
    assert inserted["severity"] == "warning"


def test_get_drift_alerts_filters_by_component(monkeypatch):
    table = _fluent_mock([{"id": 1, "component": "feature_engine"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_drift_alerts(component="feature_engine")

    assert result == [{"id": 1, "component": "feature_engine"}]
    table.eq.assert_called_with("component", "feature_engine")


# --- strategy_health_scores / strategy_versions.status ---


def test_insert_strategy_health_score(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.insert_strategy_health_score(5, 72.5, "good", {"sharpe": 80})

    assert result == {"id": 1}
    inserted = table.insert.call_args[0][0]
    assert inserted["strategy_version_id"] == 5
    assert inserted["tier"] == "good"


def test_get_latest_strategy_health_score_none_when_missing(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    assert models.get_latest_strategy_health_score(5) is None


def test_update_strategy_version_status_updates_by_id(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.update_strategy_version_status(5, "suspended")

    table.update.assert_called_with({"status": "suspended"})
    table.eq.assert_called_with("id", 5)


def test_get_active_strategy_versions_excludes_suspended(monkeypatch):
    table = _fluent_mock([{"id": 1, "status": "active"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_active_strategy_versions()

    assert result == [{"id": 1, "status": "active"}]
    table.neq.assert_called_with("status", "suspended")


# --- system_metrics ---


def test_insert_system_metrics_noop_on_empty(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    models.insert_system_metrics([])
    client.table.assert_not_called()


def test_get_recent_system_metrics_filters_by_component(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_recent_system_metrics(component="orchestrator")

    assert result == [{"id": 1}]
    table.eq.assert_called_with("component", "orchestrator")


# --- circuit_breaker_state ---


def test_get_circuit_breaker_state_none_when_missing(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    assert models.get_circuit_breaker_state("coindcx_api") is None


def test_upsert_circuit_breaker_state_upserts_with_conflict_key(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_circuit_breaker_state("coindcx_api", 3, 1234567890)

    call = table.upsert.call_args
    row = call.args[0]
    assert row["consecutive_failures"] == 3
    assert row["tripped_until"] == 1234567890
    assert call.kwargs["on_conflict"] == "component"


def test_reset_circuit_breaker_zeroes_failures(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.reset_circuit_breaker("llm")

    row = table.upsert.call_args.args[0]
    assert row["consecutive_failures"] == 0
    assert row["tripped_until"] is None


# --- trade_evaluations (drift_detection.py support) ---


def test_get_trade_evaluations_noop_on_empty(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    assert models.get_trade_evaluations([]) == []
    client.table.assert_not_called()


def test_get_trade_evaluations_returns_full_rows(monkeypatch):
    table = _fluent_mock([{"trade_id": 1, "confidence_was_accurate": True}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_trade_evaluations([1])

    assert result == [{"trade_id": 1, "confidence_was_accurate": True}]
    table.in_.assert_called_with("trade_id", [1])


# --- opportunity_evaluations: config_version/market_regime additions ---


def test_log_opportunity_evaluation_includes_market_regime_and_config_version(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={}, trend_score=None,
        momentum_score=None, volume_score=None, volatility_score=None, risk_score=None,
        opportunity_score=None, llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None,
        market_regime="strong_bull", config_version="abc123",
    )

    inserted = table.insert.call_args[0][0]
    assert inserted["market_regime"] == "strong_bull"
    assert inserted["config_version"] == "abc123"
