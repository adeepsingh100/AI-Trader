from unittest.mock import MagicMock

from src.db import models
from tests.conftest import _fake_connection, _inserted_row, _last_execute, _updated_row


# --- data_quality_log ---


def test_insert_data_quality_issues_noop_on_empty_list(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.insert_data_quality_issues([])

    conn.cursor.assert_not_called()


def test_insert_data_quality_issues_batches_rows(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    fake_execute_values = MagicMock()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    monkeypatch.setattr(models, "execute_values", fake_execute_values)

    models.insert_data_quality_issues([{"pair": "I-BTC_INR", "issue_type": "duplicate"}])

    sql, rows = fake_execute_values.call_args[0][1], fake_execute_values.call_args[0][2]
    assert "INSERT INTO data_quality_log" in sql
    assert rows[0] == ("I-BTC_INR", "duplicate")


def test_get_data_quality_log_filters_by_pair_and_source(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_data_quality_log(pair="I-BTC_INR", source="live")

    assert result == [{"id": 1}]
    sql, params = _last_execute(cur)
    assert "pair = %s" in sql and "source = %s" in sql
    assert "I-BTC_INR" in params and "live" in params


# --- drift_alerts ---


def test_insert_drift_alert_returns_inserted_row(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_drift_alert("feature_engine", "feature_distribution:rsi", "warning", 0.05, 0.20)

    assert result == {"id": 1}
    row = _inserted_row(cur)
    assert row["component"] == "feature_engine"
    assert row["severity"] == "warning"


def test_get_drift_alerts_filters_by_component(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1, "component": "feature_engine"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_drift_alerts(component="feature_engine")

    assert result == [{"id": 1, "component": "feature_engine"}]
    sql, params = _last_execute(cur)
    assert "component = %s" in sql
    assert params[0] == "feature_engine"


# --- strategy_health_scores / strategy_versions.status ---


def test_insert_strategy_health_score(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_strategy_health_score(5, 72.5, "good", {"sharpe": 80})

    assert result == {"id": 1}
    row = _inserted_row(cur)
    assert row["strategy_version_id"] == 5
    assert row["tier"] == "good"


def test_get_latest_strategy_health_score_none_when_missing(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)
    assert models.get_latest_strategy_health_score(5) is None


def test_update_strategy_version_status_updates_by_id(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.update_strategy_version_status(5, "suspended")

    row = _updated_row(cur)
    assert row == {"status": "suspended"}
    sql, params = _last_execute(cur)
    assert params[-1] == 5


def test_get_active_strategy_versions_excludes_suspended(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1, "status": "active"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_active_strategy_versions()

    assert result == [{"id": 1, "status": "active"}]
    sql, _ = _last_execute(cur)
    assert "status != 'suspended'" in sql


# --- system_metrics ---


def test_insert_system_metrics_noop_on_empty(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    models.insert_system_metrics([])
    conn.cursor.assert_not_called()


def test_get_recent_system_metrics_filters_by_component(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_recent_system_metrics(component="orchestrator")

    assert result == [{"id": 1}]
    sql, params = _last_execute(cur)
    assert "component = %s" in sql
    assert params[0] == "orchestrator"


# --- circuit_breaker_state ---


def test_get_circuit_breaker_state_none_when_missing(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)
    assert models.get_circuit_breaker_state("coindcx_api") is None


def test_upsert_circuit_breaker_state_upserts_with_conflict_key(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_circuit_breaker_state("coindcx_api", 3, 1234567890)

    sql, _ = _last_execute(cur)
    assert "ON CONFLICT (component)" in sql
    row = _inserted_row(cur)
    assert row["consecutive_failures"] == 3
    assert row["tripped_until"] == 1234567890


def test_reset_circuit_breaker_zeroes_failures(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.reset_circuit_breaker("llm")

    sql, params = _last_execute(cur)
    assert "ON CONFLICT (component)" in sql
    assert params == ("llm",)  # consecutive_failures=0/tripped_until=NULL are literals, not params


# --- trade_evaluations (drift_detection.py support) ---


def test_get_trade_evaluations_noop_on_empty(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    assert models.get_trade_evaluations([]) == []
    conn.cursor.assert_not_called()


def test_get_trade_evaluations_returns_full_rows(monkeypatch):
    conn, cur = _fake_connection(rows=[{"trade_id": 1, "confidence_was_accurate": True}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_trade_evaluations([1])

    assert result == [{"trade_id": 1, "confidence_was_accurate": True}]
    sql, params = _last_execute(cur)
    assert "trade_id = ANY(%s)" in sql
    assert params == ([1],)


# --- opportunity_evaluations: config_version/market_regime additions ---


def test_log_opportunity_evaluation_includes_market_regime_and_config_version(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={}, trend_score=None,
        momentum_score=None, volume_score=None, volatility_score=None, risk_score=None,
        opportunity_score=None, llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None,
        market_regime="strong_bull", config_version="abc123",
    )

    row = _inserted_row(cur)
    assert row["market_regime"] == "strong_bull"
    assert row["config_version"] == "abc123"
