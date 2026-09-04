from src.db import models
from tests.conftest import _fake_firestore_client


# --- data_quality_log ---


def test_insert_data_quality_issues_noop_on_empty_list(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    models.insert_data_quality_issues([])


def test_insert_data_quality_issues_dedups_via_deterministic_doc_id(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    row = {"pair": "I-BTC_INR", "interval": "5m", "issue_type": "duplicate", "candle_time": 1700000000}
    models.insert_data_quality_issues([row])

    assert store["data_quality_log"]["I-BTC_INR_5m_duplicate_1700000000"]["pair"] == "I-BTC_INR"


def test_insert_data_quality_issues_batch_level_issue_gets_auto_id(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_data_quality_issues([{"pair": "I-BTC_INR", "issue_type": "exchange_outage", "candle_time": None}])

    (row,) = store["data_quality_log"].values()
    assert row["issue_type"] == "exchange_outage"


def test_insert_data_quality_issues_drops_ignore_severity_rows(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("severity=ignore rows must never reach Firestore")

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    models.insert_data_quality_issues([
        {"pair": "I-YFI_INR", "interval": "1m", "issue_type": "zero_volume", "severity": "ignore", "candle_time": 1},
    ])


def test_insert_data_quality_issues_keeps_non_ignore_rows_alongside_ignored_ones(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_data_quality_issues([
        {"pair": "I-YFI_INR", "interval": "1m", "issue_type": "zero_volume", "severity": "ignore", "candle_time": 1},
        {"pair": "I-BTC_INR", "interval": "1m", "issue_type": "price_spike", "severity": "warn", "candle_time": 2},
    ])

    assert list(store["data_quality_log"].keys()) == ["I-BTC_INR_1m_price_spike_2"]


def test_get_data_quality_log_filters_by_pair_and_source(monkeypatch):
    seed = {"data_quality_log": {
        "1": {"pair": "I-BTC_INR", "source": "live", "created_at": 2},
        "2": {"pair": "I-BTC_INR", "source": "backtest", "created_at": 1},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_data_quality_log(pair="I-BTC_INR", source="live")

    assert [r["id"] for r in result] == ["1"]


# --- drift_alerts ---


def test_insert_drift_alert_returns_inserted_row(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.insert_drift_alert("feature_engine", "feature_distribution:rsi", "warning", 0.05, 0.20)

    assert result["component"] == "feature_engine"
    assert result["severity"] == "warning"
    assert store["drift_alerts"][result["id"]]["component"] == "feature_engine"


def test_get_drift_alerts_filters_by_component(monkeypatch):
    seed = {"drift_alerts": {
        "1": {"component": "feature_engine", "detected_at": 1},
        "2": {"component": "other", "detected_at": 2},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_drift_alerts(component="feature_engine")

    assert [r["id"] for r in result] == ["1"]


# --- strategy_health_scores / strategy_versions.status ---


def test_insert_strategy_health_score(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.insert_strategy_health_score(5, 72.5, "good", {"sharpe": 80})

    assert result["strategy_version_id"] == 5
    assert result["tier"] == "good"
    assert store["strategy_health_scores"][result["id"]]["tier"] == "good"


def test_get_latest_strategy_health_score_none_when_missing(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)
    assert models.get_latest_strategy_health_score(5) is None


def test_update_strategy_version_status_updates_by_id(monkeypatch):
    client, store = _fake_firestore_client(seed={"strategy_versions": {"5": {"status": "active"}}})
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.update_strategy_version_status(5, "suspended")

    assert store["strategy_versions"]["5"]["status"] == "suspended"


def test_get_active_strategy_versions_excludes_suspended(monkeypatch):
    seed = {"strategy_versions": {
        "1": {"status": "active", "version_number": 1},
        "2": {"status": "suspended", "version_number": 2},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_active_strategy_versions()

    assert [r["id"] for r in result] == ["1"]


# --- system_metrics ---


def test_insert_system_metrics_noop_on_empty(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)
    models.insert_system_metrics([])


def test_get_recent_system_metrics_filters_by_component(monkeypatch):
    seed = {"system_metrics": {
        "1": {"component": "orchestrator", "recorded_at": 1},
        "2": {"component": "other", "recorded_at": 2},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_recent_system_metrics(component="orchestrator")

    assert [r["id"] for r in result] == ["1"]


# --- circuit_breaker_state ---


def test_get_circuit_breaker_state_none_when_missing(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)
    assert models.get_circuit_breaker_state("coindcx_api") is None


def test_upsert_circuit_breaker_state_upserts_by_component(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_circuit_breaker_state("coindcx_api", 3, 1234567890)

    row = store["circuit_breaker_state"]["coindcx_api"]
    assert row["consecutive_failures"] == 3
    assert row["tripped_until"] == 1234567890


def test_reset_circuit_breaker_zeroes_failures(monkeypatch):
    seed = {"circuit_breaker_state": {"llm": {"consecutive_failures": 4, "tripped_until": 999}}}
    client, store = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.reset_circuit_breaker("llm")

    row = store["circuit_breaker_state"]["llm"]
    assert row["consecutive_failures"] == 0
    assert row["tripped_until"] is None


# --- trade_evaluations (drift_detection.py support) ---


def test_get_trade_evaluations_noop_on_empty(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty id list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)
    assert models.get_trade_evaluations([]) == []


def test_get_trade_evaluations_returns_full_rows(monkeypatch):
    seed = {"trade_evaluations": {"1": {"trade_id": 1, "confidence_was_accurate": True}}}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_trade_evaluations([1])

    assert result == [{"trade_id": 1, "confidence_was_accurate": True, "id": "1"}]


# --- opportunity_evaluations: config_version/market_regime additions ---


def test_log_opportunity_evaluation_includes_market_regime_and_config_version(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={}, trend_score=None,
        momentum_score=None, volume_score=None, volatility_score=None, risk_score=None,
        opportunity_score=None, llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None, strategy_type="default",
        market_regime="strong_bull", config_version="abc123",
    )

    (row,) = store["opportunity_evaluations"].values()
    assert row["market_regime"] == "strong_bull"
    assert row["config_version"] == "abc123"
