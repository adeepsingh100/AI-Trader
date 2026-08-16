from unittest.mock import Mock

from src.db import models
from src.groq_client import ModelUsageEvent


def _fluent_mock(execute_result):
    """A mock whose chained methods (select/eq/insert/update/upsert/order/
    limit) all return itself, so call args land on the same mock and
    .execute() returns a fixed result."""
    m = Mock()
    m.select.return_value = m
    m.eq.return_value = m
    m.insert.return_value = m
    m.update.return_value = m
    m.upsert.return_value = m
    m.order.return_value = m
    m.limit.return_value = m
    m.execute.return_value = Mock(data=execute_result)
    return m


def test_open_trade_inserts_expected_row(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.open_trade("paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "signal said buy")

    client.table.assert_called_with("trades")
    inserted = table.insert.call_args[0][0]
    assert inserted["mode"] == "paper"
    assert inserted["status"] == "open"
    assert inserted["symbol"] == "BTCINR"
    assert result == {"id": 1}


def test_close_trade_sets_closed_fields(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.close_trade(1, exit_price=6300000, pnl=100.0)

    updated = table.update.call_args[0][0]
    assert updated["exit_price"] == 6300000
    assert updated["pnl"] == 100.0
    assert updated["status"] == "closed"
    assert "closed_at" in updated
    table.eq.assert_called_with("id", 1)


def test_upsert_daily_pnl_conflict_key(monkeypatch):
    import datetime

    table = _fluent_mock([{"date": "2026-08-15"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_daily_pnl(
        datetime.date(2026, 8, 15), "paper", 100.0, 3, True, False
    )

    _, kwargs = table.upsert.call_args
    assert kwargs["on_conflict"] == "date,mode"


def test_log_model_usage_batches_rows(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    events = [
        ModelUsageEvent("model-a", None, 120, False),
        ModelUsageEvent("model-b", "model-a failed: 429", 340, True),
    ]
    models.log_model_usage(events)

    rows = table.insert.call_args[0][0]
    assert len(rows) == 2
    assert rows[1]["model_used"] == "model-b"
    assert rows[1]["fallback_reason"] == "model-a failed: 429"


def test_log_model_usage_skips_empty(monkeypatch):
    client = Mock()
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.log_model_usage([])

    client.table.assert_not_called()


def test_get_latest_promoted_version_filters_and_orders(monkeypatch):
    table = _fluent_mock([{"id": 2, "promoted_to_real": True}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_latest_promoted_version()

    table.eq.assert_called_with("promoted_to_real", True)
    table.order.assert_called_with("version_number", desc=True)
    assert result["id"] == 2


def test_log_opportunity_evaluation_inserts_expected_row(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.log_opportunity_evaluation(
        mode="paper",
        symbol="BTCINR",
        version_id=1,
        features={"5m": {"rsi": 55.0}},
        trend_score=80.0,
        momentum_score=70.0,
        volume_score=60.0,
        volatility_score=100.0,
        risk_score=90.0,
        opportunity_score=82.0,
        llm_decision="accept",
        llm_reasoning="strong setup",
        llm_raw_response={"decision": "accept", "reasoning": "strong setup"},
        risk_manager_result="size",
        final_decision="buy",
        reason="strong setup",
    )

    client.table.assert_called_with("opportunity_evaluations")
    inserted = table.insert.call_args[0][0]
    assert inserted["symbol"] == "BTCINR"
    assert inserted["opportunity_score"] == 82.0
    assert inserted["llm_decision"] == "accept"
    assert inserted["final_decision"] == "buy"
    assert inserted["features"] == {"5m": {"rsi": 55.0}}
