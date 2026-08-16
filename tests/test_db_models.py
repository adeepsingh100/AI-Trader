from unittest.mock import Mock

from src.db import models
from src.groq_client import ModelUsageEvent


def _fluent_mock(execute_result):
    """A mock whose chained methods (select/eq/neq/insert/update/upsert/
    order/limit/in_/gte) all return itself, so call args land on the same
    mock and .execute() returns a fixed result."""
    m = Mock()
    m.select.return_value = m
    m.eq.return_value = m
    m.neq.return_value = m
    m.insert.return_value = m
    m.update.return_value = m
    m.upsert.return_value = m
    m.order.return_value = m
    m.limit.return_value = m
    m.in_.return_value = m
    m.gte.return_value = m
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


# --- learning engine: trades extensions ---


def test_open_trade_includes_learning_engine_fields(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.open_trade(
        "paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "go",
        stop_loss_price=6076000, take_profit_price=6448000,
        entry_slippage_pct=0.05, market_regime="strong_bull",
    )

    inserted = table.insert.call_args[0][0]
    assert inserted["stop_loss_price"] == 6076000
    assert inserted["take_profit_price"] == 6448000
    assert inserted["market_regime"] == "strong_bull"


def test_close_trade_includes_exit_reason(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.close_trade(1, exit_price=6300000, pnl=100.0, exit_reason="take_profit")

    updated = table.update.call_args[0][0]
    assert updated["exit_reason"] == "take_profit"


def test_update_trade_excursion(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.update_trade_excursion(1, mfe_pct=2.5, mae_pct=1.1)

    table.update.assert_called_once_with({"mfe_pct": 2.5, "mae_pct": 1.1})
    table.eq.assert_called_with("id", 1)


def test_get_recently_closed_trades_filters_by_mode_status_and_time(monkeypatch):
    import datetime

    table = _fluent_mock([{"id": 1, "status": "closed"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    since = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    result = models.get_recently_closed_trades("paper", since)

    table.eq.assert_called_with("mode", "paper")
    table.in_.assert_called_with("status", ["closed", "flattened"])
    table.gte.assert_called_with("closed_at", since.isoformat())
    assert result == [{"id": 1, "status": "closed"}]


def test_log_opportunity_evaluation_returns_row_and_includes_trade_id(monkeypatch):
    table = _fluent_mock([{"id": 42}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={},
        trend_score=None, momentum_score=None, volume_score=None,
        volatility_score=None, risk_score=None, opportunity_score=None,
        llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None,
        trade_id=7,
    )

    inserted = table.insert.call_args[0][0]
    assert inserted["trade_id"] == 7
    assert result == {"id": 42}


# --- learning_statistics ---


def test_upsert_learning_statistics_conflict_key(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_learning_statistics("paper", "symbol", "BTCINR", {"win_rate": 0.5, "trades_count": 10})

    inserted = table.upsert.call_args[0][0]
    assert inserted["dimension_type"] == "symbol"
    assert inserted["win_rate"] == 0.5
    assert table.upsert.call_args.kwargs["on_conflict"] == "mode,dimension_type,dimension_value"


def test_get_learning_statistics_filters_by_dimension_type(monkeypatch):
    table = _fluent_mock([{"dimension_value": "BTCINR"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_learning_statistics("paper", dimension_type="symbol")

    table.eq.assert_any_call("dimension_type", "symbol")
    assert result == [{"dimension_value": "BTCINR"}]


# --- feature_importance ---


def test_upsert_feature_importance_conflict_key(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_feature_importance("paper", "rsi", 0.42, 30, "1h")

    inserted = table.upsert.call_args[0][0]
    assert inserted["timeframe"] == "1h"
    assert table.upsert.call_args.kwargs["on_conflict"] == "mode,feature_name,timeframe"


def test_get_feature_importance_filters_by_timeframe(monkeypatch):
    table = _fluent_mock([{"feature_name": "trend_score", "timeframe": "blended"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_feature_importance("paper", timeframe="blended")

    table.eq.assert_any_call("timeframe", "blended")
    assert result == [{"feature_name": "trend_score", "timeframe": "blended"}]


# --- confidence_calibration ---


def test_log_confidence_calibration_inserts_expected_row(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.log_confidence_calibration(
        opportunity_evaluation_id=42, ai_confidence=80, historical_confidence=60,
        ai_weight=0.6, historical_weight=0.4, final_confidence=72.0, similar_trades_count=10,
    )

    inserted = table.insert.call_args[0][0]
    assert inserted["opportunity_evaluation_id"] == 42
    assert inserted["final_confidence"] == 72.0


# --- recommendations ---


def test_get_latest_recommendation_orders_by_created_at(monkeypatch):
    table = _fluent_mock([{"recommended_value": 75}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE")

    table.order.assert_called_with("created_at", desc=True)
    assert result == {"recommended_value": 75}


def test_get_latest_recommendation_none_when_no_rows(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    assert models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE") is None


# --- trade_evaluations ---


def test_upsert_trade_evaluation_conflict_key(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_trade_evaluation(
        trade_id=7, predicted_confidence=80.0, predicted_opportunity_score=85.0,
        actual_outcome_won=True, confidence_was_accurate=True,
        opportunity_score_was_accurate=True, risk_assessment="appropriate",
        stop_loss_assessment="appropriate", target_assessment="realistic",
    )

    assert table.upsert.call_args.kwargs["on_conflict"] == "trade_id"


def test_get_trade_evaluation_ids_empty_input_skips_query(monkeypatch):
    client = Mock()
    monkeypatch.setattr(models, "get_client", lambda: client)

    assert models.get_trade_evaluation_ids([]) == set()
    client.table.assert_not_called()


def test_get_trade_evaluation_ids_returns_set(monkeypatch):
    table = _fluent_mock([{"trade_id": 1}, {"trade_id": 3}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    assert models.get_trade_evaluation_ids([1, 2, 3]) == {1, 3}
