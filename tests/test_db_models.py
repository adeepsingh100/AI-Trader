from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.db import models
from src.groq_client import ModelUsageEvent
from tests.conftest import _fake_connection, _inserted_row, _last_execute, _updated_row


def test_open_trade_inserts_expected_row(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.open_trade("paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "signal said buy")

    row = _inserted_row(cur)
    assert row["mode"] == "paper"
    assert row["status"] == "open"
    assert row["symbol"] == "BTCINR"
    assert result == {"id": 1}


def test_close_trade_sets_closed_fields(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.close_trade(1, exit_price=6300000, pnl=100.0)

    row = _updated_row(cur)
    assert row["exit_price"] == 6300000
    assert row["pnl"] == 100.0
    assert row["status"] == "closed"
    assert "closed_at" in row
    sql, params = _last_execute(cur)
    assert params[-1] == 1  # WHERE id = %s


def test_upsert_daily_pnl_conflict_key(monkeypatch):
    import datetime

    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_daily_pnl(datetime.date(2026, 8, 15), "paper", 100.0, 3, True, False)

    sql, params = _last_execute(cur)
    assert "ON CONFLICT (date, mode)" in sql
    row = _inserted_row(cur)
    assert row["date"] == "2026-08-15"
    assert row["mode"] == "paper"


def test_log_model_usage_batches_rows(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    fake_execute_values = MagicMock()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    monkeypatch.setattr(models, "execute_values", fake_execute_values)

    events = [
        ModelUsageEvent("model-a", None, 120, False),
        ModelUsageEvent("model-b", "model-a failed: 429", 340, True),
    ]
    models.log_model_usage(events)

    sql, rows = fake_execute_values.call_args[0][1], fake_execute_values.call_args[0][2]
    assert "INSERT INTO model_usage" in sql
    assert len(rows) == 2
    assert rows[1][0] == "model-b"  # model_used is the first column in the insert


def test_log_model_usage_skips_empty(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.log_model_usage([])

    conn.cursor.assert_not_called()


def test_get_latest_promoted_version_filters_and_orders(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 2, "promoted_to_real": True}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_latest_promoted_version()

    sql, _ = _last_execute(cur)
    assert "promoted_to_real = true" in sql
    assert "ORDER BY version_number DESC" in sql
    assert result["id"] == 2


def test_log_opportunity_evaluation_inserts_expected_row(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

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

    row = _inserted_row(cur)
    assert row["symbol"] == "BTCINR"
    assert row["opportunity_score"] == 82.0
    assert row["llm_decision"] == "accept"
    assert row["final_decision"] == "buy"
    assert row["features"] == {"5m": {"rsi": 55.0}}


# --- learning engine: trades extensions ---


def test_open_trade_includes_learning_engine_fields(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.open_trade(
        "paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "go",
        stop_loss_price=6076000, take_profit_price=6448000,
        entry_slippage_pct=0.05, market_regime="strong_bull",
    )

    row = _inserted_row(cur)
    assert row["stop_loss_price"] == 6076000
    assert row["take_profit_price"] == 6448000
    assert row["market_regime"] == "strong_bull"


def test_close_trade_includes_exit_reason(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.close_trade(1, exit_price=6300000, pnl=100.0, exit_reason="take_profit")

    row = _updated_row(cur)
    assert row["exit_reason"] == "take_profit"


def test_update_trade_excursion(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.update_trade_excursion(1, mfe_pct=2.5, mae_pct=1.1)

    row = _updated_row(cur)
    assert row == {"mfe_pct": 2.5, "mae_pct": 1.1}
    sql, params = _last_execute(cur)
    assert params[-1] == 1


def test_get_recently_closed_trades_filters_by_mode_status_and_time(monkeypatch):
    import datetime

    conn, cur = _fake_connection(rows=[{"id": 1, "status": "closed"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    since = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    result = models.get_recently_closed_trades("paper", since)

    sql, params = _last_execute(cur)
    assert "mode = %s" in sql and "status = ANY(%s)" in sql and "closed_at >= %s" in sql
    assert params == ("paper", ["closed", "flattened"], since.isoformat())
    assert result == [{"id": 1, "status": "closed"}]


def test_log_opportunity_evaluation_returns_row_and_includes_trade_id(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 42}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={},
        trend_score=None, momentum_score=None, volume_score=None,
        volatility_score=None, risk_score=None, opportunity_score=None,
        llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None,
        trade_id=7,
    )

    row = _inserted_row(cur)
    assert row["trade_id"] == 7
    assert result == {"id": 42}


# --- learning_statistics ---


def test_upsert_learning_statistics_conflict_key(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_learning_statistics("paper", "symbol", "BTCINR", {"win_rate": 0.5, "trades_count": 10})

    sql, params = _last_execute(cur)
    assert "ON CONFLICT (mode, dimension_type, dimension_value)" in sql
    row = _inserted_row(cur)
    assert row["dimension_type"] == "symbol"
    assert row["win_rate"] == 0.5


def test_get_learning_statistics_filters_by_dimension_type(monkeypatch):
    conn, cur = _fake_connection(rows=[{"dimension_value": "BTCINR"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_learning_statistics("paper", dimension_type="symbol")

    sql, params = _last_execute(cur)
    assert "dimension_type = %s" in sql
    assert "symbol" in params
    assert result == [{"dimension_value": "BTCINR"}]


# --- feature_importance ---


def test_upsert_feature_importance_conflict_key(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_feature_importance("paper", "rsi", 0.42, 30, "1h")

    sql, _ = _last_execute(cur)
    assert "ON CONFLICT (mode, feature_name, timeframe)" in sql
    row = _inserted_row(cur)
    assert row["timeframe"] == "1h"


def test_get_feature_importance_filters_by_timeframe(monkeypatch):
    conn, cur = _fake_connection(rows=[{"feature_name": "trend_score", "timeframe": "blended"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_feature_importance("paper", timeframe="blended")

    sql, params = _last_execute(cur)
    assert "timeframe = %s" in sql
    assert "blended" in params
    assert result == [{"feature_name": "trend_score", "timeframe": "blended"}]


# --- confidence_calibration ---


def test_log_confidence_calibration_inserts_expected_row(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.log_confidence_calibration(
        opportunity_evaluation_id=42, ai_confidence=80, historical_confidence=60,
        ai_weight=0.6, historical_weight=0.4, final_confidence=72.0, similar_trades_count=10,
    )

    sql, params = _last_execute(cur)
    assert "INSERT INTO confidence_calibration" in sql
    assert 42 in params
    assert 72.0 in params


# --- recommendations ---


def test_get_latest_recommendation_orders_by_created_at(monkeypatch):
    conn, cur = _fake_connection(rows=[{"recommended_value": 75}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE")

    sql, _ = _last_execute(cur)
    assert "ORDER BY created_at DESC" in sql
    assert result == {"recommended_value": 75}


def test_get_latest_recommendation_none_when_no_rows(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    assert models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE") is None


# --- trade_evaluations ---


def test_upsert_trade_evaluation_conflict_key(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_trade_evaluation(
        trade_id=7, predicted_confidence=80.0, predicted_opportunity_score=85.0,
        actual_outcome_won=True, confidence_was_accurate=True,
        opportunity_score_was_accurate=True, risk_assessment="appropriate",
        stop_loss_assessment="appropriate", target_assessment="realistic",
    )

    sql, _ = _last_execute(cur)
    assert "ON CONFLICT (trade_id)" in sql


def test_get_trade_evaluation_ids_empty_input_skips_query(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)

    assert models.get_trade_evaluation_ids([]) == set()
    conn.cursor.assert_not_called()


def test_get_trade_evaluation_ids_returns_set(monkeypatch):
    conn, _ = _fake_connection(rows=[{"trade_id": 1}, {"trade_id": 3}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    assert models.get_trade_evaluation_ids([1, 2, 3]) == {1, 3}


# --- purge_old_data (Data Retention) ---


def test_purge_old_data_deletes_rows_past_cutoff_per_table(monkeypatch):
    conn, cur = _fake_connection(rowcount=2)
    monkeypatch.setattr(models, "get_client", lambda: conn)

    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    result = models.purge_old_data({"opportunity_evaluations": cutoff, "agent_logs": cutoff})

    assert result == {"opportunity_evaluations": 2, "agent_logs": 2}
    sql, params = _last_execute(cur)
    assert "DELETE FROM agent_logs WHERE timestamp < %s" == sql
    assert params == (cutoff.isoformat(),)


def test_purge_old_data_skips_tables_without_a_cutoff(monkeypatch):
    conn, cur = _fake_connection(rowcount=0)
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.purge_old_data({"agent_logs": datetime.now(timezone.utc)})

    assert cur.execute.call_count == 1  # every other _RETENTION_TABLES entry skipped, no query at all
    assert result == {"agent_logs": 0}


def test_purge_old_data_empty_cutoffs_touches_nothing(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)

    assert models.purge_old_data({}) == {}
    conn.cursor.assert_not_called()


# --- promotion_audit (src/learning/promotion_gate.py) ---


def test_insert_promotion_audit_inserts_expected_row(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_promotion_audit(
        mode="paper",
        event_type="promotion",
        decision="PROMOTE",
        candidate_version_id=5,
        previous_champion_id=3,
        new_champion_id=5,
        promotion_score=85.0,
        gates={"g": {"passed": True}},
        breakdown={"metrics": {}},
        reasons=["all gates cleared"],
    )

    sql, _ = _last_execute(cur)
    assert "INSERT INTO promotion_audit" in sql
    row = _inserted_row(cur)
    assert row["mode"] == "paper"
    assert row["event_type"] == "promotion"
    assert row["decision"] == "PROMOTE"
    assert row["candidate_version_id"] == 5
    assert row["previous_champion_id"] == 3
    assert row["new_champion_id"] == 5
    assert row["promotion_score"] == 85.0
    assert result == {"id": 1}


def test_insert_promotion_audit_defaults_jsonb_fields(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.insert_promotion_audit(mode="paper", event_type="evaluation", decision="REJECT")

    row = _inserted_row(cur)
    assert row["gates"] == {}
    assert row["breakdown"] == {}
    # reasons is wrapped in psycopg2.extras.Json — compare its .adapted value
    assert row["reasons"].adapted == []


def test_get_latest_promotion_audit_filters_by_mode_and_event_type(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 7, "decision": "PROMOTE"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_latest_promotion_audit("paper", event_type="promotion")

    sql, params = _last_execute(cur)
    assert "mode = %s" in sql and "event_type = %s" in sql
    assert params == ("paper", "promotion")
    assert result == {"id": 7, "decision": "PROMOTE"}


def test_get_latest_promotion_audit_returns_none_when_empty(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    assert models.get_latest_promotion_audit("paper") is None
