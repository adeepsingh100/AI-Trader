from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.db import models
from src.groq_client import ModelUsageEvent
from tests.conftest import _fake_connection, _fake_firestore_client, _inserted_row, _last_execute, _updated_row


def test_open_trade_inserts_expected_row(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.open_trade(
        "paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "signal said buy", strategy_type="default"
    )

    (row,) = store["trades"].values()
    assert row["mode"] == "paper"
    assert row["status"] == "open"
    assert row["symbol"] == "BTCINR"
    assert row["strategy_type"] == "default"
    assert result["id"] in store["trades"]


def test_close_trade_sets_closed_fields(monkeypatch):
    client, store = _fake_firestore_client(seed={"trades": {"1": {"status": "open"}}})
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.close_trade(1, exit_price=6300000, pnl=100.0)

    row = store["trades"]["1"]
    assert row["exit_price"] == 6300000
    assert row["pnl"] == 100.0
    assert row["status"] == "closed"
    assert "closed_at" in row


def test_upsert_daily_pnl_conflict_key(monkeypatch):
    import datetime

    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_daily_pnl(datetime.date(2026, 8, 15), "paper", 100.0, 3, True, False)

    row = store["daily_pnl"]["2026-08-15_paper_default"]
    assert row["date"] == "2026-08-15"
    assert row["mode"] == "paper"
    assert row["strategy_type"] == "default"


def test_log_model_usage_batches_rows(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    events = [
        ModelUsageEvent("model-a", None, 120, False),
        ModelUsageEvent("model-b", "model-a failed: 429", 340, True),
    ]
    models.log_model_usage(events)

    rows = list(store["model_usage"].values())
    assert len(rows) == 2
    assert {r["model_used"] for r in rows} == {"model-a", "model-b"}


def test_log_model_usage_skips_empty(monkeypatch):
    called = False

    def _fail(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    models.log_model_usage([])

    assert not called


def test_get_latest_promoted_version_filters_and_orders(monkeypatch):
    seed = {"strategy_versions": {
        "1": {"strategy_type": "default", "status": "active", "promoted_to_real": False, "version_number": 1},
        "2": {"strategy_type": "default", "status": "active", "promoted_to_real": True, "version_number": 2},
        "3": {"strategy_type": "default", "status": "active", "promoted_to_real": True, "version_number": 1},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_latest_promoted_version()

    assert result["id"] == "2"  # promoted, and the higher version_number of the two promoted rows


def test_log_opportunity_evaluation_inserts_expected_row(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

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
        strategy_type="default",
    )

    (row,) = store["opportunity_evaluations"].values()
    assert row["symbol"] == "BTCINR"
    assert row["opportunity_score"] == 82.0
    assert row["llm_decision"] == "accept"
    assert row["final_decision"] == "buy"
    assert row["features"] == {"5m": {"rsi": 55.0}}
    assert row["strategy_type"] == "default"


# --- learning engine: trades extensions ---


def test_open_trade_includes_learning_engine_fields(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.open_trade(
        "paper", 1, "BTCINR", "buy", 0.001, 6200000, 5.0, "go", strategy_type="default",
        stop_loss_price=6076000, take_profit_price=6448000,
        entry_slippage_pct=0.05, market_regime="strong_bull",
    )

    (row,) = store["trades"].values()
    assert row["stop_loss_price"] == 6076000
    assert row["take_profit_price"] == 6448000
    assert row["market_regime"] == "strong_bull"


def test_close_trade_includes_exit_reason(monkeypatch):
    client, store = _fake_firestore_client(seed={"trades": {"1": {"status": "open"}}})
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.close_trade(1, exit_price=6300000, pnl=100.0, exit_reason="take_profit")

    assert store["trades"]["1"]["exit_reason"] == "take_profit"


def test_update_trade_excursion(monkeypatch):
    client, store = _fake_firestore_client(seed={"trades": {"1": {}}})
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.update_trade_excursion(1, mfe_pct=2.5, mae_pct=1.1)

    assert store["trades"]["1"] == {"mfe_pct": 2.5, "mae_pct": 1.1}


def test_get_recently_closed_trades_filters_by_mode_status_and_time(monkeypatch):
    import datetime

    seed = {"trades": {
        "1": {"mode": "paper", "status": "closed", "closed_at": datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)},
        "2": {"mode": "paper", "status": "open", "closed_at": None},
        "3": {"mode": "paper", "status": "closed", "closed_at": datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)},
        "4": {"mode": "real", "status": "closed", "closed_at": datetime.datetime(2026, 1, 2, tzinfo=datetime.timezone.utc)},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    since = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
    result = models.get_recently_closed_trades("paper", since)

    assert {r["id"] for r in result} == {"1"}


def test_log_opportunity_evaluation_returns_row_and_includes_trade_id(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.log_opportunity_evaluation(
        mode="paper", symbol="BTCINR", version_id=1, features={},
        trend_score=None, momentum_score=None, volume_score=None,
        volatility_score=None, risk_score=None, opportunity_score=None,
        llm_decision=None, llm_reasoning=None, llm_raw_response=None,
        risk_manager_result=None, final_decision="hold", reason=None,
        strategy_type="default", trade_id=7,
    )

    (row,) = store["opportunity_evaluations"].values()
    assert row["trade_id"] == 7
    assert result["trade_id"] == 7
    assert result["id"] in store["opportunity_evaluations"]


# --- learning_statistics ---


def test_upsert_learning_statistics_conflict_key(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_learning_statistics("paper", "symbol", "BTCINR", {"win_rate": 0.5, "trades_count": 10})

    row = store["learning_statistics"]["paper_default_symbol_BTCINR"]
    assert row["dimension_type"] == "symbol"
    assert row["win_rate"] == 0.5
    assert row["strategy_type"] == "default"


def test_get_learning_statistics_filters_by_dimension_type(monkeypatch):
    seed = {"learning_statistics": {
        "paper_default_symbol_BTCINR": {
            "mode": "paper", "strategy_type": "default", "dimension_type": "symbol", "dimension_value": "BTCINR",
        },
        "paper_default_regime_bull": {
            "mode": "paper", "strategy_type": "default", "dimension_type": "regime", "dimension_value": "bull",
        },
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_learning_statistics("paper", dimension_type="symbol")

    assert [r["dimension_value"] for r in result] == ["BTCINR"]


# --- feature_importance ---


def test_upsert_feature_importance_conflict_key(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_feature_importance("paper", "rsi", 0.42, 30, "1h")

    row = store["feature_importance"]["paper_default_rsi_1h"]
    assert row["timeframe"] == "1h"
    assert row["strategy_type"] == "default"


def test_get_feature_importance_filters_by_timeframe(monkeypatch):
    seed = {"feature_importance": {
        "paper_default_trend_score_blended": {
            "mode": "paper", "strategy_type": "default", "feature_name": "trend_score", "timeframe": "blended",
        },
        "paper_default_trend_score_1h": {
            "mode": "paper", "strategy_type": "default", "feature_name": "trend_score", "timeframe": "1h",
        },
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_feature_importance("paper", timeframe="blended")

    assert [r["timeframe"] for r in result] == ["blended"]


# --- confidence_calibration ---


def test_log_confidence_calibration_inserts_expected_row(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.log_confidence_calibration(
        opportunity_evaluation_id=42, ai_confidence=80, historical_confidence=60,
        ai_weight=0.6, historical_weight=0.4, final_confidence=72.0, similar_trades_count=10,
    )

    (row,) = store["confidence_calibration"].values()
    assert row["opportunity_evaluation_id"] == 42
    assert row["final_confidence"] == 72.0


# --- recommendations ---


def test_get_latest_recommendation_orders_by_created_at(monkeypatch):
    import datetime as dt

    seed = {"recommendations": {
        "1": {"mode": "paper", "strategy_type": "default", "metric_name": "MIN_OPPORTUNITY_SCORE",
              "recommended_value": 75, "created_at": dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)},
        "2": {"mode": "paper", "strategy_type": "default", "metric_name": "MIN_OPPORTUNITY_SCORE",
              "recommended_value": 60, "created_at": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE")

    assert result["recommended_value"] == 75


def test_get_latest_recommendation_none_when_no_rows(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    assert models.get_latest_recommendation("paper", "MIN_OPPORTUNITY_SCORE") is None


# --- trade_evaluations ---


def test_upsert_trade_evaluation_conflict_key(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_trade_evaluation(
        trade_id=7, predicted_confidence=80.0, predicted_opportunity_score=85.0,
        actual_outcome_won=True, confidence_was_accurate=True,
        opportunity_score_was_accurate=True, risk_assessment="appropriate",
        stop_loss_assessment="appropriate", target_assessment="realistic",
    )

    row = store["trade_evaluations"]["7"]  # doc ID IS the trade_id — upsert-by-doc-ID is the conflict key
    assert row["trade_id"] == 7
    assert row["actual_outcome_won"] is True


def test_get_trade_evaluation_ids_empty_input_skips_query(monkeypatch):
    called = False

    def _fail(*a, **kw):
        nonlocal called
        called = True

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    assert models.get_trade_evaluation_ids([]) == set()
    assert not called


def test_get_trade_evaluation_ids_returns_set(monkeypatch):
    seed = {"trade_evaluations": {"1": {"trade_id": 1}, "3": {"trade_id": 3}}}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    assert models.get_trade_evaluation_ids([1, 2, 3]) == {1, 3}


# --- purge_old_data (Data Retention) ---


def test_purge_old_data_deletes_docs_past_cutoff_per_collection(monkeypatch):
    cutoff = datetime(2026, 1, 1, tzinfo=timezone.utc)
    seed = {
        "opportunity_evaluations": {
            "1": {"timestamp": datetime(2025, 12, 1, tzinfo=timezone.utc)},
            "2": {"timestamp": datetime(2026, 2, 1, tzinfo=timezone.utc)},  # after cutoff, kept
        },
        "agent_logs": {
            "1": {"timestamp": datetime(2025, 11, 1, tzinfo=timezone.utc)},
        },
    }
    client, store = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.purge_old_data({"opportunity_evaluations": cutoff, "agent_logs": cutoff})

    assert result == {"opportunity_evaluations": 1, "agent_logs": 1}
    assert list(store["opportunity_evaluations"].keys()) == ["2"]
    assert store["agent_logs"] == {}


def test_purge_old_data_skips_collections_without_a_cutoff(monkeypatch):
    seed = {"agent_logs": {"1": {"timestamp": datetime(2020, 1, 1, tzinfo=timezone.utc)}}}
    client, store = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.purge_old_data({"agent_logs": datetime.now(timezone.utc)})

    assert result == {"agent_logs": 1}  # every other _RETENTION_TABLES entry skipped untouched
    assert store["agent_logs"] == {}


def test_purge_old_data_empty_cutoffs_touches_nothing(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore with no cutoffs supplied")

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    assert models.purge_old_data({}) == {}


# --- promotion_audit (src/learning/promotion_gate.py) ---


def test_insert_promotion_audit_inserts_expected_row(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

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

    row = store["promotion_audit"][result["id"]]
    assert row["mode"] == "paper"
    assert row["event_type"] == "promotion"
    assert row["decision"] == "PROMOTE"
    assert row["candidate_version_id"] == 5
    assert row["previous_champion_id"] == 3
    assert row["new_champion_id"] == 5
    assert row["promotion_score"] == 85.0


def test_insert_promotion_audit_defaults_empty_fields(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.insert_promotion_audit(mode="paper", event_type="evaluation", decision="REJECT")

    row = store["promotion_audit"][result["id"]]
    assert row["gates"] == {}
    assert row["breakdown"] == {}
    assert row["reasons"] == []


def test_get_latest_promotion_audit_filters_by_mode_and_event_type(monkeypatch):
    seed = {"promotion_audit": {
        "1": {"mode": "paper", "strategy_type": "default", "event_type": "promotion", "decision": "PROMOTE", "created_at": 2},
        "2": {"mode": "paper", "strategy_type": "default", "event_type": "evaluation", "decision": "REJECT", "created_at": 3},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_latest_promotion_audit("paper", event_type="promotion")

    assert result["id"] == "1"
    assert result["decision"] == "PROMOTE"


def test_get_latest_promotion_audit_returns_none_when_empty(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    assert models.get_latest_promotion_audit("paper") is None


# --- multi-strategy-type: strategy_type=None keeps the old mode-wide
# query byte-identical; passing a real value adds a strategy_type
# equality filter, read straight off the trade doc (no JOIN needed —
# Firestore has none, and open_trade denormalizes strategy_type onto
# every trade at write time). ---


def test_get_open_trades_without_strategy_type_is_unchanged_mode_wide_query(monkeypatch):
    seed = {"trades": {
        "1": {"mode": "paper", "status": "open", "strategy_type": "default"},
        "2": {"mode": "paper", "status": "open", "strategy_type": "swing"},
        "3": {"mode": "paper", "status": "closed", "strategy_type": "default"},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_open_trades("paper")

    assert {r["id"] for r in result} == {"1", "2"}  # both strategy_types, mode-wide


def test_get_open_trades_with_strategy_type_filters_on_the_denormalized_field(monkeypatch):
    seed = {"trades": {
        "1": {"mode": "paper", "status": "open", "strategy_type": "default"},
        "2": {"mode": "paper", "status": "open", "strategy_type": "swing"},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_open_trades("paper", "swing")

    assert {r["id"] for r in result} == {"2"}


def test_get_capital_config_filters_by_mode_and_strategy_type(monkeypatch):
    seed = {"capital_config": {"paper_swing": {"mode": "paper", "strategy_type": "swing"}}}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_capital_config("paper", "swing")

    assert result["mode"] == "paper"
    assert result["strategy_type"] == "swing"


def test_upsert_capital_config_conflict_key_is_mode_and_strategy_type(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.upsert_capital_config("paper", 1000, 1000, 50, 100, strategy_type="swing")

    row = store["capital_config"]["paper_swing"]  # doc ID IS the (mode, strategy_type) conflict key
    assert row["strategy_type"] == "swing"
