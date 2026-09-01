from datetime import date, datetime, timezone

from src.db import models
from tests.conftest import _fake_firestore_client


# --- historical_candles ---


def test_upsert_historical_candles_noop_on_empty_list(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)

    models.upsert_historical_candles("I-BTC_INR", "1m", [])


def test_upsert_historical_candles_upserts_by_deterministic_doc_id(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    candles = [{"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]
    models.upsert_historical_candles("I-BTC_INR", "1m", candles)

    row = store["historical_candles"]["I-BTC_INR_1m_1000"]
    assert row["open"] == 1 and row["volume"] == 10

    # a repeat write for the same pair/interval/time overwrites, not duplicates
    models.upsert_historical_candles("I-BTC_INR", "1m", [{**candles[0], "close": 9.9}])
    assert len(store["historical_candles"]) == 1
    assert store["historical_candles"]["I-BTC_INR_1m_1000"]["close"] == 9.9


def test_get_historical_candles_filters_by_pair_interval_and_range(monkeypatch):
    seed = {"historical_candles": {
        "I-BTC_INR_1m_500": {"pair": "I-BTC_INR", "interval": "1m", "time": 500},
        "I-BTC_INR_1m_1000": {"pair": "I-BTC_INR", "interval": "1m", "time": 1000},
        "I-BTC_INR_1m_3000": {"pair": "I-BTC_INR", "interval": "1m", "time": 3000},  # out of range
        "I-ETH_INR_1m_1000": {"pair": "I-ETH_INR", "interval": "1m", "time": 1000},  # wrong pair
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_historical_candles("I-BTC_INR", "1m", 0, 2000)

    assert [r["time"] for r in result] == [500, 1000]


def test_historical_candles_exist_true_and_false(monkeypatch):
    seed = {"historical_candles": {"I-BTC_INR_1m_1000": {"pair": "I-BTC_INR", "interval": "1m", "time": 1000}}}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    assert models.historical_candles_exist("I-BTC_INR", "1m", 0, 2000) is True
    assert models.historical_candles_exist("I-BTC_INR", "1m", 5000, 6000) is False


# --- backtest_runs ---


def test_insert_backtest_run_defaults_status_running(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.insert_backtest_run(
        symbols=["BTCINR"], start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
        warmup_buffer_days=260, starting_capital=100000, params_json={},
    )

    row = store["backtest_runs"][result["id"]]
    assert row["symbols"] == ["BTCINR"]
    assert row["status"] == "running"


def test_update_backtest_run_status_sets_completed_at_when_given(monkeypatch):
    seed = {"backtest_runs": {"run1": {"status": "running"}}}
    client, store = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    models.update_backtest_run_status("run1", "completed", completed_at=now)

    row = store["backtest_runs"]["run1"]
    assert row["status"] == "completed"
    assert row["completed_at"] == now


def test_get_backtest_run_none_when_missing(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)
    assert models.get_backtest_run("missing") is None


def test_get_backtest_runs_filters_by_status(monkeypatch):
    seed = {"backtest_runs": {
        "1": {"status": "completed", "created_at": 1},
        "2": {"status": "running", "created_at": 2},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_backtest_runs(status="completed")

    assert [r["id"] for r in result] == ["1"]


# --- backtest_trades / snapshots / execution history (run-scoped subcollections) ---


def test_insert_backtest_trade_scoped_to_run(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_backtest_trade("run5", {"symbol": "BTCINR", "pnl": 10.0, "entry_time": 1})
    models.insert_backtest_trade("run5", {"symbol": "ETHINR", "pnl": -5.0, "entry_time": 2})

    trades = models.get_backtest_trades("run5")
    assert len(trades) == 2  # two separate inserts into the same run must not collide on doc ID
    assert {t["symbol"] for t in trades} == {"BTCINR", "ETHINR"}


def test_get_backtest_trades_ordered_by_entry_time(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_backtest_trade("run5", {"symbol": "B", "entry_time": 2})
    models.insert_backtest_trade("run5", {"symbol": "A", "entry_time": 1})

    result = models.get_backtest_trades("run5")
    assert [r["symbol"] for r in result] == ["A", "B"]


def test_insert_backtest_portfolio_snapshots_noop_on_empty(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)
    models.insert_backtest_portfolio_snapshots("run5", [])


def test_insert_backtest_portfolio_snapshots_batches_all_rows(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    snapshots = [{"snapshot_time": "t1", "equity": 100}, {"snapshot_time": "t2", "equity": 110}]
    models.insert_backtest_portfolio_snapshots("run5", snapshots)

    result = models.get_backtest_portfolio_snapshots("run5")
    assert len(result) == 2


def test_insert_backtest_execution_events_noop_on_empty(monkeypatch):
    def _fail(*a, **kw):
        raise AssertionError("should not touch Firestore for an empty list")

    monkeypatch.setattr(models, "get_firestore_client", _fail)
    models.insert_backtest_execution_events("run5", [])


# --- performance metrics / walk-forward folds / strategy comparisons ---


def test_insert_and_get_backtest_performance_metrics(monkeypatch):
    seed = {"backtest_runs": {"run5": {"status": "running"}}}
    client, store = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_backtest_performance_metrics("run5", {"win_rate": 0.6})

    result = models.get_backtest_performance_metrics("run5")
    assert result == {"run_id": "run5", "metrics": {"win_rate": 0.6}}
    # the run doc's other fields survive -- this is a merge onto the run, not an overwrite
    assert store["backtest_runs"]["run5"]["status"] == "running"


def test_get_backtest_performance_metrics_none_when_missing(monkeypatch):
    seed = {"backtest_runs": {"run5": {"status": "running"}}}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)
    assert models.get_backtest_performance_metrics("run5") is None


def test_insert_backtest_walk_forward_fold_scoped_to_run(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_backtest_walk_forward_fold("run5", {"fold_number": 1, "passed": True})

    folds = models.get_backtest_walk_forward_folds("run5")
    assert len(folds) == 1
    assert folds[0]["fold_number"] == 1


def test_get_backtest_walk_forward_folds_ordered_by_fold_number(monkeypatch):
    client, _ = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    models.insert_backtest_walk_forward_fold("run5", {"fold_number": 2})
    models.insert_backtest_walk_forward_fold("run5", {"fold_number": 1})

    result = models.get_backtest_walk_forward_folds("run5")
    assert [r["fold_number"] for r in result] == [1, 2]


def test_insert_backtest_strategy_comparison(monkeypatch):
    client, store = _fake_firestore_client()
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.insert_backtest_strategy_comparison("run1", "run2", {"a": 1}, {"b": 2}, {"p": 0.01}, "b", True)

    row = store["backtest_strategy_comparisons"][result["id"]]
    assert row["run_id_a"] == "run1"
    assert row["run_id_b"] == "run2"
    assert row["promotion_recommended"] is True


def test_get_backtest_strategy_comparisons_ordered_by_created_at_desc(monkeypatch):
    seed = {"backtest_strategy_comparisons": {
        "1": {"run_id_a": "a", "run_id_b": "b", "created_at": 1},
        "2": {"run_id_a": "c", "run_id_b": "d", "created_at": 2},
    }}
    client, _ = _fake_firestore_client(seed=seed)
    monkeypatch.setattr(models, "get_firestore_client", lambda: client)

    result = models.get_backtest_strategy_comparisons()
    assert [r["id"] for r in result] == ["2", "1"]
