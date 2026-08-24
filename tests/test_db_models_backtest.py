from datetime import date, datetime, timezone
from unittest.mock import MagicMock

from src.db import models
from tests.conftest import _fake_connection, _inserted_row, _last_execute, _updated_row


# --- historical_candles ---


def test_upsert_historical_candles_noop_on_empty_list(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.upsert_historical_candles("I-BTC_INR", "1m", [])

    conn.cursor.assert_not_called()


def test_upsert_historical_candles_upserts_with_conflict_key(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    fake_execute_values = MagicMock()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    monkeypatch.setattr(models, "execute_values", fake_execute_values)

    candles = [{"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]
    models.upsert_historical_candles("I-BTC_INR", "1m", candles)

    sql, rows = fake_execute_values.call_args[0][1], fake_execute_values.call_args[0][2]
    assert "ON CONFLICT (pair, interval, time)" in sql
    assert rows[0] == ("I-BTC_INR", "1m", 1000, 1, 2, 0.5, 1.5, 10)


def test_get_historical_candles_filters_by_pair_interval_and_range(monkeypatch):
    conn, cur = _fake_connection(rows=[{"time": 1000}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_historical_candles("I-BTC_INR", "1m", 0, 2000)

    assert result == [{"time": 1000}]
    sql, params = _last_execute(cur)
    assert "pair = %s" in sql and "interval = %s" in sql and "time >= %s" in sql and "time <= %s" in sql
    assert params == ("I-BTC_INR", "1m", 0, 2000)


# --- backtest_runs ---


def test_insert_backtest_run_returns_inserted_row(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_backtest_run(
        symbols=["BTCINR"], start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
        warmup_buffer_days=260, starting_capital=100000, params_json={},
    )

    assert result == {"id": 1}
    row = _inserted_row(cur)
    assert row["symbols"] == ["BTCINR"]
    assert "status" not in row  # left to the DB default ('running'), not overridden here


def test_update_backtest_run_status_sets_completed_at_when_given(monkeypatch):
    conn, cur = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    models.update_backtest_run_status(1, "completed", completed_at=now)

    row = _updated_row(cur)
    assert row["status"] == "completed"
    assert row["completed_at"] == now.isoformat()


def test_get_backtest_run_none_when_missing(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)
    assert models.get_backtest_run(999) is None


def test_get_backtest_runs_filters_by_status(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1, "status": "completed"}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_backtest_runs(status="completed")

    assert result == [{"id": 1, "status": "completed"}]
    sql, params = _last_execute(cur)
    assert "status = %s" in sql
    assert params == ("completed",)


# --- backtest_trades / snapshots / execution history (batch inserts) ---


def test_insert_backtest_trade_attaches_run_id(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.insert_backtest_trade(5, {"symbol": "BTCINR", "pnl": 10.0})

    row = _inserted_row(cur)
    assert row["run_id"] == 5
    assert row["symbol"] == "BTCINR"


def test_insert_backtest_portfolio_snapshots_noop_on_empty(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    models.insert_backtest_portfolio_snapshots(1, [])
    conn.cursor.assert_not_called()


def test_insert_backtest_portfolio_snapshots_batches_all_rows(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    fake_execute_values = MagicMock()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    monkeypatch.setattr(models, "execute_values", fake_execute_values)

    snapshots = [{"snapshot_time": "t1", "equity": 100}, {"snapshot_time": "t2", "equity": 110}]
    models.insert_backtest_portfolio_snapshots(1, snapshots)

    sql, rows = fake_execute_values.call_args[0][1], fake_execute_values.call_args[0][2]
    assert "INSERT INTO backtest_portfolio_snapshots" in sql
    assert len(rows) == 2
    assert all(r[0] == 1 for r in rows)  # run_id is always the first column here


def test_insert_backtest_execution_events_noop_on_empty(monkeypatch):
    conn, _ = _fake_connection()
    monkeypatch.setattr(models, "get_client", lambda: conn)
    models.insert_backtest_execution_events(1, [])
    conn.cursor.assert_not_called()


# --- performance metrics / walk-forward folds / strategy comparisons ---


def test_insert_backtest_performance_metrics(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1, "run_id": 5}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_backtest_performance_metrics(5, {"win_rate": 0.6})

    assert result == {"id": 1, "run_id": 5}
    row = _inserted_row(cur)
    assert row["metrics"] == {"win_rate": 0.6}


def test_get_backtest_performance_metrics_none_when_missing(monkeypatch):
    conn, _ = _fake_connection(rows=[])
    monkeypatch.setattr(models, "get_client", lambda: conn)
    assert models.get_backtest_performance_metrics(5) is None


def test_insert_backtest_walk_forward_fold_attaches_run_id(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    models.insert_backtest_walk_forward_fold(5, {"fold_number": 1, "passed": True})

    row = _inserted_row(cur)
    assert row["run_id"] == 5
    assert row["fold_number"] == 1


def test_get_backtest_walk_forward_folds_ordered_by_fold_number(monkeypatch):
    conn, cur = _fake_connection(rows=[{"fold_number": 1}, {"fold_number": 2}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.get_backtest_walk_forward_folds(5)

    assert result == [{"fold_number": 1}, {"fold_number": 2}]
    sql, _ = _last_execute(cur)
    assert "ORDER BY fold_number" in sql


def test_insert_backtest_strategy_comparison(monkeypatch):
    conn, cur = _fake_connection(rows=[{"id": 1}])
    monkeypatch.setattr(models, "get_client", lambda: conn)

    result = models.insert_backtest_strategy_comparison(1, 2, {"a": 1}, {"b": 2}, {"p": 0.01}, "b", True)

    assert result == {"id": 1}
    row = _inserted_row(cur)
    assert row["run_id_a"] == 1
    assert row["run_id_b"] == 2
    assert row["promotion_recommended"] is True
