from datetime import date, datetime, timezone
from unittest.mock import Mock

from src.db import models
from tests.conftest import _fluent_mock


# --- historical_candles ---


def test_upsert_historical_candles_noop_on_empty_list(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.upsert_historical_candles("I-BTC_INR", "1m", [])

    client.table.assert_not_called()


def test_upsert_historical_candles_upserts_with_conflict_key(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    candles = [{"time": 1000, "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10}]
    models.upsert_historical_candles("I-BTC_INR", "1m", candles)

    client.table.assert_called_with("historical_candles")
    call = table.upsert.call_args
    rows = call.args[0]
    assert rows[0]["pair"] == "I-BTC_INR"
    assert rows[0]["interval"] == "1m"
    assert rows[0]["time"] == 1000
    assert call.kwargs["on_conflict"] == "pair,interval,time"


def test_get_historical_candles_filters_by_pair_interval_and_range(monkeypatch):
    table = _fluent_mock([{"time": 1000}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_historical_candles("I-BTC_INR", "1m", 0, 2000)

    assert result == [{"time": 1000}]
    table.eq.assert_any_call("pair", "I-BTC_INR")
    table.eq.assert_any_call("interval", "1m")
    table.gte.assert_called_with("time", 0)
    table.lte.assert_called_with("time", 2000)


# --- backtest_runs ---


def test_insert_backtest_run_returns_inserted_row(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.insert_backtest_run(
        symbols=["BTCINR"], start_date=date(2024, 1, 1), end_date=date(2024, 2, 1),
        warmup_buffer_days=260, starting_capital=100000, params_json={},
    )

    assert result == {"id": 1}
    inserted = table.insert.call_args[0][0]
    assert inserted["symbols"] == ["BTCINR"]
    assert "status" not in inserted  # left to the DB default ('running'), not overridden here


def test_update_backtest_run_status_sets_completed_at_when_given(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    models.update_backtest_run_status(1, "completed", completed_at=now)

    updated = table.update.call_args[0][0]
    assert updated["status"] == "completed"
    assert updated["completed_at"] == now.isoformat()


def test_get_backtest_run_none_when_missing(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    assert models.get_backtest_run(999) is None


def test_get_backtest_runs_filters_by_status(monkeypatch):
    table = _fluent_mock([{"id": 1, "status": "completed"}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_backtest_runs(status="completed")

    assert result == [{"id": 1, "status": "completed"}]
    table.eq.assert_called_with("status", "completed")


# --- backtest_trades / snapshots / execution history (batch inserts) ---


def test_insert_backtest_trade_attaches_run_id(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.insert_backtest_trade(5, {"symbol": "BTCINR", "pnl": 10.0})

    inserted = table.insert.call_args[0][0]
    assert inserted["run_id"] == 5
    assert inserted["symbol"] == "BTCINR"


def test_insert_backtest_portfolio_snapshots_noop_on_empty(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    models.insert_backtest_portfolio_snapshots(1, [])
    client.table.assert_not_called()


def test_insert_backtest_portfolio_snapshots_batches_all_rows(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    snapshots = [{"snapshot_time": "t1", "equity": 100}, {"snapshot_time": "t2", "equity": 110}]
    models.insert_backtest_portfolio_snapshots(1, snapshots)

    rows = table.insert.call_args[0][0]
    assert len(rows) == 2
    assert all(r["run_id"] == 1 for r in rows)


def test_insert_backtest_execution_events_noop_on_empty(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    models.insert_backtest_execution_events(1, [])
    client.table.assert_not_called()


# --- performance metrics / walk-forward folds / strategy comparisons ---


def test_insert_backtest_performance_metrics(monkeypatch):
    table = _fluent_mock([{"id": 1, "run_id": 5}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.insert_backtest_performance_metrics(5, {"win_rate": 0.6})

    assert result == {"id": 1, "run_id": 5}
    inserted = table.insert.call_args[0][0]
    assert inserted["metrics"] == {"win_rate": 0.6}


def test_get_backtest_performance_metrics_none_when_missing(monkeypatch):
    table = _fluent_mock([])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)
    assert models.get_backtest_performance_metrics(5) is None


def test_insert_backtest_walk_forward_fold_attaches_run_id(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    models.insert_backtest_walk_forward_fold(5, {"fold_number": 1, "passed": True})

    inserted = table.insert.call_args[0][0]
    assert inserted["run_id"] == 5
    assert inserted["fold_number"] == 1


def test_get_backtest_walk_forward_folds_ordered_by_fold_number(monkeypatch):
    table = _fluent_mock([{"fold_number": 1}, {"fold_number": 2}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.get_backtest_walk_forward_folds(5)

    assert result == [{"fold_number": 1}, {"fold_number": 2}]
    table.order.assert_called_with("fold_number")


def test_insert_backtest_strategy_comparison(monkeypatch):
    table = _fluent_mock([{"id": 1}])
    client = Mock(table=Mock(return_value=table))
    monkeypatch.setattr(models, "get_client", lambda: client)

    result = models.insert_backtest_strategy_comparison(1, 2, {"a": 1}, {"b": 2}, {"p": 0.01}, "b", True)

    assert result == {"id": 1}
    inserted = table.insert.call_args[0][0]
    assert inserted["run_id_a"] == 1
    assert inserted["run_id_b"] == 2
    assert inserted["promotion_recommended"] is True
