from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from src.db import models
from src.learning.drift_detection import (
    _severity,
    _split_by_window,
    detect_feature_drift,
    detect_feature_importance_drift,
    detect_performance_drift,
    population_stability_index,
    run_drift_detection,
)


def test_population_stability_index_zero_for_identical_distributions():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3
    psi = population_stability_index(values, values, buckets=5)
    assert psi == 0.0


def test_population_stability_index_large_for_shifted_distribution():
    baseline = [1.0] * 50 + [2.0] * 50
    recent = [9.0] * 50 + [10.0] * 50  # completely different range
    psi = population_stability_index(baseline, recent, buckets=5)
    assert psi is not None and psi > 1.0


def test_population_stability_index_none_below_minimum_sample():
    assert population_stability_index([1.0], [1.0, 2.0], buckets=5) is None


def test_population_stability_index_zero_for_constant_values():
    assert population_stability_index([5.0, 5.0], [5.0, 5.0], buckets=5) == 0.0


def test_severity_thresholds():
    assert _severity(0.05) is None
    assert _severity(0.15) == "warning"
    assert _severity(0.30) == "critical"


def test_split_by_window_separates_recent_from_baseline():
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 1, 30, tzinfo=timezone.utc)
    trades = [
        {"closed_at": (now - timedelta(days=2)).isoformat()},  # recent (< DRIFT_RECENT_WINDOW_DAYS=14)
        {"closed_at": (now - timedelta(days=60)).isoformat()},  # baseline
    ]
    baseline, recent = _split_by_window(trades, now)
    assert len(recent) == 1
    assert len(baseline) == 1


def test_detect_feature_importance_drift_flags_large_correlation_swing(monkeypatch):
    rows = [
        {"feature_key": "rsi", "correlation": 0.05, "computed_at": "2026-01-01T00:00:00+00:00"},
        {"feature_key": "rsi", "correlation": 0.55, "computed_at": "2026-01-30T00:00:00+00:00"},
    ]
    monkeypatch.setattr(models, "get_feature_importance", lambda mode, timeframe=None, strategy_type=None: rows)
    inserted = Mock(side_effect=lambda **kwargs: kwargs)
    monkeypatch.setattr(models, "insert_drift_alert", inserted)

    alerts = detect_feature_importance_drift("paper")

    assert len(alerts) == 1
    inserted.assert_called_once()
    assert inserted.call_args.kwargs["drift_type"] == "feature_importance:rsi"


def test_detect_feature_importance_drift_no_alert_for_stable_correlation(monkeypatch):
    rows = [
        {"feature_key": "rsi", "correlation": 0.30, "computed_at": "2026-01-01T00:00:00+00:00"},
        {"feature_key": "rsi", "correlation": 0.32, "computed_at": "2026-01-30T00:00:00+00:00"},
    ]
    monkeypatch.setattr(models, "get_feature_importance", lambda mode, timeframe=None, strategy_type=None: rows)
    monkeypatch.setattr(models, "insert_drift_alert", Mock())

    alerts = detect_feature_importance_drift("paper")
    assert alerts == []


def _trade(id_, closed_at, pnl=1.0):
    return {"id": id_, "closed_at": closed_at, "pnl": pnl}


def test_detect_feature_drift_no_crash_with_no_trades(monkeypatch):
    monkeypatch.setattr(models, "get_recently_closed_trades", lambda mode, since, strategy_type=None: [])
    assert detect_feature_drift("paper") == []


def test_detect_feature_drift_uses_entry_eval_features(monkeypatch):
    now = datetime.now(timezone.utc)
    baseline_trades = [_trade(i, (now - timedelta(days=60)).isoformat()) for i in range(1, 21)]
    recent_trades = [_trade(i, (now - timedelta(days=2)).isoformat()) for i in range(21, 41)]
    monkeypatch.setattr(models, "get_recently_closed_trades", lambda mode, since, strategy_type=None: baseline_trades + recent_trades)

    def _entry_eval(trade_id):
        # baseline rsi clustered low, recent rsi clustered high — real drift
        rsi = 20.0 if trade_id <= 20 else 80.0
        return {"features": {"1h": {"rsi": rsi}}}

    monkeypatch.setattr(models, "get_entry_evaluation_for_trade", _entry_eval)
    monkeypatch.setattr(models, "insert_drift_alert", lambda **kwargs: kwargs)

    alerts = detect_feature_drift("paper", timeframe="1h")
    assert any(a["drift_type"] == "feature_distribution:rsi" for a in alerts)


def test_detect_performance_drift_flags_significant_win_rate_drop(monkeypatch):
    now = datetime.now(timezone.utc)
    # Baseline: 25/25 wins. Recent: 0/25 wins — an unambiguous, significant drop.
    baseline_trades = [_trade(i, (now - timedelta(days=60)).isoformat(), pnl=10) for i in range(1, 26)]
    recent_trades = [_trade(i, (now - timedelta(days=2)).isoformat(), pnl=-10) for i in range(26, 51)]
    monkeypatch.setattr(models, "get_recently_closed_trades", lambda mode, since, strategy_type=None: baseline_trades + recent_trades)
    monkeypatch.setattr(models, "get_trade_evaluations", lambda ids: [])
    monkeypatch.setattr(models, "insert_drift_alert", lambda **kwargs: kwargs)

    alerts = detect_performance_drift("paper")

    win_rate_alerts = [a for a in alerts if a["drift_type"] == "win_rate"]
    assert len(win_rate_alerts) == 1
    assert win_rate_alerts[0]["severity"] == "critical"  # 100% -> 0% is a >=20pt drop


def test_detect_performance_drift_no_alert_below_sample_size_floor(monkeypatch):
    now = datetime.now(timezone.utc)
    baseline_trades = [_trade(1, (now - timedelta(days=60)).isoformat(), pnl=10)]
    recent_trades = [_trade(2, (now - timedelta(days=2)).isoformat(), pnl=-10)]
    monkeypatch.setattr(models, "get_recently_closed_trades", lambda mode, since, strategy_type=None: baseline_trades + recent_trades)
    monkeypatch.setattr(models, "get_trade_evaluations", lambda ids: [])
    monkeypatch.setattr(models, "insert_drift_alert", Mock())

    assert detect_performance_drift("paper") == []


def test_run_drift_detection_returns_summary_counts(monkeypatch):
    monkeypatch.setattr(models, "get_active_strategy_types", lambda mode: ["default"])
    monkeypatch.setattr(models, "get_recently_closed_trades", lambda mode, since, strategy_type=None: [])
    monkeypatch.setattr(models, "get_feature_importance", lambda mode, timeframe=None, strategy_type=None: [])
    result = run_drift_detection("paper")
    assert result == {
        "default": {
            "feature_drift_alerts": 0,
            "feature_importance_drift_alerts": 0,
            "performance_drift_alerts": 0,
        }
    }
