import pytest

from src.db import models
from src.monitoring.metrics import log_resource_snapshot, resource_snapshot, track


def test_resource_snapshot_has_expected_keys():
    snapshot = resource_snapshot()
    assert set(snapshot) == {
        "max_rss_kb", "user_cpu_seconds", "system_cpu_seconds", "disk_free_bytes", "disk_total_bytes",
    }


def test_track_records_success_metric(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "insert_system_metrics", lambda rows: calls.append(rows))

    with track("data_agent", "market_snapshot"):
        pass

    assert len(calls) == 1
    row = calls[0][0]
    assert row["component"] == "data_agent"
    assert row["metric_name"] == "market_snapshot_duration_ms"
    assert row["metadata"]["success"] is True


def test_track_records_failure_metric_and_reraises(monkeypatch):
    calls = []
    monkeypatch.setattr(models, "insert_system_metrics", lambda rows: calls.append(rows))

    with pytest.raises(ValueError):
        with track("data_agent", "market_snapshot"):
            raise ValueError("boom")

    row = calls[0][0]
    assert row["metadata"]["success"] is False
    assert row["metadata"]["error_type"] == "ValueError"


def test_track_fails_open_when_metrics_write_errors(monkeypatch):
    def _raise(rows):
        raise RuntimeError("db down")

    monkeypatch.setattr(models, "insert_system_metrics", _raise)

    with track("data_agent", "market_snapshot"):
        pass  # must not raise despite the metrics write failing


def test_log_resource_snapshot_fails_open_on_db_error(monkeypatch):
    def _raise(rows):
        raise RuntimeError("db down")

    monkeypatch.setattr(models, "insert_system_metrics", _raise)
    log_resource_snapshot("orchestrator")  # must not raise
