from unittest.mock import Mock

from src.audit.trail import config_version, get_decision_trail
from src.db import models


def test_config_version_deterministic():
    assert config_version() == config_version()


def test_config_version_changes_when_a_weight_changes(monkeypatch):
    before = config_version()
    monkeypatch.setattr("src.audit.trail.OPPORTUNITY_WEIGHT_TREND", 0.99)
    after = config_version()
    assert before != after


def test_config_version_is_short_hex_string():
    v = config_version()
    assert len(v) == 12
    int(v, 16)  # raises if not valid hex


def test_get_decision_trail_joins_calibration_by_evaluation_id(monkeypatch):
    eval_row = {"id": 501, "symbol": "BTCINR", "trade_id": 99, "timestamp": "2026-01-01T00:00:00+00:00"}
    calibration_row = {"opportunity_evaluation_id": 501, "final_confidence": 72.0}

    mock_query = Mock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.gte.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = Mock(data=[eval_row])
    mock_client = Mock()
    mock_client.table.return_value = mock_query
    monkeypatch.setattr(models, "get_client", lambda: mock_client)
    monkeypatch.setattr(models, "get_confidence_calibration_for_evaluation", lambda eid: calibration_row)

    trail = get_decision_trail(mode="paper", trade_id=99)

    assert len(trail) == 1
    assert trail[0]["symbol"] == "BTCINR"
    assert trail[0]["calibration"] == calibration_row


def test_get_decision_trail_calibration_none_when_not_logged(monkeypatch):
    eval_row = {"id": 502, "symbol": "ETHINR", "trade_id": None, "timestamp": "2026-01-01T00:00:00+00:00"}
    mock_query = Mock()
    mock_query.select.return_value = mock_query
    mock_query.eq.return_value = mock_query
    mock_query.order.return_value = mock_query
    mock_query.execute.return_value = Mock(data=[eval_row])
    mock_client = Mock()
    mock_client.table.return_value = mock_query
    monkeypatch.setattr(models, "get_client", lambda: mock_client)
    monkeypatch.setattr(models, "get_confidence_calibration_for_evaluation", lambda eid: None)

    trail = get_decision_trail(mode="paper", symbol="ETHINR")

    assert trail[0]["calibration"] is None
