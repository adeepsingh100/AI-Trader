from unittest.mock import patch

from src.agents.reporting_agent import (
    _mode_section_html,
    _model_usage_stats,
    _trade_log_html,
    _versions_html,
    build_report_data,
    render_html,
)


# --- model usage aggregation ---


def test_model_usage_stats_groups_and_averages():
    events = [
        {"model_used": "model-a", "success": True, "fallback_reason": None, "latency_ms": 100},
        {"model_used": "model-a", "success": False, "fallback_reason": None, "latency_ms": 300},
        {"model_used": "model-b", "success": True, "fallback_reason": "model-a failed", "latency_ms": 200},
    ]
    stats = _model_usage_stats(events)
    by_model = {s["model"]: s for s in stats}

    assert by_model["model-a"]["calls"] == 2
    assert by_model["model-a"]["success_rate"] == 0.5
    assert by_model["model-a"]["avg_latency_ms"] == 200
    assert by_model["model-b"]["fallback_rate"] == 1.0


def test_model_usage_stats_sorted_by_call_volume():
    events = [
        {"model_used": "rare", "success": True, "fallback_reason": None, "latency_ms": 1},
        {"model_used": "common", "success": True, "fallback_reason": None, "latency_ms": 1},
        {"model_used": "common", "success": True, "fallback_reason": None, "latency_ms": 1},
    ]
    stats = _model_usage_stats(events)
    assert [s["model"] for s in stats] == ["common", "rare"]


def test_model_usage_stats_empty():
    assert _model_usage_stats([]) == []


# --- XSS escaping ---


def test_trade_log_escapes_reasoning_text():
    trades = [
        {
            "symbol": "BTCINR",
            "side": "buy",
            "qty": 0.001,
            "entry_price": 100.0,
            "exit_price": None,
            "pnl": None,
            "status": "open",
            "opened_at": "2026-08-15T00:00:00Z",
            "reasoning_text": "<script>alert(1)</script>",
        }
    ]
    out = _trade_log_html(trades)
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_versions_html_escapes_notes():
    versions = [
        {
            "version_number": 1,
            "promoted_to_real": False,
            "notes": "<img src=x onerror=alert(1)>",
            "created_at": "2026-08-15T00:00:00Z",
        }
    ]
    out = _versions_html(versions)
    assert "<img src=x" not in out
    assert "&lt;img" in out


# --- section rendering ---


def test_mode_section_html_handles_missing_capital_config():
    out = _mode_section_html({"mode": "real", "capital_config": None, "daily_pnl": None, "trades": []})
    assert "No capital_config set" in out


@patch("src.agents.reporting_agent.generate_learning_report_html")
def test_mode_section_html_handles_no_daily_pnl_row_yet(mock_learning_html):
    mock_learning_html.return_value = "<section></section>"
    section = {
        "mode": "paper",
        "capital_config": {
            "capital_to_use": 10000,
            "total_capital": 10000,
            "daily_profit_target": 500,
        },
        "daily_pnl": None,
        "trades": [],
    }
    out = _mode_section_html(section)
    assert "0.00" in out  # realized defaults to 0
    assert "clear" in out  # breaker defaults to not-triggered


# --- full build + render, mocked DB ---


@patch("src.agents.reporting_agent.generate_learning_report_html")
@patch("src.agents.reporting_agent.models")
def test_build_report_data_and_render_html_smoke(mock_models, mock_learning_html):
    mock_learning_html.return_value = "<section></section>"
    mock_models.get_capital_config.return_value = {
        "capital_to_use": 10000,
        "total_capital": 10000,
        "daily_profit_target": 500,
    }
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_recent_trades.return_value = []
    mock_models.get_all_strategy_versions.return_value = []
    mock_models.get_recent_model_usage.return_value = []

    data = build_report_data()
    out = render_html(data)

    assert "<html>" in out
    assert "Paper" in out
    assert "Real" in out
