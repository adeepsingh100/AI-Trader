import json
from unittest.mock import patch

from src.backtest.report import export_run_json, export_trades_csv, generate_backtest_report_html


@patch("src.backtest.report.models")
def test_generate_backtest_report_html_missing_run(mock_models):
    mock_models.get_backtest_run.return_value = None
    out = generate_backtest_report_html(999)
    assert "No backtest run" in out


@patch("src.backtest.report.models")
def test_generate_backtest_report_html_smoke(mock_models):
    mock_models.get_backtest_run.return_value = {
        "symbols": ["BTCINR", "ETHINR"],
        "start_date": "2024-01-01",
        "end_date": "2024-02-01",
        "warmup_buffer_days": 260,
        "starting_capital": 100000,
        "use_llm_signal_agent": False,
        "status": "completed",
    }
    mock_models.get_backtest_trades.return_value = [
        {
            "symbol": "BTCINR",
            "entry_time": "2024-01-05T00:00:00Z",
            "exit_time": "2024-01-06T00:00:00Z",
            "pnl": 100.0,
            "return_pct": 5.0,
            "exit_reason": "<script>alert(1)</script>",
        }
    ]
    mock_models.get_backtest_performance_metrics.return_value = {
        "metrics": {"win_rate": 0.6, "expectancy": 10.0, "monthly_returns": {}, "annual_returns": {}}
    }
    mock_models.get_backtest_walk_forward_folds.return_value = [
        {
            "fold_number": 1,
            "train_window_start": "2024-01-01",
            "train_window_end": "2024-01-31",
            "test_window_start": "2024-01-31",
            "test_window_end": "2024-02-15",
            "p_value": 0.01,
            "passed": True,
        }
    ]
    mock_models.get_backtest_strategy_comparisons.return_value = [
        {"run_id_a": 1, "run_id_b": 2, "winner": "b", "promotion_recommended": True}
    ]

    out = generate_backtest_report_html(1)

    assert "BTCINR" in out
    assert "PASSED" in out
    assert "<script>alert(1)</script>" not in out  # XSS-escaped
    assert "&lt;script&gt;" in out
    assert "survivorship" in out.lower() or "fixed, user-supplied" in out.lower()


@patch("src.backtest.report.models")
def test_export_trades_csv_empty(mock_models):
    mock_models.get_backtest_trades.return_value = []
    assert export_trades_csv(1) == ""


@patch("src.backtest.report.models")
def test_export_trades_csv_has_header_and_rows(mock_models):
    mock_models.get_backtest_trades.return_value = [{"symbol": "BTCINR", "pnl": 10.0}]
    out = export_trades_csv(1)
    assert "symbol" in out
    assert "BTCINR" in out


@patch("src.backtest.report.models")
def test_export_run_json_is_valid_json(mock_models):
    mock_models.get_backtest_run.return_value = {"id": 1, "symbols": ["BTCINR"]}
    mock_models.get_backtest_trades.return_value = []
    mock_models.get_backtest_performance_metrics.return_value = None
    mock_models.get_backtest_walk_forward_folds.return_value = []

    out = export_run_json(1)
    parsed = json.loads(out)
    assert parsed["run"]["id"] == 1
    assert parsed["performance_metrics"] is None
