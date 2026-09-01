"""Step 13: reporting. HTML reuses the established
reporting_agent._table/local-import pattern (same precedent as the
Learning and Adaptive Strategy reports). CSV/JSON via stdlib csv/json — no
PDF: this codebase carries zero non-essential dependencies (not even
numpy/scipy despite far heavier justification), and the HTML report is
already print-to-PDF-ready in any browser. Standalone per-run artifact,
not wired into reporting_agent.py's live dashboard report."""

from __future__ import annotations

import csv
import html
import io
import json

from src.db import models


def _fmt(value, spec: str = ".2f") -> str:
    return "-" if value is None else format(value, spec)


def generate_backtest_report_html(run_id: str) -> str:
    from src.agents.reporting_agent import _table

    run = models.get_backtest_run(run_id)
    if run is None:
        return f"<p>No backtest run with id={run_id}.</p>"
    trades = models.get_backtest_trades(run_id)
    perf = models.get_backtest_performance_metrics(run_id)
    metrics = perf["metrics"] if perf else {}
    folds = models.get_backtest_walk_forward_folds(run_id)
    comparisons = [
        c
        for c in models.get_backtest_strategy_comparisons()
        if c["run_id_a"] == run_id or c["run_id_b"] == run_id
    ]

    summary_rows = [
        ["Symbols", html.escape(", ".join(run["symbols"]))],
        ["Window", f"{run['start_date']} to {run['end_date']}"],
        ["Warm-up buffer", f"{run['warmup_buffer_days']} days"],
        ["Starting capital", f"{run['starting_capital']:.2f}"],
        ["Final equity", _fmt(metrics.get("final_equity"))],
        ["Total return", f"{_fmt(metrics.get('total_return_pct'))}%"],
        ["LLM signal agent", "yes (non-reproducible)" if run["use_llm_signal_agent"] else "no (deterministic)"],
        ["Status", html.escape(run["status"])],
    ]

    metric_rows = [
        [k, _fmt(v) if isinstance(v, (int, float)) else html.escape(str(v))]
        for k, v in metrics.items()
        if k not in ("monthly_returns", "annual_returns")
    ]

    trade_rows = [
        [
            html.escape(t["symbol"]),
            html.escape(str(t["entry_time"])),
            html.escape(str(t["exit_time"]) if t["exit_time"] else ""),
            _fmt(t["pnl"]),
            _fmt(t["return_pct"]),
            html.escape(t["exit_reason"] or ""),
        ]
        for t in trades
    ]

    fold_rows = [
        [
            str(f["fold_number"]),
            f"{f['train_window_start']} to {f['train_window_end']}",
            f"{f['test_window_start']} to {f['test_window_end']}",
            "-" if f.get("p_value") is None else f"{f['p_value']:.4f}",
            "PASSED" if f.get("passed") else ("FAILED" if f.get("passed") is False else "insufficient sample"),
        ]
        for f in folds
    ]

    comparison_rows = [
        [
            str(c["run_id_a"]),
            str(c["run_id_b"]),
            html.escape(c.get("winner") or "-"),
            "yes" if c.get("promotion_recommended") else "no",
        ]
        for c in comparisons
    ]

    return f"""
    <section>
      <h2>Backtest run #{run_id}</h2>
      <p><strong>Symbol universe is a fixed, user-supplied list — not a
      historical turnover reconstruction (CoinDCX has no historical
      ticker/turnover series to replay). Results only speak to these exact
      symbols.</strong></p>
      <h3>Executive summary</h3>
      {_table(["Field", "Value"], summary_rows, "-")}
      <h3>Performance metrics</h3>
      {_table(["Metric", "Value"], metric_rows, "No metrics computed yet.")}
      <h3>Trade list</h3>
      {_table(["Symbol", "Entry", "Exit", "PnL", "Return %", "Exit reason"], trade_rows, "No trades.")}
      <h3>Walk-forward results</h3>
      {_table(["Fold", "Train window", "Test window", "p-value", "Result"], fold_rows, "No walk-forward run yet.")}
      <h3>Strategy comparisons</h3>
      {_table(["Run A", "Run B", "Winner", "Promotion recommended"], comparison_rows, "No comparisons yet.")}
    </section>
    """


def export_trades_csv(run_id: str) -> str:
    trades = models.get_backtest_trades(run_id)
    buf = io.StringIO()
    if trades:
        writer = csv.DictWriter(buf, fieldnames=list(trades[0].keys()))
        writer.writeheader()
        writer.writerows(trades)
    return buf.getvalue()


def export_run_json(run_id: str) -> str:
    run = models.get_backtest_run(run_id)
    trades = models.get_backtest_trades(run_id)
    perf = models.get_backtest_performance_metrics(run_id)
    folds = models.get_backtest_walk_forward_folds(run_id)
    return json.dumps(
        {
            "run": run,
            "trades": trades,
            "performance_metrics": perf["metrics"] if perf else None,
            "walk_forward_folds": folds,
        },
        default=str,
        indent=2,
    )
