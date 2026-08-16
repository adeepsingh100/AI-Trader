"""One HTML report covering both modes side by side: PnL vs target,
trade log (with LLM reasoning), strategy version + changelog, model
fallback stats. Plain string templating — no templating engine needed
for a single static report."""

from __future__ import annotations

import html

from src.agents.risk_manager import today_ist
from src.db import models
from src.learning.reports import generate_learning_report_html

MODES = ["paper", "real"]


def _model_usage_stats(events: list[dict]) -> list[dict]:
    by_model: dict[str, dict] = {}
    for e in events:
        stats = by_model.setdefault(
            e["model_used"], {"calls": 0, "successes": 0, "fallbacks": 0, "total_latency_ms": 0}
        )
        stats["calls"] += 1
        stats["total_latency_ms"] += e["latency_ms"]
        if e["success"]:
            stats["successes"] += 1
        if e["fallback_reason"]:
            stats["fallbacks"] += 1

    rows = [
        {
            "model": model,
            "calls": s["calls"],
            "success_rate": s["successes"] / s["calls"],
            "fallback_rate": s["fallbacks"] / s["calls"],
            "avg_latency_ms": s["total_latency_ms"] / s["calls"],
        }
        for model, s in by_model.items()
    ]
    return sorted(rows, key=lambda r: r["calls"], reverse=True)


def _mode_section(mode: str) -> dict:
    return {
        "mode": mode,
        "capital_config": models.get_capital_config(mode),
        "daily_pnl": models.get_daily_pnl(today_ist(), mode),
        "trades": models.get_recent_trades(mode, limit=50),
    }


def build_report_data() -> dict:
    return {
        "generated_at_ist": today_ist().isoformat(),
        "modes": [_mode_section(m) for m in MODES],
        "versions": models.get_all_strategy_versions(),
        "model_usage": _model_usage_stats(models.get_recent_model_usage()),
    }


def _row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _table(headers: list[str], rows: list[list[str]], empty_message: str) -> str:
    if not rows:
        return f"<p>{empty_message}</p>"
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(_row(r) for r in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _trade_log_html(trades: list[dict]) -> str:
    rows = [
        [
            html.escape(t["symbol"]),
            html.escape(t["side"]),
            f"{t['qty']:.6f}",
            f"{t['entry_price']:.2f}",
            f"{t['exit_price']:.2f}" if t["exit_price"] is not None else "-",
            f"{t['qty'] * t['entry_price']:.2f}",
            f"{t['qty'] * t['exit_price']:.2f}" if t["exit_price"] is not None else "-",
            f"{t['pnl']:.2f}" if t["pnl"] is not None else "-",
            html.escape(t["status"]),
            html.escape(t["opened_at"]),
            html.escape(t["reasoning_text"] or ""),
        ]
        for t in trades
    ]
    return _table(
        [
            "Symbol",
            "Side",
            "Qty",
            "Entry",
            "Exit",
            "Bought for",
            "Sold for",
            "PnL",
            "Status",
            "Opened",
            "Reasoning",
        ],
        rows,
        "No trades yet.",
    )


def _mode_section_html(section: dict) -> str:
    cfg = section["capital_config"]
    mode = section["mode"]
    if cfg is None:
        return f"<section><h2>{html.escape(mode.title())}</h2><p>No capital_config set.</p></section>"

    daily = section["daily_pnl"]
    realized = daily["realized_pnl"] if daily else 0
    target_hit = daily["target_hit"] if daily else False
    breaker = daily["circuit_breaker_triggered"] if daily else False

    return f"""
    <section>
      <h2>{html.escape(mode.title())}</h2>
      <p>Capital in use: {cfg['capital_to_use']:.2f} / {cfg['total_capital']:.2f}</p>
      <p>Today's PnL: {realized:.2f} (target {cfg['daily_profit_target']:.2f},
         {'HIT' if target_hit else 'not hit'})</p>
      <p>Circuit breaker: {'TRIGGERED' if breaker else 'clear'}</p>
      <h3>Trade log</h3>
      {_trade_log_html(section['trades'])}
      <h3>Learning insights</h3>
      {generate_learning_report_html(mode)}
    </section>
    """


def _versions_html(versions: list[dict]) -> str:
    rows = [
        [
            str(v["version_number"]),
            "real" if v["promoted_to_real"] else "paper-only",
            html.escape(v.get("notes") or ""),
            html.escape(v["created_at"]),
        ]
        for v in versions
    ]
    return _table(["Version", "Status", "Notes", "Created"], rows, "No strategy versions yet.")


def _model_usage_html(stats: list[dict]) -> str:
    rows = [
        [
            html.escape(s["model"]),
            str(s["calls"]),
            f"{s['success_rate'] * 100:.1f}%",
            f"{s['fallback_rate'] * 100:.1f}%",
            f"{s['avg_latency_ms']:.0f} ms",
        ]
        for s in stats
    ]
    return _table(
        ["Model", "Calls", "Success rate", "Fallback rate", "Avg latency"],
        rows,
        "No model usage recorded yet.",
    )


def render_html(data: dict) -> str:
    modes_html = "".join(_mode_section_html(s) for s in data["modes"])
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI-Trader report — {html.escape(data['generated_at_ist'])}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; font-size: 0.9rem; }}
section {{ margin-bottom: 2rem; }}
</style>
</head>
<body>
<h1>AI-Trader report — {html.escape(data['generated_at_ist'])}</h1>
{modes_html}
<section>
  <h2>Strategy versions</h2>
  {_versions_html(data['versions'])}
</section>
<section>
  <h2>Model health</h2>
  {_model_usage_html(data['model_usage'])}
</section>
</body>
</html>
"""


def generate_report() -> str:
    return render_html(build_report_data())


if __name__ == "__main__":
    print(generate_report())
