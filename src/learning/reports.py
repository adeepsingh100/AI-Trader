"""Learning Insights section for reporting_agent.py's HTML report — best/
worst regimes, symbols, score ranges, most-profitable hour/weekday,
longest win/loss streak. Reuses reporting_agent's _table/_row helpers
(imported locally to avoid a circular import — reporting_agent.py
imports generate_learning_report_html at module level) rather than a
parallel rendering path; reporting_agent.py stays the single report
entry point, still manual-only. The learning_statistics data itself
refreshes automatically every night via evolution_agent, so whatever
this renders reflects current data whenever the report is actually run."""

from __future__ import annotations

import html
from datetime import datetime, timedelta, timezone

from src.config import LEARNING_HISTORY_WINDOW_DAYS
from src.db import models


def _sorted_stats(mode: str, dimension_type: str) -> list[dict]:
    rows = [r for r in models.get_learning_statistics(mode, dimension_type) if r.get("expectancy") is not None]
    return sorted(rows, key=lambda r: r["expectancy"], reverse=True)


def _streaks(trades: list[dict]) -> dict:
    ordered = sorted((t for t in trades if t.get("pnl") is not None), key=lambda t: t["closed_at"])
    longest_win = longest_loss = current_win = current_loss = 0
    for t in ordered:
        if t["pnl"] > 0:
            current_win, current_loss = current_win + 1, 0
        else:
            current_loss, current_win = current_loss + 1, 0
        longest_win = max(longest_win, current_win)
        longest_loss = max(longest_loss, current_loss)
    return {"longest_win_streak": longest_win, "longest_loss_streak": longest_loss}


def generate_learning_report_html(mode: str) -> str:
    from src.agents.reporting_agent import _table

    def _bucket_rows(rows: list[dict]) -> str:
        table_rows = [
            [
                html.escape(str(r["dimension_value"])),
                str(r["trades_count"]),
                f"{r['win_rate'] * 100:.1f}%" if r.get("win_rate") is not None else "-",
                f"{r['expectancy']:.2f}" if r.get("expectancy") is not None else "-",
                f"{r['sharpe_ratio']:.2f}" if r.get("sharpe_ratio") is not None else "-",
            ]
            for r in rows
        ]
        return _table(["Bucket", "Trades", "Win rate", "Expectancy", "Sharpe"], table_rows, "No data yet.")

    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    streaks = _streaks(models.get_recently_closed_trades(mode, since))

    return f"""
    <section>
      <h3>Market regimes</h3>
      {_bucket_rows(_sorted_stats(mode, "market_regime"))}
      <h3>Symbols</h3>
      {_bucket_rows(_sorted_stats(mode, "symbol"))}
      <h3>Opportunity score ranges</h3>
      {_bucket_rows(sorted(_sorted_stats(mode, "opportunity_score_bucket"), key=lambda r: r["dimension_value"]))}
      <h3>Hour of day (IST)</h3>
      {_bucket_rows(_sorted_stats(mode, "hour"))}
      <h3>Weekday</h3>
      {_bucket_rows(_sorted_stats(mode, "weekday"))}
      <p>Longest winning streak: {streaks['longest_win_streak']} trades |
         Longest losing streak: {streaks['longest_loss_streak']} trades</p>
    </section>
    """
