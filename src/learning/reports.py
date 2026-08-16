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
from src.learning.statistics import streaks as _compute_streaks


def _sorted_stats(mode: str, dimension_type: str) -> list[dict]:
    rows = [r for r in models.get_learning_statistics(mode, dimension_type) if r.get("expectancy") is not None]
    return sorted(rows, key=lambda r: r["expectancy"], reverse=True)


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
    streak_data = _compute_streaks(models.get_recently_closed_trades(mode, since))

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
      <p>Longest winning streak: {streak_data['longest_win_streak']} trades |
         Longest losing streak: {streak_data['longest_loss_streak']} trades</p>
    </section>
    """


def generate_adaptive_strategy_report_html(mode: str) -> str:
    """Step 15: best/rejected/accepted recommendations, simulation
    results, current candidate/approved adaptive strategy versions,
    recommendation confidence. Same reuse pattern as
    generate_learning_report_html above — local import of the shared
    _table helper, still rendered from reporting_agent.py only."""
    from src.agents.reporting_agent import _table

    def _recommendation_rows(rows: list[dict]) -> str:
        table_rows = [
            [
                html.escape(r["category"]),
                html.escape(r["metric_name"]),
                "-" if r.get("current_value") is None else f"{r['current_value']:.3f}",
                "-" if r.get("recommended_value") is None else f"{r['recommended_value']:.3f}",
                "-" if r.get("confidence") is None else f"{r['confidence']:.1f}%",
                str(r.get("sample_size") or 0),
                html.escape(r.get("rationale") or ""),
            ]
            for r in rows
        ]
        return _table(
            ["Category", "Metric", "Current", "Recommended", "Confidence", "Samples", "Rationale"],
            table_rows,
            "No recommendations yet.",
        )

    def _simulation_rows(rows: list[dict]) -> str:
        table_rows = [
            [
                html.escape(str(r["created_at"])),
                "PASSED" if r.get("passed") else "rejected",
                "-" if r.get("p_value") is None else f"{r['p_value']:.4f}",
            ]
            for r in rows
        ]
        return _table(["Simulated", "Result", "p-value"], table_rows, "No simulations run yet.")

    def _version_rows(rows: list[dict]) -> str:
        table_rows = [
            [str(r["version_number"]), html.escape(r["status"]), html.escape(str(r["created_at"]))]
            for r in rows
        ]
        return _table(["Version", "Status", "Created"], table_rows, "No adaptive strategy candidates yet.")

    all_recs = models.get_recommendations(mode)
    pending = [r for r in all_recs if r.get("status") == "pending"]
    accepted = [r for r in all_recs if r.get("status") == "approved"]
    rejected = [r for r in all_recs if r.get("status") == "dismissed"]
    best = sorted(pending, key=lambda r: r.get("confidence") or 0, reverse=True)[:10]

    simulations = models.get_strategy_simulations(mode)
    versions = models.get_adaptive_strategy_versions(mode)

    return f"""
    <section>
      <h3>Best pending recommendations</h3>
      {_recommendation_rows(best)}
      <h3>Accepted recommendations</h3>
      {_recommendation_rows(accepted)}
      <h3>Rejected recommendations</h3>
      {_recommendation_rows(rejected)}
      <h3>Simulation results</h3>
      {_simulation_rows(simulations)}
      <h3>Adaptive strategy versions (candidate/approved — never auto-applied)</h3>
      {_version_rows(versions)}
    </section>
    """
