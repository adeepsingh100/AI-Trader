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
from src.learning.learning_status import LearningStatus, compute_learning_status
from src.learning.rejection_analysis import rejection_breakdown
from src.learning.statistics import streaks as _compute_streaks
from src.learning.weakness_detection import identify_weaknesses


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
      <h3>RSI ranges</h3>
      {_bucket_rows(_sorted_stats(mode, "rsi_bucket"))}
      <h3>Stochastic RSI ranges</h3>
      {_bucket_rows(_sorted_stats(mode, "stoch_rsi_bucket"))}
      <h3>Volatility (ATR%) ranges</h3>
      {_bucket_rows(_sorted_stats(mode, "atr_volatility_bucket"))}
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

    def _simulation_rows(rows: list[dict], fitness_by_simulation_id: dict) -> str:
        table_rows = [
            [
                html.escape(str(r["created_at"])),
                "PASSED" if r.get("passed") else "rejected",
                "-" if r.get("p_value") is None else f"{r['p_value']:.4f}",
                "-" if fitness_by_simulation_id.get(r["id"]) is None else f"{fitness_by_simulation_id[r['id']]:.1f}",
                html.escape((r.get("research_note") or "").replace("\n", " | ")),
            ]
            for r in rows
        ]
        return _table(
            ["Simulated", "Result", "p-value", "Fitness", "Research note"], table_rows, "No simulations run yet."
        )

    def _version_rows(rows: list[dict]) -> str:
        table_rows = [
            [
                str(r["version_number"]),
                html.escape(r["status"]),
                "-" if r.get("fitness_score") is None else f"{r['fitness_score']:.1f}",
                html.escape(str(r["created_at"])),
            ]
            for r in rows
        ]
        return _table(["Version", "Status", "Fitness", "Created"], table_rows, "No adaptive strategy candidates yet.")

    def _weakness_rows(weaknesses: dict) -> str:
        table_rows = [
            [html.escape(dimension_type), html.escape(str(bucket["value"])), f"{bucket['expectancy']:.2f}", str(bucket["trades_count"])]
            for dimension_type, bucket in weaknesses.get("worst_by_dimension", {}).items()
        ]
        return _table(["Dimension", "Worst bucket", "Expectancy", "Trades"], table_rows, "Not enough data yet.")

    def _rejection_rows(rows: list[dict]) -> str:
        table_rows = [
            [html.escape(str(r["reason"])), str(r["count"]), f"{r['pct_of_rejections']:.1f}%"]
            for r in rows
        ]
        return _table(["Rejection reason", "Count", "% of rejections"], table_rows, "No rejected candidates logged yet.")

    def _learning_status_rows(status: LearningStatus) -> str:
        table_rows = [
            ["Stage", html.escape(status.stage)],
            ["Evidence readiness", f"{status.evidence_readiness_pct:.1f}%"],
            ["Trades collected", str(status.trades_collected)],
            ["Winning / losing", f"{status.winning_trades} / {status.losing_trades}"],
            ["Rejected trades", str(status.rejected_trades)],
            ["Data sufficiency", f"{status.data_sufficiency_pct:.1f}%"],
            [
                "Recommendations / simulations / candidates",
                f"{status.recommendations_count} / {status.simulations_count} / {status.candidates_count}",
            ],
            ["Promotion eligible", "yes" if status.promotion_eligible else "no"],
            ["Current activity", html.escape(status.current_activity)],
            ["Next stage", html.escape(status.next_stage) if status.next_stage else "-"],
            ["Reason", html.escape(status.reason)],
        ]
        return _table(["Field", "Value"], table_rows, "Learning status unavailable.")

    def _evidence_rows(evidence: dict) -> str:
        table_rows = [
            ["Symbols covered", str(evidence["symbols_covered"])],
            ["Market regimes covered", f"{evidence['market_regimes_covered']} / 6"],
            ["Trading hours covered", f"{evidence['trading_hours_covered']} / 24"],
            ["Feature coverage", f"{evidence['feature_coverage_pct']:.1f}%"],
            ["Confidence coverage", f"{evidence['confidence_coverage_pct']:.1f}%"],
            ["Candidate opportunities scanned", str(evidence["candidate_opportunities"])],
            [
                "Symbols rarely qualifying",
                html.escape(", ".join(r["symbol"] for r in evidence["symbols_rarely_qualifying"][:5]) or "-"),
            ],
            [
                "Regimes with no candidates",
                html.escape(", ".join(evidence["regimes_with_no_candidates"]) or "-"),
            ],
        ]
        return _table(["Coverage dimension", "Value"], table_rows, "No evidence collected yet.")

    all_recs = models.get_recommendations(mode)
    pending = [r for r in all_recs if r.get("status") == "pending"]
    accepted = [r for r in all_recs if r.get("status") == "approved"]
    rejected = [r for r in all_recs if r.get("status") == "dismissed"]
    best = sorted(pending, key=lambda r: r.get("confidence") or 0, reverse=True)[:10]

    simulations = models.get_strategy_simulations(mode)
    versions = models.get_adaptive_strategy_versions(mode)
    fitness_by_simulation_id = {
        v["source_simulation_id"]: v.get("fitness_score") for v in versions if v.get("source_simulation_id")
    }
    weaknesses = identify_weaknesses(mode)
    rejections = rejection_breakdown(mode)
    learning_status = compute_learning_status(mode)

    return f"""
    <section>
      <h3>Learning status (Evidence-Driven Learning Progression)</h3>
      {_learning_status_rows(learning_status)}
      <h3>Evidence coverage breakdown</h3>
      {_evidence_rows(learning_status.evidence)}
      <h3>Weaknesses found</h3>
      {_weakness_rows(weaknesses)}
      <h3>Rejection breakdown (root cause of "no trade")</h3>
      {_rejection_rows(rejections)}
      <h3>Best pending recommendations</h3>
      {_recommendation_rows(best)}
      <h3>Accepted recommendations</h3>
      {_recommendation_rows(accepted)}
      <h3>Rejected recommendations</h3>
      {_recommendation_rows(rejected)}
      <h3>Simulation results</h3>
      {_simulation_rows(simulations, fitness_by_simulation_id)}
      <h3>Adaptive strategy versions (candidate/approved — never auto-applied)</h3>
      {_version_rows(versions)}
    </section>
    """
