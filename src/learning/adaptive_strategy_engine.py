"""Step 1: AdaptiveStrategyEngine — the single composed entry point for
the whole Adaptive Strategy Intelligence Engine, and (Scientific Strategy
Optimization Framework) now the SOLE authoritative source of strategy
change candidates — evolution_agent.py's nightly LLM prompt/param rewrite
was retired entirely, not replaced by anything here; strategy evolution
runs exclusively through this statistically-rigorous, human-approved
pipeline now. An LLM is back in the loop as of generate_ai_exit_params_
recommendations below, but only as one more candidate-value source
alongside the pure-stat sweep — every candidate, AI-proposed or not,
still goes through the same walk-forward/bootstrap/fitness gate before it
can matter. This is unrelated to (and does not reintroduce) the retired
ungated nightly rewrite; see src/orchestrator.py's module docstring for
why live trading itself makes zero LLM calls either way.

Invariants (enforced by what this module imports, not just documented):
never executes a trade (no execution-agent import), never modifies
config.py or any trading table directly (only writes to
recommendations/strategy_simulations/adaptive_strategy_versions/
feature_importance, all advisory — see PROJECT_SPEC.md §3b for why
nothing here is auto-applied).

Runs as ITS OWN nightly step (python -m src.learning.adaptive_strategy_engine),
piggybacked on the same evolution.yml cron evolution_agent.py already
uses, but as an independent workflow step. That means nothing here needs
the local-import-inside-a-function circular-import workaround
evolution_agent.py uses for compute_metrics's callers — this module simply
never imports evolution_agent.py, directly or indirectly through anything
it itself imports at module level, so the dependency graph stays a DAG.

Scope note: only the mode-wide weight recommendation, the mode-wide
MIN_OPPORTUNITY_SCORE threshold recommendation, and the exit-params
(stop_loss_pct/take_profit_pct) recommendations get walk-forward simulated
here. Regime- and symbol-scoped recommendations (generate_regime_recommendations,
generate_symbol_recommendations) stay recommend-only in this phase — each
already needs RECOMMENDATION_MIN_SAMPLE_SIZE trades in its own bucket just
to be generated once; simulating them too would need that count to
roughly double again per bucket, on top of already being the least likely
generators to fire early. Revisit once real trade volume makes it worth
the complexity — they're still fully visible in the recommendations table
either way, just without a simulated adaptive_strategy_versions candidate."""

from __future__ import annotations

from src.config import FEATURE_TIMEFRAMES
from src.db import models
from src.learning.feature_importance import compute_feature_importance
from src.learning.recommendations import (
    generate_ai_exit_params_recommendations,
    generate_exit_params_recommendations,
    generate_recommendations,
    generate_regime_recommendations,
    generate_symbol_recommendations,
    generate_weight_recommendations,
)
from src.learning.learning_status import compute_learning_status
from src.learning.rejection_analysis import rejection_breakdown
from src.learning.simulation import (
    simulate_exit_params_recommendation,
    simulate_threshold_recommendation,
    simulate_weight_recommendation,
)
from src.learning.weakness_detection import identify_weaknesses


def _build_symbol_to_pair(mode: str) -> dict[str, str] | None:
    """Best-effort mapping for simulate_exit_params_recommendation's
    optional backtest-replay validation — the one thing in this otherwise
    pure-statistics module that touches the network. Fails open (None) on
    any error (CoinDCX outage, unknown symbol); the caller degrades to its
    always-available re-partition + bootstrap checks, never a crash."""
    from datetime import datetime, timedelta, timezone

    from src.config import LEARNING_HISTORY_WINDOW_DAYS

    try:
        since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
        trades = models.get_recently_closed_trades(mode, since)
        symbols = sorted({t["symbol"] for t in trades if t.get("symbol")})
        if not symbols:
            return None
        from src.coindcx_client import get_markets_details
        from src.coindcx_client import symbol_to_pair as _symbol_to_pair

        details = get_markets_details()
        return {s: _symbol_to_pair(s, details) for s in symbols}
    except Exception:
        return None


class AdaptiveStrategyEngine:
    """Never executes trades. Never modifies config.py or any trading
    table directly — advisory-only, human-approved. See module docstring
    for the full invariant statement."""

    def analyze(self, mode: str = "paper") -> dict:
        # Computed once, threaded into every generator/simulator below —
        # each accepts status=None and would compute its own otherwise,
        # but that would mean 8 redundant EvidenceEngine passes per run.
        status = compute_learning_status(mode)
        weaknesses = identify_weaknesses(mode)
        rejections = rejection_breakdown(mode)

        weight_recs = generate_weight_recommendations(mode, status=status)
        regime_recs = generate_regime_recommendations(mode, status=status)
        symbol_recs = generate_symbol_recommendations(mode, status=status)
        threshold_recs = generate_recommendations(mode, weakness_context=weaknesses, status=status)
        exit_params_recs = generate_exit_params_recommendations(mode, status=status)
        ai_exit_params_recs = generate_ai_exit_params_recommendations(mode, status=status)
        timeframe_importance = compute_feature_importance(mode, timeframes=FEATURE_TIMEFRAMES)

        simulations = []
        if weight_recs:
            batch_id = weight_recs[0].get("batch_id")
            weight_simulation = simulate_weight_recommendation(mode, batch_id, status=status)
            if weight_simulation is not None:
                simulations.append(weight_simulation)
        if threshold_recs:
            threshold_simulation = simulate_threshold_recommendation(mode, status=status)
            if threshold_simulation is not None:
                simulations.append(threshold_simulation)
        if exit_params_recs or ai_exit_params_recs:
            symbol_to_pair = _build_symbol_to_pair(mode)
            simulations.extend(
                simulate_exit_params_recommendation(mode, symbol_to_pair=symbol_to_pair, status=status)
            )

        candidates_created = sum(1 for s in simulations if s.get("passed"))

        models.log_agent_event(
            "adaptive_strategy_engine",
            "info",
            f"stage={status.stage} trades_collected={status.trades_collected} "
            f"evidence_readiness={status.evidence_readiness_pct:.0f}% "
            f"weight_recs={len(weight_recs)} regime_recs={len(regime_recs)} "
            f"symbol_recs={len(symbol_recs)} threshold_recs={len(threshold_recs)} "
            f"exit_params_recs={len(exit_params_recs)} ai_exit_params_recs={len(ai_exit_params_recs)} "
            f"timeframe_feature_rows={len(timeframe_importance)} simulations={len(simulations)} "
            f"candidates_created={candidates_created} rejection_reasons={len(rejections)}",
        )

        return {
            "learning_status": status,
            "weaknesses": weaknesses,
            "rejection_breakdown": rejections,
            "weight_recommendations": weight_recs,
            "regime_recommendations": regime_recs,
            "symbol_recommendations": symbol_recs,
            "threshold_recommendations": threshold_recs,
            "exit_params_recommendations": exit_params_recs,
            "ai_exit_params_recommendations": ai_exit_params_recs,
            "timeframe_feature_importance": timeframe_importance,
            "simulations": simulations,
            "candidates_created": candidates_created,
        }


if __name__ == "__main__":
    AdaptiveStrategyEngine().analyze()
