"""Step 1: AdaptiveStrategyEngine — the single composed entry point for
the whole Adaptive Strategy Intelligence Engine. Analyzes learning
statistics, generates strategy improvements, recommends parameter
changes, and estimates expected improvement via walk-forward simulation.

Invariants (enforced by what this module imports, not just documented):
never executes a trade (no execution-agent import), never modifies
config.py or any trading table directly (only writes to
recommendations/strategy_simulations/adaptive_strategy_versions/
feature_importance, all advisory — see PROJECT_SPEC.md §3b for why
nothing here is auto-applied).

Runs as ITS OWN nightly step (python -m src.learning.adaptive_strategy_engine),
piggybacked on the same evolution.yml cron evolution_agent.py already
uses, but as an independent workflow step — NOT called from
run_evolution() itself. That keeps evolution_agent.py's existing
generate_recommendations()/compute_feature_importance() calls completely
untouched (removing them would be a real regression against "don't
remove existing functionality"), and it means nothing here needs the
local-import-inside-a-function circular-import workaround evolution_agent.py
uses for those two calls — this module simply never imports
evolution_agent.py, directly or indirectly through anything it itself
imports at module level, so the dependency graph stays a DAG.

Scope note: only the mode-wide weight recommendation and the mode-wide
MIN_OPPORTUNITY_SCORE threshold recommendation get walk-forward simulated
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
    generate_recommendations,
    generate_regime_recommendations,
    generate_symbol_recommendations,
    generate_weight_recommendations,
)
from src.learning.simulation import simulate_threshold_recommendation, simulate_weight_recommendation


class AdaptiveStrategyEngine:
    """Never executes trades. Never modifies config.py or any trading
    table directly — advisory-only, human-approved. See module docstring
    for the full invariant statement."""

    def analyze(self, mode: str = "paper") -> dict:
        weight_recs = generate_weight_recommendations(mode)
        regime_recs = generate_regime_recommendations(mode)
        symbol_recs = generate_symbol_recommendations(mode)
        threshold_recs = generate_recommendations(mode)
        timeframe_importance = compute_feature_importance(mode, timeframes=FEATURE_TIMEFRAMES)

        simulations = []
        if weight_recs:
            batch_id = weight_recs[0].get("batch_id")
            weight_simulation = simulate_weight_recommendation(mode, batch_id)
            if weight_simulation is not None:
                simulations.append(weight_simulation)
        if threshold_recs:
            threshold_simulation = simulate_threshold_recommendation(mode)
            if threshold_simulation is not None:
                simulations.append(threshold_simulation)

        candidates_created = sum(1 for s in simulations if s.get("passed"))

        models.log_agent_event(
            "adaptive_strategy_engine",
            "info",
            f"weight_recs={len(weight_recs)} regime_recs={len(regime_recs)} "
            f"symbol_recs={len(symbol_recs)} threshold_recs={len(threshold_recs)} "
            f"timeframe_feature_rows={len(timeframe_importance)} simulations={len(simulations)} "
            f"candidates_created={candidates_created}",
        )

        return {
            "weight_recommendations": weight_recs,
            "regime_recommendations": regime_recs,
            "symbol_recommendations": symbol_recs,
            "threshold_recommendations": threshold_recs,
            "timeframe_feature_importance": timeframe_importance,
            "simulations": simulations,
            "candidates_created": candidates_created,
        }


if __name__ == "__main__":
    AdaptiveStrategyEngine().analyze()
