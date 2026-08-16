"""Blends the LLM's own stated confidence with historical win-rate on
similar past trades — see src/learning/trade_memory.py for where the
historical figure comes from — then applies an adaptive modifier chain:
regime -> symbol -> recent-performance. Pure function throughout, no DB
access; callers (orchestrator.py) fetch whatever the modifiers need and
pass in already-computed numbers.

Collapses to AI-only when there's no historical figure (thin history) or
too few similar trades to trust it — never fabricates a historical
confidence from a handful of matches. Each modifier independently
defaults to contributing nothing when its input is None, so this stays
fully backward compatible with the base AI+historical blend alone.

Automatic and live, unlike the rest of the Adaptive Strategy Engine
(weight/threshold/regime/symbol recommendations, which stay advisory and
human-approved) — this chain is an extension of the ALREADY-automatic
calibrate_confidence gate, which has run every cycle with no approval
step since it was introduced, and stays inert in practice because
MIN_FINAL_CONFIDENCE defaults to 0."""

from __future__ import annotations

from src.config import CONFIDENCE_AI_WEIGHT, CONFIDENCE_HISTORICAL_WEIGHT, MIN_SIMILAR_TRADES


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def calibrate_confidence(
    ai_confidence: float | None,
    historical_confidence: float | None,
    sample_size: int,
    regime_modifier: float | None = None,
    symbol_modifier: float | None = None,
    recent_performance_modifier: float | None = None,
) -> dict:
    if historical_confidence is None or sample_size < MIN_SIMILAR_TRADES:
        base_confidence = ai_confidence
        ai_weight_used = 1.0 if ai_confidence is not None else 0.0
        historical_weight_used = 0.0
    elif ai_confidence is None:
        base_confidence = historical_confidence
        ai_weight_used = 0.0
        historical_weight_used = 1.0
    else:
        total_weight = CONFIDENCE_AI_WEIGHT + CONFIDENCE_HISTORICAL_WEIGHT
        ai_weight_used = CONFIDENCE_AI_WEIGHT / total_weight if total_weight else 0.5
        historical_weight_used = 1.0 - ai_weight_used
        base_confidence = ai_confidence * ai_weight_used + historical_confidence * historical_weight_used

    final_confidence = base_confidence
    if final_confidence is not None:
        modifier_total = sum(
            m for m in (regime_modifier, symbol_modifier, recent_performance_modifier) if m is not None
        )
        final_confidence = _clamp(final_confidence + modifier_total, 0, 100)

    return {
        "final_confidence": final_confidence,
        "ai_weight_used": ai_weight_used,
        "historical_weight_used": historical_weight_used,
        "regime_modifier": regime_modifier,
        "symbol_modifier": symbol_modifier,
        "recent_performance_modifier": recent_performance_modifier,
    }
