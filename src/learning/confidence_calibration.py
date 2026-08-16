"""Blends the LLM's own stated confidence with historical win-rate on
similar past trades — see src/learning/trade_memory.py for where the
historical figure comes from. Pure function, no DB access.

Collapses to AI-only when there's no historical figure (thin history) or
too few similar trades to trust it — never fabricates a historical
confidence from a handful of matches."""

from __future__ import annotations

from src.config import CONFIDENCE_AI_WEIGHT, CONFIDENCE_HISTORICAL_WEIGHT, MIN_SIMILAR_TRADES


def calibrate_confidence(
    ai_confidence: float | None, historical_confidence: float | None, sample_size: int
) -> dict:
    if historical_confidence is None or sample_size < MIN_SIMILAR_TRADES:
        return {
            "final_confidence": ai_confidence,
            "ai_weight_used": 1.0 if ai_confidence is not None else 0.0,
            "historical_weight_used": 0.0,
        }

    if ai_confidence is None:
        return {
            "final_confidence": historical_confidence,
            "ai_weight_used": 0.0,
            "historical_weight_used": 1.0,
        }

    total_weight = CONFIDENCE_AI_WEIGHT + CONFIDENCE_HISTORICAL_WEIGHT
    ai_weight = CONFIDENCE_AI_WEIGHT / total_weight if total_weight else 0.5
    historical_weight = 1.0 - ai_weight
    return {
        "final_confidence": ai_confidence * ai_weight + historical_confidence * historical_weight,
        "ai_weight_used": ai_weight,
        "historical_weight_used": historical_weight,
    }
