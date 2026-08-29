"""Evidence Engine (Evidence-Driven Learning Progression). Measures how
much evidence a mode has collected across every dimension orchestrator.py
already logs on every scanned symbol every cycle — not just closed
trades, but rejected candidates, symbols/regimes/hours seen, feature and
confidence coverage. Never changes a strategy, never writes anything;
purely a read-side measurement LearningStatus consumes to decide what's
unlocked (src/learning/learning_status.py).

Reuses 4 already-existing model queries — zero new DB functions. Real,
fixed denominators pulled from code, not guessed: 6 market regimes is
opportunity_scorer.classify_market_regime's exhaustive label set, 24 is
hours in a day. Feature coverage's denominator (len(FEATURE_KEYS)) is a
slight overcount — a few keys (volatility_regime, volume_spike,
obv_rising) are booleans/categoricals feature_importance.py never
correlates, so 100% coverage is never literally reachable; accepted as an
honest, simply-explained cap rather than importing a private exclusion
set from another module."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone

from src.agents.risk_manager import TRADING_DAY_TZ
from src.config import (
    EVIDENCE_REJECTED_COVERAGE_TARGET,
    EVIDENCE_SYMBOL_COVERAGE_TARGET,
    EVIDENCE_WEIGHT_CONFIDENCE_COVERAGE,
    EVIDENCE_WEIGHT_FEATURE_COVERAGE,
    EVIDENCE_WEIGHT_MARKET_COVERAGE,
    EVIDENCE_WEIGHT_REJECTION_EVIDENCE,
    EVIDENCE_WEIGHT_SESSION_COVERAGE,
    EVIDENCE_WEIGHT_SYMBOL_COVERAGE,
    EVIDENCE_WEIGHT_TRADE_COVERAGE,
    LEARNING_HISTORY_WINDOW_DAYS,
    LEARNING_STAGE_VALIDATION_MIN_TRADES,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
)
from src.db import models
from src.features.feature_engine import FEATURE_KEYS
from src.features.opportunity_scorer import weighted_average
from src.learning.statistics import _DIMENSION_TYPES
from src.utils import clamp, parse_timestamp

# classify_market_regime's exhaustive label set (opportunity_scorer.py) —
# the real total, not a guess, so "N of 6 regimes covered" is literal.
_ALL_MARKET_REGIMES = {"sideways", "high_volatility", "strong_bull", "weak_bull", "strong_bear", "weak_bear"}

_ACCEPTED_DECISIONS = ("buy", "sell")


def _ist_hour(timestamp: str) -> int:
    return parse_timestamp(timestamp).astimezone(TRADING_DAY_TZ).hour


class EvidenceEngine:
    """Never executes a trade, never modifies a strategy or config — a
    pure measurement pass over what's already been observed."""

    def collect(self, mode: str, strategy_type: str = "default") -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
        closed = [
            t for t in models.get_recently_closed_trades(mode, since, strategy_type) if t.get("pnl") is not None
        ]
        evaluations = models.get_opportunity_evaluations_for_trail(mode, since=since, strategy_type=strategy_type)
        feature_rows = models.get_feature_importance(mode, strategy_type=strategy_type)
        stats_rows = models.get_learning_statistics(mode, strategy_type=strategy_type)

        winning = sum(1 for t in closed if t["pnl"] > 0)
        exit_reasons_seen = {t["exit_reason"] for t in closed if t.get("exit_reason")}
        strategy_versions_seen = {t["version_id"] for t in closed if t.get("version_id") is not None}

        rejected = [e for e in evaluations if e.get("final_decision") == "hold"]
        accepted_symbols = {e["symbol"] for e in evaluations if e.get("final_decision") in _ACCEPTED_DECISIONS}
        accepted_regimes = {
            e["market_regime"]
            for e in evaluations
            if e.get("final_decision") in _ACCEPTED_DECISIONS and e.get("market_regime")
        }
        symbols_seen = {e["symbol"] for e in evaluations if e.get("symbol")}
        regimes_seen = {e["market_regime"] for e in evaluations if e.get("market_regime")}
        hours_seen = {_ist_hour(e["timestamp"]) for e in evaluations if e.get("timestamp")}
        reached_llm = sum(1 for e in evaluations if e.get("llm_decision") is not None)

        feature_keys_seen = {r["feature_name"] for r in feature_rows if r.get("timeframe") != "blended"}
        dimension_types_seen = {r["dimension_type"] for r in stats_rows}

        symbol_seen_counts: Counter = Counter(e["symbol"] for e in evaluations if e.get("symbol"))
        symbol_reject_counts: Counter = Counter(e["symbol"] for e in rejected if e.get("symbol"))
        symbols_rarely_qualifying = sorted(
            (
                {
                    "symbol": symbol,
                    "seen": count,
                    "reject_rate_pct": symbol_reject_counts.get(symbol, 0) / count * 100,
                }
                for symbol, count in symbol_seen_counts.items()
                if count >= RECOMMENDATION_MIN_SAMPLE_SIZE and symbol not in accepted_symbols
            ),
            key=lambda r: r["seen"],
            reverse=True,
        )[:10]
        regimes_with_no_candidates = sorted(regimes_seen - accepted_regimes)

        return {
            "closed_trades": len(closed),
            "winning_trades": winning,
            "losing_trades": len(closed) - winning,
            "exit_reasons_seen": len(exit_reasons_seen),
            "strategy_versions_seen": len(strategy_versions_seen),
            "rejected_opportunities": len(rejected),
            "candidate_opportunities": len(evaluations),
            "symbols_covered": len(symbols_seen),
            "market_regimes_covered": len(regimes_seen & _ALL_MARKET_REGIMES),
            "trading_hours_covered": len(hours_seen),
            "feature_coverage_pct": clamp(len(feature_keys_seen) / len(FEATURE_KEYS) * 100, 0, 100),
            "confidence_coverage_pct": clamp(reached_llm / len(evaluations) * 100, 0, 100) if evaluations else 0.0,
            "learning_coverage_pct": clamp(len(dimension_types_seen) / len(_DIMENSION_TYPES) * 100, 0, 100),
            "symbols_rarely_qualifying": symbols_rarely_qualifying,
            "regimes_with_no_candidates": regimes_with_no_candidates,
        }


def _trade_coverage_pct(closed_trades: int) -> float:
    return clamp(closed_trades / LEARNING_STAGE_VALIDATION_MIN_TRADES * 100, 0, 100)


def _market_coverage_pct(regimes_covered: int) -> float:
    return clamp(regimes_covered / len(_ALL_MARKET_REGIMES) * 100, 0, 100)


def _symbol_coverage_pct(symbols_covered: int) -> float:
    return clamp(symbols_covered / EVIDENCE_SYMBOL_COVERAGE_TARGET * 100, 0, 100)


def _session_coverage_pct(hours_covered: int) -> float:
    return clamp(hours_covered / 24 * 100, 0, 100)


def _rejection_evidence_pct(rejected_opportunities: int) -> float:
    return clamp(rejected_opportunities / EVIDENCE_REJECTED_COVERAGE_TARGET * 100, 0, 100)


def compute_evidence_readiness(evidence: dict) -> dict:
    """Reuses opportunity_scorer.weighted_average's renormalize-among-
    available blend — same primitive fitness.py already reuses for
    fitness score, no new blending logic. Returns
    {"evidence_readiness_pct", "components"}."""
    weights = {
        "trade_coverage": EVIDENCE_WEIGHT_TRADE_COVERAGE,
        "market_coverage": EVIDENCE_WEIGHT_MARKET_COVERAGE,
        "symbol_coverage": EVIDENCE_WEIGHT_SYMBOL_COVERAGE,
        "feature_coverage": EVIDENCE_WEIGHT_FEATURE_COVERAGE,
        "session_coverage": EVIDENCE_WEIGHT_SESSION_COVERAGE,
        "confidence_coverage": EVIDENCE_WEIGHT_CONFIDENCE_COVERAGE,
        "rejection_evidence": EVIDENCE_WEIGHT_REJECTION_EVIDENCE,
    }
    components = {
        "trade_coverage": _trade_coverage_pct(evidence["closed_trades"]),
        "market_coverage": _market_coverage_pct(evidence["market_regimes_covered"]),
        "symbol_coverage": _symbol_coverage_pct(evidence["symbols_covered"]),
        "feature_coverage": evidence["feature_coverage_pct"],
        "session_coverage": _session_coverage_pct(evidence["trading_hours_covered"]),
        "confidence_coverage": evidence["confidence_coverage_pct"],
        "rejection_evidence": _rejection_evidence_pct(evidence["rejected_opportunities"]),
    }
    return {"evidence_readiness_pct": weighted_average(components, weights) or 0.0, "components": components}
