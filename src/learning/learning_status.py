"""Evidence-Driven Learning Progression. LearningStatus is the single
authority every gate asks instead of comparing trade counts inline —
compute_learning_status(mode) is still the one entry point, consumed by
evolution_agent.py's and adaptive_strategy_engine.py's reports,
reports.py's HTML, and mirrored by the dashboard.

Deliberately depends on nothing that imports evolution_agent.py
(statistics.py and fitness.py both do, for compute_metrics) — evidence_engine.py
only imports models/config/pure feature+scorer helpers, so this module is
safely importable at module level from both engines with no circular-
import workaround needed.

Load-bearing design decision: evidence-readiness substitutes for trade
count only where the underlying capability doesn't need trade OUTCOMES.
Rejection analysis and coverage reporting don't need a single closed
trade, so BOOTSTRAP->OBSERVATION is fully evidence-driven (any ONE of
several dimensions clearing its bar unlocks it — see _stage_for).
Hypothesis generation (z-test on win/loss separation), simulation (train/
test split of outcomes), and candidate validation (bootstrap CI on trade
PnLs) are statistically irreducible — no amount of symbol/regime coverage
makes a win/loss z-test valid on a sample it wasn't valid on before, so
OBSERVATION->HYPOTHESIS->SIMULATION->VALIDATION stay gated on
LEARNING_STAGE_*_MIN_TRADES exactly as before, unchanged numbers,
unchanged rigor — just now expressed through the can_*() methods below
instead of scattered inline comparisons, so "single authority" holds for
all five stages even though only the first transition's logic changed."""

from __future__ import annotations

from dataclasses import dataclass

from src.config import (
    EVIDENCE_FEATURE_COVERAGE_OBSERVATION_MIN_PCT,
    EVIDENCE_HOURS_OBSERVATION_MIN_TRADES,
    EVIDENCE_READINESS_OBSERVATION_MIN_PCT,
    EVIDENCE_REGIMES_OBSERVATION_MIN,
    EVIDENCE_REJECTED_COVERAGE_TARGET,
    EVIDENCE_SYMBOL_COVERAGE_TARGET,
    LEARNING_STAGE_HYPOTHESIS_MIN_TRADES,
    LEARNING_STAGE_OBSERVATION_MIN_TRADES,
    LEARNING_STAGE_SIMULATION_MIN_TRADES,
    LEARNING_STAGE_VALIDATION_MIN_TRADES,
)
from src.db import models
from src.learning.evidence_engine import EvidenceEngine, compute_evidence_readiness

_ACTIVITY = {
    "BOOTSTRAP": "Collecting trade data, rejection reasons, and feature distributions only. No analysis yet.",
    "OBSERVATION": "Analyzing rejection reasons, feature distributions, and weakness patterns. No strategy changes yet.",
    "HYPOTHESIS": "Generating hypotheses (weight/threshold/exit-parameter recommendations) from observed weaknesses. No candidate strategies yet.",
    "SIMULATION": "Testing hypotheses via backtest and walk-forward simulation. Candidates are validated but not yet created.",
    "VALIDATION": "Full validation active — passing simulations create candidate strategies, pending human approval for promotion.",
}


@dataclass
class LearningStatus:
    stage: str
    trades_collected: int
    rejected_trades: int
    winning_trades: int
    losing_trades: int
    evidence: dict
    evidence_readiness_pct: float
    data_sufficiency_pct: float
    recommendations_count: int
    simulations_count: int
    candidates_count: int
    promotion_eligible: bool
    next_stage: str | None
    trades_to_next_stage: int
    evidence_gaps: list[str]
    current_activity: str
    reason: str

    def can_generate_hypotheses(self) -> bool:
        return self.trades_collected >= LEARNING_STAGE_HYPOTHESIS_MIN_TRADES

    def can_simulate(self) -> bool:
        return self.trades_collected >= LEARNING_STAGE_SIMULATION_MIN_TRADES

    def can_validate(self) -> bool:
        return self.trades_collected >= LEARNING_STAGE_VALIDATION_MIN_TRADES

    def can_create_candidate(self) -> bool:
        # Candidate rows are validation's OUTPUT, not a separately-
        # thresholded gate — one number behind both names, not two.
        return self.can_validate()

    def can_promote(self) -> bool:
        # Reads the fact evolution_agent.promotion_eligible() already
        # computed (paper-days + PnL + drawdown + bootstrap CI + fitness)
        # — never recomputes promotion logic here.
        return self.promotion_eligible


def _observation_ready(trades_collected: int, evidence: dict, evidence_readiness_pct: float) -> bool:
    return (
        trades_collected >= LEARNING_STAGE_OBSERVATION_MIN_TRADES
        or evidence["rejected_opportunities"] >= EVIDENCE_REJECTED_COVERAGE_TARGET
        or evidence["trading_hours_covered"] >= EVIDENCE_HOURS_OBSERVATION_MIN_TRADES
        or evidence["symbols_covered"] >= EVIDENCE_SYMBOL_COVERAGE_TARGET
        or evidence["market_regimes_covered"] >= EVIDENCE_REGIMES_OBSERVATION_MIN
        or evidence["feature_coverage_pct"] >= EVIDENCE_FEATURE_COVERAGE_OBSERVATION_MIN_PCT
        or evidence_readiness_pct >= EVIDENCE_READINESS_OBSERVATION_MIN_PCT
    )


def _stage_for(trades_collected: int, evidence: dict, evidence_readiness_pct: float) -> tuple[str, str | None, int | None]:
    """(stage, next_stage, next_stage_min_trades). next_stage_min_trades
    is None for the BOOTSTRAP->OBSERVATION edge (multiple OR-branches,
    each in its own unit — see _bootstrap_gaps) and for VALIDATION (no
    next stage)."""
    if not _observation_ready(trades_collected, evidence, evidence_readiness_pct):
        return "BOOTSTRAP", "OBSERVATION", None
    if trades_collected < LEARNING_STAGE_HYPOTHESIS_MIN_TRADES:
        return "OBSERVATION", "HYPOTHESIS", LEARNING_STAGE_HYPOTHESIS_MIN_TRADES
    if trades_collected < LEARNING_STAGE_SIMULATION_MIN_TRADES:
        return "HYPOTHESIS", "SIMULATION", LEARNING_STAGE_SIMULATION_MIN_TRADES
    if trades_collected < LEARNING_STAGE_VALIDATION_MIN_TRADES:
        return "SIMULATION", "VALIDATION", LEARNING_STAGE_VALIDATION_MIN_TRADES
    return "VALIDATION", None, None


def _bootstrap_gaps(trades_collected: int, evidence: dict, evidence_readiness_pct: float) -> list[str]:
    """Every unmet OR-branch, human-readable — the literal "Need 2
    additional market regimes OR 20 more closed trades" example. Only
    called when _observation_ready() is False, so every branch here is
    genuinely still open."""
    gaps = []
    trades_gap = LEARNING_STAGE_OBSERVATION_MIN_TRADES - trades_collected
    if trades_gap > 0:
        gaps.append(f"{trades_gap} more closed trades")
    rejected_gap = EVIDENCE_REJECTED_COVERAGE_TARGET - evidence["rejected_opportunities"]
    if rejected_gap > 0:
        gaps.append(f"{rejected_gap} more rejected candidates")
    hours_gap = EVIDENCE_HOURS_OBSERVATION_MIN_TRADES - evidence["trading_hours_covered"]
    if hours_gap > 0:
        gaps.append(f"{hours_gap} more trading hours covered")
    symbols_gap = EVIDENCE_SYMBOL_COVERAGE_TARGET - evidence["symbols_covered"]
    if symbols_gap > 0:
        gaps.append(f"{symbols_gap} more symbols covered")
    regimes_gap = EVIDENCE_REGIMES_OBSERVATION_MIN - evidence["market_regimes_covered"]
    if regimes_gap > 0:
        gaps.append(f"{regimes_gap} more market regimes covered")
    feature_gap = EVIDENCE_FEATURE_COVERAGE_OBSERVATION_MIN_PCT - evidence["feature_coverage_pct"]
    if feature_gap > 0:
        gaps.append(f"{feature_gap:.0f}% more feature coverage")
    readiness_gap = EVIDENCE_READINESS_OBSERVATION_MIN_PCT - evidence_readiness_pct
    if readiness_gap > 0:
        gaps.append(f"{readiness_gap:.0f}% more evidence readiness")
    return gaps


def _reason_for(next_stage: str | None, evidence_gaps: list[str]) -> str:
    if next_stage is None:
        return "Full validation stage reached."
    return f"Need {' OR '.join(evidence_gaps)} to reach {next_stage}."


def compute_learning_status(mode: str) -> LearningStatus:
    evidence = EvidenceEngine().collect(mode)
    readiness = compute_evidence_readiness(evidence)
    evidence_readiness_pct = readiness["evidence_readiness_pct"]
    trades_collected = evidence["closed_trades"]

    stage, next_stage, next_min = _stage_for(trades_collected, evidence, evidence_readiness_pct)

    if stage == "BOOTSTRAP":
        evidence_gaps = _bootstrap_gaps(trades_collected, evidence, evidence_readiness_pct)
        trades_to_next_stage = max(0, LEARNING_STAGE_OBSERVATION_MIN_TRADES - trades_collected)
    elif next_min is not None:
        trades_to_next_stage = max(0, next_min - trades_collected)
        evidence_gaps = [f"{trades_to_next_stage} more closed trades"]
    else:
        trades_to_next_stage = 0
        evidence_gaps = []

    version = models.get_latest_version()
    promotion_eligible = bool(version and version.get("promotion_eligible"))

    return LearningStatus(
        stage=stage,
        trades_collected=trades_collected,
        rejected_trades=evidence["rejected_opportunities"],
        winning_trades=evidence["winning_trades"],
        losing_trades=evidence["losing_trades"],
        evidence=evidence,
        evidence_readiness_pct=evidence_readiness_pct,
        data_sufficiency_pct=readiness["components"]["trade_coverage"],
        recommendations_count=len(models.get_recommendations(mode)),
        simulations_count=len(models.get_strategy_simulations(mode)),
        candidates_count=len(models.get_adaptive_strategy_versions(mode)),
        promotion_eligible=promotion_eligible,
        next_stage=next_stage,
        trades_to_next_stage=trades_to_next_stage,
        evidence_gaps=evidence_gaps,
        current_activity=_ACTIVITY[stage],
        reason=_reason_for(next_stage, evidence_gaps),
    )
