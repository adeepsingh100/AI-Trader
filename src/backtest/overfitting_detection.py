"""Step 12: overfitting detection, composed from what walk_forward_validator
and statistical_validation already computed — no new statistical machinery
here, just aggregation + verdict. "Reject weak strategies automatically"
means automatic STATUS marking only (see strategy_comparison.py's
promotion_recommended) — never automatic deletion or live application,
matching this session's established human-approval precedent throughout."""

from __future__ import annotations

from dataclasses import dataclass

from src.backtest.walk_forward_validator import Fold


@dataclass
class OverfittingReport:
    n_folds: int
    n_passed: int
    walk_forward_failure_rate: float
    in_sample_out_of_sample_gap_pct: float | None
    parameter_sensitivity: dict | None
    verdict: str  # "robust" | "marginal" | "overfit"


def _avg(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def detect(folds: list[Fold], parameter_sensitivity: dict | None = None) -> OverfittingReport:
    n_folds = len(folds)
    passed = [f for f in folds if f.passed]
    n_passed = len(passed)
    failure_rate = (1 - n_passed / n_folds) * 100 if n_folds else 100.0

    in_sample_expectancy = _avg([f.in_sample_metrics.get("expectancy") for f in folds if f.in_sample_metrics])
    out_of_sample_expectancy = _avg(
        [f.out_of_sample_metrics.get("expectancy") for f in folds if f.out_of_sample_metrics]
    )
    gap_pct = None
    if in_sample_expectancy and out_of_sample_expectancy is not None:
        gap_pct = (in_sample_expectancy - out_of_sample_expectancy) / abs(in_sample_expectancy) * 100

    unstable_params = parameter_sensitivity is not None and parameter_sensitivity.get("stable") is False

    if n_folds == 0:
        verdict = "marginal"
    elif failure_rate > 50 or unstable_params:
        verdict = "overfit"
    elif failure_rate > 20 or (gap_pct is not None and gap_pct > 50):
        verdict = "marginal"
    else:
        verdict = "robust"

    return OverfittingReport(
        n_folds=n_folds,
        n_passed=n_passed,
        walk_forward_failure_rate=failure_rate,
        in_sample_out_of_sample_gap_pct=gap_pct,
        parameter_sensitivity=parameter_sensitivity,
        verdict=verdict,
    )
