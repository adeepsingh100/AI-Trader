"""Step 11: statistical validation via seeded resampling — NOT a
parametric t-interval (see config.py's Statistical methods note: this
codebase has zero numpy/scipy, and a hand-rolled regularized-incomplete-
beta implementation for a real t-CDF is real numerical bug surface for
little gain over the existing z-test at backtest sample sizes; bootstrap
needs no distributional assumption at all, arguably the more honest answer
for small fold sizes). All randomness draws from a local
random.Random(seed) instance — never the global `random` module — so
reruns are bit-identical, satisfying "everything must be deterministic"."""

from __future__ import annotations

import random
from collections.abc import Callable

from src.config import BACKTEST_BOOTSTRAP_ITERATIONS, BACKTEST_MONTE_CARLO_ITERATIONS, BACKTEST_RANDOM_SEED
from src.utils import max_drawdown_pct as _max_drawdown_pct


def bootstrap_confidence_interval(
    values: list[float],
    statistic_fn: Callable[[list[float]], float] = lambda xs: sum(xs) / len(xs),
    iterations: int = BACKTEST_BOOTSTRAP_ITERATIONS,
    confidence_pct: float = 95.0,
    seed: int = BACKTEST_RANDOM_SEED,
) -> dict | None:
    """Resamples `values` with replacement `iterations` times, computes
    `statistic_fn` on each resample, returns the empirical CI."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    stats = sorted(statistic_fn([values[rng.randrange(n)] for _ in range(n)]) for _ in range(iterations))
    alpha = (1 - confidence_pct / 100) / 2
    lo_idx = int(alpha * iterations)
    hi_idx = min(int((1 - alpha) * iterations), iterations - 1)
    return {
        "point_estimate": statistic_fn(values),
        "ci_low": stats[lo_idx],
        "ci_high": stats[hi_idx],
        "confidence_pct": confidence_pct,
        "iterations": iterations,
    }


def monte_carlo_drawdown_distribution(
    trade_pnls: list[float],
    starting_capital: float,
    iterations: int = BACKTEST_MONTE_CARLO_ITERATIONS,
    seed: int = BACKTEST_RANDOM_SEED,
) -> dict | None:
    """Shuffles trade ORDER (not values) `iterations` times, recomputes
    max drawdown for each — tests how much the observed drawdown depends
    on the specific sequence trades happened to occur in (path
    dependency), a robustness check the realized equity curve alone can't
    answer."""
    if len(trade_pnls) < 2:
        return None
    rng = random.Random(seed)
    drawdowns = []
    for _ in range(iterations):
        shuffled = trade_pnls[:]
        rng.shuffle(shuffled)
        drawdowns.append(_max_drawdown_pct(shuffled, starting_capital))
    drawdowns.sort()
    actual_dd_pct = _max_drawdown_pct(trade_pnls, starting_capital)
    return {
        "actual_drawdown_pct": actual_dd_pct,
        "simulated_median_drawdown_pct": drawdowns[len(drawdowns) // 2],
        "simulated_worst_drawdown_pct": drawdowns[-1],
        "percentile_of_actual": sum(1 for d in drawdowns if d <= actual_dd_pct) / len(drawdowns) * 100,
        "iterations": iterations,
        # Full sorted distribution — additive (existing callers destructure
        # by key, none assert an exact key set); promotion_gate.py uses
        # this to answer "probability of a catastrophic drawdown", a
        # different question than the percentile-of-actual summary above.
        "drawdowns": drawdowns,
    }


def parameter_stability_sweep(metric_values: list[float]) -> dict:
    """Given a swept parameter's resulting metric values (e.g. expectancy
    at each candidate threshold), flags jaggedness — a smooth curve
    suggests a genuine effect, a jagged one suggests curve-fitting/noise.
    Heuristic: mean absolute second difference relative to the metric's
    own spread — no ML, just a variance-of-adjacent-differences check."""
    if len(metric_values) < 3:
        return {"stable": None, "jaggedness_score": None}
    second_diffs = [
        abs(metric_values[i + 1] - 2 * metric_values[i] + metric_values[i - 1])
        for i in range(1, len(metric_values) - 1)
    ]
    spread = max(metric_values) - min(metric_values)
    if spread == 0:
        return {"stable": True, "jaggedness_score": 0.0}
    jaggedness = (sum(second_diffs) / len(second_diffs)) / spread
    return {"stable": jaggedness < 0.5, "jaggedness_score": jaggedness}
