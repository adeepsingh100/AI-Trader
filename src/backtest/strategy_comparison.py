"""Step 10: pairwise strategy-run comparison. Reuses
src/learning/statistics.py's z_test_two_proportions (win-rate) and
z_test_two_means (expectancy) directly — "only recommend promotion if
statistically superior" means the test rejects the null in B's favor, not
just that B's raw number happens to be bigger."""

from __future__ import annotations

from statistics import mean, stdev

from src.backtest.portfolio_manager import ClosedTrade
from src.config import SIGNIFICANCE_THRESHOLD
from src.learning.statistics import z_test_two_means, z_test_two_proportions


def compare(trades_a: list[ClosedTrade], trades_b: list[ClosedTrade], metrics_a: dict, metrics_b: dict) -> dict:
    """metrics_a/metrics_b: performance_analyzer.analyze() bundles."""
    n_a, n_b = len(trades_a), len(trades_b)
    wins_a = sum(1 for t in trades_a if t.pnl > 0)
    wins_b = sum(1 for t in trades_b if t.pnl > 0)

    win_rate_p = z_test_two_proportions(wins_a, n_a, wins_b, n_b)

    expectancy_p = None
    if n_a >= 2 and n_b >= 2:
        pnls_a = [t.pnl for t in trades_a]
        pnls_b = [t.pnl for t in trades_b]
        expectancy_p = z_test_two_means(mean(pnls_a), stdev(pnls_a), n_a, mean(pnls_b), stdev(pnls_b), n_b)

    b_better_win_rate = (metrics_b.get("win_rate") or 0) > (metrics_a.get("win_rate") or 0)
    b_better_expectancy = (metrics_b.get("expectancy") or 0) > (metrics_a.get("expectancy") or 0)
    b_significantly_better = expectancy_p is not None and expectancy_p < SIGNIFICANCE_THRESHOLD and b_better_expectancy
    a_significantly_better = (
        expectancy_p is not None and expectancy_p < SIGNIFICANCE_THRESHOLD and not b_better_expectancy
    )

    winner = None
    if b_significantly_better:
        winner = "b"
    elif a_significantly_better:
        winner = "a"
    elif win_rate_p is not None and win_rate_p < SIGNIFICANCE_THRESHOLD:
        winner = "b" if b_better_win_rate else "a"

    return {
        "p_values": {"win_rate": win_rate_p, "expectancy": expectancy_p},
        "winner": winner,
        # "Only recommend promotion if statistically superior" — automatic
        # STATUS marking only (never automatic deletion/live application),
        # matching this session's established human-approval precedent.
        "promotion_recommended": winner == "b",
    }
