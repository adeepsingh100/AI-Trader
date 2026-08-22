from src.backtest.statistical_validation import (
    bootstrap_confidence_interval,
    monte_carlo_drawdown_distribution,
    moving_block_bootstrap_probability,
    parameter_stability_sweep,
)


def test_bootstrap_confidence_interval_none_below_two_samples():
    assert bootstrap_confidence_interval([1.0]) is None


def test_bootstrap_confidence_interval_deterministic_given_same_seed():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 2.5]
    result1 = bootstrap_confidence_interval(values, iterations=200, seed=42)
    result2 = bootstrap_confidence_interval(values, iterations=200, seed=42)
    assert result1 == result2


def test_bootstrap_confidence_interval_different_seed_can_differ():
    values = [1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 2.5]
    result1 = bootstrap_confidence_interval(values, iterations=200, seed=1)
    result2 = bootstrap_confidence_interval(values, iterations=200, seed=2)
    assert result1["ci_low"] != result2["ci_low"] or result1["ci_high"] != result2["ci_high"]


def test_bootstrap_confidence_interval_ci_brackets_point_estimate_roughly():
    values = [10.0] * 50  # zero variance -> CI collapses to the point estimate
    result = bootstrap_confidence_interval(values, iterations=100, seed=42)
    assert result["point_estimate"] == 10.0
    assert result["ci_low"] == 10.0
    assert result["ci_high"] == 10.0


def test_monte_carlo_drawdown_none_below_two_trades():
    assert monte_carlo_drawdown_distribution([10.0], starting_capital=1000) is None


def test_monte_carlo_drawdown_deterministic_given_same_seed():
    pnls = [10, -5, 20, -15, 8, -3, 12]
    r1 = monte_carlo_drawdown_distribution(pnls, starting_capital=1000, iterations=200, seed=7)
    r2 = monte_carlo_drawdown_distribution(pnls, starting_capital=1000, iterations=200, seed=7)
    assert r1 == r2


def test_monte_carlo_drawdown_actual_matches_realized_sequence():
    pnls = [10, -30, 5]  # running: 10, -20, -15 -> peak 10, max_dd = 30
    result = monte_carlo_drawdown_distribution(pnls, starting_capital=1000, iterations=50, seed=1)
    assert result["actual_drawdown_pct"] == 3.0  # 30/1000*100


def test_monte_carlo_drawdown_exposes_full_sorted_distribution():
    pnls = [10, -30, 5, 20, -15, 8, -3]
    result = monte_carlo_drawdown_distribution(pnls, starting_capital=1000, iterations=100, seed=1)
    assert len(result["drawdowns"]) == 100
    assert result["drawdowns"] == sorted(result["drawdowns"])
    assert result["drawdowns"][-1] == result["simulated_worst_drawdown_pct"]


def test_moving_block_bootstrap_none_below_two_diffs():
    assert moving_block_bootstrap_probability([1.0], block_length=3) is None


def test_moving_block_bootstrap_deterministic_given_same_seed():
    # TEST 1: same seed + same data -> identical result.
    diffs = [2, -1, 3, 4, -2, 1, 5, -3, 2, 1]
    r1 = moving_block_bootstrap_probability(diffs, block_length=3, iterations=200, seed=42)
    r2 = moving_block_bootstrap_probability(diffs, block_length=3, iterations=200, seed=42)
    assert r1 == r2


def test_moving_block_bootstrap_different_seed_can_differ():
    # TEST 2: different seed + same data -> potentially different result.
    # Near-zero-mean, high-variance series so the resampled outcome
    # genuinely depends on which blocks got drawn, not just the sign of
    # an overwhelmingly one-sided sum.
    diffs = [10, -9, 8, -11, 7, -6, 9, -10, 6, -7, 11, -8, 5, -12, 4, -1]
    r1 = moving_block_bootstrap_probability(diffs, block_length=3, iterations=300, seed=1)
    r2 = moving_block_bootstrap_probability(diffs, block_length=3, iterations=300, seed=2)
    assert r1["bootstrap_probability_positive_pct"] != r2["bootstrap_probability_positive_pct"]


def test_moving_block_bootstrap_records_block_length_in_metadata():
    # TEST 12: block size changes -> result metadata records block_length.
    diffs = [2, -1, 3, 4, -2, 1, 5, -3, 2, 1]
    r_small = moving_block_bootstrap_probability(diffs, block_length=2, iterations=50, seed=7)
    r_large = moving_block_bootstrap_probability(diffs, block_length=5, iterations=50, seed=7)
    assert r_small["bootstrap_block_length"] == 2
    assert r_large["bootstrap_block_length"] == 5
    assert r_small["bootstrap_method"] == r_large["bootstrap_method"] == "moving_block"


def test_moving_block_bootstrap_block_length_clamped_to_series_length():
    diffs = [1.0, 2.0, 3.0]
    result = moving_block_bootstrap_probability(diffs, block_length=100, iterations=20, seed=1)
    assert result["bootstrap_block_length"] == 3


def test_moving_block_bootstrap_consistently_positive_diffs_high_probability():
    diffs = [5, 4, 6, 5, 7, 4, 6, 5, 8, 5]
    result = moving_block_bootstrap_probability(diffs, block_length=3, iterations=200, seed=1)
    assert result["bootstrap_probability_positive_pct"] > 90.0


def test_moving_block_bootstrap_records_iterations_and_seed():
    diffs = [1.0, -1.0, 2.0, -2.0, 3.0]
    result = moving_block_bootstrap_probability(diffs, block_length=2, iterations=77, seed=9)
    assert result["bootstrap_iterations"] == 77
    assert result["seed"] == 9


def test_parameter_stability_sweep_none_below_three_points():
    assert parameter_stability_sweep([1.0, 2.0]) == {"stable": None, "jaggedness_score": None}


def test_parameter_stability_sweep_stable_for_smooth_curve():
    result = parameter_stability_sweep([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["stable"] is True
    assert result["jaggedness_score"] == 0.0


def test_parameter_stability_sweep_unstable_for_jagged_curve():
    result = parameter_stability_sweep([1.0, 10.0, 1.0, 10.0, 1.0])
    assert result["stable"] is False
