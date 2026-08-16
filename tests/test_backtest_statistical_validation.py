from src.backtest.statistical_validation import (
    bootstrap_confidence_interval,
    monte_carlo_drawdown_distribution,
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


def test_parameter_stability_sweep_none_below_three_points():
    assert parameter_stability_sweep([1.0, 2.0]) == {"stable": None, "jaggedness_score": None}


def test_parameter_stability_sweep_stable_for_smooth_curve():
    result = parameter_stability_sweep([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["stable"] is True
    assert result["jaggedness_score"] == 0.0


def test_parameter_stability_sweep_unstable_for_jagged_curve():
    result = parameter_stability_sweep([1.0, 10.0, 1.0, 10.0, 1.0])
    assert result["stable"] is False
