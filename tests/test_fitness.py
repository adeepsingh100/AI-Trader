import pytest

from src.learning.fitness import (
    compute_fitness_score,
    drawdown_component,
    expectancy_component,
    profit_factor_component,
    sharpe_component,
    win_rate_component,
)


def test_sharpe_component_maps_range_to_0_100():
    assert sharpe_component(-1.0) == pytest.approx(0.0)
    assert sharpe_component(0.0) == pytest.approx(100 / 3)
    assert sharpe_component(2.0) == pytest.approx(100.0)
    assert sharpe_component(None) is None


def test_drawdown_component_zero_drawdown_is_100(monkeypatch):
    monkeypatch.setattr("src.learning.fitness.PROMOTION_MAX_DRAWDOWN_PCT", 15)
    assert drawdown_component(0.0) == pytest.approx(100.0)
    assert drawdown_component(15.0) == pytest.approx(0.0)
    assert drawdown_component(None) is None


def test_win_rate_component_scales_to_100():
    assert win_rate_component(0.5) == pytest.approx(50.0)
    assert win_rate_component(None) is None


def test_profit_factor_component_breakeven_is_50():
    assert profit_factor_component(1.0) == pytest.approx(100 / 3)
    assert profit_factor_component(3.0) == pytest.approx(100.0)
    assert profit_factor_component(None) is None


def test_expectancy_component_neutral_at_zero_expectancy():
    assert expectancy_component(0.0, capital_to_use=10_000) == pytest.approx(50.0)


def test_expectancy_component_positive_expectancy_scores_above_neutral(monkeypatch):
    monkeypatch.setattr("src.learning.fitness.FITNESS_EXPECTANCY_SCALE", 10)
    # expectancy = 100 on 10,000 capital = 1% -> 50 + 1*10 = 60
    assert expectancy_component(100.0, capital_to_use=10_000) == pytest.approx(60.0)


def test_expectancy_component_none_on_zero_capital():
    assert expectancy_component(100.0, capital_to_use=0) is None


def test_compute_fitness_score_weighted_blend():
    stats = {
        "profit_factor": 3.0,  # -> 100
        "sharpe_ratio": 2.0,  # -> 100
        "expectancy": 0.0,  # -> 50 (neutral)
        "win_rate": 1.0,  # -> 100
        "max_drawdown_pct": 0.0,  # -> 100 (no penalty)
    }
    result = compute_fitness_score(stats, capital_to_use=10_000)
    assert result["fitness_score"] > 50  # every component is neutral-or-better
    assert set(result["components"]) == {
        "profit_factor", "sharpe", "expectancy", "win_rate", "drawdown_penalty",
    }


def test_compute_fitness_score_renormalizes_over_missing_components():
    # Only profit_factor available — score should equal that one component's
    # value (renormalized weight of 1.0), not be dragged toward 0 by the
    # missing ones.
    stats = {"profit_factor": 3.0}
    result = compute_fitness_score(stats, capital_to_use=10_000)
    assert result["fitness_score"] == pytest.approx(100.0)


def test_compute_fitness_score_none_when_nothing_available():
    result = compute_fitness_score({}, capital_to_use=10_000)
    assert result["fitness_score"] is None
