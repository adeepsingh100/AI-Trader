import pytest

from src.portfolio.capital_allocation import (
    compute_dynamic_size,
    confidence_factor,
    correlation_factor,
    drawdown_factor,
    exposure_factor,
    regime_factor,
    strategy_performance_factor,
    volatility_factor,
)


def test_all_factors_neutral_when_input_is_none():
    assert correlation_factor(None) == 1.0
    assert volatility_factor(None) == 1.0
    assert drawdown_factor(None) == 1.0
    assert exposure_factor(None) == 1.0
    assert strategy_performance_factor(None) == 1.0
    assert regime_factor(None) == 1.0
    assert confidence_factor(None) == 1.0


def test_correlation_factor_high_correlation_reduces_size():
    assert correlation_factor(1.0) < correlation_factor(0.0) < correlation_factor(-1.0)


def test_correlation_factor_clamped_to_configured_range(monkeypatch):
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_CORRELATION_MIN_MULT", 0.5)
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_CORRELATION_MAX_MULT", 1.5)
    assert correlation_factor(1.0) == pytest.approx(0.5)
    assert correlation_factor(-1.0) == pytest.approx(1.5)


def test_volatility_factor_high_volatility_reduces_size():
    low = volatility_factor(0.1)  # below VOLATILITY_LOW_MAX_PCT
    high = volatility_factor(10.0)  # above VOLATILITY_HIGH_MIN_PCT
    assert high < low


def test_drawdown_factor_larger_drawdown_reduces_size():
    assert drawdown_factor(20.0) < drawdown_factor(1.0)


def test_exposure_factor_more_committed_capital_reduces_size():
    assert exposure_factor(90.0) < exposure_factor(10.0)


def test_strategy_performance_factor_higher_win_rate_increases_size():
    assert strategy_performance_factor(0.7) > strategy_performance_factor(0.3)


def test_regime_factor_bull_larger_than_bear():
    assert regime_factor("strong_bull") > regime_factor("strong_bear")
    assert regime_factor("strong_bull") > regime_factor("high_volatility")


def test_regime_factor_unknown_regime_neutral():
    assert regime_factor("not_a_real_regime") == 1.0


def test_confidence_factor_higher_confidence_increases_size():
    assert confidence_factor(90) > confidence_factor(10)


def test_compute_dynamic_size_all_neutral_returns_base_capital():
    result = compute_dynamic_size(base_trade_capital=1000.0)
    assert result["trade_capital"] == pytest.approx(1000.0)
    assert result["combined_multiplier"] == pytest.approx(1.0)


def test_compute_dynamic_size_clamped_to_total_bounds(monkeypatch):
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_TOTAL_MIN_MULT", 0.5)
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_TOTAL_MAX_MULT", 1.5)
    result = compute_dynamic_size(
        base_trade_capital=1000.0,
        avg_correlation=-1.0,  # every factor maxed favorably
        candidate_volatility_pct=0.0,
        recent_drawdown_pct=0.0,
        current_exposure_pct=0.0,
        strategy_win_rate=1.0,
        market_regime="strong_bull",
        confidence=100.0,
    )
    # Product of 7 maxed-out factors would exceed 1.5x uncapped — the
    # combined clamp is what keeps sizing bounded.
    assert result["combined_multiplier"] == pytest.approx(1.5)
    assert result["trade_capital"] == pytest.approx(1500.0)


def test_compute_dynamic_size_worst_case_clamped_to_floor(monkeypatch):
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_TOTAL_MIN_MULT", 0.5)
    monkeypatch.setattr("src.portfolio.capital_allocation.CAPITAL_ALLOC_TOTAL_MAX_MULT", 1.5)
    result = compute_dynamic_size(
        base_trade_capital=1000.0,
        avg_correlation=1.0,
        candidate_volatility_pct=100.0,
        recent_drawdown_pct=100.0,
        current_exposure_pct=100.0,
        strategy_win_rate=0.0,
        market_regime="strong_bear",
        confidence=0.0,
    )
    assert result["combined_multiplier"] == pytest.approx(0.5)
    assert result["trade_capital"] == pytest.approx(500.0)
