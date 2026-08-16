import pytest

from src.portfolio.intelligence import (
    Position,
    category_of,
    correlation,
    correlation_matrix,
    diversification_score,
    evaluate_candidate_impact,
    expected_shortfall_pct,
    historical_var_pct,
    max_concentration_pct,
    net_gross_exposure_pct,
    portfolio_volatility,
    returns_series,
    sector_exposure,
)


def test_returns_series_basic():
    assert returns_series([100, 110, 99]) == pytest.approx([0.10, -0.1])


def test_returns_series_empty_and_single_value():
    assert returns_series([]) == []
    assert returns_series([100]) == []


def test_correlation_perfectly_correlated_series():
    a = [100, 110, 90, 120]  # varying, non-constant returns
    b = [50, 55, 45, 60]  # exactly half of a, same relative moves
    assert correlation(returns_series(a), returns_series(b)) == pytest.approx(1.0, abs=1e-6)


def test_correlation_perfectly_anti_correlated_series():
    a = returns_series([100, 110, 90, 120])
    b = [-x for x in a]
    assert correlation(a, b) == pytest.approx(-1.0, abs=1e-6)


def test_correlation_none_below_minimum_sample():
    assert correlation([0.01], [0.02]) is None


def test_correlation_matrix_diagonal_is_one():
    history = {"BTCINR": [100, 101, 102, 103], "ETHINR": [50, 49, 51, 52]}
    matrix = correlation_matrix(history, window=10)
    assert matrix[("BTCINR", "BTCINR")] == 1.0
    assert matrix[("BTCINR", "ETHINR")] == matrix[("ETHINR", "BTCINR")]


def test_category_of_uses_coin_category_map(monkeypatch):
    monkeypatch.setattr("src.portfolio.intelligence.COIN_CATEGORY_MAP", {"BTC": "layer1"})
    assert category_of("BTCINR") == "layer1"
    assert category_of("XYZINR") == "uncategorized"


def test_sector_exposure_splits_by_category(monkeypatch):
    monkeypatch.setattr(
        "src.portfolio.intelligence.COIN_CATEGORY_MAP", {"BTC": "layer1", "USDT": "stablecoin"}
    )
    positions = [
        Position("BTCINR", qty=1, entry_price=100, current_price=100),
        Position("USDTINR", qty=100, entry_price=1, current_price=1),
    ]
    # Denominated against total equity (200 = the two positions' combined
    # value here), not the sum of positions themselves — see
    # sector_exposure's docstring for why that distinction matters.
    exposure = sector_exposure(positions, equity=200)
    assert exposure["layer1"] == pytest.approx(50.0)
    assert exposure["stablecoin"] == pytest.approx(50.0)


def test_max_concentration_pct_relative_to_total_equity_not_just_positions():
    # A real bug an integration test caught: measuring concentration
    # against the sum of currently-open positions makes a lone position
    # ALWAYS 100% "concentrated" regardless of how much total capital
    # exists — the correct denominator is total equity (the standard "% of
    # NAV" convention), where a small position in a large pool is
    # correctly reported as low concentration.
    positions = [Position("BTCINR", qty=1, entry_price=100, current_price=100)]
    assert max_concentration_pct(positions, equity=100_000) == pytest.approx(0.1)
    assert max_concentration_pct(positions, equity=100) == pytest.approx(100.0)


def test_max_concentration_pct_none_when_no_positions():
    assert max_concentration_pct([], equity=1000) is None


def test_net_gross_exposure_equal_when_long_only():
    positions = [Position("BTCINR", qty=1, entry_price=100, current_price=100)]
    result = net_gross_exposure_pct(positions, equity=200)
    assert result["net_exposure_pct"] == result["gross_exposure_pct"] == pytest.approx(50.0)


def test_diversification_score_higher_for_equal_split_than_concentrated():
    concentrated = [
        Position("BTCINR", qty=9, entry_price=100, current_price=100),
        Position("ETHINR", qty=1, entry_price=100, current_price=100),
    ]
    equal = [
        Position("BTCINR", qty=5, entry_price=100, current_price=100),
        Position("ETHINR", qty=5, entry_price=100, current_price=100),
    ]
    assert diversification_score(equal) > diversification_score(concentrated)


def test_historical_var_and_expected_shortfall_positive_for_losses():
    returns = [-0.10, -0.05, 0.01, 0.02, 0.03]
    var = historical_var_pct(returns, confidence_pct=80)
    es = expected_shortfall_pct(returns, confidence_pct=80)
    assert var is not None and var > 0
    assert es is not None and es >= var  # tail average loss >= the VaR cutoff itself


def test_portfolio_volatility_none_with_insufficient_history():
    positions = [Position("BTCINR", qty=1, entry_price=100, current_price=100)]
    assert portfolio_volatility(positions, {"BTCINR": [100]}) is None


def test_evaluate_candidate_impact_no_look_ahead_uses_only_supplied_history():
    """The module must never window/slice beyond what price_history already
    contains — passing a short, explicitly truncated series must not
    error or silently pull in more data than given."""
    positions = []
    price_history = {"BTCINR": [100, 101]}  # deliberately short — "as of now"
    result = evaluate_candidate_impact(
        "BTCINR", candidate_qty=1, candidate_price=101, positions=positions,
        price_history=price_history, equity=10000, max_concurrent_positions=5,
    )
    assert result["after"]["max_concentration_pct"] == pytest.approx(101 / 10000 * 100)


def test_evaluate_candidate_impact_concentration_cap_scales_with_max_positions():
    """A single, appropriately-sized position (candidate notional at its
    equal fair share of total equity) must NOT trip the cap regardless of
    max_concurrent_positions — a real bug caught during integration
    testing where concentration was measured against the sum of currently-
    open positions (always ~100% for a lone position) instead of total
    equity, which blocked nearly every first trade regardless of book size."""
    # 10 out of 10,000 equity = 0.1% concentration — trivially fine at any
    # realistic max_concurrent_positions.
    result_small = evaluate_candidate_impact(
        "BTCINR", candidate_qty=0.1, candidate_price=100, positions=[],
        price_history={}, equity=10_000, max_concurrent_positions=2,
    )
    assert result_small["exceeds_position_concentration_cap"] is False

    # A candidate whose notional is the ENTIRE equity pool is genuinely
    # oversized relative to a 20-slot book's ~5% fair share (cap ~12.5% at
    # the default 2.5x multiple) and must be caught.
    result_oversized = evaluate_candidate_impact(
        "BTCINR", candidate_qty=100, candidate_price=100, positions=[],
        price_history={}, equity=10_000, max_concurrent_positions=20,
    )
    assert result_oversized["exceeds_position_concentration_cap"] is True
