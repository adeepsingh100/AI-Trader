"""Portfolio Intelligence Engine — correlation/exposure/risk analytics
across the whole open-positions book, not just one trade in isolation. See
PROJECT_SPEC.md §3d.

Pure functions, no DB/network access, matching execution_simulator.py's
isolation precedent — the caller (live risk_manager or the backtest
engine) supplies `positions` and a `price_history` dict that is ALREADY
truncated to the caller's current point in time. This module never
windows/slices beyond what it's handed — that's the deliberate fix for a
real look-ahead trap a Plan-agent pre-mortem caught: if this function
windowed its own lookback internally, a careless backtest call site could
hand it a full future-inclusive series "for convenience" and silently leak
future prices into a "rolling" correlation/VaR window. One test
(test_backtest_engine.py) asserts the backtest engine never does that.

All math is stdlib-only (hand-rolled covariance/variance/percentile, same
"no numpy/scipy" discipline as src/learning/statistics.py's hand-rolled
z-tests) — Python 3.9 compatible, so this deliberately does NOT use
statistics.covariance/correlation/linear_regression (3.10+ only)."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev

from src.config import (
    COIN_CATEGORY_MAP,
    MAX_POSITION_CONCENTRATION_MULT_OF_EQUAL_SHARE,
    MAX_SECTOR_CONCENTRATION_MULT_OF_EQUAL_SHARE,
    PORTFOLIO_BETA_PROXY_SYMBOL,
    PORTFOLIO_CORRELATION_LOOKBACK_BARS,
    PORTFOLIO_VAR_CONFIDENCE_PCT,
)

# Single-exchange bot (CoinDCX only) — always 100%, reported as a constant
# rather than computed, since there's nothing to aggregate over.
EXCHANGE_EXPOSURE_PCT = 100.0


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price


def _base_symbol(symbol: str) -> str:
    return symbol[:-3] if symbol.endswith("INR") else symbol


def category_of(symbol: str) -> str:
    return COIN_CATEGORY_MAP.get(_base_symbol(symbol).upper(), "uncategorized")


def returns_series(prices: list[float]) -> list[float]:
    out = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev:
            out.append((prices[i] - prev) / prev)
    return out


def _windowed(prices: list[float], window: int) -> list[float]:
    return prices[-window:] if window and len(prices) > window else prices


def _covariance(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[-n:], b[-n:]
    ma, mb = mean(a), mean(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / n


def _variance(a: list[float]) -> float | None:
    if len(a) < 2:
        return None
    return pstdev(a) ** 2


def correlation(a: list[float], b: list[float]) -> float | None:
    cov = _covariance(a, b)
    if cov is None:
        return None
    var_a, var_b = _variance(a), _variance(b)
    if not var_a or not var_b:
        return None
    denom = (var_a**0.5) * (var_b**0.5)
    return cov / denom if denom else None


def correlation_matrix(
    price_history: dict[str, list[float]], window: int = PORTFOLIO_CORRELATION_LOOKBACK_BARS
) -> dict[tuple[str, str], float | None]:
    """Windowed to the last `window` bars of whatever price_history the
    caller supplied — 'rolling' means the caller re-calls this each tick
    with its own as-of-truncated series, not that this function windows
    into data beyond what it was given."""
    symbols = list(price_history.keys())
    returns_by_symbol = {s: returns_series(_windowed(price_history[s], window + 1)) for s in symbols}
    matrix: dict[tuple[str, str], float | None] = {}
    for i, s1 in enumerate(symbols):
        for s2 in symbols[i:]:
            r = correlation(returns_by_symbol[s1], returns_by_symbol[s2]) if s1 != s2 else 1.0
            matrix[(s1, s2)] = r
            matrix[(s2, s1)] = r
    return matrix


def sector_exposure(positions: list[Position], equity: float) -> dict[str, float]:
    """Denominated against total EQUITY (the capital pool), not the sum of
    currently-open positions — a real bug an integration test caught: with
    the latter denominator, a sparse book's exposure is always ~100% of
    itself regardless of how many concurrent-position slots exist, since
    one position among one position is trivially "all of it". Total-equity
    denomination is also the standard real-world convention for a position
    limit ("no single holding over N% of NAV"), not an arbitrary choice."""
    if not equity:
        return {}
    by_category: dict[str, float] = {}
    for p in positions:
        by_category[category_of(p.symbol)] = by_category.get(category_of(p.symbol), 0.0) + p.market_value
    return {cat: value / equity * 100 for cat, value in by_category.items()}


def stablecoin_allocation_pct(positions: list[Position], equity: float) -> float:
    return sector_exposure(positions, equity).get("stablecoin", 0.0)


def max_concentration_pct(positions: list[Position], equity: float) -> float | None:
    """Denominated against total equity, same fix as sector_exposure —
    see its docstring."""
    if not equity or not positions:
        return None
    return max(p.market_value for p in positions) / equity * 100


def net_gross_exposure_pct(positions: list[Position], equity: float) -> dict[str, float | None]:
    """Spot-only, no shorting anywhere in this bot — every position is
    long, so net == gross by construction. Computed from signed qty rather
    than hardcoded equal, so this stays correct if a short-capable
    execution path is ever added without anyone having to remember to
    revisit this function."""
    if not equity:
        return {"net_exposure_pct": None, "gross_exposure_pct": None}
    signed = sum(p.qty * p.current_price for p in positions)
    gross = sum(abs(p.qty) * p.current_price for p in positions)
    return {"net_exposure_pct": signed / equity * 100, "gross_exposure_pct": gross / equity * 100}


def beta(
    price_history: dict[str, list[float]],
    proxy_symbol: str = PORTFOLIO_BETA_PROXY_SYMBOL,
    window: int = PORTFOLIO_CORRELATION_LOOKBACK_BARS,
) -> dict[str, float | None]:
    proxy_prices = price_history.get(proxy_symbol)
    if not proxy_prices:
        return {s: None for s in price_history}
    proxy_returns = returns_series(_windowed(proxy_prices, window + 1))
    var_proxy = _variance(proxy_returns)
    result: dict[str, float | None] = {}
    for symbol, prices in price_history.items():
        if symbol == proxy_symbol:
            result[symbol] = 1.0
            continue
        symbol_returns = returns_series(_windowed(prices, window + 1))
        cov = _covariance(symbol_returns, proxy_returns)
        result[symbol] = cov / var_proxy if cov is not None and var_proxy else None
    return result


def portfolio_returns(positions: list[Position], price_history: dict[str, list[float]]) -> list[float]:
    """Weighted combination of each position's return series (weights from
    current market value), aligned to the shortest common length. Empty if
    there's nothing to combine (no positions, or no overlapping history)."""
    total = sum(p.market_value for p in positions)
    if not total:
        return []
    weighted_returns: dict[str, tuple[float, list[float]]] = {}
    for p in positions:
        prices = price_history.get(p.symbol)
        if not prices or len(prices) < 2:
            continue
        weighted_returns[p.symbol] = (p.market_value / total, returns_series(prices))
    if not weighted_returns:
        return []
    n = min(len(r) for _, r in weighted_returns.values())
    if n == 0:
        return []
    combined = [0.0] * n
    for weight, series in weighted_returns.values():
        series = series[-n:]
        for i in range(n):
            combined[i] += weight * series[i]
    return combined


def portfolio_volatility(positions: list[Position], price_history: dict[str, list[float]]) -> float | None:
    returns = portfolio_returns(positions, price_history)
    return pstdev(returns) if len(returns) >= 2 else None


def risk_contribution(
    positions: list[Position], price_history: dict[str, list[float]]
) -> dict[str, float | None]:
    """Marginal contribution to portfolio variance per position: weight_i *
    cov(return_i, return_portfolio) / portfolio_variance. Sums to ~1.0
    across positions when defined."""
    port_returns = portfolio_returns(positions, price_history)
    port_var = _variance(port_returns)
    total = sum(p.market_value for p in positions)
    if not port_var or not total:
        return {p.symbol: None for p in positions}
    result: dict[str, float | None] = {}
    for p in positions:
        prices = price_history.get(p.symbol)
        weight = p.market_value / total
        if not prices or len(prices) < 2:
            result[p.symbol] = None
            continue
        n = len(port_returns)
        symbol_returns = returns_series(prices)[-n:]
        cov = _covariance(symbol_returns, port_returns)
        result[p.symbol] = (weight * cov / port_var) if cov is not None else None
    return result


def historical_var_pct(
    returns: list[float], confidence_pct: float = PORTFOLIO_VAR_CONFIDENCE_PCT
) -> float | None:
    """Historical-simulation VaR — sorted-return percentile, no
    distributional assumption (same reasoning as the backtest engine's
    seeded-bootstrap-over-parametric-t choice). Returns the loss magnitude
    as a positive percentage; None below a minimal sample size (2)."""
    if len(returns) < 2:
        return None
    ordered = sorted(returns)
    idx = int((1 - confidence_pct / 100) * len(ordered))
    idx = max(0, min(len(ordered) - 1, idx))
    return -ordered[idx] * 100 if ordered[idx] < 0 else 0.0


def expected_shortfall_pct(
    returns: list[float], confidence_pct: float = PORTFOLIO_VAR_CONFIDENCE_PCT
) -> float | None:
    """Average loss beyond the VaR cutoff — the tail the VaR percentile
    alone doesn't describe."""
    if len(returns) < 2:
        return None
    ordered = sorted(returns)
    idx = max(1, int((1 - confidence_pct / 100) * len(ordered)))
    tail = ordered[:idx]
    if not tail:
        return None
    avg = mean(tail)
    return -avg * 100 if avg < 0 else 0.0


def diversification_score(positions: list[Position]) -> float | None:
    """1 - Herfindahl-Hirschman Index of position weights, clamped [0, 1].
    0 = fully concentrated in one position, approaching 1 as capital
    spreads across more, more-equal positions."""
    total = sum(p.market_value for p in positions)
    if not total or not positions:
        return None
    hhi = sum((p.market_value / total) ** 2 for p in positions)
    return max(0.0, min(1.0, 1 - hhi))


def portfolio_snapshot(
    positions: list[Position], price_history: dict[str, list[float]], equity: float
) -> dict:
    """The full Step 3 analytics bundle in one call — what risk_manager and
    the backtest engine actually reach for."""
    port_returns = portfolio_returns(positions, price_history)
    return {
        "correlation_matrix": correlation_matrix(price_history),
        "sector_exposure": sector_exposure(positions, equity),
        "stablecoin_allocation_pct": stablecoin_allocation_pct(positions, equity),
        "exchange_exposure_pct": EXCHANGE_EXPOSURE_PCT,
        "max_concentration_pct": max_concentration_pct(positions, equity),
        **net_gross_exposure_pct(positions, equity),
        "beta": beta(price_history),
        "risk_contribution": risk_contribution(positions, price_history),
        "portfolio_volatility": portfolio_volatility(positions, price_history),
        "var_pct": historical_var_pct(port_returns),
        "expected_shortfall_pct": expected_shortfall_pct(port_returns),
        "diversification_score": diversification_score(positions),
    }


def evaluate_candidate_impact(
    candidate_symbol: str,
    candidate_qty: float,
    candidate_price: float,
    positions: list[Position],
    price_history: dict[str, list[float]],
    equity: float,
    max_concurrent_positions: int = 5,
) -> dict:
    """Projects the portfolio WITH the candidate added, alongside today's
    (without) snapshot, so a caller sees exactly what a new trade would do
    to concentration/sector exposure/diversification before it's placed —
    Step 3's "evaluate every new trade in context of the existing
    portfolio". Doesn't decide accept/reject itself; risk_manager.evaluate
    (the actual gate) is the one place a block decision gets made.

    Concentration caps are relative to an equal-weighted
    max_concurrent_positions-slot book (100/max_concurrent_positions =
    each position's "fair share"), not a fixed institutional percentage —
    a flat cap would block nearly every trade in this bot's actual 2-5
    concurrent-position range, since a single position in a small book is
    structurally a large percentage of it. `equity` is the TOTAL capital
    pool (e.g. capital_config['capital_to_use']), not cash remaining after
    commitment — buying a position converts cash to an asset of
    (approximately) equal value, it doesn't change total equity, so the
    same `equity` is used for both the before and after snapshot rather
    than inflating it by the candidate's notional."""
    equal_share_pct = 100.0 / max_concurrent_positions if max_concurrent_positions > 0 else 100.0
    position_cap_pct = equal_share_pct * MAX_POSITION_CONCENTRATION_MULT_OF_EQUAL_SHARE
    sector_cap_pct = equal_share_pct * MAX_SECTOR_CONCENTRATION_MULT_OF_EQUAL_SHARE

    before = portfolio_snapshot(positions, price_history, equity)
    projected_positions = list(positions) + [
        Position(candidate_symbol, candidate_qty, candidate_price, candidate_price)
    ]
    after = portfolio_snapshot(projected_positions, price_history, equity)

    candidate_category = category_of(candidate_symbol)
    return {
        "before": before,
        "after": after,
        "position_cap_pct": position_cap_pct,
        "sector_cap_pct": sector_cap_pct,
        "projected_max_concentration_pct": after["max_concentration_pct"],
        "exceeds_position_concentration_cap": (
            after["max_concentration_pct"] is not None and after["max_concentration_pct"] > position_cap_pct
        ),
        "projected_sector_concentration_pct": after["sector_exposure"].get(candidate_category),
        "exceeds_sector_concentration_cap": (
            after["sector_exposure"].get(candidate_category, 0.0) > sector_cap_pct
        ),
        "diversification_delta": (
            (after["diversification_score"] - before["diversification_score"])
            if after["diversification_score"] is not None and before["diversification_score"] is not None
            else None
        ),
    }
