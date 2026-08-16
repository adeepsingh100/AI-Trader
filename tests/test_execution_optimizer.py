from src.backtest.order_manager import OrderType
from src.execution_optimizer.optimizer import OrderContext, recommend


def test_recommend_market_when_spread_tight():
    ctx = OrderContext(
        symbol="BTCINR", side="buy", order_size=1000, spread_bps=2.0, bar_volume=1_000_000,
        volatility_pct=1.0, recent_fill_rate=None,
    )
    rec = recommend(ctx)
    assert rec.order_type == OrderType.MARKET
    assert rec.estimated_fill_probability == 1.0


def test_recommend_limit_when_spread_wide_and_fill_probability_high():
    ctx = OrderContext(
        symbol="BTCINR", side="buy", order_size=1000, spread_bps=50.0, bar_volume=1_000_000,
        volatility_pct=None, recent_fill_rate=0.9,
    )
    rec = recommend(ctx)
    assert rec.order_type == OrderType.LIMIT
    assert rec.estimated_fill_probability == 0.9
    assert rec.estimated_slippage_bps == 0.0


def test_recommend_falls_back_to_market_when_fill_probability_too_low():
    ctx = OrderContext(
        symbol="BTCINR", side="buy", order_size=1000, spread_bps=50.0, bar_volume=1_000_000,
        volatility_pct=None, recent_fill_rate=0.1,  # wide spread but unlikely to fill
    )
    rec = recommend(ctx)
    assert rec.order_type == OrderType.MARKET


def test_recommend_market_cost_reflects_spread():
    ctx = OrderContext(
        symbol="BTCINR", side="buy", order_size=1000, spread_bps=8.0, bar_volume=1_000_000,
    )
    rec = recommend(ctx)
    assert rec.estimated_cost_bps == 8.0


def test_recommend_slippage_scales_with_order_size_vs_liquidity():
    small = recommend(OrderContext("BTCINR", "buy", order_size=10, spread_bps=1, bar_volume=1_000_000))
    large = recommend(OrderContext("BTCINR", "buy", order_size=900_000, spread_bps=1, bar_volume=1_000_000))
    assert large.estimated_slippage_bps > small.estimated_slippage_bps


def test_recommend_zero_bar_volume_treated_as_worst_case_not_zero():
    rec = recommend(OrderContext("BTCINR", "buy", order_size=100, spread_bps=1, bar_volume=0))
    assert rec.estimated_slippage_bps > 0  # no liquidity data must not silently read as "no cost"


def test_recommend_no_fill_rate_history_uses_volatility_heuristic():
    # Both volatility levels chosen to clear EXECUTION_OPTIMIZER_MIN_FILL_
    # PROBABILITY (both land on the LIMIT branch), so the returned
    # estimated_fill_probability is directly comparable — a MARKET
    # recommendation always reports 1.0 regardless of the underlying
    # limit heuristic, which would make this comparison meaningless.
    low_vol = recommend(
        OrderContext("BTCINR", "buy", order_size=100, spread_bps=50, bar_volume=1_000_000, volatility_pct=7.0)
    )
    high_vol = recommend(
        OrderContext("BTCINR", "buy", order_size=100, spread_bps=50, bar_volume=1_000_000, volatility_pct=15.0)
    )
    assert low_vol.order_type == OrderType.LIMIT
    assert high_vol.order_type == OrderType.LIMIT
    assert high_vol.estimated_fill_probability > low_vol.estimated_fill_probability
