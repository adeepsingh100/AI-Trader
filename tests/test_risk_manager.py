from src.agents.risk_manager import (
    RiskDecision,
    circuit_breaker_triggered,
    committed_capital,
    evaluate,
    target_hit,
)


def _capital_config(**overrides):
    base = {
        "capital_to_use": 10000,
        "daily_profit_target": 500,
        "max_daily_loss": 1000,
        "position_size_pct": 10,
        "max_concurrent_positions": 5,
    }
    base.update(overrides)
    return base


# --- circuit breaker ---


def test_circuit_breaker_false_with_no_daily_pnl_row():
    assert circuit_breaker_triggered(None, _capital_config()) is False


def test_circuit_breaker_false_when_loss_under_threshold():
    daily_pnl = {"realized_pnl": -999, "circuit_breaker_triggered": False}
    assert circuit_breaker_triggered(daily_pnl, _capital_config(max_daily_loss=1000)) is False


def test_circuit_breaker_true_at_exact_threshold():
    daily_pnl = {"realized_pnl": -1000, "circuit_breaker_triggered": False}
    assert circuit_breaker_triggered(daily_pnl, _capital_config(max_daily_loss=1000)) is True


def test_circuit_breaker_true_past_threshold():
    daily_pnl = {"realized_pnl": -1500, "circuit_breaker_triggered": False}
    assert circuit_breaker_triggered(daily_pnl, _capital_config(max_daily_loss=1000)) is True


def test_circuit_breaker_sticky_even_if_pnl_recovers():
    # once flipped, stays flipped for the day regardless of recovery
    daily_pnl = {"realized_pnl": 50, "circuit_breaker_triggered": True}
    assert circuit_breaker_triggered(daily_pnl, _capital_config(max_daily_loss=1000)) is True


def test_circuit_breaker_ignores_profit():
    daily_pnl = {"realized_pnl": 5000, "circuit_breaker_triggered": False}
    assert circuit_breaker_triggered(daily_pnl, _capital_config(max_daily_loss=1000)) is False


# --- daily target (soft, tracked only) ---


def test_target_not_hit_below_threshold():
    daily_pnl = {"realized_pnl": 499}
    assert target_hit(daily_pnl, _capital_config(daily_profit_target=500)) is False


def test_target_hit_at_threshold():
    daily_pnl = {"realized_pnl": 500}
    assert target_hit(daily_pnl, _capital_config(daily_profit_target=500)) is True


def test_target_hit_with_no_daily_pnl_row_is_false():
    assert target_hit(None, _capital_config()) is False


# --- committed capital ---


def test_committed_capital_sums_open_trades():
    open_trades = [
        {"qty": 0.01, "entry_price": 1_000_000},
        {"qty": 2, "entry_price": 100},
    ]
    assert committed_capital(open_trades) == 0.01 * 1_000_000 + 2 * 100


def test_committed_capital_empty_is_zero():
    assert committed_capital([]) == 0


# --- evaluate: full decision, in priority order ---


def test_evaluate_sizes_trade_when_all_clear():
    decision = evaluate(_capital_config(), None, [], last_price=100)
    assert decision == RiskDecision(action="size", qty=10.0)  # 10% of 10000 / 100


def test_evaluate_blocks_on_circuit_breaker_before_anything_else():
    daily_pnl = {"realized_pnl": -2000, "circuit_breaker_triggered": False}
    # also over max positions and zero capital, but circuit breaker wins
    decision = evaluate(
        _capital_config(max_concurrent_positions=0), daily_pnl, [], last_price=100
    )
    assert decision.action == "block_circuit_breaker"


def test_evaluate_blocks_on_max_concurrent_positions():
    open_trades = [{"qty": 1, "entry_price": 10}, {"qty": 1, "entry_price": 10}]
    decision = evaluate(
        _capital_config(max_concurrent_positions=2), None, open_trades, last_price=100
    )
    assert decision.action == "block_max_positions"


def test_evaluate_blocks_on_capital_limit_when_committed_plus_new_exceeds_pool():
    # misconfigured: 30% per trade * 5 slots = 150% of capital_to_use
    cfg = _capital_config(position_size_pct=30, max_concurrent_positions=5)
    open_trades = [
        {"qty": 1, "entry_price": 3000},
        {"qty": 1, "entry_price": 3000},
        {"qty": 1, "entry_price": 3000},
    ]  # 9000 already committed of 10000 pool; next trade wants 3000 more
    decision = evaluate(cfg, None, open_trades, last_price=100)
    assert decision.action == "block_capital_limit"


def test_evaluate_blocks_on_invalid_price():
    decision = evaluate(_capital_config(), None, [], last_price=0)
    assert decision.action == "block_capital_limit"


def test_evaluate_qty_scales_with_price():
    decision = evaluate(_capital_config(position_size_pct=10), None, [], last_price=1000)
    assert decision.qty == 1.0  # 10% of 10000 = 1000; 1000 / 1000 price = 1.0
