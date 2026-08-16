from src.agents.risk_manager import (
    RiskDecision,
    circuit_breaker_triggered,
    committed_capital,
    evaluate,
    exit_reason,
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


# --- exit_reason: stop-loss / take-profit, decimal-fraction params_json ---


def test_exit_reason_none_when_neither_leg_configured():
    assert exit_reason(100, 90, {}) is None


def test_exit_reason_stop_loss_hit():
    assert exit_reason(100, 98, {"stop_loss_pct": 0.02}) == "stop_loss"


def test_exit_reason_stop_loss_not_yet_hit():
    assert exit_reason(100, 99, {"stop_loss_pct": 0.02}) is None


def test_exit_reason_take_profit_hit():
    assert exit_reason(100, 103, {"take_profit_pct": 0.03}) == "take_profit"


def test_exit_reason_take_profit_not_yet_hit():
    assert exit_reason(100, 102, {"take_profit_pct": 0.03}) is None


def test_exit_reason_ignores_missing_leg():
    # only take_profit_pct configured — a big drop isn't a stop-loss exit
    assert exit_reason(100, 50, {"take_profit_pct": 0.03}) is None


def test_exit_reason_none_for_invalid_prices():
    assert exit_reason(0, 100, {"stop_loss_pct": 0.02}) is None
    assert exit_reason(100, 0, {"stop_loss_pct": 0.02}) is None


# --- sizing_mode + Portfolio Intelligence integration (PROJECT_SPEC.md §3d) ---
# The single most important regression here: every EXISTING call shape
# (no new kwargs supplied) must be byte-identical to pre-§3d behavior —
# capital_config.sizing_mode defaults to 'flat' in the DB migration, and
# these new kwargs default to None, so nothing changes unless a caller
# opts in.


def test_evaluate_flat_sizing_mode_matches_pre_existing_formula():
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0)
    assert decision.action == "size"
    assert decision.qty == (10000 * 0.10) / 100.0  # unchanged flat formula


def test_evaluate_missing_sizing_mode_key_defaults_to_flat_behavior():
    # capital_config dicts from before this migration ran have no
    # 'sizing_mode' key at all — .get() must default to flat, not crash.
    capital_config = _capital_config()
    assert "sizing_mode" not in capital_config
    decision = evaluate(capital_config, None, [], last_price=100.0)
    assert decision.action == "size"


def test_evaluate_dynamic_sizing_mode_uses_capital_allocation_engine():
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10, sizing_mode="dynamic")
    base_qty = (10000 * 0.10) / 100.0

    # Every factor maxed favorably (confidence=100, win_rate=1.0) should
    # size UP relative to the flat formula.
    decision = evaluate(
        capital_config, None, [], last_price=100.0,
        sizing_context={"confidence": 100.0, "strategy_win_rate": 1.0, "market_regime": "strong_bull"},
    )
    assert decision.action == "size"
    assert decision.qty > base_qty


def test_evaluate_dynamic_sizing_still_respects_committed_capital_ceiling():
    # The dynamic multiplier must feed the SAME capital ceiling check as
    # flat sizing, never bypass it — a maxed-out multiplier on an
    # already-near-full book must still block, not oversize past the cap.
    capital_config = _capital_config(capital_to_use=1000, position_size_pct=90, sizing_mode="dynamic")
    open_trades = [{"qty": 1, "entry_price": 950}]  # 950 already committed of 1000
    decision = evaluate(
        capital_config, None, open_trades, last_price=100.0,
        sizing_context={"confidence": 100.0, "strategy_win_rate": 1.0, "market_regime": "strong_bull"},
    )
    assert decision.action == "block_capital_limit"


def test_evaluate_concentration_gate_blocks_oversized_single_position():
    # position_size_pct=50 of a 10,000 pool -> a 5,000 candidate, 50% of
    # total equity — a 20-slot book's fair share is 5% (cap 12.5% at the
    # default 2.5x multiple), so this is a genuinely oversized single bet,
    # not just "the only position that happens to exist" (that case is
    # covered by the small-book test below and must NOT block).
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=50, max_concurrent_positions=20)
    decision = evaluate(
        capital_config, None, [], last_price=100.0,
        symbol="BTCINR", portfolio_positions=[], price_history={},
    )
    assert decision.action == "block_concentration_limit"


def test_evaluate_concentration_gate_allows_first_position_in_small_book():
    # max_concurrent_positions=2 -> equal share 50%, cap = 2x that = 100% —
    # the very first trade in a small book must NOT be blocked (the real
    # bug an integration test caught: a fixed institutional-style
    # percentage cap blocked nearly every first trade in this bot's
    # actual 2-5-slot range).
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10, max_concurrent_positions=2)
    decision = evaluate(
        capital_config, None, [], last_price=100.0,
        symbol="BTCINR", portfolio_positions=[], price_history={},
    )
    assert decision.action == "size"


def test_evaluate_omitting_portfolio_context_skips_concentration_gate():
    # portfolio_positions/price_history default to None — an existing
    # caller that never passes them gets no concentration check at all,
    # not a crash and not a spurious block.
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0, symbol="BTCINR")
    assert decision.action == "size"
