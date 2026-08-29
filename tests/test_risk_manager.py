import pytest

from src.agents.risk_manager import (
    RiskDecision,
    circuit_breaker_triggered,
    committed_capital,
    compute_net_expectancy_pct,
    evaluate,
    exit_reason,
    resolve_exit_params,
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


# --- risk-based position sizing (Phase 4): stop_loss_pct is a strict
# ADDITIONAL cap, never grows qty relative to the existing flat/dynamic
# formula. ---


def test_evaluate_omitting_stop_loss_pct_matches_pre_existing_formula():
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0)
    assert decision.qty == 10.0  # unchanged: no risk-based cap applied


def test_evaluate_stop_loss_pct_caps_qty_below_flat_formula():
    # flat formula: 10% of 10000 / 100 = 10 qty (worth 1000). RISK_PER_TRADE_PCT
    # defaults to 1.0 -> max risk = 100. stop_loss_pct=0.05 -> max qty by
    # risk = 100 / (0.05 * 100) = 20... loosen the stop to force a real cap:
    # stop_loss_pct=0.5 -> max qty by risk = 100 / (0.5*100) = 2, well below 10.
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0, stop_loss_pct=0.5)
    assert decision.action == "size"
    assert decision.qty == 2.0


def test_evaluate_stop_loss_pct_never_grows_qty_past_flat_formula():
    # a tight stop (small stop_loss_pct) implies a LARGE max qty by risk —
    # the cap must never override the flat formula upward, only min() with it.
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0, stop_loss_pct=0.001)
    assert decision.qty == 10.0  # still the flat formula's qty, not the (huge) risk cap


def test_evaluate_zero_stop_loss_pct_skips_risk_cap_like_none():
    # falsy (0) stop_loss_pct is treated the same as "not supplied" —
    # never a division by zero.
    capital_config = _capital_config(capital_to_use=10000, position_size_pct=10)
    decision = evaluate(capital_config, None, [], last_price=100.0, stop_loss_pct=0)
    assert decision.qty == 10.0


# --- Net Expectancy Gate (Phase 2): fees/GST/TDS/spread/slippage netted
# against the resolved stop/target and the calibrated win-probability
# estimate. Pure percentage math — no qty/entry_price needed. ---


def test_compute_net_expectancy_pct_positive_at_high_win_probability():
    result = compute_net_expectancy_pct(stop_loss_pct=0.02, take_profit_pct=0.04, win_probability=0.8)
    assert result is not None
    assert result["net_expectancy_pct"] > 0
    assert result["risk_reward"] == 2.0  # 0.04 / 0.02


def test_compute_net_expectancy_pct_negative_at_coin_flip_win_probability():
    # Real finding: at 2%/4% (a textbook 1:2 R/R) and CoinDCX's actual
    # round-trip cost (~2.3% of notional: 0.5% fee + 18% GST on that fee,
    # both legs, plus 1% TDS on the sell leg, plus spread/slippage), a
    # coin-flip win rate is NOT enough to break even — costs eat more than
    # half the edge a naive 1:2 R/R "expects" to have. This is exactly why
    # the gate exists: opportunity_score/confidence clearing their own
    # thresholds is not the same as the trade being worth taking.
    result = compute_net_expectancy_pct(stop_loss_pct=0.02, take_profit_pct=0.04, win_probability=0.5)
    assert result is not None
    assert result["net_expectancy_pct"] < 0


def test_compute_net_expectancy_pct_none_when_stop_or_target_missing():
    assert compute_net_expectancy_pct(None, 0.04, 0.8) is None
    assert compute_net_expectancy_pct(0.02, None, 0.8) is None
    assert compute_net_expectancy_pct(0.02, 0.04, None) is None


def test_compute_net_expectancy_pct_none_for_non_positive_legs():
    assert compute_net_expectancy_pct(0.0, 0.04, 0.8) is None
    assert compute_net_expectancy_pct(0.02, -0.01, 0.8) is None


def test_compute_net_expectancy_pct_clamps_win_probability_outside_0_1():
    # a caller passing an out-of-range probability (e.g. a raw score not
    # divided by 100) must never silently produce nonsense math.
    over = compute_net_expectancy_pct(0.02, 0.04, win_probability=1.5)
    under = compute_net_expectancy_pct(0.02, 0.04, win_probability=-0.5)
    assert over["win_probability"] == 1.0
    assert under["win_probability"] == 0.0


def test_compute_net_expectancy_pct_never_nan_or_infinite():
    for stop, target, prob in ((0.01, 0.10, 0.99), (0.10, 0.01, 0.01), (0.5, 0.5, 0.5)):
        result = compute_net_expectancy_pct(stop, target, prob)
        assert result is not None
        for key in ("net_expectancy_pct", "cost_pct", "risk_reward"):
            value = result[key]
            assert value == value  # not NaN
            assert value not in (float("inf"), float("-inf"))


# --- ATR-based stop-loss fallback (Phase 5): params_json's evidence-
# validated value always wins; ATR only fills a leg params_json omits. ---


def test_resolve_exit_params_prefers_configured_params_json():
    stop, target = resolve_exit_params({"stop_loss_pct": 0.02, "take_profit_pct": 0.04}, atr_pct=10.0)
    assert (stop, target) == (0.02, 0.04)


def test_resolve_exit_params_falls_back_to_atr_when_leg_missing():
    # atr_pct=2.0 -> stop = clamp(0.02 * 1.5, 0.01, 0.10) = 0.03,
    # target = clamp(0.02 * 3.0, 0.01, 0.10) = 0.06 (default multipliers).
    stop, target = resolve_exit_params({}, atr_pct=2.0)
    assert stop == pytest.approx(0.03)
    assert target == pytest.approx(0.06)


def test_resolve_exit_params_only_fills_the_missing_leg():
    stop, target = resolve_exit_params({"stop_loss_pct": 0.02}, atr_pct=2.0)
    assert stop == 0.02  # configured value untouched
    assert target == pytest.approx(0.06)  # ATR-derived


def test_resolve_exit_params_none_when_neither_configured_nor_atr_available():
    assert resolve_exit_params({}, atr_pct=None) == (None, None)


def test_resolve_exit_params_clamps_extreme_atr_within_sweep_range():
    # a huge atr_pct must never produce an unbounded stop/target.
    stop, target = resolve_exit_params({}, atr_pct=1000.0)
    assert stop == pytest.approx(0.10)  # EXIT_PARAM_SWEEP_MAX_PCT ceiling
    assert target == pytest.approx(0.10)
    # a tiny atr_pct must never produce a near-zero stop/target either.
    stop, target = resolve_exit_params({}, atr_pct=0.0001)
    assert stop == pytest.approx(0.01)  # EXIT_PARAM_SWEEP_MIN_PCT floor
    assert target == pytest.approx(0.01)


def test_resolve_exit_params_explicit_multipliers_override_module_globals():
    # A "swing" profile's wider multipliers (STRATEGY_PROFILES) must
    # produce a different ATR-derived stop/target than the bare default
    # call on the same atr_pct -- proves the override actually reaches
    # the calc, not just accepted-and-ignored.
    default_stop, default_target = resolve_exit_params({}, atr_pct=2.0)
    swing_stop, swing_target = resolve_exit_params(
        {}, atr_pct=2.0, stop_mult=3.0, target_mult=6.0, min_pct=0.01, max_pct=0.20
    )
    assert swing_stop == pytest.approx(0.06)
    assert swing_target == pytest.approx(0.12)
    assert swing_stop != pytest.approx(default_stop)
    assert swing_target != pytest.approx(default_target)


# --- exit_reason: stored-price fallback (Phase 5) — closes the "no stop
# configured = unbounded downside" gap without disturbing the existing
# live-recomputed-from-params_json behavior when a leg IS configured. ---


def test_exit_reason_configured_leg_ignores_stored_price():
    # params_json wins even if a stored (stale/different) price would
    # otherwise have fired — no behavior change when a leg is configured.
    assert exit_reason(100, 98, {"stop_loss_pct": 0.02}, stored_stop_loss_price=50) == "stop_loss"
    assert exit_reason(100, 99, {"stop_loss_pct": 0.02}, stored_stop_loss_price=99) is None


def test_exit_reason_stored_stop_loss_fallback_fires_when_unconfigured():
    assert exit_reason(100, 89, {}, stored_stop_loss_price=90) == "stop_loss"


def test_exit_reason_stored_take_profit_fallback_fires_when_unconfigured():
    assert exit_reason(100, 111, {}, stored_take_profit_price=110) == "take_profit"


def test_exit_reason_stored_prices_not_hit_returns_none():
    assert exit_reason(100, 95, {}, stored_stop_loss_price=90, stored_take_profit_price=110) is None


def test_exit_reason_no_stored_prices_and_no_params_json_is_none():
    # the original "unbounded downside" case: neither a configured value
    # nor a stored fallback exists — nothing to check against.
    assert exit_reason(100, 50, {}) is None
