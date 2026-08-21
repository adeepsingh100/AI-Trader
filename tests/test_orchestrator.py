from unittest.mock import Mock, patch

import pytest

from src.orchestrator import _bucket_modifier, _recent_performance_modifier, run_cycle, run_risk_check


def _capital_config(**overrides):
    base = {
        "mode": "paper",
        "capital_to_use": 10000,
        "daily_profit_target": 500,
        "max_daily_loss": 1000,
        "position_size_pct": 10,
        "max_concurrent_positions": 2,
    }
    base.update(overrides)
    return base


def _version(**overrides):
    # stop_loss_pct/take_profit_pct present by default so the Net
    # Expectancy Gate (risk_manager.compute_net_expectancy_pct) has a
    # computable stop/target without needing real ATR data from
    # candles_by_timeframe (most fixtures leave that empty) — matches this
    # test suite's existing default of "a trade opens unless the test is
    # specifically about something blocking it."
    base = {
        "id": 1,
        "version_number": 1,
        "prompt_text": "be a trader",
        "params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
    }
    base.update(overrides)
    return base


def _market(symbol="BTCINR", price=1_000_000):
    return {
        "symbol": symbol,
        "pair": f"I-{symbol}",
        "last_price": price,
        "turnover_inr": 1,
        "candles_by_timeframe": {},
    }


def _scores(opportunity_score, **overrides):
    base = {
        "trend_score": opportunity_score,
        "momentum_score": opportunity_score,
        "volume_score": opportunity_score,
        "volatility_score": opportunity_score,
        "risk_score": opportunity_score,
        "opportunity_score": opportunity_score,
        "market_regime": "strong_bull",
    }
    base.update(overrides)
    return base


def _empty_similar():
    return {
        "trades": [],
        "count": 0,
        "win_rate": None,
        "avg_profit_pct": None,
        "avg_loss_pct": None,
        "avg_holding_time_seconds": None,
    }


def _permissive_calibration():
    # final_confidence=None -> the MIN_FINAL_CONFIDENCE gate treats "no
    # signal" as passing, matching every pre-existing accept-path test's
    # expectations without needing to reason about confidence at all.
    return {
        "final_confidence": None,
        "ai_weight_used": 0.0,
        "historical_weight_used": 0.0,
        "regime_modifier": None,
        "symbol_modifier": None,
        "recent_performance_modifier": None,
    }


# --- entry: candidate selection + LLM validation + risk manager ---


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_opens_trade_on_accepted_entry_candidate(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_models.open_trade.return_value = {"id": 99}
    mock_models.log_opportunity_evaluation.return_value = {"id": 501}
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = _permissive_calibration()

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == [{"id": 99}]
    assert result["closed"] == []
    assert result["circuit_breaker"] is False
    execution_agent.place_order.assert_called_once_with("BTCINR", "buy", pytest.approx(0.001), 1_000_000)
    mock_models.open_trade.assert_called_once()
    assert "quant score 85.0" in mock_models.open_trade.call_args.kwargs["reasoning_text"]
    assert mock_models.open_trade.call_args.kwargs["market_regime"] == "strong_bull"
    mock_models.log_opportunity_evaluation.assert_called_once()
    eval_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert eval_kwargs["final_decision"] == "buy"
    assert eval_kwargs["trade_id"] == 99
    mock_models.log_confidence_calibration.assert_called_once()
    assert mock_models.log_confidence_calibration.call_args.kwargs["opportunity_evaluation_id"] == 501
    mock_process.assert_called_once_with("paper")


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_net_expectancy_gate_blocks_entry_despite_high_score(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    # opportunity_score/confidence both clear their own gates, but the
    # configured stop/target is a bad enough risk/reward (a 1% target
    # against a 10% stop) that even a WIN loses money net of fees/GST/TDS/
    # spread/slippage — net expectancy is negative regardless of win
    # probability, so no order should ever be placed.
    mock_models.get_capital_config.return_value = _capital_config()
    version = _version()
    version["params_json"] = {"stop_loss_pct": 0.10, "take_profit_pct": 0.01}
    mock_models.get_latest_version.return_value = version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = _permissive_calibration()

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    mock_models.open_trade.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["final_decision"] == "hold"
    assert "net_expectancy gated" in log_kwargs["reason"]


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_skips_symbol_not_in_candidate_set(
    mock_snapshot, mock_models, mock_score, mock_select, mock_process
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(30)  # below min_score, never a candidate
    mock_select.return_value = []  # scorer's own filtering agreed: not a candidate

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["final_decision"] == "hold"
    assert log_kwargs["reason"] == "not_a_candidate"
    assert log_kwargs["llm_decision"] is None
    assert log_kwargs["trade_id"] is None


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
@patch("src.orchestrator.get_ticker")
def test_run_cycle_llm_accepts_but_risk_manager_blocks_max_positions(
    mock_ticker, mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    # unrelated open positions at flat price (0% change) so the stop-loss/
    # take-profit sweep leaves them alone — this test is about the
    # max_positions block on a NEW entry candidate, not the sweep.
    mock_ticker.return_value = [{"market": "ETHINR", "last_price": 10}, {"market": "SOLINR", "last_price": 10}]
    mock_models.get_capital_config.return_value = _capital_config(max_concurrent_positions=2)
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [
        {"id": 1, "symbol": "ETHINR", "qty": 1, "entry_price": 10},
        {"id": 2, "symbol": "SOLINR", "qty": 1, "entry_price": 10},
    ]
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = _permissive_calibration()

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["risk_manager_result"] == "block_max_positions"
    assert log_kwargs["final_decision"] == "hold"


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_confidence_gate_blocks_despite_llm_accept(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = {
        "final_confidence": 30.0,
        "ai_weight_used": 1.0,
        "historical_weight_used": 0.0,
        "regime_modifier": None,
        "symbol_modifier": None,
        "recent_performance_modifier": None,
    }

    execution_agent = Mock()
    with patch("src.orchestrator.MIN_FINAL_CONFIDENCE", 50.0):
        result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["llm_decision"] == "accept"
    assert log_kwargs["final_decision"] == "hold"
    assert "confidence gated" in log_kwargs["reason"]


# --- exit: held position + score below threshold + LLM validation ---


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_closes_held_position_when_score_drops_below_exit_threshold(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_models.log_opportunity_evaluation.return_value = {"id": 1}
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_score.return_value = _scores(20)  # below EXIT_SCORE_THRESHOLD (40)
    mock_select.return_value = []

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 999_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_called_once_with("BTCINR", "sell", 0.001, 1_000_000)
    mock_models.close_trade.assert_called_once()
    assert mock_models.close_trade.call_args.kwargs["exit_reason"] == "ai_exit"
    assert result["closed"] == [held]
    assert result["opened"] == []
    eval_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert eval_kwargs["trade_id"] == 5


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_no_exit_when_score_above_threshold(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_score.return_value = _scores(75)  # above EXIT_SCORE_THRESHOLD (40)
    mock_select.return_value = []

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["closed"] == []
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["reason"] == "score_above_exit_threshold_or_unavailable"


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_no_exit_when_score_unavailable(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    # opportunity_score None (insufficient candle history) must not crash
    # a `None < EXIT_SCORE_THRESHOLD` comparison
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_score.return_value = _scores(None)
    mock_select.return_value = []

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["closed"] == []


# --- stop-loss/take-profit sweep still runs first, unaffected by scoring ---


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_stop_loss_sweep_closes_before_scoring_pass_considers_it(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 1_000_000, "fees": 0}
    mock_ticker.return_value = [{"market": "BTCINR", "last_price": 970_000}]
    mock_models.get_capital_config.return_value = _capital_config()
    version = _version()
    version["params_json"] = {"stop_loss_pct": 0.02}
    mock_models.get_latest_version.return_value = version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=970_000)]
    mock_score.return_value = _scores(85)
    mock_select.return_value = []  # already closed by the sweep, not a candidate either way

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 970_000, "fees": 0}

    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_called_once_with("BTCINR", "sell", 0.001, 970_000)
    mock_models.close_trade.assert_called_once()
    assert mock_models.close_trade.call_args.kwargs["exit_reason"] == "stop_loss"
    assert result["closed"] == [held]
    # the sweep already closed it — the scoring pass must not also try to exit it again
    execution_agent.place_order.assert_called_once()


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_sweep_leaves_position_alone_when_no_leg_hit(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 1_000_000, "fees": 0}
    mock_ticker.return_value = [{"market": "BTCINR", "last_price": 995_000}]  # -0.5%, under 2% SL
    mock_models.get_capital_config.return_value = _capital_config()
    version = _version()
    version["params_json"] = {"stop_loss_pct": 0.02}
    mock_models.get_latest_version.return_value = version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=995_000)]
    mock_score.return_value = _scores(75)  # above exit threshold -> stays held
    mock_select.return_value = []

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_not_called()
    assert result["closed"] == []


# --- circuit breaker ---


@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_flattens_and_skips_when_breaker_already_triggered(
    mock_snapshot, mock_models, mock_score, mock_select
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = {
        "realized_pnl": -2000,
        "trades_count": 3,
        "circuit_breaker_triggered": True,
    }

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    execution_agent.flatten_all.assert_called_once_with("paper")
    mock_snapshot.assert_not_called()
    assert result == {"opened": [], "closed": [], "circuit_breaker": True}


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_trips_breaker_mid_cycle_and_stops_further_processing(
    mock_snapshot, mock_models, mock_score, mock_select, mock_ticker, mock_process
):
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "ETHINR", "qty": 1, "entry_price": 200_000, "fees": 0}
    cfg = _capital_config(max_daily_loss=100)
    mock_models.get_capital_config.return_value = cfg
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]

    losing_exit = _market(symbol="ETHINR", price=199_000)
    next_candidate = _market(symbol="BTCINR", price=1_000_000)
    mock_snapshot.return_value = [losing_exit, next_candidate]

    mock_score.return_value = _scores(20)  # below exit threshold for the held ETHINR
    mock_select.return_value = [{"symbol": "BTCINR"}]

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 199_000, "fees": 0}

    result = run_cycle(execution_agent=execution_agent)

    # the ETHINR exit alone realizes a 1000 loss, past max_daily_loss=100
    assert result["circuit_breaker"] is True
    assert result["closed"] == [held]
    assert result["opened"] == []
    execution_agent.flatten_all.assert_called_once_with("paper")
    # BTCINR's entry must never have been attempted — only ETHINR got processed
    assert mock_models.log_opportunity_evaluation.call_count == 1
    buy_calls = [c for c in execution_agent.place_order.call_args_list if c.args[1] == "buy"]
    assert buy_calls == []


# --- every scanned symbol gets logged, regardless of outcome ---


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_logs_every_scanned_symbol(
    mock_snapshot, mock_models, mock_score, mock_select, mock_process
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market("BTCINR"), _market("ETHINR")]
    mock_score.return_value = _scores(30)  # below min_score, no candidates
    mock_select.return_value = []

    execution_agent = Mock()
    run_cycle(execution_agent=execution_agent)

    assert mock_models.log_opportunity_evaluation.call_count == 2
    mock_process.assert_called_once_with("paper")


# --- per-symbol fault isolation (Resilience, PROJECT_SPEC.md §3d) ---


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_one_symbol_exception_does_not_abort_remaining_symbols(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    """A confirmed real gap before this fix: one symbol's exception used
    to crash the whole cycle, so every remaining symbol never got
    processed even though the cycle is otherwise safe to retry from a
    clean DB-read state."""
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market("BTCINR"), _market("ETHINR")]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}, {"symbol": "ETHINR"}]
    mock_models.open_trade.return_value = {"id": 99}
    mock_models.log_opportunity_evaluation.return_value = {"id": 501}
    mock_calibrate.return_value = _permissive_calibration()
    # First symbol's similarity lookup blows up; second symbol must still
    # be processed normally.
    mock_similar.side_effect = [RuntimeError("boom"), _empty_similar()]

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    # Second symbol (ETHINR) still opened a trade despite the first
    # symbol's exception.
    assert result["opened"] == [{"id": 99}]
    mock_models.log_agent_event.assert_any_call(
        "orchestrator", "error", "BTCINR: RuntimeError: boom"
    )
    # process_closed_trades still runs at the end — the cycle completed,
    # it didn't abort.
    mock_process.assert_called_once_with("paper")


# --- real mode / paused / promoted-version routing (unaffected by scoring) ---


@patch("src.orchestrator.models")
def test_run_cycle_defaults_to_real_execution_agent_when_promoted(mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_promoted_version.return_value = {
        "id": 9,
        "version_number": 5,
        "prompt_text": "p",
    }
    mock_models.get_daily_pnl.return_value = {
        "realized_pnl": 0,
        "trades_count": 0,
        "circuit_breaker_triggered": True,  # short-circuit before any real order attempt
    }

    with patch("src.orchestrator.RealExecutionAgent") as mock_real_agent_cls:
        result = run_cycle(mode="real")

    mock_real_agent_cls.assert_called_once_with()
    mock_real_agent_cls.return_value.flatten_all.assert_called_once_with("real")
    assert result["circuit_breaker"] is True


@patch("src.orchestrator.models")
def test_run_cycle_real_mode_noop_when_nothing_promoted(mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_promoted_version.return_value = None

    result = run_cycle(mode="real")

    assert result == {
        "opened": [],
        "closed": [],
        "circuit_breaker": False,
        "skipped": "no_promoted_version",
    }
    mock_models.log_agent_event.assert_called_once()


@patch("src.orchestrator.models")
def test_run_cycle_real_mode_noop_when_no_capital_config(mock_models):
    mock_models.get_capital_config.return_value = None

    result = run_cycle(mode="real")

    assert result == {
        "opened": [],
        "closed": [],
        "circuit_breaker": False,
        "skipped": "no_capital_config",
    }
    mock_models.get_latest_promoted_version.assert_not_called()


@patch("src.orchestrator.get_market_snapshot")
@patch("src.orchestrator.models")
def test_run_cycle_paper_paused_skips_without_touching_scoring(
    mock_models, mock_snapshot
):
    mock_models.get_capital_config.return_value = _capital_config(paused=True)

    result = run_cycle(mode="paper")

    assert result == {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
    mock_models.get_latest_version.assert_not_called()
    mock_snapshot.assert_not_called()


@patch("src.orchestrator.get_market_snapshot")
@patch("src.orchestrator.models")
def test_run_cycle_real_paused_skips_without_touching_scoring(
    mock_models, mock_snapshot
):
    mock_models.get_capital_config.return_value = _capital_config(paused=True)

    result = run_cycle(mode="real")

    assert result == {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
    mock_models.get_latest_promoted_version.assert_not_called()
    mock_snapshot.assert_not_called()


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_real_mode_uses_promoted_version_not_latest(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    # latest overall version is #7 (unvetted paper draft), but only #3 is
    # promoted — real mode must open trades against #3, not #7
    promoted_version = {
        "id": 3,
        "version_number": 3,
        "prompt_text": "promoted prompt",
        "params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
    }
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_promoted_version.return_value = promoted_version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = _permissive_calibration()
    mock_models.open_trade.return_value = {"id": 1}

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_000, "fees": 1.0}

    run_cycle(mode="real", execution_agent=execution_agent)

    assert mock_models.open_trade.call_args.kwargs["version_id"] == 3
    mock_models.get_latest_version.assert_not_called()
    mock_process.assert_called_once_with("real")


# --- run_risk_check: unaffected by this refactor (no LLM, no scoring at all) ---


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.models")
def test_run_risk_check_closes_stop_loss_hit_without_touching_llm(mock_models, mock_ticker):
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 1_000_000, "fees": 0}
    mock_ticker.return_value = [{"market": "BTCINR", "last_price": 970_000}]
    mock_models.get_capital_config.return_value = _capital_config()
    version = _version()
    version["params_json"] = {"stop_loss_pct": 0.02}
    mock_models.get_latest_version.return_value = version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 970_000, "fees": 0}

    result = run_risk_check(execution_agent=execution_agent)

    execution_agent.place_order.assert_called_once_with("BTCINR", "sell", 0.001, 970_000)
    assert result["closed"] == [held]


@patch("src.orchestrator.models")
def test_run_risk_check_skips_when_paused_or_unconfigured(mock_models):
    mock_models.get_capital_config.return_value = None
    result = run_risk_check()
    assert result == {"closed": [], "circuit_breaker": False, "skipped": "not_configured_or_paused"}

    mock_models.get_capital_config.return_value = _capital_config(paused=True)
    result = run_risk_check()
    assert result == {"closed": [], "circuit_breaker": False, "skipped": "not_configured_or_paused"}


@patch("src.orchestrator.models")
def test_run_risk_check_flattens_when_breaker_already_tripped(mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = {
        "realized_pnl": -2000,
        "trades_count": 3,
        "circuit_breaker_triggered": True,
    }

    execution_agent = Mock()
    result = run_risk_check(execution_agent=execution_agent)

    execution_agent.flatten_all.assert_called_once_with("paper")
    assert result == {"closed": [], "circuit_breaker": True}


# --- MFE/MAE excursion tracking (via the shared sweep, both cadences) ---


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.models")
def test_run_risk_check_updates_excursion_for_open_trades(mock_models, mock_ticker):
    held = {
        "id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 1_000_000, "fees": 0,
        "mfe_pct": 0, "mae_pct": 0,
    }
    mock_ticker.return_value = [{"market": "BTCINR", "last_price": 1_010_000}]  # +1% favorable
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]

    run_risk_check(execution_agent=Mock())

    mock_models.update_trade_excursion.assert_called_once_with(5, pytest.approx(1.0), 0.0)


# --- adaptive confidence chain: _bucket_modifier / _recent_performance_modifier ---


@patch("src.orchestrator.BUCKET_MODIFIER_SENSITIVITY", 20)
@patch("src.orchestrator.BUCKET_MODIFIER_CAP", 10)
@patch("src.orchestrator.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
def test_bucket_modifier_scales_with_win_rate_edge_over_baseline():
    stats = {"strong_bull": {"win_rate": 0.7, "trades_count": 20}}
    assert _bucket_modifier(stats, "strong_bull", overall_win_rate=0.5) == pytest.approx(4.0)


@patch("src.orchestrator.BUCKET_MODIFIER_SENSITIVITY", 20)
@patch("src.orchestrator.BUCKET_MODIFIER_CAP", 10)
@patch("src.orchestrator.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
def test_bucket_modifier_clamped_to_cap():
    stats = {"strong_bull": {"win_rate": 0.9, "trades_count": 20}}
    assert _bucket_modifier(stats, "strong_bull", overall_win_rate=0.1) == pytest.approx(10.0)


@patch("src.orchestrator.RECOMMENDATION_MIN_SAMPLE_SIZE", 20)
def test_bucket_modifier_none_below_sample_floor():
    stats = {"strong_bull": {"win_rate": 0.9, "trades_count": 5}}
    assert _bucket_modifier(stats, "strong_bull", overall_win_rate=0.1) is None


def test_bucket_modifier_none_when_value_or_baseline_missing():
    assert _bucket_modifier({}, None, overall_win_rate=0.5) is None
    assert _bucket_modifier({}, "strong_bull", overall_win_rate=None) is None
    assert _bucket_modifier({}, "unknown_regime", overall_win_rate=0.5) is None


@patch("src.orchestrator.RECENT_STREAK_WIN_MODIFIER_CAP", 8)
@patch("src.orchestrator.RECENT_STREAK_LOSS_MODIFIER_CAP", 16)
@patch("src.orchestrator.RECENT_PERFORMANCE_LOOKBACK_TRADES", 4)
@patch("src.orchestrator.models")
def test_recent_performance_modifier_full_winning_streak_hits_cap(mock_models):
    trades = [
        {"pnl": 10, "closed_at": f"2026-01-0{i}T00:00:00Z"} for i in range(1, 5)
    ]
    mock_models.get_recently_closed_trades.return_value = trades
    assert _recent_performance_modifier("paper") == pytest.approx(8.0)


@patch("src.orchestrator.RECENT_STREAK_WIN_MODIFIER_CAP", 8)
@patch("src.orchestrator.RECENT_STREAK_LOSS_MODIFIER_CAP", 16)
@patch("src.orchestrator.RECENT_PERFORMANCE_LOOKBACK_TRADES", 4)
@patch("src.orchestrator.models")
def test_recent_performance_modifier_partial_losing_streak_scales_below_cap(mock_models):
    trades = [
        {"pnl": 10, "closed_at": "2026-01-01T00:00:00Z"},
        {"pnl": 10, "closed_at": "2026-01-02T00:00:00Z"},
        {"pnl": -10, "closed_at": "2026-01-03T00:00:00Z"},
        {"pnl": -10, "closed_at": "2026-01-04T00:00:00Z"},
    ]
    mock_models.get_recently_closed_trades.return_value = trades
    # current streak = 2 losses out of a 4-trade lookback -> half the cap
    assert _recent_performance_modifier("paper") == pytest.approx(-8.0)


@patch("src.orchestrator.models")
def test_recent_performance_modifier_none_when_no_recent_trades(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    assert _recent_performance_modifier("paper") is None


@patch("src.orchestrator.process_closed_trades")
@patch("src.orchestrator.calibrate_confidence")
@patch("src.orchestrator.find_similar_trades")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_passes_regime_and_symbol_modifiers_into_calibration(
    mock_snapshot, mock_models, mock_score, mock_select, mock_similar, mock_calibrate, mock_process
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_models.open_trade.return_value = {"id": 99}
    mock_models.log_opportunity_evaluation.return_value = {"id": 501}
    mock_similar.return_value = _empty_similar()
    mock_calibrate.return_value = _permissive_calibration()

    def _learning_stats(mode, dimension_type=None):
        if dimension_type == "market_regime":
            return [{"dimension_value": "strong_bull", "win_rate": 0.8, "trades_count": 50}]
        if dimension_type == "symbol":
            return [{"dimension_value": "BTCINR", "win_rate": 0.7, "trades_count": 50}]
        if dimension_type == "strategy_version":
            return [{"dimension_value": "1", "win_rate": 0.5, "trades_count": 100}]
        return []

    mock_models.get_learning_statistics.side_effect = _learning_stats
    mock_models.get_recently_closed_trades.return_value = []

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_500, "fees": 1.0}

    run_cycle(execution_agent=execution_agent)

    calibrate_kwargs = mock_calibrate.call_args.kwargs
    # default BUCKET_MODIFIER_SENSITIVITY=20 -> (0.8-0.5)*20=6.0, (0.7-0.5)*20=4.0
    assert calibrate_kwargs["regime_modifier"] == pytest.approx(6.0)
    assert calibrate_kwargs["symbol_modifier"] == pytest.approx(4.0)
