from unittest.mock import Mock, patch

import pytest

from src.orchestrator import run_cycle, run_risk_check


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
    base = {"id": 1, "version_number": 1, "prompt_text": "be a trader"}
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
    }
    base.update(overrides)
    return base


# --- entry: candidate selection + LLM validation + risk manager ---


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_opens_trade_on_accepted_entry_candidate(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_models.open_trade.return_value = {"id": 99}
    mock_validate.return_value = ({"decision": "accept", "reasoning": "go"}, [])

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == [{"id": 99}]
    assert result["closed"] == []
    assert result["circuit_breaker"] is False
    mock_validate.assert_called_once()
    assert mock_validate.call_args.kwargs["context"] == "entry"
    execution_agent.place_order.assert_called_once_with("BTCINR", "buy", pytest.approx(0.001), 1_000_000)
    mock_models.open_trade.assert_called_once()
    assert mock_models.open_trade.call_args.kwargs["reasoning_text"] == "go"
    mock_models.log_opportunity_evaluation.assert_called_once()
    assert mock_models.log_opportunity_evaluation.call_args.kwargs["final_decision"] == "buy"


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_skips_symbol_not_in_candidate_set(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
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
    mock_validate.assert_not_called()
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["final_decision"] == "hold"
    assert log_kwargs["reason"] == "not_a_candidate"
    assert log_kwargs["llm_decision"] is None


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_does_not_buy_when_llm_rejects_entry(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_validate.return_value = ({"decision": "reject", "reasoning": "too extended"}, [])

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["llm_decision"] == "reject"
    assert log_kwargs["reason"] == "too extended"


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_llm_accepts_but_risk_manager_blocks_max_positions(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
):
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
    mock_validate.return_value = ({"decision": "accept", "reasoning": "go"}, [])

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["risk_manager_result"] == "block_max_positions"
    assert log_kwargs["final_decision"] == "hold"


# --- exit: held position + score below threshold + LLM validation ---


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_closes_held_position_when_llm_accepts_exit(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
):
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_score.return_value = _scores(20)  # below EXIT_SCORE_THRESHOLD (40)
    mock_select.return_value = []
    mock_validate.return_value = ({"decision": "accept", "reasoning": "deteriorated"}, [])

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 999_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    assert mock_validate.call_args.kwargs["context"] == "exit"
    execution_agent.place_order.assert_called_once_with("BTCINR", "sell", 0.001, 1_000_000)
    mock_models.close_trade.assert_called_once()
    assert result["closed"] == [held]
    assert result["opened"] == []


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_keeps_held_position_when_llm_rejects_exit(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
):
    mock_ticker.return_value = []
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_score.return_value = _scores(20)
    mock_select.return_value = []
    mock_validate.return_value = ({"decision": "reject", "reasoning": "still worth holding"}, [])

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_not_called()
    assert result["closed"] == []


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_no_exit_validation_when_score_above_threshold(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
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

    mock_validate.assert_not_called()
    assert result["closed"] == []
    log_kwargs = mock_models.log_opportunity_evaluation.call_args.kwargs
    assert log_kwargs["reason"] == "score_above_exit_threshold_or_unavailable"


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_no_exit_validation_when_score_unavailable(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
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

    mock_validate.assert_not_called()
    assert result["closed"] == []


# --- stop-loss/take-profit sweep still runs first, unaffected by scoring ---


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_stop_loss_sweep_closes_before_scoring_pass_considers_it(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
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
    assert result["closed"] == [held]
    # the sweep already closed it — the scoring pass must not also validate an exit for it
    mock_validate.assert_not_called()


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_sweep_leaves_position_alone_when_no_leg_hit(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
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


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_flattens_and_skips_when_breaker_already_triggered(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
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
    mock_validate.assert_not_called()
    assert result == {"opened": [], "closed": [], "circuit_breaker": True}


@patch("src.orchestrator.get_ticker")
@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_trips_breaker_mid_cycle_and_stops_further_processing(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate, mock_ticker
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
    mock_validate.return_value = ({"decision": "accept", "reasoning": "cut loss"}, [])

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 199_000, "fees": 0}

    result = run_cycle(execution_agent=execution_agent)

    # the ETHINR exit alone realizes a 1000 loss, past max_daily_loss=100
    assert result["circuit_breaker"] is True
    assert result["closed"] == [held]
    assert result["opened"] == []
    execution_agent.flatten_all.assert_called_once_with("paper")
    # BTCINR's entry validation must never have been attempted
    assert mock_validate.call_count == 1
    buy_calls = [c for c in execution_agent.place_order.call_args_list if c.args[1] == "buy"]
    assert buy_calls == []


# --- every scanned symbol gets logged, regardless of outcome ---


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_logs_every_scanned_symbol_even_without_an_llm_call(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
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
    mock_validate.assert_not_called()


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


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.get_market_snapshot")
@patch("src.orchestrator.models")
def test_run_cycle_paper_paused_skips_without_touching_scoring(
    mock_models, mock_snapshot, mock_validate
):
    mock_models.get_capital_config.return_value = _capital_config(paused=True)

    result = run_cycle(mode="paper")

    assert result == {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
    mock_models.get_latest_version.assert_not_called()
    mock_snapshot.assert_not_called()
    mock_validate.assert_not_called()


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.get_market_snapshot")
@patch("src.orchestrator.models")
def test_run_cycle_real_paused_skips_without_touching_scoring(
    mock_models, mock_snapshot, mock_validate
):
    mock_models.get_capital_config.return_value = _capital_config(paused=True)

    result = run_cycle(mode="real")

    assert result == {"opened": [], "closed": [], "circuit_breaker": False, "skipped": "paused"}
    mock_models.get_latest_promoted_version.assert_not_called()
    mock_snapshot.assert_not_called()
    mock_validate.assert_not_called()


@patch("src.orchestrator.validate_opportunity")
@patch("src.orchestrator.select_top_candidates")
@patch("src.orchestrator.score_opportunity")
@patch("src.orchestrator.models")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_real_mode_uses_promoted_version_not_latest(
    mock_snapshot, mock_models, mock_score, mock_select, mock_validate
):
    # latest overall version is #7 (unvetted paper draft), but only #3 is
    # promoted — real mode must validate against #3's prompt, not #7's
    promoted_version = {"id": 3, "version_number": 3, "prompt_text": "promoted prompt"}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_promoted_version.return_value = promoted_version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_score.return_value = _scores(85)
    mock_select.return_value = [{"symbol": "BTCINR"}]
    mock_validate.return_value = ({"decision": "accept", "reasoning": "go"}, [])
    mock_models.open_trade.return_value = {"id": 1}

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_000, "fees": 1.0}

    run_cycle(mode="real", execution_agent=execution_agent)

    assert mock_validate.call_args.args[1] == "promoted prompt"
    assert mock_models.open_trade.call_args.kwargs["version_id"] == 3
    mock_models.get_latest_version.assert_not_called()


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
