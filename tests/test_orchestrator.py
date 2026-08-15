from unittest.mock import Mock, patch

import pytest

from src.orchestrator import run_cycle


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


def _version():
    return {"id": 1, "version_number": 1, "prompt_text": "be a trader"}


def _market(symbol="BTCINR", price=1_000_000):
    return {
        "symbol": symbol,
        "pair": f"I-{symbol}",
        "last_price": price,
        "turnover_inr": 1,
        "orderbook": {},
        "candles": [],
    }


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_opens_trade_on_buy_signal(mock_snapshot, mock_get_signal, mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_get_signal.return_value = (
        {"direction": "buy", "confidence": 0.9, "reasoning": "go"},
        [],
    )
    mock_models.open_trade.return_value = {"id": 99}

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == [{"id": 99}]
    assert result["closed"] == []
    assert result["circuit_breaker"] is False
    execution_agent.place_order.assert_called_once_with("BTCINR", "buy", pytest.approx(0.001), 1_000_000)
    mock_models.open_trade.assert_called_once()
    mock_models.log_agent_event.assert_called_once()


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_skips_flat_signal(mock_snapshot, mock_get_signal, mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_get_signal.return_value = (
        {"direction": "flat", "confidence": 0.1, "reasoning": "unclear"},
        [],
    )

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_stops_at_max_concurrent_positions(mock_snapshot, mock_get_signal, mock_models):
    mock_models.get_capital_config.return_value = _capital_config(max_concurrent_positions=2)
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [
        {"id": 1, "symbol": "ETHINR", "qty": 1, "entry_price": 10},
        {"id": 2, "symbol": "SOLINR", "qty": 1, "entry_price": 10},
    ]
    mock_snapshot.return_value = [_market()]
    mock_get_signal.return_value = (
        {"direction": "buy", "confidence": 0.9, "reasoning": "go"},
        [],
    )

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    assert result["opened"] == []
    execution_agent.place_order.assert_not_called()


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_sell_signal_closes_held_position(mock_snapshot, mock_get_signal, mock_models):
    held = {"id": 5, "symbol": "BTCINR", "qty": 0.001, "entry_price": 990_000, "fees": 1.0}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]
    mock_snapshot.return_value = [_market(price=1_000_000)]
    mock_get_signal.return_value = (
        {"direction": "sell", "confidence": 0.9, "reasoning": "take profit"},
        [],
    )

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 999_500, "fees": 1.0}

    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_called_once_with("BTCINR", "sell", 0.001, 1_000_000)
    mock_models.close_trade.assert_called_once()
    mock_models.upsert_daily_pnl.assert_called_once()
    assert result["closed"] == [held]
    assert result["opened"] == []


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_sell_signal_with_nothing_held_is_noop(mock_snapshot, mock_get_signal, mock_models):
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_get_signal.return_value = (
        {"direction": "sell", "confidence": 0.9, "reasoning": "no position anyway"},
        [],
    )

    execution_agent = Mock()
    result = run_cycle(execution_agent=execution_agent)

    execution_agent.place_order.assert_not_called()
    assert result == {"opened": [], "closed": [], "circuit_breaker": False}


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_flattens_and_skips_when_breaker_already_triggered(
    mock_snapshot, mock_get_signal, mock_models
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
    mock_get_signal.assert_not_called()
    assert result == {"opened": [], "closed": [], "circuit_breaker": True}


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_trips_breaker_mid_cycle_and_stops_further_buys(
    mock_snapshot, mock_get_signal, mock_models
):
    held = {"id": 5, "symbol": "ETHINR", "qty": 1, "entry_price": 200_000, "fees": 0}
    cfg = _capital_config(max_daily_loss=100)
    mock_models.get_capital_config.return_value = cfg
    mock_models.get_latest_version.return_value = _version()
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = [held]

    losing_close = _market(symbol="ETHINR", price=199_000)
    next_buy_candidate = _market(symbol="BTCINR", price=1_000_000)
    mock_snapshot.return_value = [losing_close, next_buy_candidate]

    def signal_side_effect(market, _prompt):
        if market["symbol"] == "ETHINR":
            return {"direction": "sell", "confidence": 0.9, "reasoning": "cut loss"}, []
        return {"direction": "buy", "confidence": 0.9, "reasoning": "go"}, []

    mock_get_signal.side_effect = signal_side_effect

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 199_000, "fees": 0}

    result = run_cycle(execution_agent=execution_agent)

    # the ETHINR close alone realizes a 1000 loss, past max_daily_loss=100
    assert result["circuit_breaker"] is True
    assert result["closed"] == [held]
    assert result["opened"] == []
    execution_agent.flatten_all.assert_called_once_with("paper")
    # BTCINR buy must never have been attempted
    buy_calls = [c for c in execution_agent.place_order.call_args_list if c.args[1] == "buy"]
    assert buy_calls == []


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
    # real is expected to sit unconfigured for weeks while paper trading
    # earns promotion — this must no-op quietly, not fail every cron tick.
    mock_models.get_capital_config.return_value = None

    result = run_cycle(mode="real")

    assert result == {
        "opened": [],
        "closed": [],
        "circuit_breaker": False,
        "skipped": "no_capital_config",
    }
    mock_models.get_latest_promoted_version.assert_not_called()


@patch("src.orchestrator.models")
@patch("src.orchestrator.get_signal")
@patch("src.orchestrator.get_market_snapshot")
def test_run_cycle_real_mode_uses_promoted_version_not_latest(
    mock_snapshot, mock_get_signal, mock_models
):
    # latest overall version is #7 (unvetted paper draft), but only #3 is
    # promoted — real mode must trade #3, not #7
    promoted_version = {"id": 3, "version_number": 3, "prompt_text": "promoted prompt"}
    mock_models.get_capital_config.return_value = _capital_config()
    mock_models.get_latest_promoted_version.return_value = promoted_version
    mock_models.get_daily_pnl.return_value = None
    mock_models.get_open_trades.return_value = []
    mock_snapshot.return_value = [_market()]
    mock_get_signal.return_value = (
        {"direction": "buy", "confidence": 0.9, "reasoning": "go"},
        [],
    )
    mock_models.open_trade.return_value = {"id": 1}

    execution_agent = Mock()
    execution_agent.place_order.return_value = {"fill_price": 1_000_000, "fees": 1.0}

    run_cycle(mode="real", execution_agent=execution_agent)

    mock_get_signal.assert_called_once_with(_market(), "promoted prompt")
    assert mock_models.open_trade.call_args.kwargs["version_id"] == 3
    mock_models.get_latest_version.assert_not_called()
