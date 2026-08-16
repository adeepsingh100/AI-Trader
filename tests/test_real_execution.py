from unittest.mock import Mock, patch

import pytest

from src.agents.execution.real import RealExecutionAgent, _round_qty, _wait_for_fill


def _markets_details():
    return [
        {"symbol": "BTCINR", "step": 0.00001, "target_currency_precision": 5},
        {"symbol": "ETHINR", "step": 0.001, "target_currency_precision": 3},
    ]


# --- _round_qty ---


def test_round_qty_truncates_to_exchange_step():
    # 0.0053516... at step 0.001 -> truncate to 0.005 (not round-up to 0.006)
    assert _round_qty("ETHINR", 0.0053516, _markets_details()) == 0.005


def test_round_qty_below_one_step_becomes_zero():
    assert _round_qty("ETHINR", 0.0004, _markets_details()) == 0.0


def test_round_qty_unknown_symbol_raises():
    with pytest.raises(ValueError, match="unknown symbol"):
        _round_qty("DOGEINR", 1.0, _markets_details())


# --- _wait_for_fill ---


@patch("src.agents.execution.real.time.sleep")
@patch("src.agents.execution.real.get_order_status")
def test_wait_for_fill_polls_until_filled(mock_status, mock_sleep):
    mock_status.side_effect = [
        {"status": "open"},
        {"status": "partially_filled"},
        {"status": "filled", "avg_price": "100.5", "fee_amount": "0.5"},
    ]
    fill = _wait_for_fill("order-1")
    assert fill["status"] == "filled"
    assert mock_status.call_count == 3


@patch("src.agents.execution.real.time.sleep")
@patch("src.agents.execution.real.get_order_status")
def test_wait_for_fill_raises_immediately_on_rejected(mock_status, mock_sleep):
    mock_status.return_value = {"status": "rejected"}
    with pytest.raises(RuntimeError, match="rejected"):
        _wait_for_fill("order-1")
    assert mock_status.call_count == 1  # no pointless polling after a terminal failure


@patch("src.agents.execution.real.time.sleep")
@patch("src.agents.execution.real.get_order_status")
def test_wait_for_fill_times_out(mock_status, mock_sleep):
    mock_status.return_value = {"status": "open"}
    with pytest.raises(RuntimeError, match="did not fill"):
        _wait_for_fill("order-1")


# --- RealExecutionAgent.place_order ---


@patch("src.agents.execution.real.get_order_status")
@patch("src.agents.execution.real.create_order")
@patch("src.agents.execution.real.get_balances")
@patch("src.agents.execution.real.get_markets_details")
def test_place_order_buy_rounds_qty_and_fills(
    mock_details, mock_balances, mock_create, mock_status
):
    mock_details.return_value = _markets_details()
    mock_balances.return_value = [{"currency": "INR", "balance": 10000}]
    mock_create.return_value = {"id": "order-1", "status": "open"}
    mock_status.return_value = {"status": "filled", "avg_price": "186900", "fee_amount": "1.2"}

    agent = RealExecutionAgent()
    fill = agent.place_order("ETHINR", "buy", 0.0053516, price=186900)

    assert mock_create.call_args.kwargs["total_quantity"] == 0.005
    # buy: fee_amount passed through as-is, no TDS added
    assert fill == {"fill_price": 186900.0, "fees": 1.2}


@patch("src.agents.execution.real.get_balances")
@patch("src.agents.execution.real.get_markets_details")
def test_place_order_buy_blocked_on_insufficient_balance(mock_details, mock_balances):
    mock_details.return_value = _markets_details()
    mock_balances.return_value = [{"currency": "INR", "balance": 10}]

    agent = RealExecutionAgent()
    with pytest.raises(RuntimeError, match="insufficient INR balance"):
        agent.place_order("ETHINR", "buy", 0.01, price=186900)


@patch("src.agents.execution.real.get_markets_details")
def test_place_order_raises_when_rounded_qty_is_zero(mock_details):
    mock_details.return_value = _markets_details()
    agent = RealExecutionAgent()
    with pytest.raises(RuntimeError, match="below exchange step size"):
        agent.place_order("ETHINR", "buy", 0.0001, price=186900)


@patch("src.agents.execution.real.get_order_status")
@patch("src.agents.execution.real.create_order")
@patch("src.agents.execution.real.get_markets_details")
def test_place_order_sell_skips_balance_check(mock_details, mock_create, mock_status):
    mock_details.return_value = _markets_details()
    mock_create.return_value = {"id": "order-2", "status": "open"}
    mock_status.return_value = {"status": "filled", "avg_price": "186000", "fee_amount": "0.9"}

    agent = RealExecutionAgent()
    fill = agent.place_order("ETHINR", "sell", 0.005, price=186000)

    # sell: fee_amount + 1% TDS on notional (186000 * 0.005 * 0.01 = 9.3),
    # since TDS isn't documented as part of fee_amount
    assert fill == {"fill_price": 186000.0, "fees": pytest.approx(0.9 + 9.3)}


# --- flatten_all ---


@patch("src.agents.execution.real.RealExecutionAgent.place_order")
@patch("src.agents.execution.real.models")
def test_flatten_all_closes_every_open_trade(mock_models, mock_place_order):
    held = {"id": 1, "symbol": "ETHINR", "qty": 0.005, "entry_price": 186000, "fees": 1.0}
    mock_models.get_open_trades.return_value = [held]
    mock_place_order.return_value = {"fill_price": 185000, "fees": 0.9}

    agent = RealExecutionAgent()
    closed = agent.flatten_all("real")

    mock_place_order.assert_called_once_with("ETHINR", "sell", 0.005, price=186000)
    mock_models.close_trade.assert_called_once()
    assert mock_models.close_trade.call_args.kwargs["exit_reason"] == "circuit_breaker"
    assert closed == [held]
