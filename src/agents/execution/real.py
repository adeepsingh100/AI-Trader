"""Live CoinDCX orders. Only ever instantiated by the orchestrator when
a strategy version is actually promoted_to_real (src/orchestrator.py) —
this class alone is not the gate, don't call it directly outside that
path. Market orders only: spot has no shorting, so a "sell" always
closes a symbol already held (see src/orchestrator.py's spot-only note).

Auth (HMAC signing) and the balances endpoint are live-verified against
the real account. create_order/get_order_status field names come from
CoinDCX's published API docs, not a live fill — the account had ~₹0.91
balance at build time, under CoinDCX's ₹100 min_notional, so no order
could be placed to confirm the response shape end-to-end. Confirm with
one small real order once funds exist and before relying on this beyond
the promotion gate.

Fees: `fee_amount` on the order response is taken as CoinDCX's actual
trading-fee charge (assumed GST-inclusive, since that's a statutory
add-on to the fee itself). The 1% TDS on a sell (Income Tax Act s.194S)
is NOT documented anywhere on the order response, so it's added here
explicitly rather than assumed included — UNVERIFIED, same as the fill
shape above. Confirm the actual INR credited on a real sell matches
`fill_price * qty - fee_amount - tds` before trusting this number.
"""

from __future__ import annotations

import time

from src.agents.execution.base import ExecutionAgent
from src.coindcx_client import create_order, get_balances, get_markets_details, get_order_status
from src.db import models

FILL_POLL_INTERVAL_SECONDS = 1.0
FILL_POLL_ATTEMPTS = 10
SELL_TDS_PCT = 1


def _inr_balance() -> float:
    for b in get_balances():
        if b.get("currency") == "INR":
            return float(b["balance"])
    return 0.0


def _round_qty(symbol: str, qty: float, markets_details: list[dict]) -> float:
    for m in markets_details:
        if m["symbol"] == symbol:
            step = m["step"]
            steps = int(qty / step)
            return round(steps * step, m["target_currency_precision"])
    raise ValueError(f"unknown symbol: {symbol}")


def _extract_order(response: dict | list) -> dict:
    # Docs show create_order returning the order object directly; guard
    # against a batch-style {"orders": [...]} or bare-list response too,
    # since that shape shows up in some CoinDCX API versions.
    if isinstance(response, list):
        return response[0]
    if "orders" in response:
        return response["orders"][0]
    return response


def _wait_for_fill(order_id: str) -> dict:
    for _ in range(FILL_POLL_ATTEMPTS):
        status = get_order_status(order_id)
        if status["status"] == "filled":
            return status
        if status["status"] in ("cancelled", "rejected"):
            raise RuntimeError(f"order {order_id} ended as {status['status']}: {status}")
        time.sleep(FILL_POLL_INTERVAL_SECONDS)
    raise RuntimeError(
        f"order {order_id} did not fill within {FILL_POLL_ATTEMPTS * FILL_POLL_INTERVAL_SECONDS}s"
    )


class RealExecutionAgent(ExecutionAgent):
    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        markets_details = get_markets_details()
        qty = _round_qty(symbol, qty, markets_details)
        if qty <= 0:
            raise RuntimeError(f"rounded qty is zero for {symbol} — below exchange step size")

        if side == "buy":
            available = _inr_balance()
            estimated_cost = qty * price
            if estimated_cost > available:
                raise RuntimeError(
                    f"insufficient INR balance for {symbol}: "
                    f"need ~{estimated_cost:.2f}, have {available:.2f}"
                )

        order = _extract_order(create_order(market=symbol, side=side, total_quantity=qty))
        fill = _wait_for_fill(order["id"])
        fill_price = float(fill["avg_price"])
        fees = float(fill["fee_amount"])
        if side == "sell":
            fees += fill_price * qty * (SELL_TDS_PCT / 100)
        return {"fill_price": fill_price, "fees": fees}

    def flatten_all(self, mode: str) -> list[dict]:
        closed = []
        for trade in models.get_open_trades(mode):
            fill = self.place_order(trade["symbol"], "sell", trade["qty"], price=trade["entry_price"])
            pnl = (fill["fill_price"] - trade["entry_price"]) * trade["qty"] - fill[
                "fees"
            ] - trade["fees"]
            models.close_trade(trade["id"], fill["fill_price"], pnl, status="flattened")
            closed.append(trade)
        return closed
