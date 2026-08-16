from __future__ import annotations

from src.agents.execution.base import ExecutionAgent

# CoinDCX spot: 0.5% trading fee on trade value (both sides), +18% GST on
# that fee. Sells additionally carry 1% TDS (Income Tax Act s.194S) on
# trade value — a separate tax deduction, not something GST applies to,
# and not charged on buys (no "transfer" of the asset on acquisition).
TRADING_FEE_PCT = 0.5
GST_PCT_ON_FEE = 18
SELL_TDS_PCT = 1
SLIPPAGE_BPS = 5


def _fees(notional: float, side: str) -> float:
    trading_fee = notional * (TRADING_FEE_PCT / 100)
    total = trading_fee + trading_fee * (GST_PCT_ON_FEE / 100)
    if side == "sell":
        total += notional * (SELL_TDS_PCT / 100)
    return total


class PaperExecutionAgent(ExecutionAgent):
    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        slip = price * (SLIPPAGE_BPS / 10_000)
        fill_price = price + slip if side == "buy" else price - slip
        fees = _fees(fill_price * qty, side)
        return {"fill_price": fill_price, "fees": fees}

    def flatten_all(self, mode: str) -> list[dict]:
        from src.coindcx_client import get_ticker
        from src.db import models

        last_price = {t["market"]: float(t["last_price"]) for t in get_ticker()}

        closed = []
        for trade in models.get_open_trades(mode):
            price = last_price.get(trade["symbol"])
            if price is None:
                continue
            fill = self.place_order(trade["symbol"], "sell", trade["qty"], price)
            pnl = (fill["fill_price"] - trade["entry_price"]) * trade["qty"] - fill[
                "fees"
            ] - trade["fees"]
            models.close_trade(
                trade["id"], fill["fill_price"], pnl, status="flattened", exit_reason="circuit_breaker"
            )
            closed.append(trade)
        return closed
