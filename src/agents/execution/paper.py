from __future__ import annotations

from src.agents.execution.base import ExecutionAgent

TAKER_FEE_PCT = 0.1
SLIPPAGE_BPS = 5


class PaperExecutionAgent(ExecutionAgent):
    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        slip = price * (SLIPPAGE_BPS / 10_000)
        fill_price = price + slip if side == "buy" else price - slip
        fees = fill_price * qty * (TAKER_FEE_PCT / 100)
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
            models.close_trade(trade["id"], fill["fill_price"], pnl, status="flattened")
            closed.append(trade)
        return closed
