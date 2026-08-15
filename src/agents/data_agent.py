"""Pulls the top-N INR pairs by 24h turnover with fresh orderbook/candles
for each — the per-cycle market snapshot the Signal Agent scores."""

from __future__ import annotations

from src.coindcx_client import (
    get_candles,
    get_markets_details,
    get_orderbook,
    symbol_to_pair,
    top_inr_pairs_by_turnover,
)


def get_market_snapshot(n: int = 10) -> list[dict]:
    details = get_markets_details()
    top = top_inr_pairs_by_turnover(n)

    snapshot = []
    for t in top:
        pair = symbol_to_pair(t["market"], details)
        snapshot.append(
            {
                "symbol": t["market"],
                "pair": pair,
                "last_price": float(t["last_price"]),
                "turnover_inr": t["turnover_inr"],
                "orderbook": get_orderbook(pair),
                "candles": get_candles(pair, interval="1m", limit=20),
            }
        )
    return snapshot
