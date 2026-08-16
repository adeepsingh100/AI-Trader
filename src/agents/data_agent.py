"""Pulls the top-N INR pairs by 24h turnover with fresh multi-timeframe
candles for each — the per-cycle market snapshot the Feature Engine scores.

No orderbook fetch: nothing consumes it (PaperExecutionAgent uses a flat
slippage bps, RealExecutionAgent trades at market), so it was a dead HTTP
call — dropped per the quant-pipeline refactor's "avoid unnecessary API
calls" directive."""

from __future__ import annotations

from src.config import FEATURE_CANDLE_LIMIT, FEATURE_TIMEFRAMES
from src.coindcx_client import (
    get_candles,
    get_markets_details,
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
                "candles_by_timeframe": {
                    tf: get_candles(pair, interval=tf, limit=FEATURE_CANDLE_LIMIT)
                    for tf in FEATURE_TIMEFRAMES
                },
            }
        )
    return snapshot
