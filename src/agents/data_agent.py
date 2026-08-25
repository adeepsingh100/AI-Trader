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
from src.data_quality.repair import DataRepairEngine
from src.data_quality.validator import MarketDataValidator
from src.resilience import log_fail_open

_validator = MarketDataValidator()
_repairer = DataRepairEngine()


def _validated_candles(pair: str, interval: str) -> list[dict]:
    """Fetch -> validate -> repair, in that order, before anything reaches
    the Feature Engine (Market Data Quality Engine + Data Repair Engine,
    PROJECT_SPEC.md §3d). Every issue found is logged, whether or not it
    was repairable — a DB write failure here degrades to "log skipped",
    never blocks the trading cycle (data quality logging is advisory, the
    circuit breaker/risk manager are the actual safety gates)."""
    # CoinDCX returns candles DESCENDING by time (see feature_engine.py's
    # module docstring) — the validator's order/gap/spike checks assume
    # ascending, so sort here first or every fetch floods out_of_order.
    raw = sorted(get_candles(pair, interval=interval, limit=FEATURE_CANDLE_LIMIT), key=lambda c: c["time"])
    report = _validator.validate(raw, pair, interval, expected_pair=pair, live_fetch=True)
    repaired, repair_log = _repairer.repair(report.usable_candles, report, pair, interval)

    if report.issues:
        from src.db import models

        try:
            models.insert_data_quality_issues(
                [
                    {
                        "pair": pair,
                        "interval": interval,
                        "source": "live",
                        "issue_type": i.issue_type,
                        "severity": i.severity,
                        "detail": i.detail,
                        "repaired": any(r.candle_time == i.candle_time for r in repair_log),
                        "candle_time": i.candle_time,
                    }
                    for i in report.issues
                ]
            )
        except Exception as e:
            log_fail_open("data_agent.data_quality_log", e)

    return repaired


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
                    tf: _validated_candles(pair, tf) for tf in FEATURE_TIMEFRAMES
                },
            }
        )
    return snapshot
