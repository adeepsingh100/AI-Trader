"""CLI: backfills the historical_candles cache for a symbol universe and
date range (plus warm-up buffer). The only place in src/backtest/ that
makes a live network call — the hot loop (engine.py) never does.

    python -m src.backtest.ingest_data --symbols BTCINR,ETHINR \\
        --start 2024-01-01 --end 2024-06-01
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone

from src.backtest.data_provider import ingest
from src.coindcx_client import get_markets_details, symbol_to_pair
from src.config import BACKTEST_WARMUP_BUFFER_DAYS, FEATURE_TIMEFRAMES

# Rough per-row size estimate (bytes) for the confirmation-threshold check
# below — not exact, just enough to catch an accidental multi-year,
# multi-symbol 1m backfill before it eats Supabase free-tier storage that
# also hosts live trading data.
_EST_BYTES_PER_ROW = 120
_CONFIRM_THRESHOLD_ROWS = 2_000_000


def _date_to_ms(d: date) -> int:
    return int(datetime.combine(d, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)


def resolve_symbol_to_pair(symbols: list[str]) -> dict[str, str]:
    details = get_markets_details()
    return {s: symbol_to_pair(s, details) for s in symbols}


def estimate_row_count(n_symbols: int, n_days: int) -> int:
    minutes = n_days * 24 * 60
    per_symbol = minutes + (minutes // 15) + (minutes // 60) + n_days  # 1m + 15m + 1h + 1d
    return per_symbol * n_symbols


def ingest_universe(
    symbols: list[str], start: date, end: date, warmup_buffer_days: int = BACKTEST_WARMUP_BUFFER_DAYS
) -> dict[str, int]:
    symbol_to_pair_map = resolve_symbol_to_pair(symbols)
    warmup_start = start - timedelta(days=warmup_buffer_days)
    start_ms, end_ms = _date_to_ms(warmup_start), _date_to_ms(end + timedelta(days=1))

    counts = {}
    for symbol in symbols:
        pair = symbol_to_pair_map[symbol]
        for tf in FEATURE_TIMEFRAMES:
            counts[f"{symbol}:{tf}"] = ingest(pair, tf, start_ms, end_ms)
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", required=True, help="comma-separated, e.g. BTCINR,ETHINR")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--warmup-buffer-days", type=int, default=BACKTEST_WARMUP_BUFFER_DAYS)
    parser.add_argument("--yes", action="store_true", help="skip the row-count confirmation prompt")
    args = parser.parse_args()

    cli_symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    cli_start = date.fromisoformat(args.start)
    cli_end = date.fromisoformat(args.end)
    total_days = (cli_end - cli_start).days + args.warmup_buffer_days
    estimated_rows = estimate_row_count(len(cli_symbols), total_days)

    if not args.yes and estimated_rows > _CONFIRM_THRESHOLD_ROWS:
        estimated_mb = estimated_rows * _EST_BYTES_PER_ROW / 1_000_000
        confirm = input(
            f"This will ingest an estimated {estimated_rows:,} rows (~{estimated_mb:.0f} MB) into "
            f"historical_candles — a free-tier Supabase project already hosts live trading data. "
            f"Continue? [y/N] "
        )
        if confirm.strip().lower() != "y":
            print("Aborted.")
            raise SystemExit(1)

    ingested_counts = ingest_universe(cli_symbols, cli_start, cli_end, args.warmup_buffer_days)
    for key, count in ingested_counts.items():
        print(f"{key}: {count} candles")
