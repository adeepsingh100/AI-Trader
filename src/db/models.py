"""Supabase data access. One function per read/write the agents need —
see PROJECT_SPEC.md §6 for the schema these map to."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from src.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from src.groq_client import ModelUsageEvent

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


# --- capital_config ---


def get_capital_config(mode: str) -> dict | None:
    res = get_client().table("capital_config").select("*").eq("mode", mode).execute()
    return res.data[0] if res.data else None


def upsert_capital_config(
    mode: str,
    total_capital: float,
    capital_to_use: float,
    daily_profit_target: float,
    max_daily_loss: float,
    position_size_pct: float = 10,
    max_concurrent_positions: int = 5,
) -> None:
    get_client().table("capital_config").upsert(
        {
            "mode": mode,
            "total_capital": total_capital,
            "capital_to_use": capital_to_use,
            "daily_profit_target": daily_profit_target,
            "max_daily_loss": max_daily_loss,
            "position_size_pct": position_size_pct,
            "max_concurrent_positions": max_concurrent_positions,
        },
        on_conflict="mode",
    ).execute()


# --- strategy_versions (immutable once created, see spec §3) ---


def get_latest_version() -> dict | None:
    res = (
        get_client()
        .table("strategy_versions")
        .select("*")
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def get_latest_promoted_version() -> dict | None:
    res = (
        get_client()
        .table("strategy_versions")
        .select("*")
        .eq("promoted_to_real", True)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def insert_strategy_version(
    version_number: int, prompt_text: str, params_json: dict, notes: str | None = None
) -> dict:
    res = (
        get_client()
        .table("strategy_versions")
        .insert(
            {
                "version_number": version_number,
                "prompt_text": prompt_text,
                "params_json": params_json,
                "notes": notes,
            }
        )
        .execute()
    )
    return res.data[0]


def promote_version(version_id: int) -> None:
    get_client().table("strategy_versions").update({"promoted_to_real": True}).eq(
        "id", version_id
    ).execute()


def get_all_strategy_versions() -> list[dict]:
    res = (
        get_client()
        .table("strategy_versions")
        .select("*")
        .order("version_number", desc=True)
        .execute()
    )
    return res.data


# --- trades ---


def open_trade(
    mode: str,
    version_id: int,
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    fees: float,
    reasoning_text: str,
) -> dict:
    res = (
        get_client()
        .table("trades")
        .insert(
            {
                "mode": mode,
                "version_id": version_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "entry_price": entry_price,
                "fees": fees,
                "status": "open",
                "reasoning_text": reasoning_text,
            }
        )
        .execute()
    )
    return res.data[0]


def close_trade(trade_id: int, exit_price: float, pnl: float, status: str = "closed") -> None:
    get_client().table("trades").update(
        {
            "exit_price": exit_price,
            "pnl": pnl,
            "status": status,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", trade_id).execute()


def get_open_trades(mode: str) -> list[dict]:
    res = get_client().table("trades").select("*").eq("mode", mode).eq("status", "open").execute()
    return res.data


def get_recent_trades(mode: str, limit: int = 50) -> list[dict]:
    res = (
        get_client()
        .table("trades")
        .select("*")
        .eq("mode", mode)
        .order("opened_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data


def get_closed_trades(mode: str, version_id: int) -> list[dict]:
    res = (
        get_client()
        .table("trades")
        .select("*")
        .eq("mode", mode)
        .eq("version_id", version_id)
        .in_("status", ["closed", "flattened"])
        .execute()
    )
    return res.data


# --- daily_pnl ---


def get_daily_pnl(day: Date, mode: str) -> dict | None:
    res = (
        get_client()
        .table("daily_pnl")
        .select("*")
        .eq("date", day.isoformat())
        .eq("mode", mode)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_daily_pnl(
    day: Date,
    mode: str,
    realized_pnl: float,
    trades_count: int,
    target_hit: bool,
    circuit_breaker_triggered: bool,
) -> None:
    get_client().table("daily_pnl").upsert(
        {
            "date": day.isoformat(),
            "mode": mode,
            "realized_pnl": realized_pnl,
            "trades_count": trades_count,
            "target_hit": target_hit,
            "circuit_breaker_triggered": circuit_breaker_triggered,
        },
        on_conflict="date,mode",
    ).execute()


# --- agent_logs ---


def log_agent_event(
    agent_name: str, level: str, message: str, raw_llm_response: Any = None
) -> None:
    get_client().table("agent_logs").insert(
        {
            "agent_name": agent_name,
            "level": level,
            "message": message,
            "raw_llm_response": raw_llm_response,
        }
    ).execute()


# --- model_usage ---


def log_model_usage(events: list[ModelUsageEvent]) -> None:
    if not events:
        return
    rows = [
        {
            "model_used": e.model_used,
            "fallback_reason": e.fallback_reason,
            "latency_ms": e.latency_ms,
            "success": e.success,
        }
        for e in events
    ]
    get_client().table("model_usage").insert(rows).execute()


def get_recent_model_usage(limit: int = 500) -> list[dict]:
    res = (
        get_client()
        .table("model_usage")
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data
