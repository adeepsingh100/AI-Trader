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
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    entry_slippage_pct: float | None = None,
    market_regime: str | None = None,
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
                "stop_loss_price": stop_loss_price,
                "take_profit_price": take_profit_price,
                "entry_slippage_pct": entry_slippage_pct,
                "market_regime": market_regime,
            }
        )
        .execute()
    )
    return res.data[0]


def close_trade(
    trade_id: int, exit_price: float, pnl: float, status: str = "closed", exit_reason: str | None = None
) -> None:
    get_client().table("trades").update(
        {
            "exit_price": exit_price,
            "pnl": pnl,
            "status": status,
            "exit_reason": exit_reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
        }
    ).eq("id", trade_id).execute()


def update_trade_excursion(trade_id: int, mfe_pct: float, mae_pct: float) -> None:
    get_client().table("trades").update({"mfe_pct": mfe_pct, "mae_pct": mae_pct}).eq(
        "id", trade_id
    ).execute()


def get_open_trades(mode: str) -> list[dict]:
    res = get_client().table("trades").select("*").eq("mode", mode).eq("status", "open").execute()
    return res.data


def get_recently_closed_trades(mode: str, since: datetime) -> list[dict]:
    """Closed/flattened trades since `since` — mode-scoped, not
    version-scoped (unlike get_closed_trades), since strategy versions
    rotate and the learning engine's catch-up pass needs to see across
    versions. Bounded by the caller's `since` (LEARNING_CATCHUP_LOOKBACK_HOURS
    for process_closed_trades, LEARNING_HISTORY_WINDOW_DAYS for stats
    bucket recompute) so this never becomes a full-table scan."""
    res = (
        get_client()
        .table("trades")
        .select("*")
        .eq("mode", mode)
        .in_("status", ["closed", "flattened"])
        .gte("closed_at", since.isoformat())
        .execute()
    )
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


# --- opportunity_evaluations ---


def log_opportunity_evaluation(
    mode: str,
    symbol: str,
    version_id: int,
    features: dict,
    trend_score: float | None,
    momentum_score: float | None,
    volume_score: float | None,
    volatility_score: float | None,
    risk_score: float | None,
    opportunity_score: float | None,
    llm_decision: str | None,
    llm_reasoning: str | None,
    llm_raw_response: Any,
    risk_manager_result: str | None,
    final_decision: str,
    reason: str | None,
    trade_id: int | None = None,
) -> dict:
    res = (
        get_client()
        .table("opportunity_evaluations")
        .insert(
            {
                "mode": mode,
                "symbol": symbol,
                "version_id": version_id,
                "features": features,
                "trend_score": trend_score,
                "momentum_score": momentum_score,
                "volume_score": volume_score,
                "volatility_score": volatility_score,
                "risk_score": risk_score,
                "opportunity_score": opportunity_score,
                "llm_decision": llm_decision,
                "llm_reasoning": llm_reasoning,
                "llm_raw_response": llm_raw_response,
                "risk_manager_result": risk_manager_result,
                "final_decision": final_decision,
                "reason": reason,
                "trade_id": trade_id,
            }
        )
        .execute()
    )
    return res.data[0]


# --- learning_statistics ---


def upsert_learning_statistics(mode: str, dimension_type: str, dimension_value: str, stats: dict) -> None:
    get_client().table("learning_statistics").upsert(
        {
            "mode": mode,
            "dimension_type": dimension_type,
            "dimension_value": dimension_value,
            **stats,
        },
        on_conflict="mode,dimension_type,dimension_value",
    ).execute()


def get_learning_statistics(mode: str, dimension_type: str | None = None) -> list[dict]:
    query = get_client().table("learning_statistics").select("*").eq("mode", mode)
    if dimension_type is not None:
        query = query.eq("dimension_type", dimension_type)
    return query.execute().data


# --- feature_importance ---


def upsert_feature_importance(mode: str, feature_name: str, correlation_score: float, sample_count: int) -> None:
    get_client().table("feature_importance").upsert(
        {
            "mode": mode,
            "feature_name": feature_name,
            "correlation_score": correlation_score,
            "sample_count": sample_count,
        },
        on_conflict="mode,feature_name",
    ).execute()


def get_feature_importance(mode: str) -> list[dict]:
    res = get_client().table("feature_importance").select("*").eq("mode", mode).execute()
    return res.data


def get_entry_evaluation_for_trade(trade_id: int) -> dict | None:
    res = (
        get_client()
        .table("opportunity_evaluations")
        .select("*")
        .eq("trade_id", trade_id)
        .eq("final_decision", "buy")
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


# --- confidence_calibration ---


def get_confidence_calibration_for_evaluation(opportunity_evaluation_id: int) -> dict | None:
    res = (
        get_client()
        .table("confidence_calibration")
        .select("*")
        .eq("opportunity_evaluation_id", opportunity_evaluation_id)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def log_confidence_calibration(
    opportunity_evaluation_id: int,
    ai_confidence: float | None,
    historical_confidence: float | None,
    ai_weight: float,
    historical_weight: float,
    final_confidence: float | None,
    similar_trades_count: int,
) -> None:
    get_client().table("confidence_calibration").insert(
        {
            "opportunity_evaluation_id": opportunity_evaluation_id,
            "ai_confidence": ai_confidence,
            "historical_confidence": historical_confidence,
            "ai_weight": ai_weight,
            "historical_weight": historical_weight,
            "final_confidence": final_confidence,
            "similar_trades_count": similar_trades_count,
        }
    ).execute()


# --- recommendations ---


def get_latest_recommendation(mode: str, metric_name: str) -> dict | None:
    res = (
        get_client()
        .table("recommendations")
        .select("*")
        .eq("mode", mode)
        .eq("metric_name", metric_name)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def insert_recommendation(
    mode: str,
    metric_name: str,
    current_value: float,
    recommended_value: float,
    rationale: str,
    sample_size: int,
) -> None:
    get_client().table("recommendations").insert(
        {
            "mode": mode,
            "metric_name": metric_name,
            "current_value": current_value,
            "recommended_value": recommended_value,
            "rationale": rationale,
            "sample_size": sample_size,
        }
    ).execute()


def get_recommendations(mode: str, status: str | None = None) -> list[dict]:
    query = get_client().table("recommendations").select("*").eq("mode", mode)
    if status is not None:
        query = query.eq("status", status)
    return query.order("created_at", desc=True).execute().data


# --- trade_evaluations ---


def upsert_trade_evaluation(
    trade_id: int,
    predicted_confidence: float | None,
    predicted_opportunity_score: float | None,
    actual_outcome_won: bool,
    confidence_was_accurate: bool | None,
    opportunity_score_was_accurate: bool | None,
    risk_assessment: str | None,
    stop_loss_assessment: str | None,
    target_assessment: str | None,
) -> None:
    get_client().table("trade_evaluations").upsert(
        {
            "trade_id": trade_id,
            "predicted_confidence": predicted_confidence,
            "predicted_opportunity_score": predicted_opportunity_score,
            "actual_outcome_won": actual_outcome_won,
            "confidence_was_accurate": confidence_was_accurate,
            "opportunity_score_was_accurate": opportunity_score_was_accurate,
            "risk_assessment": risk_assessment,
            "stop_loss_assessment": stop_loss_assessment,
            "target_assessment": target_assessment,
        },
        on_conflict="trade_id",
    ).execute()


def get_trade_evaluation_ids(trade_ids: list[int]) -> set[int]:
    if not trade_ids:
        return set()
    res = get_client().table("trade_evaluations").select("trade_id").in_("trade_id", trade_ids).execute()
    return {row["trade_id"] for row in res.data}


def get_trades_by_ids(trade_ids: list[int]) -> list[dict]:
    if not trade_ids:
        return []
    res = get_client().table("trades").select("*").in_("id", trade_ids).execute()
    return res.data


def get_entry_evaluations_since(mode: str, since: datetime) -> list[dict]:
    """Entry-time opportunity_evaluations rows (final_decision='buy', so
    trade_id is always set) since `since` — the candidate pool
    find_similar_trades() ranks by distance. No embedded join to `trades`
    for the outcome (pnl/closed_at) — this codebase has no precedent for
    PostgREST embeds anywhere in models.py; callers fetch outcomes
    separately via get_trades_by_ids() and match in Python, same pattern
    as process_closed_trades()'s diff."""
    res = (
        get_client()
        .table("opportunity_evaluations")
        .select("*")
        .eq("mode", mode)
        .eq("final_decision", "buy")
        .gte("timestamp", since.isoformat())
        .execute()
    )
    return res.data
