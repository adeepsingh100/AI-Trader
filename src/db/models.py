"""Supabase data access. One function per read/write the agents need —
see PROJECT_SPEC.md §6 for the schema these map to."""

from __future__ import annotations

from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from src.config import SUPABASE_SERVICE_KEY, SUPABASE_URL
from src.groq_client import ModelUsageEvent
from src.resilience import retry_with_backoff

_client: Client | None = None


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


def ping() -> None:
    """Trivial reachability check — used by monitoring/diagnostics.py's
    database health check, which needs no real data, just confirmation the
    connection works."""
    get_client().table("capital_config").select("mode").limit(1).execute()


def _execute(builder):
    """Retries a read/upsert query builder on a transient Supabase error —
    safe here because select/upsert are naturally idempotent (a repeat
    doesn't create a second row). Plain .insert() calls are NOT routed
    through this: a retry after a request whose response was lost but
    which actually succeeded server-side would insert a duplicate row
    (a trade, a log line) — a real correctness risk this codebase's own
    "never fabricate/duplicate financial state" ethos rules out. Applied at
    the handful of call sites that gate whether a cycle can start at all
    (capital_config/version/daily_pnl/open_trades reads) rather than
    mechanically retrofitted across all 55 functions in this file."""
    return retry_with_backoff(builder.execute)


# --- capital_config ---


def get_capital_config(mode: str) -> dict | None:
    res = _execute(get_client().table("capital_config").select("*").eq("mode", mode))
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
    # Excludes suspended versions (Strategy Health Engine, PROJECT_SPEC.md
    # §3d) — without this filter, auto-suspension would be a silent no-op
    # since this is still an unfiltered "newest row" query otherwise.
    res = _execute(
        get_client()
        .table("strategy_versions")
        .select("*")
        .neq("status", "suspended")
        .order("version_number", desc=True)
        .limit(1)
    )
    return res.data[0] if res.data else None


def get_latest_promoted_version() -> dict | None:
    res = _execute(
        get_client()
        .table("strategy_versions")
        .select("*")
        .eq("promoted_to_real", True)
        .neq("status", "suspended")
        .order("version_number", desc=True)
        .limit(1)
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


def set_strategy_version_promotion_eligible(version_id: int, eligible: bool) -> None:
    """Code sets this flag only — never promoted_to_real itself (Scientific
    Strategy Optimization Framework). A human reviews eligible rows in
    Supabase and flips promoted_to_real themselves."""
    get_client().table("strategy_versions").update({"promotion_eligible": eligible}).eq(
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
    res = _execute(get_client().table("trades").select("*").eq("mode", mode).eq("status", "open"))
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
    res = _execute(
        get_client()
        .table("daily_pnl")
        .select("*")
        .eq("date", day.isoformat())
        .eq("mode", mode)
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
    market_regime: str | None = None,
    config_version: str | None = None,
) -> dict:
    """market_regime/config_version (Audit System, PROJECT_SPEC.md §3d) —
    the two fields Step 9's decision-trail needed that weren't already
    columns here; everything else in the audit spec (timestamp/component/
    input/decision/output/reason/strategy-version/confidence/trade-id) was
    already captured by this table plus confidence_calibration/trades, so
    src.audit.trail reads those three tables rather than adding a new
    write path."""
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
                "market_regime": market_regime,
                "config_version": config_version,
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


def upsert_feature_importance(
    mode: str, feature_name: str, correlation_score: float, sample_count: int, timeframe: str
) -> None:
    get_client().table("feature_importance").upsert(
        {
            "mode": mode,
            "feature_name": feature_name,
            "correlation_score": correlation_score,
            "sample_count": sample_count,
            "timeframe": timeframe,
        },
        on_conflict="mode,feature_name,timeframe",
    ).execute()


def get_feature_importance(mode: str, timeframe: str | None = None) -> list[dict]:
    query = get_client().table("feature_importance").select("*").eq("mode", mode)
    if timeframe is not None:
        query = query.eq("timeframe", timeframe)
    return query.execute().data


def get_opportunity_evaluations_for_trail(
    mode: str,
    trade_id: int | None = None,
    symbol: str | None = None,
    since=None,
) -> list[dict]:
    """Chronological rows for src.audit.trail.get_decision_trail — the one
    query that module needs, routed through here (rather than a raw
    get_client().table() call in audit/trail.py) so every DB access in
    src/ goes through this file, per this repo's own convention."""
    query = get_client().table("opportunity_evaluations").select("*").eq("mode", mode)
    if trade_id is not None:
        query = query.eq("trade_id", trade_id)
    if symbol is not None:
        query = query.eq("symbol", symbol)
    if since is not None:
        query = query.gte("timestamp", since.isoformat())
    return query.order("timestamp").execute().data


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
    regime_modifier: float | None = None,
    symbol_modifier: float | None = None,
    recent_performance_modifier: float | None = None,
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
            "regime_modifier": regime_modifier,
            "symbol_modifier": symbol_modifier,
            "recent_performance_modifier": recent_performance_modifier,
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
    category: str = "threshold",
    confidence: float | None = None,
    evidence: dict | None = None,
    batch_id: str | None = None,
) -> None:
    get_client().table("recommendations").insert(
        {
            "mode": mode,
            "metric_name": metric_name,
            "current_value": current_value,
            "recommended_value": recommended_value,
            "rationale": rationale,
            "sample_size": sample_size,
            "category": category,
            "confidence": confidence,
            "evidence": evidence,
            "batch_id": batch_id,
        }
    ).execute()


def get_recommendations(
    mode: str, status: str | None = None, category: str | None = None
) -> list[dict]:
    query = get_client().table("recommendations").select("*").eq("mode", mode)
    if status is not None:
        query = query.eq("status", status)
    if category is not None:
        query = query.eq("category", category)
    return query.order("created_at", desc=True).execute().data


# --- strategy_simulations ---


def insert_strategy_simulation(
    recommendation_batch_id: str | None,
    mode: str,
    train_window_start: datetime,
    train_window_end: datetime,
    test_window_start: datetime,
    test_window_end: datetime,
    baseline_metrics: dict | None,
    candidate_metrics: dict | None,
    p_value: float | None,
    passed: bool,
    research_note: str | None = None,
    validation_detail: dict | None = None,
) -> dict:
    """research_note/validation_detail (Scientific Strategy Optimization
    Framework) — a narrative Observation/Weakness/Hypothesis/Simulation/
    Walk Forward/Decision report and the raw numbers behind it (bootstrap
    CI, walk-forward folds, strategy-comparison result where run)."""
    res = (
        get_client()
        .table("strategy_simulations")
        .insert(
            {
                "recommendation_batch_id": recommendation_batch_id,
                "mode": mode,
                "train_window_start": train_window_start.isoformat(),
                "train_window_end": train_window_end.isoformat(),
                "test_window_start": test_window_start.isoformat(),
                "test_window_end": test_window_end.isoformat(),
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "p_value": p_value,
                "passed": passed,
                "research_note": research_note,
                "validation_detail": validation_detail,
            }
        )
        .execute()
    )
    return res.data[0]


def get_strategy_simulations(mode: str, passed: bool | None = None) -> list[dict]:
    query = get_client().table("strategy_simulations").select("*").eq("mode", mode)
    if passed is not None:
        query = query.eq("passed", passed)
    return query.order("created_at", desc=True).execute().data


# --- adaptive_strategy_versions ---


def insert_adaptive_strategy_version(
    mode: str,
    version_number: int,
    params_json: dict,
    source_recommendation_batch_id: str | None,
    source_simulation_id: int | None,
    notes: str | None = None,
    fitness_score: float | None = None,
) -> dict:
    res = (
        get_client()
        .table("adaptive_strategy_versions")
        .insert(
            {
                "mode": mode,
                "version_number": version_number,
                "params_json": params_json,
                "source_recommendation_batch_id": source_recommendation_batch_id,
                "source_simulation_id": source_simulation_id,
                "notes": notes,
                "fitness_score": fitness_score,
            }
        )
        .execute()
    )
    return res.data[0]


def get_adaptive_strategy_versions(mode: str, status: str | None = None) -> list[dict]:
    query = get_client().table("adaptive_strategy_versions").select("*").eq("mode", mode)
    if status is not None:
        query = query.eq("status", status)
    return query.order("created_at", desc=True).execute().data


def get_latest_adaptive_strategy_version(mode: str) -> dict | None:
    res = (
        get_client()
        .table("adaptive_strategy_versions")
        .select("*")
        .eq("mode", mode)
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


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


def get_trade_evaluations(trade_ids: list[int]) -> list[dict]:
    """Full trade_evaluations rows (confidence_was_accurate/
    opportunity_score_was_accurate) for drift_detection.py — distinct from
    get_trade_evaluation_ids, which only returns the id set for the
    already-evaluated membership check process_closed_trades() needs."""
    if not trade_ids:
        return []
    res = get_client().table("trade_evaluations").select("*").in_("trade_id", trade_ids).execute()
    return res.data


def get_trades_by_ids(trade_ids: list[int]) -> list[dict]:
    if not trade_ids:
        return []
    res = get_client().table("trades").select("*").in_("id", trade_ids).execute()
    return res.data


# --- historical_candles ---


def upsert_historical_candles(pair: str, interval: str, candles: list[dict]) -> None:
    """candles: list of {"time","open","high","low","close","volume"} dicts,
    CoinDCX's own raw shape — caller passes them through unchanged."""
    if not candles:
        return
    rows = [
        {
            "pair": pair,
            "interval": interval,
            "time": c["time"],
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["volume"],
        }
        for c in candles
    ]
    get_client().table("historical_candles").upsert(rows, on_conflict="pair,interval,time").execute()


def get_historical_candles(pair: str, interval: str, start_time_ms: int, end_time_ms: int) -> list[dict]:
    res = (
        get_client()
        .table("historical_candles")
        .select("*")
        .eq("pair", pair)
        .eq("interval", interval)
        .gte("time", start_time_ms)
        .lte("time", end_time_ms)
        .order("time")
        .execute()
    )
    return res.data


# --- backtest_runs ---


def insert_backtest_run(
    symbols: list[str],
    start_date: Date,
    end_date: Date,
    warmup_buffer_days: int,
    starting_capital: float,
    params_json: dict,
    use_llm_signal_agent: bool = False,
    source_adaptive_strategy_version_id: int | None = None,
    name: str | None = None,
) -> dict:
    res = (
        get_client()
        .table("backtest_runs")
        .insert(
            {
                "name": name,
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "warmup_buffer_days": warmup_buffer_days,
                "starting_capital": starting_capital,
                "params_json": params_json,
                "use_llm_signal_agent": use_llm_signal_agent,
                "source_adaptive_strategy_version_id": source_adaptive_strategy_version_id,
            }
        )
        .execute()
    )
    return res.data[0]


def update_backtest_run_status(run_id: int, status: str, completed_at: datetime | None = None) -> None:
    update = {"status": status}
    if completed_at is not None:
        update["completed_at"] = completed_at.isoformat()
    get_client().table("backtest_runs").update(update).eq("id", run_id).execute()


def get_backtest_run(run_id: int) -> dict | None:
    res = get_client().table("backtest_runs").select("*").eq("id", run_id).execute()
    return res.data[0] if res.data else None


def get_backtest_runs(status: str | None = None) -> list[dict]:
    query = get_client().table("backtest_runs").select("*")
    if status is not None:
        query = query.eq("status", status)
    return query.order("created_at", desc=True).execute().data


# --- backtest_trades ---


def insert_backtest_trade(run_id: int, trade: dict) -> dict:
    res = get_client().table("backtest_trades").insert({"run_id": run_id, **trade}).execute()
    return res.data[0]


def get_backtest_trades(run_id: int) -> list[dict]:
    res = (
        get_client()
        .table("backtest_trades")
        .select("*")
        .eq("run_id", run_id)
        .order("entry_time")
        .execute()
    )
    return res.data


# --- backtest_portfolio_snapshots ---


def insert_backtest_portfolio_snapshots(run_id: int, snapshots: list[dict]) -> None:
    """Batch insert — a multi-month equity curve is thousands of points,
    one-row-per-network-call would be needlessly slow."""
    if not snapshots:
        return
    rows = [{"run_id": run_id, **s} for s in snapshots]
    get_client().table("backtest_portfolio_snapshots").insert(rows).execute()


def get_backtest_portfolio_snapshots(run_id: int) -> list[dict]:
    res = (
        get_client()
        .table("backtest_portfolio_snapshots")
        .select("*")
        .eq("run_id", run_id)
        .order("snapshot_time")
        .execute()
    )
    return res.data


# --- backtest_execution_history ---


def insert_backtest_execution_events(run_id: int, events: list[dict]) -> None:
    if not events:
        return
    rows = [{"run_id": run_id, **e} for e in events]
    get_client().table("backtest_execution_history").insert(rows).execute()


def get_backtest_execution_history(run_id: int) -> list[dict]:
    res = (
        get_client()
        .table("backtest_execution_history")
        .select("*")
        .eq("run_id", run_id)
        .order("event_time")
        .execute()
    )
    return res.data


# --- backtest_performance_metrics ---


def insert_backtest_performance_metrics(run_id: int, metrics: dict) -> dict:
    res = (
        get_client()
        .table("backtest_performance_metrics")
        .insert({"run_id": run_id, "metrics": metrics})
        .execute()
    )
    return res.data[0]


def get_backtest_performance_metrics(run_id: int) -> dict | None:
    res = (
        get_client()
        .table("backtest_performance_metrics")
        .select("*")
        .eq("run_id", run_id)
        .execute()
    )
    return res.data[0] if res.data else None


# --- backtest_walk_forward_folds ---


def insert_backtest_walk_forward_fold(run_id: int, fold: dict) -> dict:
    res = (
        get_client()
        .table("backtest_walk_forward_folds")
        .insert({"run_id": run_id, **fold})
        .execute()
    )
    return res.data[0]


def get_backtest_walk_forward_folds(run_id: int) -> list[dict]:
    res = (
        get_client()
        .table("backtest_walk_forward_folds")
        .select("*")
        .eq("run_id", run_id)
        .order("fold_number")
        .execute()
    )
    return res.data


# --- backtest_strategy_comparisons ---


def insert_backtest_strategy_comparison(
    run_id_a: int,
    run_id_b: int,
    metrics_a: dict | None,
    metrics_b: dict | None,
    p_values: dict | None,
    winner: str | None,
    promotion_recommended: bool | None,
) -> dict:
    res = (
        get_client()
        .table("backtest_strategy_comparisons")
        .insert(
            {
                "run_id_a": run_id_a,
                "run_id_b": run_id_b,
                "metrics_a": metrics_a,
                "metrics_b": metrics_b,
                "p_values": p_values,
                "winner": winner,
                "promotion_recommended": promotion_recommended,
            }
        )
        .execute()
    )
    return res.data[0]


def get_backtest_strategy_comparisons() -> list[dict]:
    res = (
        get_client()
        .table("backtest_strategy_comparisons")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
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


def get_hold_evaluations_since(mode: str, since: datetime) -> list[dict]:
    """Non-trade opportunity_evaluations rows (final_decision='hold') since
    `since` — every scanned-but-not-traded candidate, each carrying its own
    reason/risk_manager_result (Root Cause Analysis, Scientific Strategy
    Optimization Framework). Same shape as get_entry_evaluations_since,
    filtering the opposite final_decision value."""
    res = (
        get_client()
        .table("opportunity_evaluations")
        .select("*")
        .eq("mode", mode)
        .eq("final_decision", "hold")
        .gte("timestamp", since.isoformat())
        .execute()
    )
    return res.data


# --- data_quality_log (Market Data Quality Engine + Data Repair Engine,
# PROJECT_SPEC.md §3d) ---


def insert_data_quality_issues(rows: list[dict]) -> None:
    if not rows:
        return
    get_client().table("data_quality_log").insert(rows).execute()


def get_data_quality_log(pair: str | None = None, source: str | None = None, limit: int = 200) -> list[dict]:
    query = get_client().table("data_quality_log").select("*")
    if pair is not None:
        query = query.eq("pair", pair)
    if source is not None:
        query = query.eq("source", source)
    res = query.order("created_at", desc=True).limit(limit).execute()
    return res.data


# --- drift_alerts (Feature Drift Detection, PROJECT_SPEC.md §3d) ---


def insert_drift_alert(
    component: str,
    drift_type: str,
    severity: str,
    baseline_value: float | None,
    recent_value: float | None,
    detail: dict | None = None,
) -> dict:
    res = (
        get_client()
        .table("drift_alerts")
        .insert(
            {
                "component": component,
                "drift_type": drift_type,
                "severity": severity,
                "baseline_value": baseline_value,
                "recent_value": recent_value,
                "detail": detail or {},
            }
        )
        .execute()
    )
    return res.data[0]


def get_drift_alerts(component: str | None = None, limit: int = 200) -> list[dict]:
    query = get_client().table("drift_alerts").select("*")
    if component is not None:
        query = query.eq("component", component)
    res = query.order("detected_at", desc=True).limit(limit).execute()
    return res.data


# --- strategy_health_scores (Strategy Health Engine, PROJECT_SPEC.md §3d) ---


def insert_strategy_health_score(
    strategy_version_id: int, health_score: float | None, tier: str, breakdown: dict
) -> dict:
    res = (
        get_client()
        .table("strategy_health_scores")
        .insert(
            {
                "strategy_version_id": strategy_version_id,
                "health_score": health_score,
                "tier": tier,
                "breakdown": breakdown,
            }
        )
        .execute()
    )
    return res.data[0]


def get_latest_strategy_health_score(strategy_version_id: int) -> dict | None:
    res = (
        get_client()
        .table("strategy_health_scores")
        .select("*")
        .eq("strategy_version_id", strategy_version_id)
        .order("computed_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def update_strategy_version_status(version_id: int, status: str) -> None:
    """Status-only marking (active/suspended) — never a delete. A human can
    always flip it back in Supabase; nothing in code reverses it the other
    direction automatically."""
    get_client().table("strategy_versions").update({"status": status}).eq("id", version_id).execute()


def get_active_strategy_versions() -> list[dict]:
    res = (
        get_client()
        .table("strategy_versions")
        .select("*")
        .neq("status", "suspended")
        .order("version_number", desc=True)
        .execute()
    )
    return res.data


# --- system_metrics (Production Monitoring + Self-Diagnostics,
# PROJECT_SPEC.md §3d) — one generic table, not N single-purpose ones,
# matching the jsonb-bundle precedent elsewhere in this schema. ---


def insert_system_metrics(rows: list[dict]) -> None:
    if not rows:
        return
    get_client().table("system_metrics").insert(rows).execute()


def get_recent_system_metrics(component: str | None = None, limit: int = 200) -> list[dict]:
    query = get_client().table("system_metrics").select("*")
    if component is not None:
        query = query.eq("component", component)
    res = query.order("recorded_at", desc=True).limit(limit).execute()
    return res.data


# --- circuit_breaker_state (src/resilience.py) ---


def get_circuit_breaker_state(component: str) -> dict | None:
    res = (
        get_client()
        .table("circuit_breaker_state")
        .select("*")
        .eq("component", component)
        .execute()
    )
    return res.data[0] if res.data else None


def upsert_circuit_breaker_state(component: str, consecutive_failures: int, tripped_until: int | None) -> None:
    get_client().table("circuit_breaker_state").upsert(
        {
            "component": component,
            "consecutive_failures": consecutive_failures,
            "tripped_until": tripped_until,
        },
        on_conflict="component",
    ).execute()


def reset_circuit_breaker(component: str) -> None:
    get_client().table("circuit_breaker_state").upsert(
        {"component": component, "consecutive_failures": 0, "tripped_until": None},
        on_conflict="component",
    ).execute()


# --- Data Retention ---
# Keeps the free-tier Supabase disk from maxing out again the way it did
# earlier — opportunity_evaluations is written every scanned symbol every
# cycle, the highest-volume table by far. trades/strategy_versions/
# recommendations/adaptive_strategy_versions/strategy_simulations/
# learning_statistics/feature_importance/drift_alerts/
# strategy_health_scores/historical_candles are deliberately NOT here: the
# actual ledger, small-row-count decision history, compact rollups, or
# low-volume/valuable backtest data respectively — see src/config.py's
# Data Retention section.
_RETENTION_TABLES = (
    ("opportunity_evaluations", "timestamp"),
    ("confidence_calibration", "created_at"),
    ("agent_logs", "timestamp"),
    ("model_usage", "timestamp"),
    ("system_metrics", "recorded_at"),
    ("data_quality_log", "created_at"),
)


# --- promotion_audit (src/learning/promotion_gate.py) ---


def insert_promotion_audit(
    mode: str,
    event_type: str,
    decision: str,
    candidate_version_id: int | None = None,
    previous_champion_id: int | None = None,
    new_champion_id: int | None = None,
    promotion_score: float | None = None,
    gates: dict | None = None,
    breakdown: dict | None = None,
    reasons: list | None = None,
) -> dict:
    res = (
        get_client()
        .table("promotion_audit")
        .insert(
            {
                "mode": mode,
                "event_type": event_type,
                "decision": decision,
                "candidate_version_id": candidate_version_id,
                "previous_champion_id": previous_champion_id,
                "new_champion_id": new_champion_id,
                "promotion_score": promotion_score,
                "gates": gates or {},
                "breakdown": breakdown or {},
                "reasons": reasons or [],
            }
        )
        .execute()
    )
    return res.data[0]


def get_latest_promotion_audit(mode: str, event_type: str | None = None) -> dict | None:
    query = get_client().table("promotion_audit").select("*").eq("mode", mode)
    if event_type is not None:
        query = query.eq("event_type", event_type)
    res = query.order("created_at", desc=True).limit(1).execute()
    return res.data[0] if res.data else None


def purge_old_data(cutoffs: dict[str, datetime]) -> dict[str, int]:
    """Deletes rows older than `cutoffs[table]` for every table in
    _RETENTION_TABLES a cutoff was supplied for (a table with no entry in
    `cutoffs` is skipped, not purged with some default). Delete is
    naturally idempotent (re-deleting an already-gone row is a no-op), so
    this goes through _execute's retry like every other idempotent write
    in this module. Returns {table: rows_deleted} for the caller to log —
    the Supabase Python client's .delete() returns the deleted rows by
    default (Prefer: return=representation), so the count is read straight
    off that response with no separate count query."""
    deleted: dict[str, int] = {}
    for table, column in _RETENTION_TABLES:
        cutoff = cutoffs.get(table)
        if cutoff is None:
            continue
        res = _execute(get_client().table(table).delete().lt(column, cutoff.isoformat()))
        deleted[table] = len(res.data or [])
    return deleted
