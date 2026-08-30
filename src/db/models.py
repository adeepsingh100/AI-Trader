"""Neon (Postgres) data access. One function per read/write the agents
need — see PROJECT_SPEC.md §6 for the schema these map to.

Migrated off Supabase (its free-tier disk filled and its instance got
stuck in Postgres crash-recovery, unrecoverable from this codebase's
side) onto Neon, a plain managed Postgres with no PostgREST/Auth/RLS
layer — every function below is now hand-written parameterized SQL via
psycopg2 instead of the Supabase client's chainable query builder.
Every function's name, signature, and return shape (list[dict] / dict /
None) is unchanged from before, so every caller in this repo (all ~30+
modules that import this file) needed zero changes."""

from __future__ import annotations

import psycopg2
import psycopg2.extensions
from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import Json, RealDictCursor, execute_values

from src.config import DATABASE_URL
from src.groq_client import ModelUsageEvent
from src.resilience import retry_with_backoff

_conn: psycopg2.extensions.connection | None = None

# psycopg2 defaults NUMERIC to Decimal; every downstream caller
# (features/learning/agents) does float arithmetic on these values, and
# Supabase's JSON/PostgREST layer always returned floats — register this
# once so return shapes stay byte-identical to before.
_DEC2FLOAT = psycopg2.extensions.new_type(
    psycopg2.extensions.DECIMAL.values,
    "DEC2FLOAT",
    lambda value, curs: float(value) if value is not None else None,
)
psycopg2.extensions.register_type(_DEC2FLOAT)

# psycopg2 doesn't adapt a plain dict to jsonb on write by default. Every
# dict-typed parameter in this file is destined for a jsonb column
# (features/params_json/raw_llm_response/breakdown/gates/evidence/detail/
# metadata/...) — registering this globally covers all of them. NOT done
# for `list` (register_adapter(list, Json)) on purpose: several functions
# pass a plain list of ids/values into `= ANY(%s)` for a native Postgres
# array (psycopg2's built-in list adapter), and a global list->Json
# adapter would silently break every one of those. The one list-typed
# jsonb column this file writes (promotion_audit.reasons) wraps its value
# in Json(...) explicitly at that single call site instead.
psycopg2.extensions.register_adapter(dict, Json)


def get_client() -> psycopg2.extensions.connection:
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(DATABASE_URL)  # pooled connection string — see src/config.py
    return _conn


def ping() -> None:
    """Trivial reachability check — used by monitoring/diagnostics.py's
    database health check, which needs no real data, just confirmation the
    connection works."""
    with get_client().cursor() as cur:
        cur.execute("select mode from capital_config limit 1")
    get_client().commit()


def _run_query(sql_text: str, params: tuple = ()) -> list[dict]:
    """SELECT helper — every read in this file goes through this (or
    _run_write for a read that follows a write in the same call). Commits
    after every query, even reads: this connection is long-lived across
    many calls in one process run, and leaving a transaction open would
    hold a backend connection under Neon's pooled (PgBouncer transaction-
    mode) connection string longer than necessary."""
    conn = get_client()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_text, params)
            rows = [dict(r) for r in cur.fetchall()]
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise


def _run_write(sql_text: str, params: tuple = ()) -> list[dict]:
    """INSERT/UPDATE/UPSERT/DELETE helper, optionally with RETURNING.
    Returns the RETURNING rows (or [] if the statement had none)."""
    conn = get_client()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_text, params)
            rows = [dict(r) for r in cur.fetchall()] if cur.description else []
        conn.commit()
        return rows
    except Exception:
        conn.rollback()
        raise


def _insert_row(table: str, row: dict) -> dict:
    """Dynamic single-row insert (table/column names come from this
    file's own call sites — internal Python dict keys, never external
    input — safe to interpolate directly, same reasoning this file
    already applies to ORDER BY/LIMIT clauses below)."""
    cols = list(row.keys())
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    query = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) RETURNING *"
    return _run_write(query, tuple(row.values()))[0]


def _insert_rows(table: str, rows: list[dict]) -> None:
    """Dynamic batch insert — a multi-month equity curve or a cycle's
    worth of log rows is many rows, one-row-per-round-trip would be
    needlessly slow. All rows in one call are assumed to share the same
    keys (true at every call site below)."""
    if not rows:
        return
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    query = f"INSERT INTO {table} ({col_list}) VALUES %s"
    conn = get_client()
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, [tuple(r[c] for c in cols) for r in rows])
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def _execute(fn):
    """Retries fn() on a transient connection error — safe here because
    every call site below is naturally idempotent (a repeat select/
    upsert/delete doesn't create a second row). Plain single-row inserts
    are NOT routed through this: a retry after a request whose response
    was lost but which actually succeeded server-side would insert a
    duplicate row (a trade, a log line) — a real correctness risk this
    codebase's own "never fabricate/duplicate financial state" ethos
    rules out. Also resets the module-level connection on
    OperationalError before re-raising — psycopg2 doesn't health-check a
    connection per-call the way the old Supabase HTTP client implicitly
    did, so a connection Neon dropped (e.g. after a scale-to-zero
    suspend) needs a fresh one on the next attempt, not a retry against
    the same dead socket."""
    def attempt():
        try:
            return fn()
        except psycopg2.OperationalError:
            global _conn
            if _conn is not None:
                _conn.close()
            _conn = None
            raise
    return retry_with_backoff(attempt)


# --- capital_config ---


def get_capital_config(mode: str, strategy_type: str = "default") -> dict | None:
    rows = _execute(lambda: _run_query(
        "SELECT * FROM capital_config WHERE mode = %s AND strategy_type = %s", (mode, strategy_type)
    ))
    return rows[0] if rows else None


def upsert_capital_config(
    mode: str,
    total_capital: float,
    capital_to_use: float,
    daily_profit_target: float,
    max_daily_loss: float,
    position_size_pct: float = 10,
    max_concurrent_positions: int = 5,
    strategy_type: str = "default",
) -> None:
    _run_write(
        """
        INSERT INTO capital_config
            (mode, strategy_type, total_capital, capital_to_use, daily_profit_target, max_daily_loss,
             position_size_pct, max_concurrent_positions)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (mode, strategy_type) DO UPDATE SET
            total_capital = EXCLUDED.total_capital,
            capital_to_use = EXCLUDED.capital_to_use,
            daily_profit_target = EXCLUDED.daily_profit_target,
            max_daily_loss = EXCLUDED.max_daily_loss,
            position_size_pct = EXCLUDED.position_size_pct,
            max_concurrent_positions = EXCLUDED.max_concurrent_positions
        """,
        (mode, strategy_type, total_capital, capital_to_use, daily_profit_target, max_daily_loss,
         position_size_pct, max_concurrent_positions),
    )


def get_active_strategy_types(mode: str) -> list[str]:
    """Strategy types with a seeded capital_config row for this mode --
    "active" = someone ran seed_config.py for it (src/seed_config.py).
    Callers intersect this with src.config.STRATEGY_PROFILES; this module
    stays DB-only and doesn't import config's registry."""
    rows = _execute(lambda: _run_query(
        "SELECT DISTINCT strategy_type FROM capital_config WHERE mode = %s", (mode,)
    ))
    return [r["strategy_type"] for r in rows]


# --- strategy_versions (immutable once created, see spec §3) ---


def get_latest_version(strategy_type: str = "default") -> dict | None:
    # Excludes suspended versions (Strategy Health Engine, PROJECT_SPEC.md
    # §3d) — without this filter, auto-suspension would be a silent no-op
    # since this is still an unfiltered "newest row" query otherwise.
    rows = _execute(lambda: _run_query(
        "SELECT * FROM strategy_versions WHERE status != 'suspended' AND strategy_type = %s "
        "ORDER BY version_number DESC LIMIT 1",
        (strategy_type,),
    ))
    return rows[0] if rows else None


def get_latest_promoted_version(strategy_type: str = "default") -> dict | None:
    rows = _execute(lambda: _run_query(
        "SELECT * FROM strategy_versions WHERE promoted_to_real = true AND status != 'suspended' "
        "AND strategy_type = %s ORDER BY version_number DESC LIMIT 1",
        (strategy_type,),
    ))
    return rows[0] if rows else None


def insert_strategy_version(
    version_number: int,
    prompt_text: str,
    params_json: dict,
    notes: str | None = None,
    strategy_type: str = "default",
) -> dict:
    return _insert_row("strategy_versions", {
        "version_number": version_number,
        "prompt_text": prompt_text,
        "params_json": params_json,
        "notes": notes,
        "strategy_type": strategy_type,
    })


def promote_version(version_id: int) -> None:
    _run_write("UPDATE strategy_versions SET promoted_to_real = true WHERE id = %s", (version_id,))


def set_strategy_version_promotion_eligible(version_id: int, eligible: bool) -> None:
    """Mirrors src/learning/promotion_gate.py::evaluate_promotion()'s
    PROMOTE/REJECT/EXTEND_VALIDATION verdict onto the row for visibility —
    evolution_agent.py calls promote_version() itself immediately after on
    PROMOTE, fully automatically, no human step. This flag is a record of
    the decision, not a queue awaiting manual action."""
    _run_write("UPDATE strategy_versions SET promotion_eligible = %s WHERE id = %s", (eligible, version_id))


def get_all_strategy_versions() -> list[dict]:
    return _run_query("SELECT * FROM strategy_versions ORDER BY version_number DESC")


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
    return _insert_row("trades", {
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
    })


def close_trade(
    trade_id: int, exit_price: float, pnl: float, status: str = "closed", exit_reason: str | None = None
) -> None:
    _run_write(
        "UPDATE trades SET exit_price = %s, pnl = %s, status = %s, exit_reason = %s, closed_at = %s "
        "WHERE id = %s",
        (exit_price, pnl, status, exit_reason, datetime.now(timezone.utc).isoformat(), trade_id),
    )


def update_trade_excursion(trade_id: int, mfe_pct: float, mae_pct: float) -> None:
    _run_write("UPDATE trades SET mfe_pct = %s, mae_pct = %s WHERE id = %s", (mfe_pct, mae_pct, trade_id))


def get_open_trades(mode: str, strategy_type: str | None = None) -> list[dict]:
    if strategy_type is None:
        return _execute(lambda: _run_query(
            "SELECT * FROM trades WHERE mode = %s AND status = 'open'", (mode,)
        ))
    return _execute(lambda: _run_query(
        "SELECT t.* FROM trades t JOIN strategy_versions sv ON t.version_id = sv.id "
        "WHERE t.mode = %s AND t.status = 'open' AND sv.strategy_type = %s",
        (mode, strategy_type),
    ))


def get_recently_closed_trades(mode: str, since: datetime, strategy_type: str | None = None) -> list[dict]:
    """Closed/flattened trades since `since` — mode-scoped, not
    version-scoped (unlike get_closed_trades), since strategy versions
    rotate and the learning engine's catch-up pass needs to see across
    versions of the SAME strategy_type. Bounded by the caller's `since`
    (LEARNING_CATCHUP_LOOKBACK_HOURS for process_closed_trades,
    LEARNING_HISTORY_WINDOW_DAYS for stats bucket recompute) so this
    never becomes a full-table scan. strategy_type is additive/optional —
    omitted keeps today's exact mode-wide (cross-strategy-type) query."""
    if strategy_type is None:
        return _run_query(
            "SELECT * FROM trades WHERE mode = %s AND status = ANY(%s) AND closed_at >= %s",
            (mode, ["closed", "flattened"], since.isoformat()),
        )
    return _run_query(
        "SELECT t.* FROM trades t JOIN strategy_versions sv ON t.version_id = sv.id "
        "WHERE t.mode = %s AND t.status = ANY(%s) AND t.closed_at >= %s AND sv.strategy_type = %s",
        (mode, ["closed", "flattened"], since.isoformat(), strategy_type),
    )


def get_recent_trades(mode: str, limit: int = 50, strategy_type: str | None = None) -> list[dict]:
    if strategy_type is None:
        return _run_query(
            "SELECT * FROM trades WHERE mode = %s ORDER BY opened_at DESC LIMIT %s", (mode, limit)
        )
    return _run_query(
        "SELECT t.* FROM trades t JOIN strategy_versions sv ON t.version_id = sv.id "
        "WHERE t.mode = %s AND sv.strategy_type = %s ORDER BY t.opened_at DESC LIMIT %s",
        (mode, strategy_type, limit),
    )


def get_closed_trades(mode: str, version_id: int) -> list[dict]:
    return _run_query(
        "SELECT * FROM trades WHERE mode = %s AND version_id = %s AND status = ANY(%s)",
        (mode, version_id, ["closed", "flattened"]),
    )


# --- risk_check_lock ---


def try_acquire_risk_check_lock(mode: str, stale_after_seconds: int = 180) -> bool:
    """Non-blocking mutex for orchestrator.run_risk_check(mode) — a
    tightened polling cadence means one run can still be in flight when
    the next fires. `stale_after_seconds` self-heals a lock left held by
    a crashed process rather than deadlocking the mode forever (ponytail:
    fixed 3min ceiling, revisit only if a real run ever legitimately
    takes that long)."""
    rows = _run_write(
        "UPDATE risk_check_lock SET locked_at = now() "
        "WHERE mode = %s AND (locked_at IS NULL OR locked_at < now() - make_interval(secs => %s)) "
        "RETURNING mode",
        (mode, stale_after_seconds),
    )
    return bool(rows)


def release_risk_check_lock(mode: str) -> None:
    _run_write("UPDATE risk_check_lock SET locked_at = NULL WHERE mode = %s", (mode,))


# --- daily_pnl ---


def get_daily_pnl(day: Date, mode: str, strategy_type: str = "default") -> dict | None:
    rows = _execute(lambda: _run_query(
        "SELECT * FROM daily_pnl WHERE date = %s AND mode = %s AND strategy_type = %s",
        (day.isoformat(), mode, strategy_type),
    ))
    return rows[0] if rows else None


def upsert_daily_pnl(
    day: Date,
    mode: str,
    realized_pnl: float,
    trades_count: int,
    target_hit: bool,
    circuit_breaker_triggered: bool,
    strategy_type: str = "default",
) -> None:
    _run_write(
        """
        INSERT INTO daily_pnl
            (date, mode, strategy_type, realized_pnl, trades_count, target_hit, circuit_breaker_triggered)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (date, mode, strategy_type) DO UPDATE SET
            realized_pnl = EXCLUDED.realized_pnl,
            trades_count = EXCLUDED.trades_count,
            target_hit = EXCLUDED.target_hit,
            circuit_breaker_triggered = EXCLUDED.circuit_breaker_triggered
        """,
        (day.isoformat(), mode, strategy_type, realized_pnl, trades_count, target_hit, circuit_breaker_triggered),
    )


# --- agent_logs ---


def log_agent_event(
    agent_name: str, level: str, message: str, raw_llm_response: Any = None
) -> None:
    _run_write(
        "INSERT INTO agent_logs (agent_name, level, message, raw_llm_response) VALUES (%s, %s, %s, %s)",
        (agent_name, level, message, raw_llm_response),
    )


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
    _insert_rows("model_usage", rows)


def get_recent_model_usage(limit: int = 500) -> list[dict]:
    return _run_query("SELECT * FROM model_usage ORDER BY timestamp DESC LIMIT %s", (limit,))


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
    return _insert_row("opportunity_evaluations", {
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
    })


# --- learning_statistics ---


def upsert_learning_statistics(
    mode: str, dimension_type: str, dimension_value: str, stats: dict, strategy_type: str = "default"
) -> None:
    cols = ["mode", "strategy_type", "dimension_type", "dimension_value", *stats.keys()]
    vals = [mode, strategy_type, dimension_type, dimension_value, *stats.values()]
    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    set_clause = ", ".join(f"{k} = EXCLUDED.{k}" for k in stats)
    _run_write(
        f"INSERT INTO learning_statistics ({col_list}) VALUES ({placeholders}) "
        f"ON CONFLICT (mode, strategy_type, dimension_type, dimension_value) DO UPDATE SET {set_clause}",
        tuple(vals),
    )


def get_learning_statistics(
    mode: str, dimension_type: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    clauses, params = ["mode = %s", "strategy_type = %s"], [mode, strategy_type]
    if dimension_type is not None:
        clauses.append("dimension_type = %s")
        params.append(dimension_type)
    return _run_query(f"SELECT * FROM learning_statistics WHERE {' AND '.join(clauses)}", tuple(params))


# --- feature_importance ---


def upsert_feature_importance(
    mode: str,
    feature_name: str,
    correlation_score: float,
    sample_count: int,
    timeframe: str,
    strategy_type: str = "default",
) -> None:
    _run_write(
        """
        INSERT INTO feature_importance
            (mode, strategy_type, feature_name, correlation_score, sample_count, timeframe)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (mode, strategy_type, feature_name, timeframe) DO UPDATE SET
            correlation_score = EXCLUDED.correlation_score,
            sample_count = EXCLUDED.sample_count
        """,
        (mode, strategy_type, feature_name, correlation_score, sample_count, timeframe),
    )


def get_feature_importance(
    mode: str, timeframe: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    clauses, params = ["mode = %s", "strategy_type = %s"], [mode, strategy_type]
    if timeframe is not None:
        clauses.append("timeframe = %s")
        params.append(timeframe)
    return _run_query(f"SELECT * FROM feature_importance WHERE {' AND '.join(clauses)}", tuple(params))


def get_opportunity_evaluations_for_trail(
    mode: str,
    trade_id: int | None = None,
    symbol: str | None = None,
    since=None,
    strategy_type: str | None = None,
) -> list[dict]:
    """Chronological rows for src.audit.trail.get_decision_trail — the one
    query that module needs, routed through here (rather than a raw
    query in audit/trail.py) so every DB access in src/ goes through this
    file, per this repo's own convention. strategy_type is additive/
    optional — omitted keeps today's exact mode-wide (cross-strategy-type)
    query, matching the audit trail's own "show everything" default."""
    if strategy_type is None:
        clauses, params = ["mode = %s"], [mode]
        if trade_id is not None:
            clauses.append("trade_id = %s")
            params.append(trade_id)
        if symbol is not None:
            clauses.append("symbol = %s")
            params.append(symbol)
        if since is not None:
            clauses.append("timestamp >= %s")
            params.append(since.isoformat())
        query = f"SELECT * FROM opportunity_evaluations WHERE {' AND '.join(clauses)} ORDER BY timestamp"
        return _run_query(query, tuple(params))

    clauses, params = ["oe.mode = %s", "sv.strategy_type = %s"], [mode, strategy_type]
    if trade_id is not None:
        clauses.append("oe.trade_id = %s")
        params.append(trade_id)
    if symbol is not None:
        clauses.append("oe.symbol = %s")
        params.append(symbol)
    if since is not None:
        clauses.append("oe.timestamp >= %s")
        params.append(since.isoformat())
    query = (
        "SELECT oe.* FROM opportunity_evaluations oe JOIN strategy_versions sv ON oe.version_id = sv.id "
        f"WHERE {' AND '.join(clauses)} ORDER BY oe.timestamp"
    )
    return _run_query(query, tuple(params))


def get_entry_evaluation_for_trade(trade_id: int) -> dict | None:
    rows = _run_query(
        "SELECT * FROM opportunity_evaluations WHERE trade_id = %s AND final_decision = 'buy' LIMIT 1",
        (trade_id,),
    )
    return rows[0] if rows else None


# --- confidence_calibration ---


def get_confidence_calibration_for_evaluation(opportunity_evaluation_id: int) -> dict | None:
    rows = _run_query(
        "SELECT * FROM confidence_calibration WHERE opportunity_evaluation_id = %s LIMIT 1",
        (opportunity_evaluation_id,),
    )
    return rows[0] if rows else None


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
    _run_write(
        """
        INSERT INTO confidence_calibration
            (opportunity_evaluation_id, ai_confidence, historical_confidence, ai_weight,
             historical_weight, final_confidence, similar_trades_count, regime_modifier,
             symbol_modifier, recent_performance_modifier)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (opportunity_evaluation_id, ai_confidence, historical_confidence, ai_weight, historical_weight,
         final_confidence, similar_trades_count, regime_modifier, symbol_modifier, recent_performance_modifier),
    )


# --- recommendations ---


def get_latest_recommendation(mode: str, metric_name: str, strategy_type: str = "default") -> dict | None:
    rows = _run_query(
        "SELECT * FROM recommendations WHERE mode = %s AND metric_name = %s AND strategy_type = %s "
        "ORDER BY created_at DESC LIMIT 1",
        (mode, metric_name, strategy_type),
    )
    return rows[0] if rows else None


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
    strategy_type: str = "default",
) -> None:
    _run_write(
        """
        INSERT INTO recommendations
            (mode, strategy_type, metric_name, current_value, recommended_value, rationale, sample_size,
             category, confidence, evidence, batch_id)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (mode, strategy_type, metric_name, current_value, recommended_value, rationale, sample_size,
         category, confidence, evidence, batch_id),
    )


def get_recommendations(
    mode: str, status: str | None = None, category: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    clauses, params = ["mode = %s", "strategy_type = %s"], [mode, strategy_type]
    if status is not None:
        clauses.append("status = %s")
        params.append(status)
    if category is not None:
        clauses.append("category = %s")
        params.append(category)
    query = f"SELECT * FROM recommendations WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    return _run_query(query, tuple(params))


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
    strategy_type: str = "default",
) -> dict:
    """research_note/validation_detail (Scientific Strategy Optimization
    Framework) — a narrative Observation/Weakness/Hypothesis/Simulation/
    Walk Forward/Decision report and the raw numbers behind it (bootstrap
    CI, walk-forward folds, strategy-comparison result where run)."""
    return _insert_row("strategy_simulations", {
        "recommendation_batch_id": recommendation_batch_id,
        "mode": mode,
        "strategy_type": strategy_type,
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
    })


def get_strategy_simulations(mode: str, passed: bool | None = None, strategy_type: str = "default") -> list[dict]:
    clauses, params = ["mode = %s", "strategy_type = %s"], [mode, strategy_type]
    if passed is not None:
        clauses.append("passed = %s")
        params.append(passed)
    query = f"SELECT * FROM strategy_simulations WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    return _run_query(query, tuple(params))


# --- adaptive_strategy_versions ---


def insert_adaptive_strategy_version(
    mode: str,
    version_number: int,
    params_json: dict,
    source_recommendation_batch_id: str | None,
    source_simulation_id: int | None,
    notes: str | None = None,
    fitness_score: float | None = None,
    strategy_type: str = "default",
) -> dict:
    return _insert_row("adaptive_strategy_versions", {
        "mode": mode,
        "strategy_type": strategy_type,
        "version_number": version_number,
        "params_json": params_json,
        "source_recommendation_batch_id": source_recommendation_batch_id,
        "source_simulation_id": source_simulation_id,
        "notes": notes,
        "fitness_score": fitness_score,
    })


def get_adaptive_strategy_versions(
    mode: str, status: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    clauses, params = ["mode = %s", "strategy_type = %s"], [mode, strategy_type]
    if status is not None:
        clauses.append("status = %s")
        params.append(status)
    query = f"SELECT * FROM adaptive_strategy_versions WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
    return _run_query(query, tuple(params))


def get_latest_adaptive_strategy_version(mode: str, strategy_type: str = "default") -> dict | None:
    rows = _run_query(
        "SELECT * FROM adaptive_strategy_versions WHERE mode = %s AND strategy_type = %s "
        "ORDER BY version_number DESC LIMIT 1",
        (mode, strategy_type),
    )
    return rows[0] if rows else None


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
    _run_write(
        """
        INSERT INTO trade_evaluations
            (trade_id, predicted_confidence, predicted_opportunity_score, actual_outcome_won,
             confidence_was_accurate, opportunity_score_was_accurate, risk_assessment,
             stop_loss_assessment, target_assessment)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (trade_id) DO UPDATE SET
            predicted_confidence = EXCLUDED.predicted_confidence,
            predicted_opportunity_score = EXCLUDED.predicted_opportunity_score,
            actual_outcome_won = EXCLUDED.actual_outcome_won,
            confidence_was_accurate = EXCLUDED.confidence_was_accurate,
            opportunity_score_was_accurate = EXCLUDED.opportunity_score_was_accurate,
            risk_assessment = EXCLUDED.risk_assessment,
            stop_loss_assessment = EXCLUDED.stop_loss_assessment,
            target_assessment = EXCLUDED.target_assessment
        """,
        (trade_id, predicted_confidence, predicted_opportunity_score, actual_outcome_won,
         confidence_was_accurate, opportunity_score_was_accurate, risk_assessment,
         stop_loss_assessment, target_assessment),
    )


def get_trade_evaluation_ids(trade_ids: list[int]) -> set[int]:
    if not trade_ids:
        return set()
    rows = _run_query("SELECT trade_id FROM trade_evaluations WHERE trade_id = ANY(%s)", (trade_ids,))
    return {row["trade_id"] for row in rows}


def get_trade_evaluations(trade_ids: list[int]) -> list[dict]:
    """Full trade_evaluations rows (confidence_was_accurate/
    opportunity_score_was_accurate) for drift_detection.py — distinct from
    get_trade_evaluation_ids, which only returns the id set for the
    already-evaluated membership check process_closed_trades() needs."""
    if not trade_ids:
        return []
    return _run_query("SELECT * FROM trade_evaluations WHERE trade_id = ANY(%s)", (trade_ids,))


def get_trades_by_ids(trade_ids: list[int]) -> list[dict]:
    if not trade_ids:
        return []
    return _run_query("SELECT * FROM trades WHERE id = ANY(%s)", (trade_ids,))


# --- historical_candles ---


def upsert_historical_candles(pair: str, interval: str, candles: list[dict]) -> None:
    """candles: list of {"time","open","high","low","close","volume"} dicts,
    CoinDCX's own raw shape — caller passes them through unchanged."""
    if not candles:
        return
    rows = [(pair, interval, c["time"], c["open"], c["high"], c["low"], c["close"], c["volume"]) for c in candles]
    conn = get_client()
    try:
        with conn.cursor() as cur:
            execute_values(
                cur,
                "INSERT INTO historical_candles (pair, interval, time, open, high, low, close, volume) VALUES %s "
                "ON CONFLICT (pair, interval, time) DO UPDATE SET "
                "open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, "
                "close = EXCLUDED.close, volume = EXCLUDED.volume",
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_historical_candles(pair: str, interval: str, start_time_ms: int, end_time_ms: int) -> list[dict]:
    return _run_query(
        "SELECT * FROM historical_candles WHERE pair = %s AND interval = %s AND time >= %s AND time <= %s "
        "ORDER BY time",
        (pair, interval, start_time_ms, end_time_ms),
    )


def historical_candles_exist(pair: str, interval: str, start_time_ms: int, end_time_ms: int) -> bool:
    """Existence-only check for the same range get_historical_candles
    queries — callers that only need a yes/no (promotion_gate.py,
    simulation.py, both gating an expensive backtest replay on "is there
    any data at all") were pulling the full result set just to check
    non-emptiness, at BACKTEST_TICK_TIMEFRAME granularity over a range
    that can span a whole strategy version's trade history — real,
    unnecessary Neon egress. This transfers one boolean instead."""
    rows = _run_query(
        "SELECT EXISTS (SELECT 1 FROM historical_candles "
        "WHERE pair = %s AND interval = %s AND time >= %s AND time <= %s) AS exists_",
        (pair, interval, start_time_ms, end_time_ms),
    )
    return rows[0]["exists_"]


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
    return _insert_row("backtest_runs", {
        "name": name,
        "symbols": symbols,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "warmup_buffer_days": warmup_buffer_days,
        "starting_capital": starting_capital,
        "params_json": params_json,
        "use_llm_signal_agent": use_llm_signal_agent,
        "source_adaptive_strategy_version_id": source_adaptive_strategy_version_id,
    })


def update_backtest_run_status(run_id: int, status: str, completed_at: datetime | None = None) -> None:
    if completed_at is not None:
        _run_write(
            "UPDATE backtest_runs SET status = %s, completed_at = %s WHERE id = %s",
            (status, completed_at.isoformat(), run_id),
        )
    else:
        _run_write("UPDATE backtest_runs SET status = %s WHERE id = %s", (status, run_id))


def get_backtest_run(run_id: int) -> dict | None:
    rows = _run_query("SELECT * FROM backtest_runs WHERE id = %s", (run_id,))
    return rows[0] if rows else None


def get_backtest_runs(status: str | None = None) -> list[dict]:
    if status is not None:
        return _run_query(
            "SELECT * FROM backtest_runs WHERE status = %s ORDER BY created_at DESC", (status,)
        )
    return _run_query("SELECT * FROM backtest_runs ORDER BY created_at DESC")


# --- backtest_trades ---


def insert_backtest_trade(run_id: int, trade: dict) -> dict:
    return _insert_row("backtest_trades", {"run_id": run_id, **trade})


def get_backtest_trades(run_id: int) -> list[dict]:
    return _run_query("SELECT * FROM backtest_trades WHERE run_id = %s ORDER BY entry_time", (run_id,))


# --- backtest_portfolio_snapshots ---


def insert_backtest_portfolio_snapshots(run_id: int, snapshots: list[dict]) -> None:
    """Batch insert — a multi-month equity curve is thousands of points,
    one-row-per-network-call would be needlessly slow."""
    if not snapshots:
        return
    _insert_rows("backtest_portfolio_snapshots", [{"run_id": run_id, **s} for s in snapshots])


def get_backtest_portfolio_snapshots(run_id: int) -> list[dict]:
    return _run_query(
        "SELECT * FROM backtest_portfolio_snapshots WHERE run_id = %s ORDER BY snapshot_time", (run_id,)
    )


# --- backtest_execution_history ---


def insert_backtest_execution_events(run_id: int, events: list[dict]) -> None:
    if not events:
        return
    _insert_rows("backtest_execution_history", [{"run_id": run_id, **e} for e in events])


def get_backtest_execution_history(run_id: int) -> list[dict]:
    return _run_query(
        "SELECT * FROM backtest_execution_history WHERE run_id = %s ORDER BY event_time", (run_id,)
    )


# --- backtest_performance_metrics ---


def insert_backtest_performance_metrics(run_id: int, metrics: dict) -> dict:
    return _insert_row("backtest_performance_metrics", {"run_id": run_id, "metrics": metrics})


def get_backtest_performance_metrics(run_id: int) -> dict | None:
    rows = _run_query("SELECT * FROM backtest_performance_metrics WHERE run_id = %s", (run_id,))
    return rows[0] if rows else None


# --- backtest_walk_forward_folds ---


def insert_backtest_walk_forward_fold(run_id: int, fold: dict) -> dict:
    return _insert_row("backtest_walk_forward_folds", {"run_id": run_id, **fold})


def get_backtest_walk_forward_folds(run_id: int) -> list[dict]:
    return _run_query(
        "SELECT * FROM backtest_walk_forward_folds WHERE run_id = %s ORDER BY fold_number", (run_id,)
    )


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
    return _insert_row("backtest_strategy_comparisons", {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "metrics_a": metrics_a,
        "metrics_b": metrics_b,
        "p_values": p_values,
        "winner": winner,
        "promotion_recommended": promotion_recommended,
    })


def get_backtest_strategy_comparisons() -> list[dict]:
    return _run_query("SELECT * FROM backtest_strategy_comparisons ORDER BY created_at DESC")


def get_entry_evaluations_since(mode: str, since: datetime, strategy_type: str | None = None) -> list[dict]:
    """Entry-time opportunity_evaluations rows (final_decision='buy', so
    trade_id is always set) since `since` — the candidate pool
    find_similar_trades() ranks by distance. No embedded join to `trades`
    for the outcome (pnl/closed_at) — this codebase has no precedent for
    a join across models.py functions; callers fetch outcomes separately
    via get_trades_by_ids() and match in Python, same pattern as
    process_closed_trades()'s diff. strategy_type is additive/optional —
    omitted keeps today's exact mode-wide (cross-strategy-type) query."""
    if strategy_type is None:
        return _run_query(
            "SELECT * FROM opportunity_evaluations WHERE mode = %s AND final_decision = 'buy' AND timestamp >= %s",
            (mode, since.isoformat()),
        )
    return _run_query(
        "SELECT oe.* FROM opportunity_evaluations oe JOIN strategy_versions sv ON oe.version_id = sv.id "
        "WHERE oe.mode = %s AND oe.final_decision = 'buy' AND oe.timestamp >= %s AND sv.strategy_type = %s",
        (mode, since.isoformat(), strategy_type),
    )


def get_hold_evaluations_since(mode: str, since: datetime, strategy_type: str | None = None) -> list[dict]:
    """Non-trade opportunity_evaluations rows (final_decision='hold') since
    `since` — every scanned-but-not-traded candidate, each carrying its own
    reason/risk_manager_result (Root Cause Analysis, Scientific Strategy
    Optimization Framework). Same shape as get_entry_evaluations_since,
    filtering the opposite final_decision value."""
    if strategy_type is None:
        return _run_query(
            "SELECT * FROM opportunity_evaluations WHERE mode = %s AND final_decision = 'hold' AND timestamp >= %s",
            (mode, since.isoformat()),
        )
    return _run_query(
        "SELECT oe.* FROM opportunity_evaluations oe JOIN strategy_versions sv ON oe.version_id = sv.id "
        "WHERE oe.mode = %s AND oe.final_decision = 'hold' AND oe.timestamp >= %s AND sv.strategy_type = %s",
        (mode, since.isoformat(), strategy_type),
    )


# --- data_quality_log (Market Data Quality Engine + Data Repair Engine,
# PROJECT_SPEC.md §3d) ---


def insert_data_quality_issues(rows: list[dict]) -> None:
    """ON CONFLICT DO NOTHING on (pair, interval, issue_type, candle_time) --
    the same rolling candle window is re-validated every cycle, so an
    unrepaired issue on a still-in-window candle would otherwise be
    re-logged every cycle forever (the root cause of a real disk-fill
    incident)."""
    if not rows:
        return
    cols = list(rows[0].keys())
    col_list = ", ".join(cols)
    query = f"INSERT INTO data_quality_log ({col_list}) VALUES %s ON CONFLICT (pair, interval, issue_type, candle_time) DO NOTHING"
    conn = get_client()
    try:
        with conn.cursor() as cur:
            execute_values(cur, query, [tuple(r[c] for c in cols) for r in rows])
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def get_data_quality_log(pair: str | None = None, source: str | None = None, limit: int = 200) -> list[dict]:
    clauses, params = [], []
    if pair is not None:
        clauses.append("pair = %s")
        params.append(pair)
    if source is not None:
        clauses.append("source = %s")
        params.append(source)
    where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
    params.append(limit)
    return _run_query(f"SELECT * FROM data_quality_log {where}ORDER BY created_at DESC LIMIT %s", tuple(params))


# --- drift_alerts (Feature Drift Detection, PROJECT_SPEC.md §3d) ---


def insert_drift_alert(
    component: str,
    drift_type: str,
    severity: str,
    baseline_value: float | None,
    recent_value: float | None,
    detail: dict | None = None,
) -> dict:
    return _insert_row("drift_alerts", {
        "component": component,
        "drift_type": drift_type,
        "severity": severity,
        "baseline_value": baseline_value,
        "recent_value": recent_value,
        "detail": detail or {},
    })


def get_drift_alerts(component: str | None = None, limit: int = 200) -> list[dict]:
    if component is not None:
        return _run_query(
            "SELECT * FROM drift_alerts WHERE component = %s ORDER BY detected_at DESC LIMIT %s",
            (component, limit),
        )
    return _run_query("SELECT * FROM drift_alerts ORDER BY detected_at DESC LIMIT %s", (limit,))


# --- strategy_health_scores (Strategy Health Engine, PROJECT_SPEC.md §3d) ---


def insert_strategy_health_score(
    strategy_version_id: int, health_score: float | None, tier: str, breakdown: dict
) -> dict:
    return _insert_row("strategy_health_scores", {
        "strategy_version_id": strategy_version_id,
        "health_score": health_score,
        "tier": tier,
        "breakdown": breakdown,
    })


def get_latest_strategy_health_score(strategy_version_id: int) -> dict | None:
    rows = _run_query(
        "SELECT * FROM strategy_health_scores WHERE strategy_version_id = %s "
        "ORDER BY computed_at DESC LIMIT 1",
        (strategy_version_id,),
    )
    return rows[0] if rows else None


def update_strategy_version_status(version_id: int, status: str) -> None:
    """Status-only marking (active/suspended) — never a delete. A human can
    always flip it back; nothing in code reverses it the other direction
    automatically."""
    _run_write("UPDATE strategy_versions SET status = %s WHERE id = %s", (status, version_id))


def get_active_strategy_versions() -> list[dict]:
    return _run_query(
        "SELECT * FROM strategy_versions WHERE status != 'suspended' ORDER BY version_number DESC"
    )


# --- system_metrics (Production Monitoring + Self-Diagnostics,
# PROJECT_SPEC.md §3d) — one generic table, not N single-purpose ones,
# matching the jsonb-bundle precedent elsewhere in this schema. ---


def insert_system_metrics(rows: list[dict]) -> None:
    if not rows:
        return
    _insert_rows("system_metrics", rows)


def get_recent_system_metrics(component: str | None = None, limit: int = 200) -> list[dict]:
    if component is not None:
        return _run_query(
            "SELECT * FROM system_metrics WHERE component = %s ORDER BY recorded_at DESC LIMIT %s",
            (component, limit),
        )
    return _run_query("SELECT * FROM system_metrics ORDER BY recorded_at DESC LIMIT %s", (limit,))


# --- circuit_breaker_state (src/resilience.py) ---


def get_circuit_breaker_state(component: str) -> dict | None:
    rows = _run_query("SELECT * FROM circuit_breaker_state WHERE component = %s", (component,))
    return rows[0] if rows else None


def upsert_circuit_breaker_state(component: str, consecutive_failures: int, tripped_until: int | None) -> None:
    _run_write(
        """
        INSERT INTO circuit_breaker_state (component, consecutive_failures, tripped_until)
        VALUES (%s, %s, %s)
        ON CONFLICT (component) DO UPDATE SET
            consecutive_failures = EXCLUDED.consecutive_failures,
            tripped_until = EXCLUDED.tripped_until
        """,
        (component, consecutive_failures, tripped_until),
    )


def reset_circuit_breaker(component: str) -> None:
    _run_write(
        """
        INSERT INTO circuit_breaker_state (component, consecutive_failures, tripped_until)
        VALUES (%s, 0, NULL)
        ON CONFLICT (component) DO UPDATE SET
            consecutive_failures = EXCLUDED.consecutive_failures,
            tripped_until = EXCLUDED.tripped_until
        """,
        (component,),
    )


# --- Data Retention ---
# Keeps the free-tier disk from maxing out again the way it did on
# Supabase — opportunity_evaluations is written every scanned symbol
# every cycle, the highest-volume table by far. trades/strategy_versions/
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
    strategy_type: str = "default",
) -> dict:
    return _insert_row("promotion_audit", {
        "mode": mode,
        "strategy_type": strategy_type,
        "event_type": event_type,
        "decision": decision,
        "candidate_version_id": candidate_version_id,
        "previous_champion_id": previous_champion_id,
        "new_champion_id": new_champion_id,
        "promotion_score": promotion_score,
        "gates": gates or {},
        "breakdown": breakdown or {},
        "reasons": Json(reasons or []),  # jsonb column holding a list, not a dict — the one
                                          # call site needing an explicit wrap (see the global
                                          # dict->Json adapter note near the top of this file)
    })


def get_latest_promotion_audit(
    mode: str, event_type: str | None = None, strategy_type: str = "default"
) -> dict | None:
    if event_type is not None:
        rows = _run_query(
            "SELECT * FROM promotion_audit WHERE mode = %s AND strategy_type = %s AND event_type = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (mode, strategy_type, event_type),
        )
    else:
        rows = _run_query(
            "SELECT * FROM promotion_audit WHERE mode = %s AND strategy_type = %s "
            "ORDER BY created_at DESC LIMIT 1",
            (mode, strategy_type),
        )
    return rows[0] if rows else None


def purge_old_data(cutoffs: dict[str, datetime]) -> dict[str, int]:
    """Deletes rows older than `cutoffs[table]` for every table in
    _RETENTION_TABLES a cutoff was supplied for (a table with no entry in
    `cutoffs` is skipped, not purged with some default). Delete is
    naturally idempotent (re-deleting an already-gone row is a no-op), so
    this goes through _execute's retry like every other idempotent write
    in this module. Returns {table: rows_deleted} for the caller to log —
    cur.rowcount after a DELETE gives the count directly, no RETURNING/
    representation trick needed the way the old Supabase client required."""
    deleted: dict[str, int] = {}
    for table, column in _RETENTION_TABLES:
        cutoff = cutoffs.get(table)
        if cutoff is None:
            continue

        def _delete(table=table, column=column, cutoff=cutoff):
            conn = get_client()
            try:
                with conn.cursor() as cur:
                    cur.execute(f"DELETE FROM {table} WHERE {column} < %s", (cutoff.isoformat(),))
                    count = cur.rowcount
                conn.commit()
                return count
            except Exception:
                conn.rollback()
                raise

        deleted[table] = _execute(_delete)
    return deleted
