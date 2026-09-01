"""Data access — one function per read/write the agents need, see
PROJECT_SPEC.md §6 for the schema these map to.

Migrated off Supabase (its free-tier disk filled and its instance got
stuck in Postgres crash-recovery) onto Neon, a plain managed Postgres.
Now mid-migration OFF Neon too (its free-tier data-transfer quota
exhausted, hard-blocking the bot with no way to restore access short of
paying — see PROJECT_SPEC.md) onto Firebase/Firestore, genuinely free at
this scale. Migrating table-by-table, not a single cutover: functions
for capital_config/strategy_versions/trades/daily_pnl/risk_check_lock/
agent_logs/opportunity_evaluations/confidence_calibration/
learning_statistics/trade_evaluations/feature_importance (the live
trading hot path — Phase 1) talk to Firestore via get_firestore_client();
everything else still talks to Neon via get_client()/psycopg2, until its
own phase. Two backends in one file, but never two sources of truth for
the same table — every table has exactly one backend at a time, tracked
per-function below, not a dual-write. Every function's name, signature,
and return shape (list[dict] / dict / None) stays unchanged across the
swap, same discipline as the Supabase→Neon migration, so callers need
zero changes — except open_trade/log_opportunity_evaluation, which
gained a required strategy_type param: Firestore has no JOIN, so
trades/opportunity_evaluations now carry strategy_type as a real field
instead of deriving it via version_id → strategy_versions at read time."""

from __future__ import annotations

import json
import os
import psycopg2
import psycopg2.extensions
from datetime import date as Date
from datetime import datetime, timezone
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore
from psycopg2.extras import Json, RealDictCursor, execute_values

from src.config import DATABASE_URL, FIREBASE_SERVICE_ACCOUNT_JSON
from src.groq_client import ModelUsageEvent
from src.resilience import retry_with_backoff

_conn: psycopg2.extensions.connection | None = None
_firebase_app: firebase_admin.App | None = None

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


def get_firestore_client() -> firestore.Client:
    """Separate seam from get_client() — deliberately, not a rename.
    Phase 1 of the Neon→Firestore migration (see module docstring) moves
    ~20 of this file's ~81 functions to Firestore; the rest keep using
    get_client()/psycopg2 unchanged until their own phase. One shared
    get_client() returning "whichever backend" would either break the
    untouched functions or force a big-bang cutover — neither is what was
    approved. FIREBASE_SERVICE_ACCOUNT_JSON is the whole service-account
    key file's contents (Cloud Run Jobs' --set-secrets gives you a env
    var, not a mounted file path)."""
    global _firebase_app
    if _firebase_app is None:
        cred = credentials.Certificate(json.loads(FIREBASE_SERVICE_ACCOUNT_JSON))
        _firebase_app = firebase_admin.initialize_app(cred)
    return firestore.client()


def _doc_to_dict(snap: firestore.DocumentSnapshot) -> dict:
    data = snap.to_dict() or {}
    data["id"] = snap.id
    return data


def _get_doc(collection: str, doc_id: str) -> dict | None:
    snap = get_firestore_client().collection(collection).document(doc_id).get()
    return _doc_to_dict(snap) if snap.exists else None


def _set_doc(collection: str, doc_id: str, fields: dict) -> None:
    """merge=True always — every SQL UPSERT this replaces has an explicit
    ON CONFLICT DO UPDATE SET listing specific columns, never a blind
    full-row overwrite; a plain (non-merge) .set() would silently delete
    any field not in `fields` (e.g. capital_config.paused, set by a
    different call site than upsert_capital_config) on every repeat
    write. merge=True is the faithful translation of "only touch these
    columns," matching Postgres's column-scoped UPDATE semantics."""
    get_firestore_client().collection(collection).document(doc_id).set(fields, merge=True)


def _insert_doc(collection: str, fields: dict) -> dict:
    ref = get_firestore_client().collection(collection).document()
    ref.set(fields)
    return fields | {"id": ref.id}


def _query(
    collection: str,
    filters: list[tuple] | None = None,
    order_by: str | None = None,
    desc: bool = False,
    limit: int | None = None,
) -> list[dict]:
    q = get_firestore_client().collection(collection)
    for field, op, value in filters or []:
        q = q.where(filter=firestore.FieldFilter(field, op, value))
    if order_by:
        q = q.order_by(
            order_by, direction=firestore.Query.DESCENDING if desc else firestore.Query.ASCENDING
        )
    if limit is not None:
        q = q.limit(limit)
    return [_doc_to_dict(d) for d in q.stream()]


def _get_docs_by_ids(collection: str, doc_ids: list) -> list[firestore.DocumentSnapshot]:
    """Batch get by doc ID — the natural translation of Postgres's
    `id = ANY(%s)`. Unlike the `in` query operator, Client.get_all() has
    no 30-item cap, so no chunking needed here."""
    if not doc_ids:
        return []
    client = get_firestore_client()
    refs = [client.collection(collection).document(str(doc_id)) for doc_id in doc_ids]
    return [snap for snap in client.get_all(refs) if snap.exists]


def _batch_set(collection: str, rows: list[dict], doc_id_fn=None) -> None:
    """Batch insert/upsert, chunked at Firestore's 500-writes-per-batch
    limit — the Firestore counterpart to _insert_rows' execute_values."""
    client = get_firestore_client()
    coll_ref = client.collection(collection)
    for i in range(0, len(rows), 500):
        batch = client.batch()
        for row in rows[i:i + 500]:
            doc_ref = coll_ref.document(doc_id_fn(row)) if doc_id_fn else coll_ref.document()
            batch.set(doc_ref, row)
        batch.commit()


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
    return _execute(lambda: _get_doc("capital_config", f"{mode}_{strategy_type}"))


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
    _set_doc("capital_config", f"{mode}_{strategy_type}", {
        "mode": mode,
        "strategy_type": strategy_type,
        "total_capital": total_capital,
        "capital_to_use": capital_to_use,
        "daily_profit_target": daily_profit_target,
        "max_daily_loss": max_daily_loss,
        "position_size_pct": position_size_pct,
        "max_concurrent_positions": max_concurrent_positions,
        "updated_at": datetime.now(timezone.utc),
    })


def get_active_strategy_types(mode: str) -> list[str]:
    """Strategy types with a seeded capital_config row for this mode --
    "active" = someone ran seed_config.py for it (src/seed_config.py).
    Callers intersect this with src.config.STRATEGY_PROFILES; this module
    stays DB-only and doesn't import config's registry. Dedup happens
    Python-side — Firestore has no SELECT DISTINCT, and this collection
    is small (one doc per mode+strategy_type combo)."""
    rows = _execute(lambda: _query("capital_config", [("mode", "==", mode)]))
    return sorted({r["strategy_type"] for r in rows})


# --- strategy_versions (immutable once created, see spec §3) ---


def get_latest_version(strategy_type: str = "default") -> dict | None:
    # status is only ever 'active'/'suspended' (update_strategy_version_status
    # is the sole writer) — 'active' equality reads the same as the old
    # 'status != suspended' filter but avoids a Firestore inequality-filter/
    # order_by interaction. Excludes suspended versions (Strategy Health
    # Engine, PROJECT_SPEC.md §3d) — without this filter, auto-suspension
    # would be a silent no-op since this is still an unfiltered "newest
    # row" query otherwise.
    rows = _execute(lambda: _query(
        "strategy_versions",
        [("strategy_type", "==", strategy_type), ("status", "==", "active")],
        order_by="version_number", desc=True, limit=1,
    ))
    return rows[0] if rows else None


def get_latest_promoted_version(strategy_type: str = "default") -> dict | None:
    rows = _execute(lambda: _query(
        "strategy_versions",
        [("strategy_type", "==", strategy_type), ("promoted_to_real", "==", True), ("status", "==", "active")],
        order_by="version_number", desc=True, limit=1,
    ))
    return rows[0] if rows else None


def insert_strategy_version(
    version_number: int,
    prompt_text: str,
    params_json: dict,
    notes: str | None = None,
    strategy_type: str = "default",
) -> dict:
    return _insert_doc("strategy_versions", {
        "version_number": version_number,
        "prompt_text": prompt_text,
        "params_json": params_json,
        "promoted_to_real": False,
        "notes": notes,
        "status": "active",
        "promotion_eligible": False,
        "strategy_type": strategy_type,
        "created_at": datetime.now(timezone.utc),
    })


def promote_version(version_id: int) -> None:
    _set_doc("strategy_versions", str(version_id), {"promoted_to_real": True})


def set_strategy_version_promotion_eligible(version_id: int, eligible: bool) -> None:
    """Mirrors src/learning/promotion_gate.py::evaluate_promotion()'s
    PROMOTE/REJECT/EXTEND_VALIDATION verdict onto the row for visibility —
    evolution_agent.py calls promote_version() itself immediately after on
    PROMOTE, fully automatically, no human step. This flag is a record of
    the decision, not a queue awaiting manual action."""
    _set_doc("strategy_versions", str(version_id), {"promotion_eligible": eligible})


def get_all_strategy_versions() -> list[dict]:
    return _query("strategy_versions", order_by="version_number", desc=True)


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
    strategy_type: str,
    stop_loss_price: float | None = None,
    take_profit_price: float | None = None,
    entry_slippage_pct: float | None = None,
    market_regime: str | None = None,
) -> dict:
    """strategy_type is a required param here (not optional/additive like
    the read-side functions below) — Firestore has no JOIN, so trades
    carries its own strategy_type field instead of deriving one via
    version_id -> strategy_versions at read time. Sole call site
    (orchestrator.py) is already inside the per-strategy_type loop."""
    return _insert_doc("trades", {
        "mode": mode,
        "version_id": version_id,
        "strategy_type": strategy_type,
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
        "opened_at": datetime.now(timezone.utc),
        "mfe_pct": 0,
        "mae_pct": 0,
    })


def close_trade(
    trade_id: int, exit_price: float, pnl: float, status: str = "closed", exit_reason: str | None = None
) -> None:
    _set_doc("trades", str(trade_id), {
        "exit_price": exit_price,
        "pnl": pnl,
        "status": status,
        "exit_reason": exit_reason,
        "closed_at": datetime.now(timezone.utc),
    })


def update_trade_excursion(trade_id: int, mfe_pct: float, mae_pct: float) -> None:
    _set_doc("trades", str(trade_id), {"mfe_pct": mfe_pct, "mae_pct": mae_pct})


def get_open_trades(mode: str, strategy_type: str | None = None) -> list[dict]:
    filters = [("mode", "==", mode), ("status", "==", "open")]
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _execute(lambda: _query("trades", filters))


def get_recently_closed_trades(mode: str, since: datetime, strategy_type: str | None = None) -> list[dict]:
    """Closed/flattened trades since `since` — mode-scoped, not
    version-scoped (unlike get_closed_trades), since strategy versions
    rotate and the learning engine's catch-up pass needs to see across
    versions of the SAME strategy_type. Bounded by the caller's `since`
    (LEARNING_CATCHUP_LOOKBACK_HOURS for process_closed_trades,
    LEARNING_HISTORY_WINDOW_DAYS for stats bucket recompute) so this
    never becomes a full-table scan. strategy_type is additive/optional —
    omitted keeps today's exact mode-wide (cross-strategy-type) query."""
    filters = [("mode", "==", mode), ("status", "in", ["closed", "flattened"]), ("closed_at", ">=", since)]
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _query("trades", filters)


def get_recent_trades(mode: str, limit: int = 50, strategy_type: str | None = None) -> list[dict]:
    filters = [("mode", "==", mode)]
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _query("trades", filters, order_by="opened_at", desc=True, limit=limit)


def get_closed_trades(mode: str, version_id: int) -> list[dict]:
    return _query("trades", [
        ("mode", "==", mode), ("version_id", "==", version_id), ("status", "in", ["closed", "flattened"]),
    ])


# --- risk_check_lock ---


def try_acquire_risk_check_lock(mode: str, stale_after_seconds: int = 180) -> bool:
    """Non-blocking mutex for orchestrator.run_risk_check(mode) — a
    tightened polling cadence means one run can still be in flight when
    the next fires. `stale_after_seconds` self-heals a lock left held by
    a crashed process rather than deadlocking the mode forever (ponytail:
    fixed 3min ceiling, revisit only if a real run ever legitimately
    takes that long).

    A plain get-then-set would be a TOCTOU race — two concurrent runs
    could both read "unlocked" before either writes. Firestore's
    optimistic-concurrency transaction closes that gap: if a concurrent
    transaction commits a write to this doc between this one's read and
    its own commit, this one aborts and the client library retries
    _attempt, which re-reads the winner's fresh write and correctly
    returns False. (Migration 0015's reason for avoiding
    pg_advisory_lock — PgBouncer transaction-mode pooling breaking
    session-level locks — doesn't apply to Firestore, which has no
    connection-pooling session concept at all.)"""
    doc_ref = get_firestore_client().collection("risk_check_lock").document(mode)

    @firestore.transactional
    def _attempt(transaction: firestore.Transaction) -> bool:
        snapshot = doc_ref.get(transaction=transaction)
        locked_at = snapshot.get("locked_at") if snapshot.exists else None
        now = datetime.now(timezone.utc)
        if locked_at is not None and (now - locked_at).total_seconds() < stale_after_seconds:
            return False
        transaction.set(doc_ref, {"mode": mode, "locked_at": now}, merge=True)
        return True

    return _attempt(get_firestore_client().transaction())


def release_risk_check_lock(mode: str) -> None:
    _set_doc("risk_check_lock", mode, {"locked_at": None})


# --- daily_pnl ---


def _daily_pnl_doc_id(day: Date, mode: str, strategy_type: str) -> str:
    return f"{day.isoformat()}_{mode}_{strategy_type}"


def get_daily_pnl(day: Date, mode: str, strategy_type: str = "default") -> dict | None:
    return _execute(lambda: _get_doc("daily_pnl", _daily_pnl_doc_id(day, mode, strategy_type)))


def upsert_daily_pnl(
    day: Date,
    mode: str,
    realized_pnl: float,
    trades_count: int,
    target_hit: bool,
    circuit_breaker_triggered: bool,
    strategy_type: str = "default",
) -> None:
    _set_doc("daily_pnl", _daily_pnl_doc_id(day, mode, strategy_type), {
        "date": day.isoformat(),
        "mode": mode,
        "strategy_type": strategy_type,
        "realized_pnl": realized_pnl,
        "trades_count": trades_count,
        "target_hit": target_hit,
        "circuit_breaker_triggered": circuit_breaker_triggered,
    })


# --- agent_logs ---


def log_agent_event(
    agent_name: str, level: str, message: str, raw_llm_response: Any = None
) -> None:
    _insert_doc("agent_logs", {
        "timestamp": datetime.now(timezone.utc),
        "agent_name": agent_name,
        "level": level,
        "message": message,
        "raw_llm_response": raw_llm_response,
    })


# --- model_usage ---


def log_model_usage(events: list[ModelUsageEvent]) -> None:
    if not events:
        return
    now = datetime.now(timezone.utc)
    rows = [
        {
            "timestamp": now,
            "model_used": e.model_used,
            "fallback_reason": e.fallback_reason,
            "latency_ms": e.latency_ms,
            "success": e.success,
        }
        for e in events
    ]
    _batch_set("model_usage", rows)


def get_recent_model_usage(limit: int = 500) -> list[dict]:
    return _query("model_usage", order_by="timestamp", desc=True, limit=limit)


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
    strategy_type: str,
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
    write path. strategy_type is required (not optional/additive) for the
    same reason as open_trade's — Firestore has no JOIN, so this table
    carries its own strategy_type field. Sole call site (orchestrator.py)
    is already inside the per-strategy_type loop."""
    return _insert_doc("opportunity_evaluations", {
        "timestamp": datetime.now(timezone.utc),
        "mode": mode,
        "symbol": symbol,
        "version_id": version_id,
        "strategy_type": strategy_type,
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
    doc_id = f"{mode}_{strategy_type}_{dimension_type}_{dimension_value}"
    _set_doc("learning_statistics", doc_id, {
        "mode": mode,
        "strategy_type": strategy_type,
        "dimension_type": dimension_type,
        "dimension_value": dimension_value,
        **stats,
        "computed_at": datetime.now(timezone.utc),
    })


def get_learning_statistics(
    mode: str, dimension_type: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    filters = [("mode", "==", mode), ("strategy_type", "==", strategy_type)]
    if dimension_type is not None:
        filters.append(("dimension_type", "==", dimension_type))
    return _query("learning_statistics", filters)


# --- feature_importance ---


def upsert_feature_importance(
    mode: str,
    feature_name: str,
    correlation_score: float,
    sample_count: int,
    timeframe: str,
    strategy_type: str = "default",
) -> None:
    doc_id = f"{mode}_{strategy_type}_{feature_name}_{timeframe}"
    _set_doc("feature_importance", doc_id, {
        "mode": mode,
        "strategy_type": strategy_type,
        "feature_name": feature_name,
        "timeframe": timeframe,
        "correlation_score": correlation_score,
        "sample_count": sample_count,
        "computed_at": datetime.now(timezone.utc),
    })


def get_feature_importance(
    mode: str, timeframe: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    filters = [("mode", "==", mode), ("strategy_type", "==", strategy_type)]
    if timeframe is not None:
        filters.append(("timeframe", "==", timeframe))
    return _query("feature_importance", filters)


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
    query, matching the audit trail's own "show everything" default. No
    JOIN needed (unlike the old Postgres version) — Firestore reads
    strategy_type straight off the row, written there by
    log_opportunity_evaluation."""
    filters = [("mode", "==", mode)]
    if trade_id is not None:
        filters.append(("trade_id", "==", trade_id))
    if symbol is not None:
        filters.append(("symbol", "==", symbol))
    if since is not None:
        filters.append(("timestamp", ">=", since))
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _query("opportunity_evaluations", filters, order_by="timestamp")


def get_entry_evaluation_for_trade(trade_id: int) -> dict | None:
    rows = _query(
        "opportunity_evaluations", [("trade_id", "==", trade_id), ("final_decision", "==", "buy")], limit=1
    )
    return rows[0] if rows else None


# --- confidence_calibration ---


def get_confidence_calibration_for_evaluation(opportunity_evaluation_id: int) -> dict | None:
    rows = _query(
        "confidence_calibration", [("opportunity_evaluation_id", "==", opportunity_evaluation_id)], limit=1
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
    _insert_doc("confidence_calibration", {
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
        "created_at": datetime.now(timezone.utc),
    })


# --- recommendations ---


def get_latest_recommendation(mode: str, metric_name: str, strategy_type: str = "default") -> dict | None:
    rows = _query(
        "recommendations",
        [("mode", "==", mode), ("metric_name", "==", metric_name), ("strategy_type", "==", strategy_type)],
        order_by="created_at", desc=True, limit=1,
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
    _insert_doc("recommendations", {
        "mode": mode,
        "strategy_type": strategy_type,
        "metric_name": metric_name,
        "current_value": current_value,
        "recommended_value": recommended_value,
        "rationale": rationale,
        "sample_size": sample_size,
        "status": "pending",
        "category": category,
        "confidence": confidence,
        "evidence": evidence,
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc),
    })


def get_recommendations(
    mode: str, status: str | None = None, category: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    filters = [("mode", "==", mode), ("strategy_type", "==", strategy_type)]
    if status is not None:
        filters.append(("status", "==", status))
    if category is not None:
        filters.append(("category", "==", category))
    return _query("recommendations", filters, order_by="created_at", desc=True)


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
    return _insert_doc("strategy_simulations", {
        "recommendation_batch_id": recommendation_batch_id,
        "mode": mode,
        "strategy_type": strategy_type,
        "train_window_start": train_window_start,
        "train_window_end": train_window_end,
        "test_window_start": test_window_start,
        "test_window_end": test_window_end,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "p_value": p_value,
        "passed": passed,
        "research_note": research_note,
        "validation_detail": validation_detail,
        "created_at": datetime.now(timezone.utc),
    })


def get_strategy_simulations(mode: str, passed: bool | None = None, strategy_type: str = "default") -> list[dict]:
    filters = [("mode", "==", mode), ("strategy_type", "==", strategy_type)]
    if passed is not None:
        filters.append(("passed", "==", passed))
    return _query("strategy_simulations", filters, order_by="created_at", desc=True)


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
    return _insert_doc("adaptive_strategy_versions", {
        "mode": mode,
        "strategy_type": strategy_type,
        "version_number": version_number,
        "params_json": params_json,
        "source_recommendation_batch_id": source_recommendation_batch_id,
        "source_simulation_id": source_simulation_id,
        "status": "candidate",
        "notes": notes,
        "fitness_score": fitness_score,
        "created_at": datetime.now(timezone.utc),
    })


def get_adaptive_strategy_versions(
    mode: str, status: str | None = None, strategy_type: str = "default"
) -> list[dict]:
    filters = [("mode", "==", mode), ("strategy_type", "==", strategy_type)]
    if status is not None:
        filters.append(("status", "==", status))
    return _query("adaptive_strategy_versions", filters, order_by="created_at", desc=True)


def get_latest_adaptive_strategy_version(mode: str, strategy_type: str = "default") -> dict | None:
    rows = _query(
        "adaptive_strategy_versions",
        [("mode", "==", mode), ("strategy_type", "==", strategy_type)],
        order_by="version_number", desc=True, limit=1,
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
    _set_doc("trade_evaluations", str(trade_id), {
        "trade_id": trade_id,
        "predicted_confidence": predicted_confidence,
        "predicted_opportunity_score": predicted_opportunity_score,
        "actual_outcome_won": actual_outcome_won,
        "confidence_was_accurate": confidence_was_accurate,
        "opportunity_score_was_accurate": opportunity_score_was_accurate,
        "risk_assessment": risk_assessment,
        "stop_loss_assessment": stop_loss_assessment,
        "target_assessment": target_assessment,
        "evaluated_at": datetime.now(timezone.utc),
    })


def get_trade_evaluation_ids(trade_ids: list[int]) -> set[int]:
    return {int(snap.id) for snap in _get_docs_by_ids("trade_evaluations", trade_ids)}


def get_trade_evaluations(trade_ids: list[int]) -> list[dict]:
    """Full trade_evaluations rows (confidence_was_accurate/
    opportunity_score_was_accurate) for drift_detection.py — distinct from
    get_trade_evaluation_ids, which only returns the id set for the
    already-evaluated membership check process_closed_trades() needs."""
    return [_doc_to_dict(snap) for snap in _get_docs_by_ids("trade_evaluations", trade_ids)]


def get_trades_by_ids(trade_ids: list[int]) -> list[dict]:
    return [_doc_to_dict(snap) for snap in _get_docs_by_ids("trades", trade_ids)]


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
    filters = [("mode", "==", mode), ("final_decision", "==", "buy"), ("timestamp", ">=", since)]
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _query("opportunity_evaluations", filters)


def get_hold_evaluations_since(mode: str, since: datetime, strategy_type: str | None = None) -> list[dict]:
    """Non-trade opportunity_evaluations rows (final_decision='hold') since
    `since` — every scanned-but-not-traded candidate, each carrying its own
    reason/risk_manager_result (Root Cause Analysis, Scientific Strategy
    Optimization Framework). Same shape as get_entry_evaluations_since,
    filtering the opposite final_decision value."""
    filters = [("mode", "==", mode), ("final_decision", "==", "hold"), ("timestamp", ">=", since)]
    if strategy_type is not None:
        filters.append(("strategy_type", "==", strategy_type))
    return _query("opportunity_evaluations", filters)


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
    _set_doc("strategy_versions", str(version_id), {"status": status})


def get_active_strategy_versions() -> list[dict]:
    # status is only ever 'active'/'suspended' — see get_latest_version's
    # comment on why 'active' equality replaces the old '!= suspended'.
    return _query("strategy_versions", [("status", "==", "active")], order_by="version_number", desc=True)


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
