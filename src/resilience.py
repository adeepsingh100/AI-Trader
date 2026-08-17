"""Generic retry/backoff + a DB-backed circuit breaker, shared by every
external call in this repo that talks to CoinDCX or Supabase. Scoped to
what's real for a stateless ~10-minute GitHub Actions cron job (no
persistent daemon exists to checkpoint) — see PROJECT_SPEC.md §3d.

groq_client.py's LLM chain keeps its own retry loop (it needs a
ModelUsageEvent per attempt for model_usage logging, a genuinely different
shape from a flat retry), but calls this module's backoff_delay() so the
exponential-backoff formula itself lives in one place."""

from __future__ import annotations

import sys
import time
from typing import Callable, TypeVar

from src.config import (
    CIRCUIT_BREAKER_COOLDOWN_SECONDS,
    CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_ATTEMPTS,
)

T = TypeVar("T")


def backoff_delay(base: float, attempt: int) -> float:
    return base * (2**attempt)


def log_fail_open(component: str, err: Exception) -> None:
    """A fail-open `except Exception: pass` site's failure is otherwise
    invisible — this repo runs as GitHub Actions cron, so stderr already
    surfaces in the run log, no new logging framework needed. Never raises,
    never changes what the caller returns."""
    print(f"[fail-open] {component}: {type(err).__name__}: {err}", file=sys.stderr)


def retry_with_backoff(
    fn: Callable[[], T],
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY_SECONDS,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Calls fn() up to max_attempts times, sleeping backoff_delay(base_delay,
    attempt) between attempts. Re-raises the last exception if every attempt
    fails."""
    last_exc: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except exceptions as e:
            last_exc = e
            if attempt < max_attempts - 1:
                time.sleep(backoff_delay(base_delay, attempt))
    assert last_exc is not None
    raise last_exc


class CircuitBreakerOpenError(RuntimeError):
    """Raised by check_circuit_breaker() when a component is currently
    tripped — the caller should skip that component's work gracefully
    rather than let requests pile up against something already failing."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def check_circuit_breaker(component: str) -> None:
    """Raises CircuitBreakerOpenError if `component` is currently tripped.
    Reads circuit_breaker_state via src.db.models — a local import so this
    module has no hard dependency on the DB layer for retry_with_backoff's
    callers that don't need the breaker. Fails OPEN (does nothing) if the
    read itself errors — a DB outage can't block itself from being
    recorded, and can't lock the bot out of trying."""
    from src.db import models

    try:
        state = models.get_circuit_breaker_state(component)
    except Exception as e:
        log_fail_open(component, e)
        return
    if state is None:
        return
    tripped_until = state.get("tripped_until")
    if tripped_until and tripped_until > _now_ms():
        raise CircuitBreakerOpenError(f"{component} circuit breaker open until {tripped_until}")


def record_success(component: str) -> None:
    """Resets a component's consecutive-failure count. Fails open (log and
    continue) on its own write error."""
    from src.db import models

    try:
        models.reset_circuit_breaker(component)
    except Exception as e:
        log_fail_open(component, e)


def record_failure(component: str) -> None:
    """Increments a component's consecutive-failure count, tripping the
    breaker for CIRCUIT_BREAKER_COOLDOWN_SECONDS once
    CIRCUIT_BREAKER_FAILURE_THRESHOLD is reached. Fails open on its own
    write error — never raises."""
    from src.db import models

    try:
        state = models.get_circuit_breaker_state(component) or {}
        consecutive_failures = (state.get("consecutive_failures") or 0) + 1
        tripped_until = None
        if consecutive_failures >= CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            tripped_until = _now_ms() + CIRCUIT_BREAKER_COOLDOWN_SECONDS * 1000
        models.upsert_circuit_breaker_state(component, consecutive_failures, tripped_until)
    except Exception as e:
        log_fail_open(component, e)


def call_with_circuit_breaker(
    component: str,
    fn: Callable[[], T],
    max_attempts: int = RETRY_MAX_ATTEMPTS,
    base_delay: float = RETRY_BASE_DELAY_SECONDS,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """The combined pattern most callers actually want: check the breaker,
    retry-with-backoff, record the outcome. Raises CircuitBreakerOpenError
    without attempting a call if the breaker is already open."""
    check_circuit_breaker(component)
    try:
        result = retry_with_backoff(fn, max_attempts, base_delay, exceptions)
    except exceptions:
        record_failure(component)
        raise
    record_success(component)
    return result
