from unittest.mock import Mock, patch

import pytest

from src.db import models
from src.resilience import (
    CircuitBreakerOpenError,
    call_with_circuit_breaker,
    check_circuit_breaker,
    record_failure,
    record_success,
    retry_with_backoff,
)


def test_retry_with_backoff_returns_on_first_success():
    fn = Mock(return_value="ok")
    assert retry_with_backoff(fn, max_attempts=3, base_delay=0) == "ok"
    assert fn.call_count == 1


def test_retry_with_backoff_retries_then_succeeds():
    fn = Mock(side_effect=[ValueError("boom"), "ok"])
    with patch("src.resilience.time.sleep") as mock_sleep:
        result = retry_with_backoff(fn, max_attempts=3, base_delay=1, exceptions=(ValueError,))
    assert result == "ok"
    assert fn.call_count == 2
    mock_sleep.assert_called_once_with(1)  # base_delay * 2**0


def test_retry_with_backoff_raises_last_exception_after_exhausting_attempts():
    fn = Mock(side_effect=ValueError("always fails"))
    with patch("src.resilience.time.sleep"):
        with pytest.raises(ValueError, match="always fails"):
            retry_with_backoff(fn, max_attempts=3, base_delay=0, exceptions=(ValueError,))
    assert fn.call_count == 3


def test_retry_with_backoff_backoff_grows_exponentially():
    fn = Mock(side_effect=[ValueError(), ValueError(), "ok"])
    with patch("src.resilience.time.sleep") as mock_sleep:
        retry_with_backoff(fn, max_attempts=3, base_delay=2, exceptions=(ValueError,))
    mock_sleep.assert_any_call(2)  # 2 * 2**0
    mock_sleep.assert_any_call(4)  # 2 * 2**1


def test_check_circuit_breaker_noop_when_not_tripped(monkeypatch):
    monkeypatch.setattr(
        models, "get_circuit_breaker_state", lambda c: {"consecutive_failures": 1, "tripped_until": None}
    )
    check_circuit_breaker("coindcx_api")  # must not raise


def test_check_circuit_breaker_raises_when_tripped(monkeypatch):
    monkeypatch.setattr(
        models,
        "get_circuit_breaker_state",
        lambda c: {"consecutive_failures": 5, "tripped_until": 99999999999999},
    )
    with pytest.raises(CircuitBreakerOpenError):
        check_circuit_breaker("coindcx_api")


def test_check_circuit_breaker_fails_open_on_db_error(monkeypatch):
    def _raise(c):
        raise RuntimeError("db down")

    monkeypatch.setattr(models, "get_circuit_breaker_state", _raise)
    check_circuit_breaker("supabase")  # must not raise — fails open


def test_record_failure_fails_open_on_db_error(monkeypatch):
    def _raise(c):
        raise RuntimeError("db down")

    monkeypatch.setattr(models, "get_circuit_breaker_state", _raise)
    record_failure("supabase")  # must not raise


def test_record_failure_trips_breaker_at_threshold(monkeypatch):
    monkeypatch.setattr(models, "get_circuit_breaker_state", lambda c: {"consecutive_failures": 4})
    upsert_mock = Mock()
    monkeypatch.setattr(models, "upsert_circuit_breaker_state", upsert_mock)
    with patch("src.resilience.CIRCUIT_BREAKER_FAILURE_THRESHOLD", 5):
        record_failure("coindcx_api")
    upserted = upsert_mock.call_args
    assert upserted.args[1] == 5  # consecutive_failures
    assert upserted.args[2] is not None  # tripped_until now set


def test_record_success_resets_breaker(monkeypatch):
    reset_mock = Mock()
    monkeypatch.setattr(models, "reset_circuit_breaker", reset_mock)
    record_success("llm")
    reset_mock.assert_called_once_with("llm")


def test_call_with_circuit_breaker_records_failure_and_reraises(monkeypatch):
    monkeypatch.setattr(models, "get_circuit_breaker_state", lambda c: None)
    upsert_mock = Mock()
    monkeypatch.setattr(models, "upsert_circuit_breaker_state", upsert_mock)
    fn = Mock(side_effect=ValueError("boom"))
    with patch("src.resilience.time.sleep"):
        with pytest.raises(ValueError):
            call_with_circuit_breaker("coindcx_api", fn, max_attempts=1, exceptions=(ValueError,))
    upsert_mock.assert_called_once()


def test_call_with_circuit_breaker_records_success(monkeypatch):
    monkeypatch.setattr(models, "get_circuit_breaker_state", lambda c: None)
    reset_mock = Mock()
    monkeypatch.setattr(models, "reset_circuit_breaker", reset_mock)
    fn = Mock(return_value="ok")
    result = call_with_circuit_breaker("coindcx_api", fn)
    assert result == "ok"
    reset_mock.assert_called_once_with("coindcx_api")
