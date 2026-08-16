import pytest

from src.backtest.simulation_clock import SimulationClock, is_bar_closed, timeframe_duration_ms


def test_timeframe_duration_ms_known_values():
    assert timeframe_duration_ms("1m") == 60_000
    assert timeframe_duration_ms("15m") == 15 * 60_000
    assert timeframe_duration_ms("1h") == 60 * 60_000
    assert timeframe_duration_ms("1d") == 24 * 60 * 60_000


def test_timeframe_duration_ms_rejects_unsupported_interval():
    with pytest.raises(ValueError):
        timeframe_duration_ms("5m")


# --- the single most safety-critical property in this feature ---


def test_is_bar_closed_still_forming_bar_not_visible():
    bar_open = 1_000_000_000
    duration = timeframe_duration_ms("1m")
    # exactly at open time -> definitely still forming
    assert is_bar_closed(bar_open, "1m", as_of_ms=bar_open) is False
    # one ms before it would close -> still forming
    assert is_bar_closed(bar_open, "1m", as_of_ms=bar_open + duration - 1) is False


def test_is_bar_closed_fully_closed_bar_is_visible():
    bar_open = 1_000_000_000
    duration = timeframe_duration_ms("1m")
    # exactly at close time -> closed
    assert is_bar_closed(bar_open, "1m", as_of_ms=bar_open + duration) is True
    # well after close -> closed
    assert is_bar_closed(bar_open, "1m", as_of_ms=bar_open + duration + 999_999) is True


def test_clock_ticks_advance_by_timeframe_duration_inclusive_of_end():
    clock = SimulationClock(start_ms=0, end_ms=180_000, tick_timeframe="1m")
    ticks = list(clock.ticks())
    assert ticks == [0, 60_000, 120_000, 180_000]


def test_clock_now_reflects_current_tick():
    clock = SimulationClock(start_ms=0, end_ms=120_000, tick_timeframe="1m")
    for t in clock.ticks():
        assert clock.now_ms == t


def test_clock_rejects_start_ge_end():
    with pytest.raises(ValueError):
        SimulationClock(start_ms=100, end_ms=100, tick_timeframe="1m")


def test_today_ist_is_simulation_time_not_wall_clock():
    # 2024-01-01T00:00:00Z is 2024-01-01 05:30 IST — same calendar day.
    clock = SimulationClock(start_ms=1704067200000, end_ms=1704067200000 + 60_000, tick_timeframe="1m")
    assert clock.today_ist().isoformat() == "2024-01-01"
