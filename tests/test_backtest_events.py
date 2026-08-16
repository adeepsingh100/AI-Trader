from datetime import datetime, timezone

from src.backtest.events import EventQueue, MarketEvent, TimeEvent


def test_event_queue_fifo_order():
    q = EventQueue()
    t1 = TimeEvent(time=datetime(2024, 1, 1, tzinfo=timezone.utc))
    t2 = TimeEvent(time=datetime(2024, 1, 2, tzinfo=timezone.utc))
    q.put(t1)
    q.put(t2)
    assert q.get() is t1
    assert q.get() is t2
    assert q.get() is None


def test_event_queue_bool_and_len():
    q = EventQueue()
    assert not q
    assert len(q) == 0
    q.put(TimeEvent(time=datetime(2024, 1, 1, tzinfo=timezone.utc)))
    assert q
    assert len(q) == 1


def test_market_event_is_frozen():
    ev = MarketEvent(time=datetime(2024, 1, 1, tzinfo=timezone.utc), symbol="BTCINR", features_by_tf={}, last_price=100.0)
    try:
        ev.last_price = 200.0
        assert False, "should not be able to mutate a frozen dataclass"
    except AttributeError:
        pass
