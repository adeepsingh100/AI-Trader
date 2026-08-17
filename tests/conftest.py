from unittest.mock import Mock

_FLUENT_METHODS = (
    "select", "eq", "neq", "insert", "update", "upsert", "order", "limit", "in_", "gte", "lte",
)


def _fluent_mock(execute_result):
    """A mock whose chained Supabase query-builder methods all return
    itself, so call args land on the same mock and .execute() returns a
    fixed result. Was independently redefined in test_db_models.py,
    test_db_models_reliability.py, and test_db_models_backtest.py — this
    is the union of chained methods any of the three needed."""
    m = Mock()
    for method in _FLUENT_METHODS:
        getattr(m, method).return_value = m
    m.execute.return_value = Mock(data=execute_result)
    return m
