import re
from unittest.mock import MagicMock


def _fake_cursor(rows=None, rowcount=0):
    """A psycopg2-cursor-shaped mock: execute() records call args,
    fetchall() returns `rows`, description is truthy iff the statement
    produced columns (SELECT/RETURNING) — inferred from whether `rows`
    was given, matching every call site in src/db/models.py (a plain
    DELETE never passes rows here, everything else does, even an empty
    list for a SELECT that matched nothing)."""
    cur = MagicMock()
    cur.__enter__.return_value = cur
    cur.__exit__.return_value = False
    cur.fetchall.return_value = rows if rows is not None else []
    cur.rowcount = rowcount
    cur.description = rows is not None
    return cur


def _fake_connection(rows=None, rowcount=0, cur=None):
    """Builds a fake connection whose .cursor(...) returns a fake cursor
    (rows/rowcount configure it directly, or pass `cur` explicitly to
    reuse one across multiple calls in a single test). monkeypatch this
    onto models.get_client — every _run_query/_run_write call in
    src/db/models.py goes through conn.cursor()/.commit()/.rollback(),
    all of which are plain MagicMock no-ops here."""
    cur = cur or _fake_cursor(rows=rows, rowcount=rowcount)
    conn = MagicMock()
    conn.cursor.return_value = cur
    conn.closed = False
    return conn, cur


def _last_execute(cur):
    """(sql, params) from the most recent cur.execute(...) call."""
    args = cur.execute.call_args[0]
    return args[0], (args[1] if len(args) > 1 else ())


def _inserted_row(cur):
    """Reconstructs {column: value} from the most recent `INSERT INTO t
    (a, b, ...) VALUES (%s, %s, ...)` call's SQL + positional params —
    lets tests assert on column values by name, same ergonomics the
    pre-migration Supabase-builder-based tests had (asserting on a plain
    inserted dict), instead of every test hand-counting positional tuple
    indices. Works for both plain INSERT and INSERT ... ON CONFLICT
    (upsert) statements, since the column list is always the first
    parenthesized group right after the table name."""
    sql, params = _last_execute(cur)
    match = re.search(r"INSERT INTO \w+\s*\(([^)]+)\)", sql)
    cols = [c.strip() for c in match.group(1).split(",")]
    return dict(zip(cols, params))


def _updated_row(cur):
    """Reconstructs {column: value} from the SET clause of the most
    recent `UPDATE t SET a = %s, b = %s WHERE ...` call — WHERE-clause
    params (always last) are left out; index into the raw params tuple
    directly if a test needs one of those."""
    sql, params = _last_execute(cur)
    set_match = re.search(r"SET (.+?) WHERE", sql, re.DOTALL)
    set_cols = [c.split("=")[0].strip() for c in set_match.group(1).split(",")]
    return dict(zip(set_cols, params[: len(set_cols)]))
