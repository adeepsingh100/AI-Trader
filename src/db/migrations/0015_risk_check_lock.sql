-- Mutex for run_risk_check(mode), one row per mode. Cloud Scheduler's
-- risk-check cadence is dropping from 5min to 1min (see deploy/README.md)
-- to close the gap between an exit condition hitting and the next poll —
-- a stale/overlapping run must not double-close the same position. A
-- plain row + UPDATE...WHERE, not pg_advisory_lock: DATABASE_URL is
-- Neon's pooled (PgBouncer transaction-mode) connection string, and
-- session-level advisory locks aren't safe across a pooled connection
-- (a later query in the same "session" can land on a different physical
-- backend). This table works correctly under pooling because each
-- acquire/release is a single self-contained UPDATE.

create table if not exists risk_check_lock (
  mode text primary key,
  locked_at timestamptz
);

insert into risk_check_lock (mode, locked_at) values
  ('paper', null),
  ('real', null)
on conflict (mode) do nothing;
