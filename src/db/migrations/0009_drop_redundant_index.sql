-- Refinement pass cleanup. 0007_backtesting_engine.sql created both a
-- unique index (backtest_portfolio_snapshots_run_time_key, on run_id +
-- snapshot_time) and a separate plain index on the exact same column
-- tuple (backtest_portfolio_snapshots_run_idx) — fully redundant, any
-- query the plain index could serve is already served by the unique one.
-- Pure cleanup, no behavior change.
drop index if exists backtest_portfolio_snapshots_run_idx;
