-- data_agent.py re-fetches and re-validates the same rolling
-- FEATURE_CANDLE_LIMIT-bar window every cycle, so any still-in-window
-- candle with a real issue (e.g. zero_volume on a thin pair) was being
-- re-logged every cycle forever -- unbounded growth, same disk-fill
-- failure mode the out_of_order sort bug caused in 26 hours, just
-- slower. candle_time + a dedup unique index makes repeat logging of
-- the same candle's same issue a no-op (ON CONFLICT DO NOTHING).
-- NULL candle_time (batch-level issues, e.g. exchange_outage) is never
-- deduped -- Postgres treats NULLs as distinct in a unique index, which
-- is fine, those are rare.

alter table data_quality_log add column candle_time bigint;

create unique index data_quality_log_dedup_idx
  on data_quality_log (pair, interval, issue_type, candle_time);
