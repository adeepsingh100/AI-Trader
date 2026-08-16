-- Quant-first pipeline: one row per scanned symbol per cycle, logged
-- regardless of outcome. llm_decision/llm_reasoning/llm_raw_response are
-- null whenever a symbol never reached LLM validation (below score
-- threshold or not a top candidate) — most rows, by design, since the
-- whole point of the refactor is fewer LLM calls. See PROJECT_SPEC.md §3.

create table opportunity_evaluations (
  id                  bigserial primary key,
  timestamp           timestamptz not null default now(),
  mode                text not null,
  symbol              text not null,
  version_id          bigint not null references strategy_versions(id),
  features            jsonb not null,             -- compute_multi_timeframe_features() output
  trend_score         numeric,
  momentum_score      numeric,
  volume_score        numeric,
  volatility_score    numeric,
  risk_score          numeric,
  opportunity_score   numeric,
  llm_decision        text,                       -- 'accept' | 'reject' | null (null = never reached LLM)
  llm_reasoning       text,
  llm_raw_response    jsonb,                      -- full parsed verdict, null when no LLM call
  risk_manager_result text,                       -- 'size' | 'block_circuit_breaker' | 'block_max_positions' | 'block_capital_limit' | null
  final_decision      text not null,              -- 'buy' | 'sell' | 'hold' | 'circuit_breaker'
  reason              text
);

create index opportunity_evaluations_mode_timestamp_idx on opportunity_evaluations (mode, timestamp);
create index opportunity_evaluations_symbol_idx on opportunity_evaluations (symbol);

alter table opportunity_evaluations enable row level security;

create policy "public read" on opportunity_evaluations for select using (true);

grant select on opportunity_evaluations to anon, authenticated;
