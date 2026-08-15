-- Initial schema. See PROJECT_SPEC.md §6 for field-by-field rationale.

create table capital_config (
  mode                    text primary key,       -- 'paper' | 'real'
  total_capital           numeric not null,
  capital_to_use          numeric not null,
  daily_profit_target     numeric not null,
  max_daily_loss          numeric not null,
  position_size_pct       numeric not null default 10,
  max_concurrent_positions int not null default 5,
  updated_at              timestamptz not null default now()
);

create table strategy_versions (
  id                 bigserial primary key,
  version_number     int not null,
  prompt_text        text not null,
  params_json        jsonb not null default '{}',
  promoted_to_real   boolean not null default false,
  notes              text,
  created_at         timestamptz not null default now()
);

create table trades (
  id             bigserial primary key,
  mode           text not null,
  version_id     bigint not null references strategy_versions(id),
  symbol         text not null,
  side           text not null,
  qty            numeric not null,
  entry_price    numeric not null,
  exit_price     numeric,
  pnl            numeric,
  fees           numeric not null default 0,
  status         text not null,
  opened_at      timestamptz not null default now(),
  closed_at      timestamptz,
  reasoning_text text
);

create table daily_pnl (
  date                      date not null,
  mode                      text not null,
  realized_pnl              numeric not null default 0,
  trades_count              int not null default 0,
  target_hit                boolean not null default false,
  circuit_breaker_triggered boolean not null default false,
  primary key (date, mode)
);

create table agent_logs (
  id               bigserial primary key,
  timestamp        timestamptz not null default now(),
  agent_name       text not null,
  level            text not null,
  message          text not null,
  raw_llm_response jsonb
);

create table model_usage (
  id              bigserial primary key,
  timestamp       timestamptz not null default now(),
  model_used      text not null,
  fallback_reason text,
  latency_ms      int not null,
  success         boolean not null
);

create index trades_mode_status_idx on trades (mode, status);
create index trades_opened_at_idx on trades (opened_at);
create index agent_logs_timestamp_idx on agent_logs (timestamp);
create index model_usage_timestamp_idx on model_usage (timestamp);
