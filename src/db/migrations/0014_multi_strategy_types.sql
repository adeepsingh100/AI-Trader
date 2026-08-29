-- Multi-strategy-type support: the bot previously ran exactly one
-- strategy lineage, period (strategy_versions had no mode column at
-- all, let alone anything narrower). strategy_type is a new dimension
-- orthogonal to mode, letting several genuinely different strategies
-- (e.g. today's short-term "default" plus a longer-horizon "swing")
-- run concurrently with independent capital, circuit breaker, and
-- learning evidence -- so one strategy's trades/stats never corrupt
-- another's. Ships with every existing row backfilled 'default';
-- adding a second strategy_type is a seed_config.py run, not a
-- migration.
--
-- strategy_versions/opportunity_evaluations/trades: no new column on
-- the latter two -- both already carry version_id -> strategy_versions,
-- so strategy_type is derivable via join, matching this repo's existing
-- no-duplicate-derivable-fact convention (CLAUDE.md).

alter table strategy_versions add column strategy_type text not null default 'default';

-- capital_config: composite PK -- one capital sleeve per (mode, strategy_type).
alter table capital_config drop constraint capital_config_pkey;
alter table capital_config add column strategy_type text not null default 'default';
alter table capital_config add primary key (mode, strategy_type);

-- daily_pnl: independent circuit breaker / daily P&L bucket per strategy type.
alter table daily_pnl drop constraint daily_pnl_pkey;
alter table daily_pnl add column strategy_type text not null default 'default';
alter table daily_pnl add primary key (date, mode, strategy_type);

-- promotion_audit: independent promotion cooldown clock per strategy type.
alter table promotion_audit add column strategy_type text not null default 'default';
drop index promotion_audit_mode_created_idx;
create index promotion_audit_mode_strategy_type_created_idx
  on promotion_audit (mode, strategy_type, created_at);

-- learning_statistics / feature_importance: required, not optional --
-- without widening these unique constraints, a second strategy type's
-- stats would ON CONFLICT-overwrite the first's bucket row (silent
-- data loss, not a blend), and orchestrator.py reads these buckets to
-- compute regime/symbol confidence modifiers that gate live trades.
alter table learning_statistics drop constraint learning_statistics_mode_dimension_type_dimension_value_key;
alter table learning_statistics add column strategy_type text not null default 'default';
alter table learning_statistics add constraint learning_statistics_mode_strategy_type_dim_key
  unique (mode, strategy_type, dimension_type, dimension_value);
drop index learning_statistics_lookup_idx;
create index learning_statistics_lookup_idx on learning_statistics (mode, strategy_type, dimension_type);

alter table feature_importance drop constraint feature_importance_mode_feature_name_timeframe_key;
alter table feature_importance add column strategy_type text not null default 'default';
alter table feature_importance add constraint feature_importance_mode_strategy_type_feature_key
  unique (mode, strategy_type, feature_name, timeframe);
drop index feature_importance_mode_idx;
create index feature_importance_mode_strategy_type_idx on feature_importance (mode, strategy_type);

alter table recommendations add column strategy_type text not null default 'default';
drop index recommendations_mode_category_status_idx;
create index recommendations_mode_strategy_type_category_status_idx
  on recommendations (mode, strategy_type, category, status);

alter table strategy_simulations add column strategy_type text not null default 'default';
drop index strategy_simulations_mode_idx;
create index strategy_simulations_mode_strategy_type_idx
  on strategy_simulations (mode, strategy_type, created_at);

alter table adaptive_strategy_versions add column strategy_type text not null default 'default';
drop index adaptive_strategy_versions_mode_status_idx;
create index adaptive_strategy_versions_mode_strategy_type_status_idx
  on adaptive_strategy_versions (mode, strategy_type, status);
