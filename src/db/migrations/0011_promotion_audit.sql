-- Promotion Audit (src/learning/promotion_gate.py). One row per
-- EVALUATION, not just per promotion, so REJECT/EXTEND_VALIDATION
-- decisions are equally auditable — "no promotion may occur without a
-- complete audit record" is trivially true if only PROMOTE rows existed,
-- so this logs every run of evaluate_promotion(). Same shape/RLS pattern
-- as drift_alerts/strategy_health_scores (0008).
create table promotion_audit (
  id                   bigserial primary key,
  mode                 text not null,
  event_type           text not null,  -- 'evaluation' | 'promotion' | 'rollback'
  candidate_version_id bigint references strategy_versions(id),
  previous_champion_id bigint references strategy_versions(id),
  new_champion_id      bigint references strategy_versions(id),
  decision             text not null,  -- 'PROMOTE' | 'REJECT' | 'EXTEND_VALIDATION'
  promotion_score      numeric,
  gates                jsonb not null default '{}',
  breakdown            jsonb not null default '{}',
  reasons              jsonb not null default '[]',
  created_at           timestamptz not null default now()
);

create index promotion_audit_mode_created_idx on promotion_audit (mode, created_at);
create index promotion_audit_event_type_idx on promotion_audit (event_type, created_at);

alter table promotion_audit enable row level security;

create policy "public read" on promotion_audit for select using (true);

grant select on promotion_audit to anon, authenticated;
