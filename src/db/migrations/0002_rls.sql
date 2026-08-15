-- Dashboard access. Agents write with the service key, which bypasses
-- RLS entirely — these policies only govern the browser's anon/
-- authenticated access. See PROJECT_SPEC.md §8.

alter table capital_config enable row level security;
alter table strategy_versions enable row level security;
alter table trades enable row level security;
alter table daily_pnl enable row level security;
alter table agent_logs enable row level security;
alter table model_usage enable row level security;

create policy "public read" on capital_config for select using (true);
create policy "public read" on strategy_versions for select using (true);
create policy "public read" on trades for select using (true);
create policy "public read" on daily_pnl for select using (true);
create policy "public read" on agent_logs for select using (true);
create policy "public read" on model_usage for select using (true);

-- Only the Config panel writes from the browser, only to
-- capital_config, and only when signed in.
create policy "authenticated update capital_config" on capital_config
  for update using (auth.role() = 'authenticated');

-- RLS policies only take effect on top of a base GRANT — tables created
-- via the SQL Editor don't reliably inherit Supabase's default
-- schema-level grants, so these are explicit rather than assumed.
grant select on capital_config, strategy_versions, trades, daily_pnl, agent_logs, model_usage
  to anon, authenticated;
grant update on capital_config to authenticated;
