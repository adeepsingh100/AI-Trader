-- Start/stop control from the dashboard. orchestrator.py checks this
-- before touching the Signal Agent (Groq) or market data at all, so a
-- paused mode makes zero LLM/exchange calls that cycle.
alter table capital_config add column paused boolean not null default false;
