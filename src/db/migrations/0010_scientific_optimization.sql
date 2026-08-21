-- Scientific Strategy Optimization Framework. Retires evolution_agent.py's
-- unconstrained nightly LLM prompt/param rewrite and its zero-approval
-- auto-promotion to real trading; the adaptive_strategy_versions candidate
-- pipeline (already advisory-only, human-approved in Supabase) becomes the
-- sole authoritative source of strategy changes, extended with a fitness
-- score, bootstrap-CI/walk-forward validation, and narrative research
-- notes. All columns below are nullable/safe-defaulted — zero behavior
-- change until the corresponding code ships, same deployment-order safety
-- as every migration except 0008.

-- Historical note: at the time this migration shipped, evolution_agent.py
-- flagged eligibility here and a human flipped promoted_to_real manually.
-- That manual step was later retired (see src/learning/promotion_gate.py)
-- once promotion required strict multi-gate evidence instead of 3 raw
-- thresholds — promoted_to_real is now auto-flipped the same run PROMOTE
-- is decided, still fully automatic, just gated on far stricter evidence.
alter table strategy_versions add column promotion_eligible boolean not null default false;

-- Narrative "research report" version notes (Observation/Weakness/
-- Hypothesis/Simulation/Walk Forward/Decision), replacing changelog-style
-- notes; validation_detail carries the raw numbers (bootstrap CI,
-- walk-forward fold summary, strategy-comparison result) behind the prose.
alter table strategy_simulations add column research_note text;
alter table strategy_simulations add column validation_detail jsonb;

-- Lets Supabase's table editor sort/filter candidates by fitness directly
-- instead of reading candidate_metrics jsonb by hand.
alter table adaptive_strategy_versions add column fitness_score numeric;
