# Graph Report - AI-Trader  (2026-08-22)

## Corpus Check
- 191 files · ~109,380 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1949 nodes · 5134 edges · 103 communities (84 shown, 19 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 197 edges (avg confidence: 0.52)
- Token cost: 198,042 input · 0 output

## Community Hubs (Navigation)
- Test Promotion Gate
- Test Backtest Execution Simulator
- Test Risk Manager
- Models
- Coindcx Client
- Test Backtest Statistical Validation
- Test Learning Statistics
- Intelligence
- Test Simulation
- Test Orchestrator
- Test Opportunity Scorer
- Test Data Quality Validator
- Test Feature Engine
- Reporting Agent
- Recommendations
- Package
- Test Recommendations
- Resilience
- Learning Status
- Performance Analyzer
- Capital Allocation
- Tsconfig
- Test Evolution Agent
- Test Drift Detection
- Fitness
- Test Feature Importance
- Learningaggregates
- Groq Client
- Data Provider
- Portfolio Manager
- Test Db Models
- Test Db Models Backtest
- Events
- Simulation Clock
- Diagnostics
- Engine
- Test Db Models Reliability
- Page
- Metrics
- Project Spec
- Test Backtest Engine
- Project Spec
- Rejection Analysis
- Test Evidence Engine
- Test Strategy Health
- Test Lenient Json
- Project Spec
- Test Backtest Overfitting Detection
- Trade Analysis
- Test Learning Status
- Config Client
- Types
- Statcard
- Test Confidence Calibration
- Evidence Engine
- Readme
- Requirements
- Test Audit Trail
- Test Backtest Strategy Comparison
- Test Backtest Walk Forward Validator
- Trades Client
- Layout
- Test Adaptive Strategy Engine
- Models
- Seed Config
- Route
- Project Spec
- Simulation Clock
- Test Db Models
- Test Db Models
- Test Db Models
- Eslint.Config
- Next.Config
- Postcss.Config
- Run Evolution
- Run Risk Check
- Run Trading Cycle
- Caveman
- Copilot Instructions
- Agents
- Caveman
- Agents
- File
- Globe
- Next
- Vercel
- Window
- Project Spec

## God Nodes (most connected - your core abstractions)
1. `get_client()` - 80 edges
2. `_fluent_mock()` - 64 edges
3. `run_cycle()` - 57 edges
4. `evaluate_promotion()` - 44 edges
5. `BacktestEngine` - 42 edges
6. `compute_bucket_statistics()` - 38 edges
7. `simulate_exit_params_recommendation()` - 37 edges
8. `compute_learning_status()` - 36 edges
9. `evaluate()` - 34 edges
10. `LearningStatus` - 33 edges

## Surprising Connections (you probably didn't know these)
- `groq>=0.11.0` --conceptually_related_to--> `chat()`  [INFERRED]
  requirements.txt → src/groq_client.py
- `requests>=2.31.0` --conceptually_related_to--> `RealExecutionAgent`  [INFERRED]
  requirements.txt → src/agents/execution/real.py
- `BacktestEngine` --references--> `Feature Engine`  [EXTRACTED]
  src/backtest/engine.py → PROJECT_SPEC.md
- `Table: model_usage` --shares_data_with--> `chat()`  [EXTRACTED]
  PROJECT_SPEC.md → src/groq_client.py
- `evaluate_promotion()` --references--> `fitness.py (multi-objective fitness score)`  [EXTRACTED]
  src/learning/promotion_gate.py → PROJECT_SPEC.md

## Import Cycles
- 3-file cycle: `src/agents/risk_manager.py -> src/portfolio/intelligence.py -> src/learning/statistics.py -> src/agents/risk_manager.py`
- 4-file cycle: `src/agents/evolution_agent.py -> src/agents/risk_manager.py -> src/portfolio/intelligence.py -> src/learning/statistics.py -> src/agents/evolution_agent.py`

## Hyperedges (group relationships)
- **Caveman Mode Duplicated Across AI Tool Configs** — _clinerules_caveman_rule, _github_copilot_instructions_rule, _opencode_agents_rule, _windsurf_rules_caveman_rule, agents_rule [INFERRED 0.95]
- **Trading Pipeline: Data Agent to Execution Agent** — src_agents_data_agent_module, src_features_feature_engine_module, src_features_opportunity_scorer_module, src_agents_risk_manager_module, src_agents_execution_base_executionagent [EXTRACTED 1.00]
- **Scientific Strategy Optimization Research Pipeline** — project_spec_trade_memory_learning_engine, src_learning_statistics_compute_bucket_statistics, src_learning_weakness_detection_identify_weaknesses, src_learning_recommendations_module, project_spec_table_adaptive_strategy_versions, src_learning_simulation_module, src_learning_promotion_gate_evaluate_promotion [EXTRACTED 1.00]

## Communities (103 total, 19 thin omitted)

### Community 0 - "Test Promotion Gate"
Cohesion: 0.07
Nodes (91): ExitStack, OverfittingReport, _backtest_evidence(), _bootstrap_probability_of_profit(), _champion_improvement_gate(), _complexity_delta(), _cooldown_gate(), _dedup_by_snapshot_time() (+83 more)

### Community 1 - "Test Backtest Execution Simulator"
Cohesion: 0.06
Nodes (72): Enum, fees(), PaperExecutionAgent, Public — reused as-is by src/backtest/execution_simulator.py for commission…, order_context is new, additive, optional (Execution Optimizer, PROJECT_SPEC.md…, datetime, BacktestEngine — the event reactor. Not a reuse of orchestrator.run_cycle…, check_resting_order_fill() (+64 more)

### Community 2 - "Test Risk Manager"
Cohesion: 0.05
Nodes (81): Position, Circuit Breaker (daily loss limit), CoinDCX Spot API Has No Exchange-Side Stop Order, Table: circuit_breaker_state, circuit_breaker_triggered(), committed_capital(), compute_net_expectancy_pct(), evaluate() (+73 more)

### Community 3 - "Models"
Cohesion: 0.05
Nodes (78): Client, Free-Tier Supabase Disk Fill Incident, _mode_section(), Runs a backtest end to end and persists the run/trades/snapshots/ execution…, run_and_persist(), _execute(), get_active_strategy_versions(), get_adaptive_strategy_versions() (+70 more)

### Community 4 - "Coindcx Client"
Cohesion: 0.06
Nodes (66): ABC, get_market_snapshot(), ExecutionAgent, Force-close every open position for mode. Returns closed trades., Returns a fill: {"fill_price": float, "fees": float}., 1% TDS (Income Tax Act s.194S) on a sell's trade value — public, reused as-is…, sell_tds(), _extract_order() (+58 more)

### Community 5 - "Test Backtest Statistical Validation"
Cohesion: 0.07
Nodes (57): _created_date(), _max_drawdown_pct(), Hourly: evaluate the active strategy version against…, bootstrap_confidence_interval(), monte_carlo_drawdown_distribution(), moving_block_bootstrap_probability(), parameter_stability_sweep(), Step 11: statistical validation via seeded resampling — NOT a parametric… (+49 more)

### Community 6 - "Test Learning Statistics"
Cohesion: 0.07
Nodes (58): _accuracy_pct(), accuracy_rates(), _assess_risk(), _assess_stop_loss(), _assess_target(), _bucket_label(), _bucket_memberships(), compute_bucket_statistics() (+50 more)

### Community 7 - "Intelligence"
Cohesion: 0.09
Nodes (53): _avg_correlation_with_book(), _base_symbol(), beta(), category_of(), correlation(), correlation_matrix(), _covariance(), diversification_score() (+45 more)

### Community 8 - "Test Simulation"
Cohesion: 0.09
Nodes (50): Table: adaptive_strategy_versions, Table: strategy_versions, get_latest_recommendation(), _activate_exit_params_candidate(), _backtest_replay_gate(), _bootstrap_gate(), _build_research_note(), _create_candidate_version() (+42 more)

### Community 9 - "Test Orchestrator"
Cohesion: 0.16
Nodes (50): _bucket_modifier(), _empty_daily_pnl(), _price_history_from_snapshot(), Adaptive confidence chain (Step 7): how much better/worse this regime's or…, Adaptive confidence chain (Step 7): current win/loss streak over the last…, Daily closes for every symbol scanned this cycle — already-fetched data…, No LLM calls, no market snapshot — just the circuit breaker and the stop-…, _recent_performance_modifier() (+42 more)

### Community 10 - "Test Opportunity Scorer"
Cohesion: 0.12
Nodes (48): _blend_across_timeframes(), _bool_score(), classify_market_regime(), _linear_score(), Opportunity Scorer, Turns Feature Engine output into a deterministic 0-100 opportunity score. No…, Despite the "risk" name (kept for the DB column/OPPORTUNITY_WEIGHT_RISK…, Deterministic composite label, reusing score_trend() (already computed) plus… (+40 more)

### Community 11 - "Test Data Quality Validator"
Cohesion: 0.10
Nodes (39): Pulls the top-N INR pairs by 24h turnover with fresh multi-timeframe candles…, DataRepairEngine, _dedup_keep_latest(), _interpolate_gap(), Auto-repairs what validator.py flagged as safely fixable. Never touches a…, Exact-duplicate-timestamp merge: keeps the last-seen row for a given time (the…, Linear interpolation between the two known-good bars bracketing the gap —…, RepairLogEntry (+31 more)

### Community 12 - "Test Feature Engine"
Cohesion: 0.10
Nodes (44): adx(), atr(), bollinger_bands(), compute_features(), compute_multi_timeframe_features(), ema(), _ema_series(), macd() (+36 more)

### Community 13 - "Reporting Agent"
Cohesion: 0.11
Nodes (36): build_report_data(), generate_report(), _mode_section_html(), _model_usage_html(), _model_usage_stats(), One HTML report covering both modes side by side: PnL vs target, trade log…, render_html(), _row() (+28 more)

### Community 14 - "Recommendations"
Cohesion: 0.15
Nodes (31): callable, get_capital_config(), insert_recommendation(), Step 1: AdaptiveStrategyEngine — the single composed entry point for the whole…, compute_learning_status(), LearningStatus, _avoid_bucket_recommendations(), _find_optimal_threshold() (+23 more)

### Community 15 - "Package"
Cohesion: 0.05
Nodes (36): dependencies, next, react, react-dom, recharts, @supabase/supabase-js, devDependencies, eslint (+28 more)

### Community 16 - "Test Recommendations"
Cohesion: 0.12
Nodes (35): current_weights(), Approximates this trade's pnl under a candidate stop_loss_pct/ take_profit_pct…, _simulate_exit_pnl(), _ai_trades(), patch, Evidence-Driven Learning Progression: a valid candidate weight set still…, Explicit LearningStatus for every test below — passed straight into the status=…, _status() (+27 more)

### Community 17 - "Resilience"
Cohesion: 0.12
Nodes (30): BaseException, get_orderbook(), call_with_circuit_breaker(), check_circuit_breaker(), CircuitBreakerOpenError, _now_ms(), RuntimeError, Generic retry/backoff + a DB-backed circuit breaker, shared by every external… (+22 more)

### Community 18 - "Learning Status"
Cohesion: 0.11
Nodes (25): get_learning_statistics(), _bootstrap_gaps(), _observation_ready(), Evidence-Driven Learning Progression. LearningStatus is the single authority…, (stage, next_stage, next_stage_min_trades). next_stage_min_trades is None for…, Every unmet OR-branch, human-readable — the literal "Need 2 additional market…, _reason_for(), _stage_for() (+17 more)

### Community 19 - "Performance Analyzer"
Cohesion: 0.14
Nodes (28): analyze(), annual_returns(), capital_utilization_pct(), exposure_time_pct(), gross_profit_loss(), monthly_returns(), omega_ratio(), Step 6/7: portfolio-level performance metrics + equity/drawdown curve analysis.… (+20 more)

### Community 20 - "Capital Allocation"
Cohesion: 0.16
Nodes (28): compute_dynamic_size(), confidence_factor(), correlation_factor(), drawdown_factor(), exposure_factor(), _linear(), Capital Allocation Engine — dynamic position sizing as a function of portfolio…, 0.3 win rate -> MIN, 0.7 -> MAX, linear between — deliberately not anchored at… (+20 more)

### Community 21 - "Tsconfig"
Cohesion: 0.07
Nodes (28): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+20 more)

### Community 22 - "Test Evolution Agent"
Cohesion: 0.22
Nodes (27): compute_metrics(), promotion_ready(), Promotion monitor for the one live strategy_versions row — no LLM call, no new…, run_evolution(), PromotionDecision, _metrics(), patch, _status() (+19 more)

### Community 23 - "Test Drift Detection"
Cohesion: 0.16
Nodes (26): detect_feature_drift(), detect_feature_importance_drift(), detect_performance_drift(), population_stability_index(), _proportion_drift_alert(), datetime, Feature Drift Detection (Step 6, PROJECT_SPEC.md §3d). Compares a recent window…, feature_importance rows are nightly snapshots (already timestamped); compares… (+18 more)

### Community 24 - "Fitness"
Cohesion: 0.15
Nodes (25): compute_fitness_score(), drawdown_component(), expectancy_component(), profit_factor_component(), Multi-Objective Fitness Score (Step 8, Scientific Strategy Optimization…, Framed as a penalty term: 0% drawdown -> 100 (no penalty),…, expectancy as a % of capital_to_use, linear-mapped around a neutral 50…, stats: a compute_bucket_statistics()-shaped dict. Returns {"fitness_score":… (+17 more)

### Community 25 - "Test Feature Importance"
Cohesion: 0.16
Nodes (25): get_entry_evaluation_for_trade(), upsert_feature_importance(), compute_feature_importance(), compute_subscore_correlation_weights(), feature_importance.py, pearson_correlation(), Point-biserial correlation between Feature Engine/Opportunity Scorer values and…, Correlates the 5 already-flat opportunity_evaluations sub-score columns against… (+17 more)

### Community 26 - "Learningaggregates"
Cohesion: 0.11
Nodes (21): fmt(), LearningClient(), ACCEPTED_DECISIONS, ALL_MARKET_REGIMES, bootstrapGaps(), clamp(), collectEvidence(), computeEvidenceReadiness() (+13 more)

### Community 27 - "Groq Client"
Cohesion: 0.18
Nodes (22): Groq, AllModelsFailedError, chat(), _gemini_completion(), _groq_completion(), ModelUsageEvent, RuntimeError, LLM chat wrapper: retries each model in a chain with backoff, then falls back… (+14 more)

### Community 28 - "Data Provider"
Cohesion: 0.18
Nodes (21): CandleStore, fetch_historical_candles_paginated(), _fetch_page(), ingest(), Historical OHLCV: paginated network fetch (CoinDCX's public candles endpoint…, Up to `limit` most-recent candles fully closed as of `as_of_ms`. Cutoff =…, Ticker-price proxy for `now` (mirrors live's real-time get_ticker() last_price)…, Walks backward from end_ms in BACKTEST_CANDLE_PAGE_SIZE-candle pages (the… (+13 more)

### Community 29 - "Portfolio Manager"
Cohesion: 0.13
Nodes (12): PortfolioManager, Position, datetime, Cash/equity/position/exposure tracking with a mark-to-market equity curve —…, Entry-basis cost of open positions (matches risk_manager.committed_capital's…, test_buying_power_and_leverage_are_spot_only(), test_close_position_credits_cash_and_computes_pnl(), test_committed_capital_uses_entry_basis() (+4 more)

### Community 30 - "Test Db Models"
Cohesion: 0.09
Nodes (23): get_latest_promotion_audit(), get_trade_evaluation_ids(), test_get_feature_importance_filters_by_timeframe(), test_get_latest_promoted_version_filters_and_orders(), test_get_latest_promotion_audit_filters_by_mode_and_event_type(), test_get_latest_promotion_audit_returns_none_when_empty(), test_get_latest_recommendation_none_when_no_rows(), test_get_latest_recommendation_orders_by_created_at() (+15 more)

### Community 31 - "Test Db Models Backtest"
Cohesion: 0.15
Nodes (22): insert_backtest_portfolio_snapshots(), candles: list of {"time","open","high","low","close","volume"} dicts, CoinDCX's…, Batch insert — a multi-month equity curve is thousands of points, one-row-per-…, upsert_historical_candles(), _fluent_mock(), A mock whose chained Supabase query-builder methods all return itself, so call…, test_get_backtest_performance_metrics_none_when_missing(), test_get_backtest_run_none_when_missing() (+14 more)

### Community 32 - "Events"
Cohesion: 0.12
Nodes (17): Event, EventQueue, FillEvent, MarketEvent, OrderEvent, PortfolioEvent, PositionEvent, Event types the BacktestEngine reactor dispatches on, plus a thin EventQueue.… (+9 more)

### Community 33 - "Simulation Clock"
Cohesion: 0.17
Nodes (14): is_bar_closed(), datetime, Drives the BacktestEngine's chronological replay. `is_bar_closed` is the no-…, Yields each tick's epoch-ms timestamp in order, advancing `now` as it goes, up…, SimulationClock, timeframe_duration_ms(), test_clock_now_reflects_current_tick(), test_clock_rejects_start_ge_end() (+6 more)

### Community 34 - "Diagnostics"
Cohesion: 0.18
Nodes (19): ping(), Trivial reachability check — used by monitoring/diagnostics.py's database…, _check_database(), _check_execution_engine(), _check_learning_engine(), _check_market_feed(), _check_portfolio_engine(), _check_recommendation_engine() (+11 more)

### Community 35 - "Engine"
Cohesion: 0.18
Nodes (8): IntelligencePosition, BacktestEngine, Daily closes per symbol, visible as of as_of_ms — mirrors…, Mirrors orchestrator._sweep_stop_loss_take_profit + its bracketing circuit-…, Mirrors orchestrator.run_cycle's Pass 1 + Pass 2 (checkpoints 2 and 3 of 3:…, backtest/events.py (Event/EventQueue), backtest/execution_simulator.py, backtest/portfolio_manager.py

### Community 36 - "Test Db Models Reliability"
Cohesion: 0.11
Nodes (19): get_trade_evaluations(), insert_data_quality_issues(), Full trade_evaluations rows (confidence_was_accurate/…, test_get_active_strategy_versions_excludes_suspended(), test_get_circuit_breaker_state_none_when_missing(), test_get_data_quality_log_filters_by_pair_and_source(), test_get_drift_alerts_filters_by_component(), test_get_latest_strategy_health_score_none_when_missing() (+11 more)

### Community 37 - "Page"
Cohesion: 0.19
Nodes (8): EvolutionClient(), ModelHealthClient(), aggregateModelUsage(), ModelStat, CHROME, SERIES, supabase, ModelUsage

### Community 38 - "Metrics"
Cohesion: 0.18
Nodes (16): Exception, Fetch -> validate -> repair, in that order, before anything reaches the Feature…, _validated_candles(), log_resource_snapshot(), Production Monitoring (Step 8, PROJECT_SPEC.md §3d). Scoped to what's real for…, Wraps a block, recording its duration and success/failure to system_metrics on…, CPU/memory via stdlib `resource` (this process's own usage — the closest…, resource_snapshot() (+8 more)

### Community 39 - "Project Spec"
Cohesion: 0.18
Nodes (18): Evidence-Driven Learning Progression, PROJECT_SPEC.md System Architecture Spec, Table: agent_logs, Table: confidence_calibration, Table: daily_pnl, Table: data_quality_log, Table: feature_importance, Table: learning_statistics (+10 more)

### Community 40 - "Test Backtest Engine"
Cohesion: 0.20
Nodes (16): _date_to_ms(), date, _engine(), patch, Backtest/live parity (PROJECT_SPEC.md §3d): risk_manager.evaluate()'s…, test_decision_pass_blocks_oversized_candidate_via_concentration_gate(), test_decision_pass_opens_position_for_qualifying_candidate(), test_decision_pass_stops_before_later_candidate_once_breaker_trips_mid_loop() (+8 more)

### Community 41 - "Project Spec"
Cohesion: 0.13
Nodes (17): CLAUDE.md Project Guide, Vercel /api/cron/[workflow] Trigger Route, Dashboard Architecture (Next.js on Vercel), Groq Rate-Limit Multi-Day Outage Incident, Retirement of Nightly LLM Strategy Rewrite Loop, No-Duplicate-Fact Precedent, Scientific Strategy Optimization Framework, Trade Memory + Learning Engine (+9 more)

### Community 42 - "Rejection Analysis"
Cohesion: 0.20
Nodes (15): Adaptive Strategy Intelligence Engine, AdaptiveStrategyEngine.analyze(), adaptive_strategy_engine.py, datetime, Root Cause Analysis (Step 2, Scientific Strategy Optimization Framework).…, risk_manager_result, when present, is the more specific reason (e.g.…, Ranked [{"reason", "count", "pct_of_rejections"}, ...], descending by count —…, rejection_breakdown() (+7 more)

### Community 43 - "Test Evidence Engine"
Cohesion: 0.26
Nodes (16): EvidenceEngine, Never executes a trade, never modifies a strategy or config — a pure…, _evaluation(), _evidence(), patch, test_collect_confidence_coverage_is_fraction_reaching_llm(), test_collect_confidence_coverage_zero_when_no_evaluations(), test_collect_counts_closed_winning_losing() (+8 more)

### Community 44 - "Test Strategy Health"
Cohesion: 0.20
Nodes (11): run_strategy_health(), _tier(), _mock_critical_setup(), Shared setup: one active version with a clearly critical book (all losses, no…, test_compute_health_score_no_trades_returns_none_score(), test_run_strategy_health_never_suspends_when_auto_suspend_disabled(), test_run_strategy_health_no_rollback_when_suspended_version_is_not_champion(), test_run_strategy_health_rollback_audit_failure_fails_open() (+3 more)

### Community 45 - "Test Lenient Json"
Cohesion: 0.21
Nodes (15): parse_llm_json(), Any, Best-effort cleanup for JSON an LLM claims to have written but didn't quite:…, _strip_code_fence(), _strip_comments(), _strip_trailing_commas(), test_does_not_strip_double_slash_inside_string_value(), test_genuinely_broken_json_still_raises() (+7 more)

### Community 46 - "Project Spec"
Cohesion: 0.15
Nodes (15): Evolution Hourly Cadence Tolerant of Cron Drift, Evolution GitHub Actions Workflow, Automatic Rollback, Flat Concentration Cap Bug Fix (scales with max_concurrent_positions), Institutional Reliability Layer, Portfolio Intelligence Look-Ahead Trap Fix, Table: capital_config, Table: drift_alerts (+7 more)

### Community 47 - "Test Backtest Overfitting Detection"
Cohesion: 0.30
Nodes (12): _avg(), detect(), Step 12: overfitting detection, composed from what walk_forward_validator and…, Fold, Step 9: real rolling multi-fold walk-forward validation — genuinely new,…, _fold(), test_detect_all_folds_pass_is_robust(), test_detect_computes_in_sample_out_of_sample_gap() (+4 more)

### Community 48 - "Trade Analysis"
Cohesion: 0.30
Nodes (11): ClosedTrade, Step 8: per-trade analytics. Most fields (MFE/MAE/slippage/commission/…, MFE/MAE-based proxy: how much favorable excursion was captured relative to…, risk_reward(), to_row(), to_rows(), test_risk_reward_none_when_never_adverse(), test_risk_reward_ratio_of_mfe_to_mae() (+3 more)

### Community 49 - "Test Learning Status"
Cohesion: 0.30
Nodes (14): _evidence(), patch, Evidence-Driven Learning Progression: only 3 closed trades, but 500+ rejected…, Deeper stages are NOT evidence-substitutable — strong coverage with too few…, _status(), test_can_promote_reads_promotion_eligible_without_recomputing(), test_current_activity_and_reason_are_nonempty_strings(), test_fields_reflect_evidence_wins_losses_rejected() (+6 more)

### Community 50 - "Config Client"
Cohesion: 0.20
Nodes (7): ConfigClient(), ConfigForm(), FIELDS, LoginForm(), ModeToggle(), useMode(), Mode

### Community 51 - "Types"
Cohesion: 0.26
Nodes (9): LearningData, STAGE_ORDER, AdaptiveStrategyVersion, LearningStatistic, LearningStatus, OpportunityEvaluationRow, Recommendation, StrategySimulation (+1 more)

### Community 52 - "Statcard"
Cohesion: 0.19
Nodes (9): OverviewClient(), load(), StatCard(), Tone, TONE_COLOR, todayIst(), STATUS, CapitalConfig (+1 more)

### Community 53 - "Test Confidence Calibration"
Cohesion: 0.29
Nodes (12): calibrate_confidence(), Blends the LLM's own stated confidence with historical win-rate on similar past…, patch, test_blended_confidence_known_value(), test_modifiers_clamped_to_0_100(), test_modifiers_default_to_none_and_dont_change_base_confidence(), test_modifiers_never_applied_when_base_confidence_is_none(), test_modifiers_sum_onto_base_confidence() (+4 more)

### Community 54 - "Evidence Engine"
Cohesion: 0.33
Nodes (10): compute_evidence_readiness(), _ist_hour(), _market_coverage_pct(), Evidence Engine (Evidence-Driven Learning Progression). Measures how much…, Reuses opportunity_scorer.weighted_average's renormalize-among- available blend…, _rejection_evidence_pct(), _session_coverage_pct(), _symbol_coverage_pct() (+2 more)

### Community 55 - "Readme"
Cohesion: 0.31
Nodes (11): GitHub Actions Cron Unreliable Below ~15min Cadence, Risk Check GitHub Actions Workflow, Trading Cycle GitHub Actions Workflow, Cloud Run Job: evolution, Cloud Run Job: risk-check, Cloud Run Job: trading-cycle, Cloud Run Jobs Migration Guide, GitHub Actions Free-Tier Minutes Exhaustion (+3 more)

### Community 56 - "Requirements"
Cohesion: 0.20
Nodes (9): Python Dependencies (requirements.txt), groq>=0.11.0, pytest>=8.0.0, python-dotenv>=1.0.0, requests>=2.31.0, supabase>=2.0.0, Data Agent, 0001_init.sql (Supabase schema migration) (+1 more)

### Community 57 - "Test Audit Trail"
Cohesion: 0.29
Nodes (8): config_version(), Audit System (Step 9, PROJECT_SPEC.md §3d) — reuse first. Everything Step 9…, Short, stable hash of the live scoring/threshold constants that determine an…, test_config_version_changes_when_a_weight_changes(), test_config_version_deterministic(), test_config_version_is_short_hex_string(), test_get_decision_trail_calibration_none_when_not_logged(), test_get_decision_trail_joins_calibration_by_evaluation_id()

### Community 58 - "Test Backtest Strategy Comparison"
Cohesion: 0.38
Nodes (8): compare(), Step 10: pairwise strategy-run comparison. Reuses src/learning/statistics.py's…, metrics_a/metrics_b: performance_analyzer.analyze() bundles., patch, test_compare_no_winner_when_not_significant(), test_compare_picks_b_when_significantly_better_expectancy(), test_compare_returns_p_values_dict(), _trades()

### Community 59 - "Test Backtest Walk Forward Validator"
Cohesion: 0.38
Nodes (9): date, params_json is applied UNCHANGED to both train and test windows in each fold —…, run_walk_forward(), _run_window(), patch, test_run_walk_forward_insufficient_sample_reports_none_pvalue(), test_run_walk_forward_passes_when_out_of_sample_beats_in_sample(), test_run_walk_forward_produces_one_fold_per_window_that_fits() (+1 more)

### Community 60 - "Trades Client"
Cohesion: 0.31
Nodes (6): latestClose(), STATUS_COLOR, timeframeMinutes(), TradesClient(), loadPrices(), Trade

### Community 61 - "Layout"
Cohesion: 0.29
Nodes (5): geistMono, geistSans, metadata, LINKS, Nav()

### Community 62 - "Test Adaptive Strategy Engine"
Cohesion: 0.52
Nodes (6): AdaptiveStrategyEngine, Never executes trades. Never modifies config.py or any trading table directly —…, patch, _status(), test_analyze_composes_all_generators_and_simulations(), test_analyze_skips_simulation_when_no_recommendations_generated()

### Community 63 - "Models"
Cohesion: 0.33
Nodes (6): log_opportunity_evaluation(), Any, market_regime/config_version (Audit System, PROJECT_SPEC.md §3d) — the two…, test_log_opportunity_evaluation_includes_market_regime_and_config_version(), test_log_opportunity_evaluation_inserts_expected_row(), test_log_opportunity_evaluation_returns_row_and_includes_trade_id()

### Community 64 - "Seed Config"
Cohesion: 0.53
Nodes (5): main(), _prompt_float(), _prompt_int(), Ad-hoc CLI to set capital_config for a mode, and to bootstrap strategy_versions…, seed_initial_strategy_version()

### Community 66 - "Project Spec"
Cohesion: 0.67
Nodes (3): Table: historical_candles, backtest/data_provider.py (CandleStore), backtest/ingest_data.py (CLI)

### Community 68 - "Test Db Models"
Cohesion: 0.67
Nodes (3): close_trade(), test_close_trade_includes_exit_reason(), test_close_trade_sets_closed_fields()

### Community 69 - "Test Db Models"
Cohesion: 0.67
Nodes (3): insert_promotion_audit(), test_insert_promotion_audit_defaults_jsonb_fields(), test_insert_promotion_audit_inserts_expected_row()

### Community 70 - "Test Db Models"
Cohesion: 0.67
Nodes (3): open_trade(), test_open_trade_includes_learning_engine_fields(), test_open_trade_inserts_expected_row()

## Knowledge Gaps
- **97 isolated node(s):** `eslintConfig`, `nextConfig`, `name`, `version`, `private` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BacktestEngine` connect `Engine` to `Test Promotion Gate`, `Test Backtest Execution Simulator`, `Test Risk Manager`, `Project Spec`, `Simulation Clock`, `Models`, `Project Spec`, `Test Backtest Engine`, `Project Spec`, `Test Opportunity Scorer`, `Intelligence`, `Test Simulation`, `Test Backtest Overfitting Detection`, `Test Backtest Walk Forward Validator`, `Data Provider`, `Portfolio Manager`?**
  _High betweenness centrality (0.047) - this node is a cross-community bridge._
- **Why does `run_cycle()` connect `Test Orchestrator` to `Test Backtest Execution Simulator`, `Test Risk Manager`, `Models`, `Coindcx Client`, `Test Backtest Statistical Validation`, `Metrics`, `Test Learning Statistics`, `Project Spec`, `Test Opportunity Scorer`, `Test Feature Engine`, `Recommendations`, `Learning Status`, `Test Confidence Calibration`, `Readme`, `Requirements`, `Test Audit Trail`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Why does `CLAUDE.md Project Guide` connect `Project Spec` to `Test Promotion Gate`, `Test Backtest Execution Simulator`, `Test Risk Manager`, `Models`, `Coindcx Client`, `Project Spec`, `Test Simulation`, `Test Orchestrator`, `Rejection Analysis`, `Test Opportunity Scorer`, `Project Spec`, `Recommendations`, `Test Confidence Calibration`, `Readme`, `Requirements`, `Groq Client`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `run_cycle()` (e.g. with `PaperExecutionAgent` and `RealExecutionAgent`) actually correct?**
  _`run_cycle()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `BacktestEngine` (e.g. with `RiskDecision` and `CandleStore`) actually correct?**
  _`BacktestEngine` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `eslintConfig`, `nextConfig`, `name` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Test Promotion Gate` be split into smaller, more focused modules?**
  _Cohesion score 0.07183710821322352 - nodes in this community are weakly interconnected._