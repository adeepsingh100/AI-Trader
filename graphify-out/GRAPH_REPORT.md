# Graph Report - AI-Trader  (2026-08-16)

## Corpus Check
- Corpus is ~18,118 words - fits in a single context window. You may not need a graph.

## Summary
- 468 nodes · 993 edges · 37 communities (19 shown, 18 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 46 edges (avg confidence: 0.62)
- Token cost: 264,607 input · 0 output

## Community Hubs (Navigation)
- Next.js Dashboard Pages & Components
- CoinDCX Client & Real Execution
- Database Models & Config Seeding
- Project Docs & CI Workflows
- Dashboard NPM Dependencies
- LLM Client & Signal Agent
- Risk Manager & Stop-Loss/Take-Profit
- Evolution Agent & Promotion Logic
- Dashboard TypeScript Config
- Orchestrator Cycle & Tests
- HTML Reporting Agent
- Execution Agents & Fee Model
- Lenient JSON Parser
- Dashboard Layout & Navigation
- Agent Instruction Docs (CLAUDE.md)
- Dashboard ESLint Config
- Dashboard Next.js Config
- Dashboard PostCSS Config
- Principle: Autonomous Bug Fixing
- Principle: Core Principles
- Principle: Demand Elegance
- Principle: Plan Mode Default
- Principle: Self-Improvement Loop
- Principle: Subagent Strategy
- Principle: Task Management
- Icon Asset: file.svg
- Icon Asset: globe.svg
- Logo Asset: next.svg
- Logo Asset: vercel.svg
- Icon Asset: window.svg
- ExecutionAgent.get_fill (interface stub)
- ExecutionAgent.place_order (interface stub)

## God Nodes (most connected - your core abstractions)
1. `run_cycle()` - 36 edges
2. `chat()` - 23 edges
3. `get_client()` - 19 edges
4. `parse_llm_json()` - 19 edges
5. `run_risk_check()` - 19 edges
6. `RealExecutionAgent` - 18 edges
7. `_capital_config()` - 18 edges
8. `compilerOptions` - 16 edges
9. `run_evolution()` - 16 edges
10. `ModelUsageEvent` - 16 edges

## Surprising Connections (you probably didn't know these)
- `Risk Check Workflow (real job)` --references--> `RealExecutionAgent`  [INFERRED]
  .github/workflows/risk_check.yml → src/agents/execution/real.py
- `requests>=2.31.0` --conceptually_related_to--> `RealExecutionAgent`  [INFERRED]
  requirements.txt → src/agents/execution/real.py
- `chat()` --shares_data_with--> `model_usage table`  [EXTRACTED]
  src/groq_client.py → PROJECT_SPEC.md
- `Risk Check Workflow (paper job)` --references--> `run_risk_check()`  [EXTRACTED]
  .github/workflows/risk_check.yml → src/orchestrator.py
- `PaperExecutionAgent` --implements--> `Paper Trading Fee/Slippage Simulation Model`  [EXTRACTED]
  src/agents/execution/paper.py → PROJECT_SPEC.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **GitHub Actions Free-Tier Cron Deployment** — _github_workflows_trading_cycle_paper, _github_workflows_trading_cycle_real, _github_workflows_risk_check_paper, _github_workflows_risk_check_real, _github_workflows_evolution_evolve, project_spec_deployment_topology [EXTRACTED 1.00]
- **One Full Trading Cycle (Data to Execution)** — src_orchestrator_module, src_agents_data_agent_module, src_agents_signal_agent_module, src_agents_risk_manager_module, src_agents_execution_paper_paperexecutionagent [EXTRACTED 1.00]
- **Execution Agent Interface (base.py) and Implementors** — src_agents_execution_base_place_order, src_agents_execution_base_get_fill, src_agents_execution_base_flatten_all, src_agents_execution_paper_paperexecutionagent, src_agents_execution_real_realexecutionagent [EXTRACTED 1.00]

## Communities (37 total, 18 thin omitted)

### Community 0 - "Next.js Dashboard Pages & Components"
Cohesion: 0.08
Nodes (28): ConfigClient(), ConfigForm(), FIELDS, LoginForm(), EvolutionClient(), ModelHealthClient(), OverviewClient(), load() (+20 more)

### Community 1 - "CoinDCX Client & Real Execution"
Cohesion: 0.11
Nodes (38): get_market_snapshot(), Pulls the top-N INR pairs by 24h turnover with fresh orderbook/candles for each…, _extract_order(), _inr_balance(), Live CoinDCX orders. Only ever instantiated by the orchestrator when a strategy…, RealExecutionAgent, _round_qty(), _wait_for_fill() (+30 more)

### Community 2 - "Database Models & Config Seeding"
Cohesion: 0.10
Nodes (38): Client, run_evolution(), build_report_data(), _mode_section(), close_trade(), get_all_strategy_versions(), get_capital_config(), get_client() (+30 more)

### Community 3 - "Project Docs & CI Workflows"
Cohesion: 0.07
Nodes (37): Evolution Workflow (evolve job), Risk Check Workflow (paper job), Risk Check Workflow (real job), Trading Cycle Workflow (paper job), Trading Cycle Workflow (real job), agent_logs table, 11-Step Build Order, capital_config table (+29 more)

### Community 4 - "Dashboard NPM Dependencies"
Cohesion: 0.05
Nodes (36): dependencies, next, react, react-dom, recharts, @supabase/supabase-js, devDependencies, eslint (+28 more)

### Community 5 - "LLM Client & Signal Agent"
Cohesion: 0.13
Nodes (29): Verification Before Done, Groq, LLM Provider Abstraction (Groq/Ollama fallback chain), groq>=0.11.0, RuntimeError, get_signal(), _messages_for(), Scores one market snapshot with the LLM, using the active strategy version's… (+21 more)

### Community 6 - "Risk Manager & Stop-Loss/Take-Profit"
Cohesion: 0.14
Nodes (33): Stop-Loss/Take-Profit Polling Enforcement, circuit_breaker_triggered(), committed_capital(), evaluate(), exit_reason(), stop_loss_pct/take_profit_pct from the active strategy version's params_json,…, RiskDecision, target_hit() (+25 more)

### Community 7 - "Evolution Agent & Promotion Logic"
Cohesion: 0.16
Nodes (29): compute_metrics(), _created_date(), _max_drawdown_pct(), promotion_ready(), propose_next_version(), Nightly: score the active strategy version's paper trades, ask the LLM for an…, date, Safety-critical. Enforces, in order: circuit breaker -> capital limit ->… (+21 more)

### Community 8 - "Dashboard TypeScript Config"
Cohesion: 0.07
Nodes (28): compilerOptions, allowJs, esModuleInterop, incremental, isolatedModules, jsx, lib, module (+20 more)

### Community 9 - "Orchestrator Cycle & Tests"
Cohesion: 0.28
Nodes (26): _empty_daily_pnl(), No LLM calls, no market snapshot — just the circuit breaker and the stop-…, run_cycle(), run_risk_check(), _capital_config(), _market(), patch, test_run_cycle_defaults_to_real_execution_agent_when_promoted() (+18 more)

### Community 10 - "HTML Reporting Agent"
Cohesion: 0.20
Nodes (19): generate_report(), _mode_section_html(), _model_usage_html(), _model_usage_stats(), One HTML report covering both modes side by side: PnL vs target, trade log…, render_html(), _row(), _table() (+11 more)

### Community 11 - "Execution Agents & Fee Model"
Cohesion: 0.18
Nodes (11): ABC, Paper Trading Fee/Slippage Simulation Model, ExecutionAgent, Force-close every open position for mode. Returns closed trades., Returns a fill: {"fill_price": float, "fees": float}., _fees(), PaperExecutionAgent, test_place_order_buy_applies_slippage_and_fee() (+3 more)

### Community 12 - "Lenient JSON Parser"
Cohesion: 0.21
Nodes (15): parse_llm_json(), Any, Best-effort cleanup for JSON an LLM claims to have written but didn't quite:…, _strip_code_fence(), _strip_comments(), _strip_trailing_commas(), test_does_not_strip_double_slash_inside_string_value(), test_genuinely_broken_json_still_raises() (+7 more)

### Community 13 - "Dashboard Layout & Navigation"
Cohesion: 0.29
Nodes (5): geistMono, geistSans, metadata, LINKS, Nav()

## Knowledge Gaps
- **72 isolated node(s):** `eslintConfig`, `nextConfig`, `name`, `version`, `private` (+67 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `run_cycle()` connect `Orchestrator Cycle & Tests` to `CoinDCX Client & Real Execution`, `Database Models & Config Seeding`, `LLM Client & Signal Agent`, `Risk Manager & Stop-Loss/Take-Profit`, `Evolution Agent & Promotion Logic`, `Execution Agents & Fee Model`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `RealExecutionAgent` connect `CoinDCX Client & Real Execution` to `Execution Agents & Fee Model`, `Orchestrator Cycle & Tests`, `Project Docs & CI Workflows`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `chat()` connect `LLM Client & Signal Agent` to `Project Docs & CI Workflows`, `Evolution Agent & Promotion Logic`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `run_cycle()` (e.g. with `RuntimeError` and `PaperExecutionAgent`) actually correct?**
  _`run_cycle()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `run_risk_check()` (e.g. with `PaperExecutionAgent` and `RealExecutionAgent`) actually correct?**
  _`run_risk_check()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `eslintConfig`, `nextConfig`, `name` to the rest of the system?**
  _72 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Next.js Dashboard Pages & Components` be split into smaller, more focused modules?**
  _Cohesion score 0.07686932215234102 - nodes in this community are weakly interconnected._