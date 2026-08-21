"""Promotion Gate — the composite, multi-dimensional, evidence-gated
decision engine for auto-promotion (Scientific Strategy Optimization
Framework, extended). Promotion stays fully automatic — no human-approval
step anywhere — but a candidate must clear sample-size floors, risk/
statistical/Monte-Carlo gates, regime/symbol robustness, and a
significant, same-market-data improvement over the current real-mode
champion before PROMOTE. Missing required evidence (e.g. no historical
candles ingested yet for walk-forward/champion-challenger backtest
replay) always yields EXTEND_VALIDATION, never a silent skip and never a
promotion on partial evidence — see PROJECT_SPEC.md §2/§3e.

Pure composition over existing statistical primitives, no new statistical
machinery: bootstrap CI + Monte Carlo drawdown distribution
(src.backtest.statistical_validation), real multi-fold walk-forward
(src.backtest.walk_forward_validator), pairwise same-market-data
comparison (src.backtest.strategy_comparison), overfitting verdict
(src.backtest.overfitting_detection), multi-dimensional fitness
(src.learning.fitness). Never executes a trade, never modifies
params_json/config.py directly — only decides PROMOTE/REJECT/
EXTEND_VALIDATION and returns it for the caller (evolution_agent.py) to
act on and audit.

Hardening pass (audit findings, same 3-way decision, same full automation
— no new gate category, no manual-approval step added anywhere):
- The backtest trade-count sample-size gate (PROMOTION_MIN_BACKTEST_TRADES)
  is now actually enforced — it was imported/configured but never checked.
- A missing Sharpe improvement is missing evidence (EXTEND_VALIDATION),
  never silently treated as a pass.
- The primary champion-vs-challenger significance test is now PAIRED —
  candidate-minus-champion equity delta at matching backtest-replay
  snapshots (same symbols/date range/decision-cycle grid), gated at
  PROMOTION_MIN_CONFIDENCE_PCT — not "is the candidate profitable alone".
- execution_quality in the Promotion Score is real per-trade slippage data
  (or None + reweighted among the rest) — never a neutral 50 placeholder.

Second hardening pass (closes remaining loopholes in the first pass):
- The paired champion-vs-challenger comparison is now the ONLY
  significance test — no fallback to the older unpaired win-rate/
  expectancy z-test when a paired comparison can't be computed. Missing
  paired evidence is missing evidence, full stop (EXTEND_VALIDATION).
- Observations are paired by their shared decision-cycle identifier
  (backtest-replay snapshot_time — see PortfolioManager.snapshot), never
  by list index/position. A length or ordering mismatch between the two
  snapshot lists must never silently mispair an observation.
- PROMOTION_MIN_PAIRED_OBSERVATIONS gates the TRUE matched-observation
  count, replacing the old min(champion_trades, challenger_trades) proxy
  (PROMOTION_MIN_CHAMPION_CHALLENGER_TRADES, retired — independent trade
  counts don't imply matched market observations at all).
- The statistic is explicitly named bootstrap_probability_candidate_
  better_pct (fraction of bootstrap resamples of the paired candidate-
  minus-champion diffs whose sum is positive) — never a generic
  "confidence" label.
- A bot's first-ever promotion (no champion) marks BOTH the champion-
  improvement gate AND the paired-observations sample gate NOT_APPLICABLE
  — previously the sample gate stayed permanently None with no champion
  to pair against, deadlocking every first promotion at EXTEND_VALIDATION
  regardless of how good every other gate looked.

Two gate categories, deliberately never conflated:
- "extend-only" gates (sample sizes): a definite below-floor count is
  EXTEND_VALIDATION, never REJECT — "not enough evidence yet" is not the
  same claim as "the evidence says no".
- "reject-capable" gates (risk/statistical/robustness/overfitting/
  champion-improvement): an explicit False is a genuine REJECT; missing
  evidence (None) is EXTEND_VALIDATION, same as sample sizes.
A REJECT from any reject-capable gate wins over an EXTEND_VALIDATION
elsewhere — real evidence of failure, even on a partial sample, is worth
surfacing immediately rather than waiting for more.

Imports evolution_agent.py at module level (compute_metrics/
promotion_ready) — safe because evolution_agent.py only imports this
module locally inside run_evolution(), keeping the dependency graph a DAG
(same local-import pattern evolution_agent.py already uses for
src.learning.statistics/fitness/learning_status)."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median

from src.agents.evolution_agent import compute_metrics, promotion_ready
from src.backtest.overfitting_detection import OverfittingReport, detect as detect_overfitting
from src.backtest.statistical_validation import bootstrap_confidence_interval, monte_carlo_drawdown_distribution
from src.backtest.strategy_comparison import compare as compare_strategies
from src.backtest.walk_forward_validator import run_walk_forward
from src.config import (
    BACKTEST_BOOTSTRAP_ITERATIONS,
    BACKTEST_RANDOM_SEED,
    BACKTEST_TICK_TIMEFRAME,
    LEARNING_HISTORY_WINDOW_DAYS,
    PROMOTION_COOLDOWN_DAYS,
    PROMOTION_MAX_DRAWDOWN_INCREASE_PCT,
    PROMOTION_MAX_REGIME_DEGRADATION_PCT,
    PROMOTION_MAX_SYMBOL_PROFIT_CONCENTRATION_PCT,
    PROMOTION_MC_CATASTROPHIC_DD_THRESHOLD_PCT,
    PROMOTION_MC_MAX_CATASTROPHIC_DD_PROBABILITY_PCT,
    PROMOTION_MC_MAX_WORST_DRAWDOWN_PCT,
    PROMOTION_MC_MIN_PROFITABLE_PCT,
    PROMOTION_MIN_BACKTEST_TRADES,
    PROMOTION_MIN_CONFIDENCE_PCT,
    PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT,
    PROMOTION_MIN_FITNESS_SCORE,
    PROMOTION_MIN_PAIRED_OBSERVATIONS,
    PROMOTION_MIN_PAPER_TRADES,
    PROMOTION_MIN_PROFITABLE_SYMBOLS,
    PROMOTION_MIN_SCORE,
    PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT,
    PROMOTION_MIN_WALK_FORWARD_TRADES,
    PROMOTION_SCORE_WEIGHT_CHAMPION_IMPROVEMENT,
    PROMOTION_SCORE_WEIGHT_EXECUTION_QUALITY,
    PROMOTION_SCORE_WEIGHT_OUT_OF_SAMPLE,
    PROMOTION_SCORE_WEIGHT_REGIME_ROBUSTNESS,
    PROMOTION_SCORE_WEIGHT_RISK,
    PROMOTION_SCORE_WEIGHT_SIMPLICITY,
    PROMOTION_SCORE_WEIGHT_STABILITY,
    PROMOTION_SCORE_WEIGHT_STATISTICAL_SIGNIFICANCE,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SLIPPAGE_BPS,
)
from src.db import models
from src.features.opportunity_scorer import weighted_average
from src.learning.fitness import compute_fitness_score, drawdown_component
from src.learning.statistics import compute_bucket_statistics
from src.utils import clamp
from src.utils import parse_timestamp as _parse_ts

_EXTEND_ONLY_GATES = ("paper_trades", "backtest_trades", "walk_forward_trades", "paired_observations")
_REJECT_CAPABLE_GATES = (
    "paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo",
    "regime_robustness", "symbol_robustness", "overfitting", "champion_improvement",
)


@dataclass
class PromotionDecision:
    decision: str  # "PROMOTE" | "REJECT" | "EXTEND_VALIDATION"
    promotion_score: float | None
    gates: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)


def build_symbol_to_pair(mode: str) -> dict[str, str] | None:
    """Best-effort mapping for the backtest-replay-based gates below — the
    one thing in this otherwise pure-statistics module that touches the
    network. Fails open (None) on any error (CoinDCX outage, unknown
    symbol); every gate that needs it degrades to EXTEND_VALIDATION, never
    a crash. Same shape as adaptive_strategy_engine.py's
    _build_symbol_to_pair — a separate small copy rather than a cross-
    import, since that module is the top-level orchestrator and importing
    a helper back out of it here would point the dependency the wrong way."""
    try:
        since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
        trades = models.get_recently_closed_trades(mode, since)
        symbols = sorted({t["symbol"] for t in trades if t.get("symbol")})
        if not symbols:
            return None
        from src.coindcx_client import get_markets_details
        from src.coindcx_client import symbol_to_pair as _symbol_to_pair

        details = get_markets_details()
        return {s: _symbol_to_pair(s, details) for s in symbols}
    except Exception:
        return None


def _has_historical_candles(symbols, symbol_to_pair, start, end) -> bool:
    if not symbols or not symbol_to_pair.get(symbols[0]):
        return False
    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    rows = models.get_historical_candles(symbol_to_pair[symbols[0]], BACKTEST_TICK_TIMEFRAME, start_ms, end_ms)
    return bool(rows)


def _bootstrap_probability_of_profit(
    pnls: list[float], iterations: int = BACKTEST_BOOTSTRAP_ITERATIONS, seed: int = BACKTEST_RANDOM_SEED
) -> float | None:
    """Fraction of bootstrap resamples (with replacement) whose summed pnl
    is positive. Same resampling mechanics as statistical_validation.
    bootstrap_confidence_interval, but "probability of profit" needs the
    raw per-resample statistic, not a percentile of that function's CI
    output — a small dedicated pass rather than widening the shared,
    heavily-reused CI function's return shape for one caller."""
    if len(pnls) < 2:
        return None
    rng = random.Random(seed)
    n = len(pnls)
    positive = sum(1 for _ in range(iterations) if sum(pnls[rng.randrange(n)] for _ in range(n)) > 0)
    return positive / iterations * 100


def _paired_champion_comparison(champion_result: dict, candidate_result: dict) -> dict | None:
    """THE primary champion-vs-challenger significance test — candidate
    MINUS champion, not candidate-alone profitability, and the only one:
    no unpaired fallback exists anywhere downstream of this function.

    Observations are paired by their shared decision-cycle identifier —
    each backtest-replay snapshot's `snapshot_time` (BacktestEngine.run()
    fires it off `self.clock`'s deterministic tick grid, identical for
    both engines since both replay the same symbols/date range/decision-
    cycle cadence, only params_json differs) — matched by explicit
    identifier equality via dict-key intersection, NEVER by list index. A
    length or ordering mismatch between the two snapshot lists (e.g. one
    run circuit-breaking out earlier) must never silently mispair an
    observation the way a blind zip(a, b) would.

    paired_observation_count is the count of matched (champion, challenger)
    snapshot pairs itself — the number actually compared against
    PROMOTION_MIN_PAIRED_OBSERVATIONS — not the one-fewer count of period-
    over-period diffs the bootstrap consumes internally. Reuses
    _bootstrap_probability_of_profit's resampling (built for the Monte
    Carlo risk gate below) against those diffs — "is this positive" is the
    same question either way."""
    champ_by_time = {s["snapshot_time"]: s["equity"] for s in champion_result["snapshots"]}
    cand_by_time = {s["snapshot_time"]: s["equity"] for s in candidate_result["snapshots"]}
    shared_times = sorted(set(champ_by_time) & set(cand_by_time))
    paired_observation_count = len(shared_times)
    if paired_observation_count < 3:  # need >= 2 diffs for the bootstrap below to mean anything
        return None
    champ_eq = [champ_by_time[t] for t in shared_times]
    cand_eq = [cand_by_time[t] for t in shared_times]
    diffs = [(cand_eq[i] - cand_eq[i - 1]) - (champ_eq[i] - champ_eq[i - 1]) for i in range(1, paired_observation_count)]
    bootstrap_probability_candidate_better_pct = _bootstrap_probability_of_profit(diffs)
    if bootstrap_probability_candidate_better_pct is None:
        return None
    return {
        "paired_observation_count": paired_observation_count,
        "mean_difference": sum(diffs) / len(diffs),
        "median_difference": median(diffs),
        "bootstrap_probability_candidate_better_pct": bootstrap_probability_candidate_better_pct,
        "significant": bootstrap_probability_candidate_better_pct >= PROMOTION_MIN_CONFIDENCE_PCT,
    }


def _cooldown_gate(mode: str) -> dict:
    latest = models.get_latest_promotion_audit(mode, event_type="promotion")
    if latest is None:
        return {"passed": True, "detail": "no prior promotion"}
    last_time = _parse_ts(latest["created_at"])
    elapsed = datetime.now(timezone.utc) - last_time
    if elapsed < timedelta(days=PROMOTION_COOLDOWN_DAYS):
        cooldown_end = (last_time + timedelta(days=PROMOTION_COOLDOWN_DAYS)).date()
        return {"passed": False, "detail": f"cooldown active until {cooldown_end}"}
    return {"passed": True, "detail": f"{elapsed.days}d since last promotion"}


def _sample_size_gates(
    paper_count: int,
    backtest_count: int | None,
    walk_forward_count: int | None,
    paired_observation_count: int | None,
    champion_present: bool,
) -> dict:
    gates = {
        "paper_trades": {
            "passed": paper_count >= PROMOTION_MIN_PAPER_TRADES,
            "count": paper_count,
            "minimum": PROMOTION_MIN_PAPER_TRADES,
        }
    }
    # Issue 1: the actual backtest trade count (BacktestEngine's own
    # closed_trades, never paper/walk-forward/paired-observation counts
    # substituted in) must independently clear PROMOTION_MIN_BACKTEST_TRADES
    # — previously imported/configured but never enforced.
    if backtest_count is None:
        gates["backtest_trades"] = {"passed": None, "detail": "backtest not available (no historical data)"}
    else:
        gates["backtest_trades"] = {
            "passed": backtest_count >= PROMOTION_MIN_BACKTEST_TRADES,
            "count": backtest_count,
            "minimum": PROMOTION_MIN_BACKTEST_TRADES,
        }
    if walk_forward_count is None:
        gates["walk_forward_trades"] = {"passed": None, "detail": "walk-forward not available (no historical data)"}
    else:
        gates["walk_forward_trades"] = {
            "passed": walk_forward_count >= PROMOTION_MIN_WALK_FORWARD_TRADES,
            "count": walk_forward_count,
            "minimum": PROMOTION_MIN_WALK_FORWARD_TRADES,
        }
    if not champion_present:
        # Fix 3: nothing to pair against on a bot's first-ever promotion —
        # NOT_APPLICABLE, never a perpetual EXTEND_VALIDATION deadlock.
        gates["paired_observations"] = {
            "passed": True,
            "status": "NOT_APPLICABLE",
            "detail": "no current champion — nothing to pair against (first-ever promotion)",
        }
    elif paired_observation_count is None:
        gates["paired_observations"] = {
            "passed": None,
            "detail": "paired champion-challenger comparison not available (no historical data)",
        }
    else:
        gates["paired_observations"] = {
            "passed": paired_observation_count >= PROMOTION_MIN_PAIRED_OBSERVATIONS,
            "count": paired_observation_count,
            "minimum": PROMOTION_MIN_PAIRED_OBSERVATIONS,
        }
    return gates


def _risk_gates(candidate: dict, metrics: dict, trades: list[dict], fitness_score: float | None, capital_to_use: float) -> dict:
    gates = {"paper_days_pnl_drawdown": {"passed": promotion_ready(candidate, metrics), "detail": metrics}}

    pnls = [t["pnl"] for t in trades if t.get("pnl") is not None]
    ci = bootstrap_confidence_interval(pnls)
    gates["bootstrap_ci"] = {"passed": ci is not None and ci["ci_low"] > 0, "detail": ci}
    gates["fitness_floor"] = {
        "passed": fitness_score is not None and fitness_score >= PROMOTION_MIN_FITNESS_SCORE,
        "value": fitness_score,
    }

    if len(pnls) >= 2:
        mc = monte_carlo_drawdown_distribution(pnls, capital_to_use)
        prob_profit = _bootstrap_probability_of_profit(pnls)
        catastrophic_prob = None
        if mc is not None:
            catastrophic_prob = (
                sum(1 for d in mc["drawdowns"] if d >= PROMOTION_MC_CATASTROPHIC_DD_THRESHOLD_PCT)
                / len(mc["drawdowns"])
                * 100
            )
        mc_passed = (
            mc is not None
            and prob_profit is not None
            and prob_profit >= PROMOTION_MC_MIN_PROFITABLE_PCT
            and catastrophic_prob is not None
            and catastrophic_prob <= PROMOTION_MC_MAX_CATASTROPHIC_DD_PROBABILITY_PCT
            and mc["simulated_worst_drawdown_pct"] <= PROMOTION_MC_MAX_WORST_DRAWDOWN_PCT
        )
        gates["monte_carlo"] = {
            "passed": mc_passed,
            "probability_of_profit_pct": prob_profit,
            "catastrophic_drawdown_probability_pct": catastrophic_prob,
            "worst_simulated_drawdown_pct": mc["simulated_worst_drawdown_pct"] if mc else None,
        }
    else:
        gates["monte_carlo"] = {"passed": None, "detail": "insufficient trades for Monte Carlo"}

    return gates


def _regime_robustness_gate(mode: str, candidate_trades: list[dict], champion_id: int | None, capital_to_use: float) -> dict:
    candidate_by_regime: dict[str, list[dict]] = {}
    for t in candidate_trades:
        regime = t.get("market_regime")
        if regime:
            candidate_by_regime.setdefault(regime, []).append(t)

    champion_stats_by_regime: dict[str, dict] = {}
    if champion_id is not None:
        champion_by_regime: dict[str, list[dict]] = {}
        for t in models.get_closed_trades(mode, champion_id):
            regime = t.get("market_regime")
            if regime:
                champion_by_regime.setdefault(regime, []).append(t)
        champion_stats_by_regime = {
            regime: compute_bucket_statistics(rs, capital_to_use) for regime, rs in champion_by_regime.items()
        }

    degraded_regimes, detail = [], {}
    for regime, rs in candidate_by_regime.items():
        if len(rs) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        cand_stats = compute_bucket_statistics(rs, capital_to_use)
        champ_stats = champion_stats_by_regime.get(regime)
        entry = {"candidate_expectancy": cand_stats["expectancy"], "trades_count": len(rs)}
        # No champion baseline for this regime yet, or champion itself
        # isn't profitable here -> nothing meaningful to degrade below.
        if champ_stats and champ_stats.get("expectancy") and champ_stats["expectancy"] > 0 and cand_stats["expectancy"] is not None:
            degradation_pct = (champ_stats["expectancy"] - cand_stats["expectancy"]) / champ_stats["expectancy"] * 100
            entry["champion_expectancy"] = champ_stats["expectancy"]
            entry["degradation_pct"] = degradation_pct
            if degradation_pct > PROMOTION_MAX_REGIME_DEGRADATION_PCT:
                degraded_regimes.append(regime)
        detail[regime] = entry

    return {"passed": len(degraded_regimes) == 0, "degraded_regimes": degraded_regimes, "detail": detail}


def _symbol_robustness_gate(candidate_trades: list[dict]) -> dict:
    by_symbol: dict[str, float] = {}
    for t in candidate_trades:
        pnl = t.get("pnl")
        if pnl is not None:
            by_symbol[t["symbol"]] = by_symbol.get(t["symbol"], 0.0) + pnl

    profitable = {s: p for s, p in by_symbol.items() if p > 0}
    total_profit = sum(profitable.values())
    max_concentration_pct = max(profitable.values()) / total_profit * 100 if total_profit > 0 else None

    concentration_ok = max_concentration_pct is None or max_concentration_pct <= PROMOTION_MAX_SYMBOL_PROFIT_CONCENTRATION_PCT
    diversity_ok = len(profitable) >= PROMOTION_MIN_PROFITABLE_SYMBOLS

    return {
        "passed": concentration_ok and diversity_ok,
        "max_symbol_profit_concentration_pct": max_concentration_pct,
        "profitable_symbols_count": len(profitable),
        "profitable_symbols": sorted(profitable),
    }


def _backtest_evidence(candidate: dict, champion: dict | None, candidate_trades: list[dict], symbol_to_pair: dict[str, str] | None) -> dict:
    """None fields throughout when historical data isn't available —
    every caller treats None as "can't evaluate this gate yet", never a
    failure. Reuses BacktestEngine/walk_forward_validator/
    strategy_comparison/overfitting_detection exactly as the existing
    exit-params candidate pipeline (simulation.py) already does."""
    result = {
        "backtest_trades_count": None,
        "walk_forward_folds": None,
        "walk_forward_trades_count": None,
        "overfitting_report": None,
        "champion_challenger": None,
        "paired_observation_count": None,
    }
    closed = [t for t in candidate_trades if t.get("closed_at")]
    if not closed or not symbol_to_pair:
        return result

    symbols = sorted({t["symbol"] for t in closed if symbol_to_pair.get(t["symbol"])})
    if not symbols:
        return result
    start = min(_parse_ts(t["closed_at"]) for t in closed).date()
    end = max(_parse_ts(t["closed_at"]) for t in closed).date()
    if not _has_historical_candles(symbols, symbol_to_pair, start, end):
        return result

    from src.backtest.engine import BacktestEngine
    from src.backtest.performance_analyzer import analyze

    candidate_params = candidate.get("params_json") or {}
    candidate_engine = BacktestEngine(symbols, symbol_to_pair, start, end, params_json=candidate_params)
    candidate_result = candidate_engine.run()
    result["backtest_trades_count"] = len(candidate_result["closed_trades"])

    folds = run_walk_forward(symbols, symbol_to_pair, start, end, candidate_params)
    if folds:
        result["walk_forward_folds"] = folds
        result["walk_forward_trades_count"] = sum((f.out_of_sample_metrics or {}).get("trades_count") or 0 for f in folds)
        result["overfitting_report"] = detect_overfitting(folds)

    if champion is not None:
        champion_params = champion.get("params_json") or {}
        champion_engine = BacktestEngine(symbols, symbol_to_pair, start, end, params_json=champion_params)
        champion_result = champion_engine.run()

        candidate_metrics = analyze(
            candidate_result["closed_trades"], candidate_result["snapshots"], candidate_engine.portfolio.starting_capital
        )
        champion_metrics = analyze(
            champion_result["closed_trades"], champion_result["snapshots"], champion_engine.portfolio.starting_capital
        )
        paired_comparison = _paired_champion_comparison(champion_result, candidate_result)
        result["champion_challenger"] = {
            "comparison": compare_strategies(
                champion_result["closed_trades"], candidate_result["closed_trades"], champion_metrics, candidate_metrics
            ),
            "paired_comparison": paired_comparison,
            "candidate_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
        }
        # Fix 2: the TRUE matched-observation count, not
        # min(champion_trades, candidate_trades) — independent trade
        # counts don't imply matched market observations at all.
        result["paired_observation_count"] = paired_comparison["paired_observation_count"] if paired_comparison else None

    return result


def _pct_improvement(old: float | None, new: float | None) -> float | None:
    if old is None or new is None or old == 0:
        return None
    return (new - old) / abs(old) * 100


def _champion_improvement_gate(champion: dict | None, backtest_evidence: dict) -> dict:
    if champion is None:
        # Fix 3: first-ever promotion has nothing to beat — vacuously
        # passes this ONE gate, every other gate (sample sizes, risk,
        # regime/symbol robustness, overfitting) still runs independently.
        return {
            "passed": True,
            "status": "NOT_APPLICABLE",
            "result": "NOT_APPLICABLE",
            "detail": "no current champion — vacuously passes (first-ever promotion)",
        }

    cc = backtest_evidence.get("champion_challenger")
    if cc is None:
        return {
            "passed": None,
            "status": "UNAVAILABLE",
            "detail": "champion-challenger comparison not available (no historical data)",
        }

    comparison = cc["comparison"]  # informational only — see Fix 1, never drives `passed`
    cand, champ = cc["candidate_metrics"], cc["champion_metrics"]
    expectancy_improvement_pct = _pct_improvement(champ.get("expectancy"), cand.get("expectancy"))
    if expectancy_improvement_pct is None:
        return {
            "passed": None,
            "status": "UNAVAILABLE",
            "detail": "expectancy improvement not computable yet",
            "comparison": comparison,
        }

    sharpe_improvement_pct = _pct_improvement(champ.get("sharpe_ratio"), cand.get("sharpe_ratio"))
    if sharpe_improvement_pct is None:
        # Issue 2: a missing Sharpe improvement is missing EVIDENCE, never
        # a pass and never treated as 0 — the old `is None or >=` check
        # let a null silently clear this gate, which is exactly backwards.
        return {
            "passed": None,
            "status": "UNAVAILABLE",
            "detail": "Insufficient Sharpe evidence",
            "expectancy_improvement_pct": expectancy_improvement_pct,
            "comparison": comparison,
        }

    sortino_improvement_pct = _pct_improvement(champ.get("sortino_ratio"), cand.get("sortino_ratio"))
    profit_factor_change_pct = _pct_improvement(champ.get("profit_factor"), cand.get("profit_factor"))
    drawdown_increase_pct = (cand.get("max_drawdown_pct") or 0) - (champ.get("max_drawdown_pct") or 0)

    # Fix 1: the paired comparison is the ONLY significance test — no
    # fallback to the older unpaired win-rate/expectancy z-test
    # (`comparison`, still carried above for the audit record only) when
    # the paired series can't be computed. Missing paired evidence is
    # missing evidence, never a substitute pass and never inferred.
    paired = cc.get("paired_comparison")
    if paired is None:
        return {
            "passed": None,
            "status": "UNAVAILABLE",
            "detail": "paired champion-challenger comparison not available (insufficient matched observations)",
            "expectancy_improvement_pct": expectancy_improvement_pct,
            "sharpe_improvement_pct": sharpe_improvement_pct,
            "sortino_improvement_pct": sortino_improvement_pct,
            "profit_factor_change_pct": profit_factor_change_pct,
            "drawdown_increase_pct": drawdown_increase_pct,
            "comparison": comparison,
        }

    significant = paired["significant"]
    meets_minimums = (
        expectancy_improvement_pct >= PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT
        and sharpe_improvement_pct >= PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT
        and drawdown_increase_pct <= PROMOTION_MAX_DRAWDOWN_INCREASE_PCT
    )
    passed = significant and meets_minimums
    return {
        "passed": passed,
        "status": "AVAILABLE",
        "result": "candidate_significantly_better" if significant else "not_significant",
        "significant": significant,
        "paired_observation_count": paired["paired_observation_count"],
        "mean_difference": paired["mean_difference"],
        "median_difference": paired["median_difference"],
        "bootstrap_probability_candidate_better_pct": paired["bootstrap_probability_candidate_better_pct"],
        "sharpe_improvement_pct": sharpe_improvement_pct,
        "sortino_improvement_pct": sortino_improvement_pct,
        "expectancy_improvement_pct": expectancy_improvement_pct,
        "profit_factor_change_pct": profit_factor_change_pct,
        "drawdown_increase_pct": drawdown_increase_pct,
        "candidate_metrics": cand,
        "champion_metrics": champ,
        "comparison": comparison,
    }


def _complexity_delta(candidate: dict, champion: dict | None) -> int:
    """Count of params_json keys that differ between candidate and
    champion — today's only auto-versioning path is exit-params
    (stop_loss_pct/take_profit_pct), so this is almost always 0-2. Real
    code, not a no-op, feeding the promotion score's simplicity component
    — not a separate hard gate (Phase 12: prefer simpler unless the
    improvement is statistically meaningful, which the score already
    encodes by weighing champion_improvement 4x simplicity)."""
    candidate_params = candidate.get("params_json") or {}
    champion_params = (champion or {}).get("params_json") or {}
    all_keys = set(candidate_params) | set(champion_params)
    return sum(1 for k in all_keys if candidate_params.get(k) != champion_params.get(k))


def _execution_quality_component(trades: list[dict]) -> float | None:
    """Real per-trade entry slippage — orchestrator.py already records
    entry_slippage_pct on every opened trade — scored against the assumed
    SLIPPAGE_BPS baseline the net-expectancy gate already prices every
    trade against. Issue 4: None (never a neutral 50) when no trade in
    this candidate's history has slippage recorded; weighted_average
    excludes a None component and redistributes its weight, same
    convention every other aggregation step in this codebase already uses."""
    slippages = [abs(t["entry_slippage_pct"]) for t in trades if t.get("entry_slippage_pct") is not None]
    if not slippages or not SLIPPAGE_BPS:
        return None
    avg_slippage_bps = (sum(slippages) / len(slippages)) * 100  # pct -> bps
    return clamp(100 - (avg_slippage_bps / SLIPPAGE_BPS) * 50, 0, 100)


def _promotion_score(
    gates: dict,
    champion_gate: dict,
    overfitting_report: OverfittingReport | None,
    complexity_delta: int,
    trades: list[dict],
) -> tuple[float | None, dict]:
    """`gates` is the full merged gate dict evaluate_promotion built (sample-
    size + risk + regime/symbol/overfitting all live in one dict there) —
    a single parameter, not artificially split, since every component
    below reads from that same merged evidence. `trades` is the
    candidate's own closed trades, used only for the execution-quality
    component."""
    out_of_sample = None
    if gates["paper_trades"]["passed"]:
        out_of_sample = 100.0 if gates.get("walk_forward_trades", {}).get("passed") else 60.0

    champion_improvement = None
    if champion_gate.get("expectancy_improvement_pct") is not None:
        champion_improvement = clamp(50 + champion_gate["expectancy_improvement_pct"], 0, 100)

    mc_gate = gates.get("monte_carlo", {})
    risk = None
    if mc_gate.get("passed") is not None:
        parts = [c for c in (drawdown_component(gates["paper_days_pnl_drawdown"]["detail"].get("max_drawdown_pct")),
                              mc_gate.get("probability_of_profit_pct")) if c is not None]
        risk = sum(parts) / len(parts) if parts else None

    ci = gates.get("bootstrap_ci", {}).get("detail")
    statistical_significance = None if ci is None else (100.0 if ci["ci_low"] > 0 else 0.0)

    regime_gate = gates.get("regime_robustness", {})
    regime_robustness = None
    if regime_gate.get("detail"):
        regime_robustness = 0.0 if regime_gate.get("degraded_regimes") else 100.0

    execution_quality = _execution_quality_component(trades)

    stability = None if overfitting_report is None else clamp(100 - overfitting_report.walk_forward_failure_rate, 0, 100)

    simplicity = clamp(100 - complexity_delta * 10, 0, 100)

    components = {
        "out_of_sample": out_of_sample,
        "champion_improvement": champion_improvement,
        "risk": risk,
        "statistical_significance": statistical_significance,
        "regime_robustness": regime_robustness,
        "execution_quality": execution_quality,
        "stability": stability,
        "simplicity": simplicity,
    }
    weights = {
        "out_of_sample": PROMOTION_SCORE_WEIGHT_OUT_OF_SAMPLE,
        "champion_improvement": PROMOTION_SCORE_WEIGHT_CHAMPION_IMPROVEMENT,
        "risk": PROMOTION_SCORE_WEIGHT_RISK,
        "statistical_significance": PROMOTION_SCORE_WEIGHT_STATISTICAL_SIGNIFICANCE,
        "regime_robustness": PROMOTION_SCORE_WEIGHT_REGIME_ROBUSTNESS,
        "execution_quality": PROMOTION_SCORE_WEIGHT_EXECUTION_QUALITY,
        "stability": PROMOTION_SCORE_WEIGHT_STABILITY,
        "simplicity": PROMOTION_SCORE_WEIGHT_SIMPLICITY,
    }
    return weighted_average(components, weights), components


def evaluate_promotion(
    mode: str,
    candidate: dict,
    trades: list[dict],
    capital_to_use: float,
    champion: dict | None = None,
    symbol_to_pair: dict[str, str] | None = None,
) -> PromotionDecision:
    """The single entry point evolution_agent.py calls. `trades` is the
    candidate's own closed trades (models.get_closed_trades(mode,
    candidate["id"])); `champion` is models.get_latest_promoted_version()
    (or None — a bot's first-ever promotion has nothing to beat yet).
    Every gate that CAN run does, regardless of what else fails/is
    pending, so `.gates`/`.breakdown` are a complete audit record even on
    REJECT/EXTEND_VALIDATION — see promotion_audit (§9 of the plan)."""
    gates: dict = {"cooldown": _cooldown_gate(mode)}

    closed_trades = [t for t in trades if t.get("pnl") is not None]
    metrics = compute_metrics(closed_trades, capital_to_use)
    fitness = compute_fitness_score(compute_bucket_statistics(closed_trades, capital_to_use), capital_to_use)
    fitness_score = fitness["fitness_score"]

    backtest_evidence = _backtest_evidence(candidate, champion, closed_trades, symbol_to_pair)

    gates.update(_sample_size_gates(
        len(closed_trades),
        backtest_evidence["backtest_trades_count"],
        backtest_evidence["walk_forward_trades_count"],
        backtest_evidence["paired_observation_count"],
        champion is not None,
    ))
    gates.update(_risk_gates(candidate, metrics, closed_trades, fitness_score, capital_to_use))

    champion_id = champion["id"] if champion else None
    gates["regime_robustness"] = _regime_robustness_gate(mode, closed_trades, champion_id, capital_to_use)
    gates["symbol_robustness"] = _symbol_robustness_gate(closed_trades)

    overfitting_report = backtest_evidence["overfitting_report"]
    gates["overfitting"] = (
        {"passed": overfitting_report.verdict != "overfit", "verdict": overfitting_report.verdict,
         "walk_forward_failure_rate_pct": overfitting_report.walk_forward_failure_rate}
        if overfitting_report is not None
        else {"passed": None, "detail": "walk-forward not available (no historical data)"}
    )

    champion_gate = _champion_improvement_gate(champion, backtest_evidence)
    gates["champion_improvement"] = champion_gate

    complexity_delta = _complexity_delta(candidate, champion)
    score, score_components = _promotion_score(gates, champion_gate, overfitting_report, complexity_delta, closed_trades)

    # Issue 7: mandatory gates decide first, unconditionally — the score
    # below only ever gates a candidate that has ALREADY cleared every one
    # of them; it can never rescue a candidate that failed one.
    reasons: list[str] = []
    if not gates["cooldown"]["passed"]:
        decision = "EXTEND_VALIDATION"
        reasons.append(gates["cooldown"]["detail"])
    else:
        reject_failed = [n for n in _REJECT_CAPABLE_GATES if gates.get(n, {}).get("passed") is False]
        if reject_failed:
            decision = "REJECT"
            reasons.extend(f"{n} failed" for n in reject_failed)
        else:
            extend_unmet = [n for n in _EXTEND_ONLY_GATES if gates.get(n, {}).get("passed") is not True]
            reject_pending = [n for n in _REJECT_CAPABLE_GATES if gates.get(n, {}).get("passed") is None]
            if extend_unmet or reject_pending:
                decision = "EXTEND_VALIDATION"
                reasons.extend(f"{n}: insufficient evidence — {gates[n].get('detail')}" for n in extend_unmet + reject_pending)
            elif score is None or score < PROMOTION_MIN_SCORE:
                decision = "EXTEND_VALIDATION"
                reasons.append(f"promotion_score {score} below minimum {PROMOTION_MIN_SCORE}")
            else:
                decision = "PROMOTE"
                reasons.append(f"all gates cleared, promotion_score={score:.1f}")

    # Issue 6: one explicit, flat, named-field decision record — everything
    # a human (or the promotion_audit row §9/Issue 10 persists it into)
    # needs to answer "why", without re-deriving it from nested gates.
    risk_gate_names = ("paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo")
    risk_results = [gates[n].get("passed") for n in risk_gate_names]
    risk_status = "fail" if any(r is False for r in risk_results) else "pending" if any(r is None for r in risk_results) else "pass"

    summary = {
        "candidate_id": candidate.get("id"),
        "version": candidate.get("version_number"),
        "champion_id": champion.get("id") if champion else None,
        "champion_version": champion.get("version_number") if champion else None,
        "backtest_trade_count": backtest_evidence["backtest_trades_count"],
        "walkforward_trade_count": backtest_evidence["walk_forward_trades_count"],
        "paper_trade_count": len(closed_trades),
        "paired_observation_count": backtest_evidence["paired_observation_count"],
        "champion_comparison_status": champion_gate.get("status"),
        "champion_comparison_result": champion_gate.get("result"),
        "mean_difference": champion_gate.get("mean_difference"),
        "median_difference": champion_gate.get("median_difference"),
        "bootstrap_probability_candidate_better_pct": champion_gate.get("bootstrap_probability_candidate_better_pct"),
        "sharpe_improvement_pct": champion_gate.get("sharpe_improvement_pct"),
        "sortino_improvement_pct": champion_gate.get("sortino_improvement_pct"),
        "expectancy_improvement_pct": champion_gate.get("expectancy_improvement_pct"),
        "drawdown_change_pct": champion_gate.get("drawdown_increase_pct"),
        "profit_factor_change_pct": champion_gate.get("profit_factor_change_pct"),
        "execution_quality": score_components.get("execution_quality"),
        "regime_robustness": gates["regime_robustness"]["passed"],
        "symbol_robustness": gates["symbol_robustness"]["passed"],
        "overfitting_status": overfitting_report.verdict if overfitting_report is not None else None,
        "risk_status": risk_status,
        "promotion_score": score,
        "passed_gates": [n for n, g in gates.items() if g.get("passed") is True],
        "failed_gates": [n for n, g in gates.items() if g.get("passed") is False],
        "missing_gates": [n for n, g in gates.items() if g.get("passed") is None],
        "promotion_status": decision,
        "promotion_reason": "; ".join(reasons),
    }

    breakdown = {
        "metrics": metrics,
        "fitness": fitness,
        "complexity_delta": complexity_delta,
        "score_components": score_components,
        "backtest_evidence_available": backtest_evidence["backtest_trades_count"] is not None,
        "walk_forward_fold_count": len(backtest_evidence["walk_forward_folds"] or []),
        "summary": summary,
    }

    return PromotionDecision(decision, score, gates, reasons, breakdown)
