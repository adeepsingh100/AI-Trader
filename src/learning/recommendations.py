"""Advisory recommendations — never auto-applied to config, human approval
required. Four generators, all following the same idempotency pattern
(skip inserting if the latest existing recommendation for that metric_name
hasn't moved materially — otherwise this would write near-duplicate rows
every night forever) and the same "only from trades already taken" limit:

- generate_recommendations(): the original threshold sweep on
  MIN_OPPORTUNITY_SCORE (unchanged, still called from evolution_agent's
  cron exactly as before this module grew).
- generate_weight_recommendations(): Step 2 — candidate OPPORTUNITY_WEIGHT_*
  values from feature_importance.compute_subscore_correlation_weights(),
  accepted only if they separate winners from losers (via
  feature_importance.score_separation_p_value) significantly better than
  the current weights do.
- generate_regime_recommendations(): Step 4 — "avoid regime X" plus
  regime-conditioned weight recommendations, reusing the same primitives
  scoped to each regime's trade subset.
- generate_symbol_recommendations(): Step 5 — "avoid symbol X" plus a
  per-symbol optimal-threshold sweep (confidence/opportunity_score/
  stop_distance/volatility) via a generalized _find_optimal_threshold.

Real limitation, stated once here for all four: none of this ever
discovers a trade that wasn't taken — a candidate weight/threshold can
only be checked against the outcomes of trades that already happened
under the CURRENT weights/thresholds. That's a selection-bias ceiling,
not a bug; a real backtester (candle replay) would be a much bigger
feature and isn't being built here."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

from src.config import (
    EXIT_PARAM_SWEEP_MAX_PCT,
    EXIT_PARAM_SWEEP_MIN_PCT,
    EXIT_PARAM_SWEEP_STEP_PCT,
    LEARNING_HISTORY_WINDOW_DAYS,
    MIN_EXPECTANCY_DELTA,
    MIN_OPPORTUNITY_SCORE,
    OPPORTUNITY_SCORE_BUCKET_WIDTH,
    OPPORTUNITY_WEIGHT_MOMENTUM,
    OPPORTUNITY_WEIGHT_RISK,
    OPPORTUNITY_WEIGHT_TREND,
    OPPORTUNITY_WEIGHT_VOLATILITY,
    OPPORTUNITY_WEIGHT_VOLUME,
    RECOMMENDATION_MIN_IMPROVEMENT_PCT,
    RECOMMENDATION_MIN_SAMPLE_SIZE,
    SIGNIFICANCE_THRESHOLD,
)
from src.db import models
from src.features.opportunity_scorer import PRIMARY_TIMEFRAME
from src.learning.feature_importance import compute_subscore_correlation_weights, score_separation_p_value
from src.learning.learning_status import LearningStatus, compute_learning_status
from src.learning.statistics import compute_bucket_statistics, z_test_two_proportions

_SUBSCORE_TO_WEIGHT_CONFIG = {
    "trend_score": "OPPORTUNITY_WEIGHT_TREND",
    "momentum_score": "OPPORTUNITY_WEIGHT_MOMENTUM",
    "volume_score": "OPPORTUNITY_WEIGHT_VOLUME",
    "volatility_score": "OPPORTUNITY_WEIGHT_VOLATILITY",
    "risk_score": "OPPORTUNITY_WEIGHT_RISK",
}


def current_weights() -> dict[str, float]:
    return {
        "trend_score": OPPORTUNITY_WEIGHT_TREND,
        "momentum_score": OPPORTUNITY_WEIGHT_MOMENTUM,
        "volume_score": OPPORTUNITY_WEIGHT_VOLUME,
        "volatility_score": OPPORTUNITY_WEIGHT_VOLATILITY,
        "risk_score": OPPORTUNITY_WEIGHT_RISK,
    }


def _recently_closed(mode: str) -> list[dict]:
    since = datetime.now(timezone.utc) - timedelta(days=LEARNING_HISTORY_WINDOW_DAYS)
    return [t for t in models.get_recently_closed_trades(mode, since) if t.get("pnl") is not None]


def _not_materially_different(mode: str, metric_name: str, candidate_value: float) -> bool:
    latest = models.get_latest_recommendation(mode, metric_name)
    if latest is None or latest.get("recommended_value") is None:
        return False
    prior = latest["recommended_value"] or 1.0
    moved_pct = abs(latest["recommended_value"] - candidate_value) / abs(prior) * 100
    return moved_pct < RECOMMENDATION_MIN_IMPROVEMENT_PCT


def _find_optimal_threshold(
    scored_trades: list[tuple[float, dict]],
    candidate_thresholds: list[float],
    baseline_stats: dict,
    capital_to_use: float,
) -> tuple[float, dict, float] | None:
    """Sweeps candidate_thresholds, keeping trades with score >= threshold,
    and returns the threshold whose expectancy improves on baseline_stats
    by BOTH the relative RECOMMENDATION_MIN_IMPROVEMENT_PCT check AND an
    absolute MIN_EXPECTANCY_DELTA floor — the absolute floor matters once
    baselines get small/noisy (per-symbol/per-regime buckets), where a
    near-zero baseline expectancy would make the relative check trivially
    pass on a meaningless swing. None if nothing clears both bars."""
    best_threshold, best_stats, best_expectancy = None, None, baseline_stats["expectancy"]
    for threshold in candidate_thresholds:
        higher = [t for score, t in scored_trades if score >= threshold]
        if len(higher) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        stats = compute_bucket_statistics(higher, capital_to_use)
        if stats["expectancy"] is not None and stats["expectancy"] > best_expectancy:
            best_threshold, best_stats, best_expectancy = threshold, stats, stats["expectancy"]

    if best_threshold is None:
        return None

    delta = best_expectancy - baseline_stats["expectancy"]
    if delta < MIN_EXPECTANCY_DELTA:
        return None

    if baseline_stats["expectancy"]:
        improvement_pct = delta / abs(baseline_stats["expectancy"]) * 100
    else:
        improvement_pct = float("inf") if best_expectancy > 0 else 0.0

    if improvement_pct < RECOMMENDATION_MIN_IMPROVEMENT_PCT:
        return None

    return best_threshold, best_stats, improvement_pct


def generate_recommendations(
    mode: str, weakness_context: dict | None = None, status: LearningStatus | None = None
) -> list[dict]:
    """weakness_context (Scientific Strategy Optimization Framework,
    optional): weakness_detection.identify_weaknesses(mode)'s output —
    when its worst opportunity_score_bucket finding is available, the
    rationale cites it as supporting evidence rather than standing alone
    as a bare number. status (Evidence-Driven Learning Progression,
    optional): a pre-computed LearningStatus — adaptive_strategy_engine.py
    computes one and threads it into all 8 generator/simulator calls per
    run rather than each recomputing it; standalone/test callers omitting
    it get one computed here."""
    status = status or compute_learning_status(mode)
    if not status.can_generate_hypotheses():
        return []
    closed = _recently_closed(mode)

    scored_trades = []
    for trade in closed:
        entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
        if entry_eval and entry_eval.get("opportunity_score") is not None:
            scored_trades.append((entry_eval["opportunity_score"], trade))
    if len(scored_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
        return []

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    baseline_trades = [t for score, t in scored_trades if score >= MIN_OPPORTUNITY_SCORE]
    baseline_stats = compute_bucket_statistics(baseline_trades, capital_to_use)
    if baseline_stats["expectancy"] is None:
        return []

    candidate_thresholds = sorted(
        {
            math.floor(score / OPPORTUNITY_SCORE_BUCKET_WIDTH) * OPPORTUNITY_SCORE_BUCKET_WIDTH
            for score, _ in scored_trades
            if score > MIN_OPPORTUNITY_SCORE
        }
    )

    found = _find_optimal_threshold(scored_trades, candidate_thresholds, baseline_stats, capital_to_use)
    if found is None:
        return []
    best_threshold, best_stats, _improvement_pct = found

    if _not_materially_different(mode, "MIN_OPPORTUNITY_SCORE", best_threshold):
        return []

    rationale = (
        f"Trades scoring >= {best_threshold} (n={best_stats['trades_count']}) show expectancy "
        f"{best_stats['expectancy']:.2f} vs {baseline_stats['expectancy']:.2f} at the current "
        f"MIN_OPPORTUNITY_SCORE={MIN_OPPORTUNITY_SCORE} (n={len(baseline_trades)})."
    )
    weak_bucket = (weakness_context or {}).get("worst_by_dimension", {}).get("opportunity_score_bucket")
    if weak_bucket is not None:
        rationale += (
            f" Corroborated by weakness detection: bucket {weak_bucket['value']} is the worst-performing "
            f"opportunity_score_bucket (expectancy {weak_bucket['expectancy']:.2f}, n={weak_bucket['trades_count']})."
        )
    models.insert_recommendation(
        mode=mode,
        metric_name="MIN_OPPORTUNITY_SCORE",
        current_value=MIN_OPPORTUNITY_SCORE,
        recommended_value=best_threshold,
        rationale=rationale,
        sample_size=best_stats["trades_count"],
        category="threshold",
    )
    return [{"metric_name": "MIN_OPPORTUNITY_SCORE", "recommended_value": best_threshold, "rationale": rationale}]


def generate_weight_recommendations(mode: str, status: LearningStatus | None = None) -> list[dict]:
    """Step 2: candidate weight set from sub-score/outcome correlation,
    accepted only if it separates winners from losers significantly
    better (lower p-value) than the CURRENT weights do on the same trades
    — never just "is the correlation positive," which says nothing about
    whether it beats what's already configured."""
    candidate_weights = compute_subscore_correlation_weights(mode)
    if candidate_weights is None:
        return []

    status = status or compute_learning_status(mode)
    if not status.can_generate_hypotheses():
        return []

    trades = _recently_closed(mode)

    candidate_sep = score_separation_p_value(trades, candidate_weights)
    if candidate_sep is None or candidate_sep["p_value"] is None or candidate_sep["p_value"] >= SIGNIFICANCE_THRESHOLD:
        return []

    live_weights = current_weights()
    current_sep = score_separation_p_value(trades, live_weights)
    if current_sep is not None and current_sep["p_value"] is not None and candidate_sep["p_value"] >= current_sep["p_value"]:
        return []  # candidate doesn't separate winners/losers any better than current

    confidence = (1 - candidate_sep["p_value"]) * 100
    current_p_display = (
        f"{current_sep['p_value']:.4f}" if current_sep and current_sep["p_value"] is not None else "n/a"
    )
    batch_id = None
    results = []
    for key, recommended_weight in candidate_weights.items():
        metric_name = _SUBSCORE_TO_WEIGHT_CONFIG[key]
        if _not_materially_different(mode, metric_name, recommended_weight):
            continue
        if batch_id is None:
            batch_id = str(uuid.uuid4())

        rationale = (
            f"Candidate weight set separates winners (mean recomputed score "
            f"{candidate_sep['mean_win_score']:.1f}) from losers ({candidate_sep['mean_loss_score']:.1f}) "
            f"with p={candidate_sep['p_value']:.4f} (n={len(trades)}), vs current weights' p={current_p_display}."
        )
        evidence = {
            "sub_score": key,
            "sample_size": len(trades),
            "candidate_p_value": candidate_sep["p_value"],
            "current_p_value": current_sep["p_value"] if current_sep else None,
        }
        models.insert_recommendation(
            mode=mode,
            metric_name=metric_name,
            current_value=live_weights[key],
            recommended_value=recommended_weight,
            rationale=rationale,
            sample_size=len(trades),
            category="weight",
            confidence=confidence,
            evidence=evidence,
            batch_id=batch_id,
        )
        results.append(
            {"metric_name": metric_name, "recommended_value": recommended_weight, "rationale": rationale, "batch_id": batch_id}
        )

    return results


def _avoid_bucket_recommendations(
    mode: str,
    dimension_type: str,
    metric_prefix: str,
    category: str,
    wins_overall: int,
    n_overall: int,
) -> list[dict]:
    """Shared "is this bucket reliably worse than the overall baseline"
    check, used for both market_regime and symbol buckets — same
    two-proportion z-test, only the dimension/metric-name prefix differ."""
    overall_win_rate = wins_overall / n_overall
    results = []
    batch_id = None
    for row in models.get_learning_statistics(mode, dimension_type=dimension_type):
        bucket = row["dimension_value"]
        n_bucket = row.get("trades_count") or 0
        win_rate = row.get("win_rate")
        if n_bucket < RECOMMENDATION_MIN_SAMPLE_SIZE or win_rate is None or win_rate >= overall_win_rate:
            continue
        wins_bucket = round(win_rate * n_bucket)
        p_value = z_test_two_proportions(wins_bucket, n_bucket, wins_overall, n_overall)
        if p_value is None or p_value >= SIGNIFICANCE_THRESHOLD:
            continue

        metric_name = f"{metric_prefix}:{bucket}"
        latest = models.get_latest_recommendation(mode, metric_name)
        if latest is not None and latest.get("recommended_value") == 0.0:
            continue  # already on record as "avoid" — append-only idempotency

        if batch_id is None:
            batch_id = str(uuid.uuid4())
        confidence = (1 - p_value) * 100
        rationale = (
            f"{dimension_type}={bucket}: win rate {win_rate * 100:.1f}% (n={n_bucket}) vs overall "
            f"{overall_win_rate * 100:.1f}% (n={n_overall}), p={p_value:.4f}."
        )
        evidence = {dimension_type: bucket, "win_rate": win_rate, "overall_win_rate": overall_win_rate, "trades_count": n_bucket}
        models.insert_recommendation(
            mode=mode,
            metric_name=metric_name,
            current_value=1.0,
            recommended_value=0.0,
            rationale=rationale,
            sample_size=n_bucket,
            category=category,
            confidence=confidence,
            evidence=evidence,
            batch_id=batch_id,
        )
        results.append({"metric_name": metric_name, "recommended_value": 0.0, "rationale": rationale})
    return results


def generate_regime_recommendations(mode: str, status: LearningStatus | None = None) -> list[dict]:
    """Step 4: "avoid regime X" plus regime-conditioned weight
    recommendations (e.g. trend weight matters more in a bull regime than
    in a sideways one), reusing generate_weight_recommendations'
    primitives scoped to each regime's own trade subset."""
    status = status or compute_learning_status(mode)
    if not status.can_generate_hypotheses():
        return []

    all_trades = _recently_closed(mode)

    wins_overall = sum(1 for t in all_trades if t["pnl"] > 0)
    n_overall = len(all_trades)

    results = _avoid_bucket_recommendations(
        mode, "market_regime", "avoid_regime", "regime", wins_overall, n_overall
    )

    regimes = {t.get("market_regime") for t in all_trades if t.get("market_regime")}
    for regime in regimes:
        regime_trades = [t for t in all_trades if t.get("market_regime") == regime]
        if len(regime_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue
        regime_weights = compute_subscore_correlation_weights(mode, trades=regime_trades, cache=False)
        if regime_weights is None:
            continue
        candidate_sep = score_separation_p_value(regime_trades, regime_weights)
        if candidate_sep is None or candidate_sep["p_value"] is None or candidate_sep["p_value"] >= SIGNIFICANCE_THRESHOLD:
            continue

        batch_id = str(uuid.uuid4())
        confidence = (1 - candidate_sep["p_value"]) * 100
        for key, recommended_weight in regime_weights.items():
            metric_name = f"{_SUBSCORE_TO_WEIGHT_CONFIG[key]}:{regime}"
            if _not_materially_different(mode, metric_name, recommended_weight):
                continue
            rationale = (
                f"Within regime {regime} (n={len(regime_trades)}), a {key} weight of "
                f"{recommended_weight:.3f} separates winners/losers with p={candidate_sep['p_value']:.4f}."
            )
            evidence = {"regime": regime, "sub_score": key, "sample_size": len(regime_trades)}
            models.insert_recommendation(
                mode=mode,
                metric_name=metric_name,
                current_value=None,
                recommended_value=recommended_weight,
                rationale=rationale,
                sample_size=len(regime_trades),
                category="regime",
                confidence=confidence,
                evidence=evidence,
                batch_id=batch_id,
            )
            results.append({"metric_name": metric_name, "recommended_value": recommended_weight, "rationale": rationale})

    return results


def _symbol_metric_extractors() -> dict[str, callable]:
    return {
        "optimal_opportunity_score": lambda trade, entry_eval, calibration: (
            entry_eval.get("opportunity_score") if entry_eval else None
        ),
        "optimal_confidence": lambda trade, entry_eval, calibration: (
            calibration.get("final_confidence") if calibration else None
        ),
        "optimal_stop_distance_pct": lambda trade, entry_eval, calibration: (
            abs(trade["stop_loss_price"] - trade["entry_price"]) / trade["entry_price"] * 100
            if trade.get("stop_loss_price") and trade.get("entry_price")
            else None
        ),
        "optimal_volatility_atr_pct": lambda trade, entry_eval, calibration: (
            ((entry_eval.get("features") or {}).get(PRIMARY_TIMEFRAME) or {}).get("atr_pct")
            if entry_eval
            else None
        ),
    }


def generate_symbol_recommendations(mode: str, status: LearningStatus | None = None) -> list[dict]:
    """Step 5: "avoid symbol X" plus a per-symbol optimal-threshold sweep
    for confidence/opportunity_score/stop_distance/volatility — the same
    _find_optimal_threshold sweep generate_recommendations() uses
    globally, generalized and scoped per symbol."""
    status = status or compute_learning_status(mode)
    if not status.can_generate_hypotheses():
        return []

    all_trades = _recently_closed(mode)

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0

    wins_overall = sum(1 for t in all_trades if t["pnl"] > 0)
    n_overall = len(all_trades)

    results = _avoid_bucket_recommendations(mode, "symbol", "avoid_symbol", "symbol", wins_overall, n_overall)

    symbols = {t["symbol"] for t in all_trades}
    extractors = _symbol_metric_extractors()

    for symbol in symbols:
        symbol_trades = [t for t in all_trades if t["symbol"] == symbol]
        if len(symbol_trades) < RECOMMENDATION_MIN_SAMPLE_SIZE:
            continue

        baseline_stats = compute_bucket_statistics(symbol_trades, capital_to_use)
        if baseline_stats["expectancy"] is None:
            continue

        for metric_suffix, extractor in extractors.items():
            scored = []
            for trade in symbol_trades:
                entry_eval = models.get_entry_evaluation_for_trade(trade["id"])
                calibration = (
                    models.get_confidence_calibration_for_evaluation(entry_eval["id"]) if entry_eval else None
                )
                value = extractor(trade, entry_eval, calibration)
                if value is not None:
                    scored.append((value, trade))
            if len(scored) < RECOMMENDATION_MIN_SAMPLE_SIZE:
                continue

            # Candidate cutoffs: every distinct observed value except the
            # minimum (a threshold at the minimum filters nothing).
            candidate_thresholds = sorted({v for v, _ in scored})[1:]
            if not candidate_thresholds:
                continue

            found = _find_optimal_threshold(scored, candidate_thresholds, baseline_stats, capital_to_use)
            if found is None:
                continue
            best_threshold, best_stats, improvement_pct = found

            metric_name = f"{metric_suffix}:{symbol}"
            if _not_materially_different(mode, metric_name, best_threshold):
                continue

            rationale = (
                f"{symbol}: trades with {metric_suffix.replace('optimal_', '')} >= {best_threshold:.2f} "
                f"(n={best_stats['trades_count']}) show expectancy {best_stats['expectancy']:.2f} vs "
                f"symbol baseline {baseline_stats['expectancy']:.2f} (n={len(symbol_trades)})."
            )
            evidence = {"symbol": symbol, "sample_size": best_stats["trades_count"], "improvement_pct": improvement_pct}
            models.insert_recommendation(
                mode=mode,
                metric_name=metric_name,
                current_value=None,
                recommended_value=best_threshold,
                rationale=rationale,
                sample_size=best_stats["trades_count"],
                category="symbol",
                confidence=None,
                evidence=evidence,
                batch_id=None,
            )
            results.append({"metric_name": metric_name, "recommended_value": best_threshold, "rationale": rationale})

    return results


def _sweep_range() -> list[float]:
    values, pct = [], EXIT_PARAM_SWEEP_MIN_PCT
    while pct <= EXIT_PARAM_SWEEP_MAX_PCT + 1e-9:
        values.append(round(pct, 6))
        pct += EXIT_PARAM_SWEEP_STEP_PCT
    return values


def _simulate_exit_pnl(trade: dict, stop_loss_pct: float | None, take_profit_pct: float | None) -> float:
    """Approximates this trade's pnl under a candidate stop_loss_pct/
    take_profit_pct using its recorded mfe_pct/mae_pct (max favorable/
    adverse excursion) rather than a full price-path replay — the same
    approximation _assess_stop_loss/_assess_target already make per-trade
    in statistics.py, generalized here into a sweep. Does not model
    excursion ORDER (whether the favorable or adverse extreme happened
    first) — a stated limitation, not a full backtest."""
    notional = trade["entry_price"] * trade["qty"]
    mae_pct, mfe_pct = trade.get("mae_pct") or 0.0, trade.get("mfe_pct") or 0.0
    if stop_loss_pct and mae_pct >= stop_loss_pct * 100:
        return -stop_loss_pct * notional
    if take_profit_pct and mfe_pct >= take_profit_pct * 100:
        return take_profit_pct * notional
    return trade["pnl"]


def generate_exit_params_recommendations(mode: str, status: LearningStatus | None = None) -> list[dict]:
    """New candidate type (Scientific Strategy Optimization Framework):
    sweeps stop_loss_pct/take_profit_pct — previously only ever
    LLM-guessed via the now-retired evolution_agent.propose_next_version —
    against expectancy via _simulate_exit_pnl. Each leg is swept
    independently, holding the other at its current configured value."""
    status = status or compute_learning_status(mode)
    if not status.can_generate_hypotheses():
        return []

    closed = _recently_closed(mode)

    capital_config = models.get_capital_config(mode)
    capital_to_use = capital_config["capital_to_use"] if capital_config else 0
    version = models.get_latest_version()
    current_params = (version.get("params_json") or {}) if version else {}
    current_stop, current_target = current_params.get("stop_loss_pct"), current_params.get("take_profit_pct")

    baseline_stats = compute_bucket_statistics(closed, capital_to_use)
    if baseline_stats["expectancy"] is None:
        return []

    candidates = _sweep_range()
    results = []
    for param_name, fixed_other in (("stop_loss_pct", current_target), ("take_profit_pct", current_stop)):
        best_value, best_expectancy = None, baseline_stats["expectancy"]
        for candidate_value in candidates:
            stop = candidate_value if param_name == "stop_loss_pct" else fixed_other
            target = candidate_value if param_name == "take_profit_pct" else fixed_other
            simulated = [{**t, "pnl": _simulate_exit_pnl(t, stop, target)} for t in closed]
            stats = compute_bucket_statistics(simulated, capital_to_use)
            if stats["expectancy"] is not None and stats["expectancy"] > best_expectancy:
                best_value, best_expectancy = candidate_value, stats["expectancy"]

        if best_value is None:
            continue
        delta = best_expectancy - baseline_stats["expectancy"]
        if delta < MIN_EXPECTANCY_DELTA:
            continue
        improvement_pct = (
            delta / abs(baseline_stats["expectancy"]) * 100
            if baseline_stats["expectancy"]
            else (float("inf") if best_expectancy > 0 else 0.0)
        )
        if improvement_pct < RECOMMENDATION_MIN_IMPROVEMENT_PCT:
            continue
        if _not_materially_different(mode, param_name, best_value):
            continue

        current_value = current_stop if param_name == "stop_loss_pct" else current_target
        rationale = (
            f"Simulated {param_name}={best_value:.3f} (mfe_pct/mae_pct approximation, n={len(closed)}) "
            f"shows expectancy {best_expectancy:.2f} vs current {baseline_stats['expectancy']:.2f} "
            f"(current {param_name}={current_value})."
        )
        models.insert_recommendation(
            mode=mode,
            metric_name=param_name,
            current_value=current_value,
            recommended_value=best_value,
            rationale=rationale,
            sample_size=len(closed),
            category="exit_params",
        )
        results.append({"metric_name": param_name, "recommended_value": best_value, "rationale": rationale})

    return results
