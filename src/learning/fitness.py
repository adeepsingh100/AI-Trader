"""Multi-Objective Fitness Score (Step 8, Scientific Strategy Optimization
Framework). A configurable weighted blend of profit factor / Sharpe /
expectancy / win rate / drawdown penalty — never a single metric, so a
candidate can't win on trade count or one flattering number alone.

The 4 metric->0-100 component functions below were previously private
copies inside strategy_health.py; moved here as the one authoritative
implementation (strategy_health.py imports them back) since both modules
need the identical "raw metric -> normalized 0-100 score" mapping."""

from __future__ import annotations

from src.config import (
    FITNESS_EXPECTANCY_SCALE,
    FITNESS_WEIGHT_DRAWDOWN_PENALTY,
    FITNESS_WEIGHT_EXPECTANCY,
    FITNESS_WEIGHT_PROFIT_FACTOR,
    FITNESS_WEIGHT_SHARPE,
    FITNESS_WEIGHT_WIN_RATE,
    PROMOTION_MAX_DRAWDOWN_PCT,
)
from src.features.opportunity_scorer import weighted_average
from src.utils import clamp


def sharpe_component(sharpe_ratio: float | None) -> float | None:
    if sharpe_ratio is None:
        return None
    # Sharpe of 2.0+ is excellent by conventional standards, 0 is neutral,
    # negative is bad — maps [-1, 2] -> [0, 100].
    return clamp((sharpe_ratio + 1) / 3 * 100, 0, 100)


def drawdown_component(max_drawdown_pct: float | None) -> float | None:
    """Framed as a penalty term: 0% drawdown -> 100 (no penalty),
    PROMOTION_MAX_DRAWDOWN_PCT (the existing "acceptable" bar) -> 0
    (full penalty) — reused, not a second drawdown anchor."""
    if max_drawdown_pct is None:
        return None
    return clamp(100 - (max_drawdown_pct / PROMOTION_MAX_DRAWDOWN_PCT) * 100, 0, 100)


def win_rate_component(win_rate: float | None) -> float | None:
    return clamp(win_rate * 100, 0, 100) if win_rate is not None else None


def profit_factor_component(profit_factor: float | None) -> float | None:
    if profit_factor is None:
        return None
    # profit_factor of 1.0 (breakeven) -> 50, 3.0+ -> 100, 0 -> 0.
    return clamp(profit_factor / 3 * 100, 0, 100)


def expectancy_component(expectancy: float | None, capital_to_use: float) -> float | None:
    """expectancy as a % of capital_to_use, linear-mapped around a neutral
    50 (breakeven) using FITNESS_EXPECTANCY_SCALE as the sensitivity
    anchor — expectancy is in raw currency, unlike the other 0-100-native
    inputs, so it needs this scale step before it can share a scale with
    them."""
    if expectancy is None or not capital_to_use:
        return None
    expectancy_pct = expectancy / capital_to_use * 100
    return clamp(50 + expectancy_pct * FITNESS_EXPECTANCY_SCALE, 0, 100)


def compute_fitness_score(stats: dict, capital_to_use: float, weights: dict | None = None) -> dict:
    """stats: a compute_bucket_statistics()-shaped dict. Returns
    {"fitness_score": float | None, "components": {...}} — renormalized
    among whatever components are available (weighted_average's existing
    convention), so a bucket missing e.g. Sharpe (too few trades) doesn't
    skew the score toward the components it does have."""
    weights = weights or {
        "profit_factor": FITNESS_WEIGHT_PROFIT_FACTOR,
        "sharpe": FITNESS_WEIGHT_SHARPE,
        "expectancy": FITNESS_WEIGHT_EXPECTANCY,
        "win_rate": FITNESS_WEIGHT_WIN_RATE,
        "drawdown_penalty": FITNESS_WEIGHT_DRAWDOWN_PENALTY,
    }
    components = {
        "profit_factor": profit_factor_component(stats.get("profit_factor")),
        "sharpe": sharpe_component(stats.get("sharpe_ratio")),
        "expectancy": expectancy_component(stats.get("expectancy"), capital_to_use),
        "win_rate": win_rate_component(stats.get("win_rate")),
        "drawdown_penalty": drawdown_component(stats.get("max_drawdown_pct")),
    }
    return {"fitness_score": weighted_average(components, weights), "components": components}
