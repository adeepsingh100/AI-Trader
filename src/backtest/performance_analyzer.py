"""Step 6/7: portfolio-level performance metrics + equity/drawdown curve
analysis. Reuses src/learning/statistics.py's compute_bucket_statistics
(Sharpe/Sortino/Calmar/win-rate/profit-factor/expectancy — itself already
wrapping evolution_agent.compute_metrics for trade-level parity with the
live system's own reporting) directly — zero reimplementation. New (no
existing equivalent): gross profit/loss, Omega ratio, Ulcer index, rolling
Sharpe/volatility/drawdown, monthly/annual returns, exposure time, capital
utilization. Recovery factor is computed here, NOT stored anywhere — it's
numerically identical to Calmar, matching this codebase's own explicit
precedent against storing one fact under two names."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import stdev

from src.backtest.portfolio_manager import ClosedTrade
from src.learning.statistics import compute_bucket_statistics


def _trade_to_dict(t: ClosedTrade) -> dict:
    return {"pnl": t.pnl, "opened_at": t.entry_time.isoformat(), "closed_at": t.exit_time.isoformat()}


def gross_profit_loss(trades: list[ClosedTrade]) -> dict:
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    return {
        "gross_profit": sum(wins),
        "gross_loss": sum(losses),
        "avg_trade": sum(t.pnl for t in trades) / len(trades) if trades else None,
        "largest_win": max(wins) if wins else None,
        "largest_loss": min(losses) if losses else None,
    }


def omega_ratio(returns: list[float], threshold: float = 0.0) -> float | None:
    gains = sum(r - threshold for r in returns if r > threshold)
    losses = sum(threshold - r for r in returns if r < threshold)
    return gains / losses if losses else None


def ulcer_index(equity_curve: list[float]) -> float | None:
    if not equity_curve:
        return None
    peak = equity_curve[0]
    sq_drawdowns = []
    for e in equity_curve:
        peak = max(peak, e)
        dd_pct = (peak - e) / peak * 100 if peak else 0.0
        sq_drawdowns.append(dd_pct**2)
    return math.sqrt(sum(sq_drawdowns) / len(sq_drawdowns))


def exposure_time_pct(snapshots: list[dict]) -> float | None:
    if not snapshots:
        return None
    exposed = sum(1 for s in snapshots if s["open_positions_count"] > 0)
    return exposed / len(snapshots) * 100


def capital_utilization_pct(snapshots: list[dict]) -> float | None:
    if not snapshots:
        return None
    return sum(s["exposure_pct"] for s in snapshots) / len(snapshots)


def monthly_returns(snapshots: list[dict]) -> dict[str, float]:
    by_month: dict[str, list[dict]] = defaultdict(list)
    for s in snapshots:
        by_month[s["snapshot_time"].strftime("%Y-%m")].append(s)
    result = {}
    for month, snaps in sorted(by_month.items()):
        snaps = sorted(snaps, key=lambda s: s["snapshot_time"])
        first, last = snaps[0]["equity"], snaps[-1]["equity"]
        result[month] = (last - first) / first * 100 if first else None
    return result


def annual_returns(monthly: dict[str, float]) -> dict[str, float]:
    by_year: dict[str, list[float]] = defaultdict(list)
    for month, ret in monthly.items():
        if ret is not None:
            by_year[month[:4]].append(ret)
    # additive approximation over monthly pct returns — documented, not a
    # compounded geometric return, consistent with this codebase's
    # existing simple-pct conventions elsewhere (e.g. cumulative_pnl_pct).
    return {year: sum(rets) for year, rets in by_year.items()}


def rolling_sharpe(returns: list[float], window: int) -> list[float | None]:
    result = []
    for i in range(len(returns)):
        if i + 1 < window:
            result.append(None)
            continue
        window_returns = returns[i + 1 - window : i + 1]
        sd = stdev(window_returns) if len(window_returns) >= 2 else 0
        result.append((sum(window_returns) / len(window_returns)) / sd if sd else None)
    return result


def rolling_volatility(returns: list[float], window: int) -> list[float | None]:
    result = []
    for i in range(len(returns)):
        if i + 1 < window:
            result.append(None)
            continue
        window_returns = returns[i + 1 - window : i + 1]
        result.append(stdev(window_returns) if len(window_returns) >= 2 else None)
    return result


def rolling_drawdown(equity_curve: list[float]) -> list[float]:
    peak = float("-inf")
    result = []
    for e in equity_curve:
        peak = max(peak, e)
        result.append((peak - e) / peak * 100 if peak > 0 else 0.0)
    return result


def analyze(trades: list[ClosedTrade], snapshots: list[dict], starting_capital: float) -> dict:
    """The full Step 6/7 metrics bundle — what
    backtest_performance_metrics.metrics stores (jsonb, matching
    strategy_simulations' existing bundle pattern)."""
    trade_dicts = [_trade_to_dict(t) for t in trades]
    base = compute_bucket_statistics(trade_dicts, starting_capital)
    gpl = gross_profit_loss(trades)

    equity_curve = [s["equity"] for s in snapshots]
    returns = [t.pnl / starting_capital for t in trades] if starting_capital else []

    recovery_factor = None
    if base["max_drawdown_pct"]:
        recovery_factor = (sum(t.pnl for t in trades) / starting_capital * 100) / base["max_drawdown_pct"]

    monthly = monthly_returns(snapshots)

    return {
        **base,
        **gpl,
        "recovery_factor": recovery_factor,
        "omega_ratio": omega_ratio(returns),
        "ulcer_index": ulcer_index(equity_curve),
        "exposure_time_pct": exposure_time_pct(snapshots),
        "capital_utilization_pct": capital_utilization_pct(snapshots),
        "monthly_returns": monthly,
        "annual_returns": annual_returns(monthly),
        "final_equity": equity_curve[-1] if equity_curve else starting_capital,
        "total_return_pct": ((equity_curve[-1] - starting_capital) / starting_capital * 100) if equity_curve else 0.0,
    }
