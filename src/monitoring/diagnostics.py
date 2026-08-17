"""Self Diagnostics (Step 10, PROJECT_SPEC.md §3d). Startup + periodic
health checks across every engine, run as a new step in risk_check.yml
(every 5 min, already the finest-grained cron) — no new workflow file, per
a Plan-agent pre-mortem's pushback on adding a 4th cron that itself could
silently stop running unnoticed."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.config import SYSTEM_METRICS_LEARNING_STALE_HOURS, SYSTEM_METRICS_MARKET_FEED_STALE_MINUTES
from src.resilience import log_fail_open


def _check_database() -> dict:
    from src.db import models

    try:
        models.ping()
        return {"healthy": True}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def _check_market_feed(mode: str = "paper") -> dict:
    """Freshness proxy: does this mode have a recent opportunity_evaluations
    row (written every cycle the market feed successfully ran)? A stale
    feed means the trading cycle stopped scanning, not necessarily that
    CoinDCX itself is down."""
    from src.db import models

    since = datetime.now(timezone.utc) - timedelta(minutes=SYSTEM_METRICS_MARKET_FEED_STALE_MINUTES)
    try:
        recent = models.get_entry_evaluations_since(mode, since)
        return {"healthy": True} if recent else {"healthy": False, "detail": "no evaluations in freshness window"}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def _check_learning_engine(mode: str = "paper") -> dict:
    from src.db import models

    try:
        stats = models.get_learning_statistics(mode)
        if not stats:
            return {"healthy": None, "detail": "no learning_statistics rows yet"}
        stale_cutoff = datetime.now(timezone.utc) - timedelta(hours=SYSTEM_METRICS_LEARNING_STALE_HOURS)
        newest = max(s["updated_at"] for s in stats if s.get("updated_at"))
        return {"healthy": newest >= stale_cutoff.isoformat()}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def _check_execution_engine(mode: str = "paper") -> dict:
    from src.db import models

    try:
        capital_config = models.get_capital_config(mode)
        if capital_config is None:
            return {"healthy": None, "detail": "not configured yet"}
        return {"healthy": True}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def _check_portfolio_engine(mode: str = "paper") -> dict:
    """Positions reconcile: every open trade has a positive qty and
    entry_price — a cheap sanity check that catches obviously corrupted
    rows without recomputing full portfolio analytics here."""
    from src.db import models

    try:
        open_trades = models.get_open_trades(mode)
        bad = [t for t in open_trades if not t.get("qty") or t["qty"] <= 0 or not t.get("entry_price")]
        return {"healthy": not bad, "detail": f"{len(bad)} malformed open trades" if bad else None}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def _check_recommendation_engine(mode: str = "paper") -> dict:
    """Reachability check only (does the query succeed), not a freshness
    check — recommendations only generate once RECOMMENDATION_MIN_SAMPLE_SIZE
    is cleared, so an empty result on a young dataset is expected, not
    unhealthy."""
    from src.db import models

    try:
        models.get_recommendations(mode)
        return {"healthy": True}
    except Exception as e:
        return {"healthy": False, "detail": str(e)}


def run_health_check(mode: str = "paper") -> dict:
    from src.db import models

    checks = {
        "database": _check_database(),
        "market_feed": _check_market_feed(mode),
        "learning_engine": _check_learning_engine(mode),
        "execution_engine": _check_execution_engine(mode),
        "portfolio_engine": _check_portfolio_engine(mode),
        "recommendation_engine": _check_recommendation_engine(mode),
    }
    overall_healthy = all(c.get("healthy") is not False for c in checks.values())

    try:
        models.insert_system_metrics(
            [
                {
                    "component": "diagnostics",
                    "metric_name": f"{name}_healthy",
                    "value": 1.0 if check.get("healthy") else 0.0,
                    "metadata": {"detail": check.get("detail")},
                }
                for name, check in checks.items()
            ]
        )
    except Exception as e:
        log_fail_open("diagnostics.system_metrics", e)

    return {"overall_healthy": overall_healthy, "checks": checks}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper", "real"])
    args = parser.parse_args()
    print(json.dumps(run_health_check(mode=args.mode), default=str, indent=2))
