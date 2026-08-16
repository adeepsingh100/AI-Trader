"""Production Monitoring (Step 8, PROJECT_SPEC.md §3d). Scoped to what's
real for short-lived GitHub Actions cron invocations (stateless, Supabase
as the only durable state) rather than invented long-running-server
metaphors: cycle timing/success per component, plus cheap
resource/disk snapshots via stdlib (no new dependency, no persistent
metrics backend). One generic system_metrics table, not N single-purpose
ones — matching the jsonb-bundle precedent elsewhere in this schema."""

from __future__ import annotations

import resource
import shutil
import time
from contextlib import contextmanager


@contextmanager
def track(component: str, name: str, extra: dict | None = None):
    """Wraps a block, recording its duration and success/failure to
    system_metrics on exit. Fails open on its own write error (a metrics
    write failure must never take down the block it's measuring) — same
    fail-open discipline as src/resilience.py's circuit breaker."""
    from src.db import models

    start = time.monotonic()
    success = True
    error_type = None
    try:
        yield
    except Exception as e:
        success = False
        error_type = type(e).__name__
        raise
    finally:
        duration_ms = (time.monotonic() - start) * 1000
        try:
            models.insert_system_metrics(
                [
                    {
                        "component": component,
                        "metric_name": f"{name}_duration_ms",
                        "value": duration_ms,
                        "metadata": {"success": success, "error_type": error_type, **(extra or {})},
                    }
                ]
            )
        except Exception:
            pass


def resource_snapshot() -> dict:
    """CPU/memory via stdlib `resource` (this process's own usage — the
    closest meaningful thing to "CPU/memory usage" for a script that runs
    for a few seconds and exits, not a long-running server with its own
    process to monitor), disk via stdlib `shutil.disk_usage` on the
    current working directory (the GitHub Actions runner's ephemeral
    disk)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    disk = shutil.disk_usage(".")
    return {
        "max_rss_kb": usage.ru_maxrss,
        "user_cpu_seconds": usage.ru_utime,
        "system_cpu_seconds": usage.ru_stime,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
    }


def log_resource_snapshot(component: str = "orchestrator") -> None:
    from src.db import models

    snapshot = resource_snapshot()
    try:
        models.insert_system_metrics(
            [{"component": component, "metric_name": k, "value": v, "metadata": {}} for k, v in snapshot.items()]
        )
    except Exception:
        pass
