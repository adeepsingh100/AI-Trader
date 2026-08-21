#!/bin/bash
# Mirrors .github/workflows/trading_cycle.yml's two independent GH jobs
# (paper, real) — both always attempt, overall exit is non-zero if
# either failed, so one mode's failure never silently skips the other.
status=0
python -m src.orchestrator --mode=paper || status=1
python -m src.orchestrator --mode=real || status=1
exit $status
