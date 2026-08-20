#!/bin/bash
# Mirrors .github/workflows/risk_check.yml's two independent GH jobs
# (paper, real), each running risk-only then diagnostics as a chain that
# stops at its own first failure. A paper failure must not skip the real
# chain, and vice versa — same independence as the two GH jobs today.
status=0

if python -m src.orchestrator --mode=paper --risk-only; then
  python -m src.monitoring.diagnostics --mode=paper || status=1
else
  status=1
fi

if python -m src.orchestrator --mode=real --risk-only; then
  python -m src.monitoring.diagnostics --mode=real || status=1
else
  status=1
fi

exit $status
