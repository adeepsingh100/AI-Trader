#!/bin/bash
# Mirrors .github/workflows/evolution.yml's single "evolve" job — one
# sequential chain, stop on first failure (default GH Actions step
# behavior, no continue-on-error set on any of these steps).
set -e
python -m src.agents.evolution_agent
python -m src.learning.adaptive_strategy_engine
python -m src.learning.drift_detection
python -m src.learning.strategy_health
