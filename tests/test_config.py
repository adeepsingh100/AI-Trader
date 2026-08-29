"""STRATEGY_PROFILES (multi-strategy-type support) — "default" must stay
byte-identical to the bare env-var-driven globals it's built from (proves
zero behavior change for the existing single-strategy path), and every
profile must independently pass the same exit_score_threshold <
min_opportunity_score hysteresis check the top-level constants already
enforce at import time."""

import importlib

import pytest


def test_default_profile_matches_bare_globals():
    from src import config

    profile = config.STRATEGY_PROFILES["default"]
    assert profile["timeframe_weights"] == config.TIMEFRAME_WEIGHTS
    assert profile["opportunity_weights"] == {
        "trend": config.OPPORTUNITY_WEIGHT_TREND,
        "momentum": config.OPPORTUNITY_WEIGHT_MOMENTUM,
        "volume": config.OPPORTUNITY_WEIGHT_VOLUME,
        "volatility": config.OPPORTUNITY_WEIGHT_VOLATILITY,
        "risk": config.OPPORTUNITY_WEIGHT_RISK,
    }
    assert profile["min_opportunity_score"] == config.MIN_OPPORTUNITY_SCORE
    assert profile["top_n_candidates"] == config.TOP_N_CANDIDATES
    assert profile["exit_score_threshold"] == config.EXIT_SCORE_THRESHOLD
    assert profile["stop_loss_atr_multiplier"] == config.STOP_LOSS_ATR_MULTIPLIER
    assert profile["take_profit_atr_multiplier"] == config.TAKE_PROFIT_ATR_MULTIPLIER
    assert profile["exit_param_sweep_min_pct"] == config.EXIT_PARAM_SWEEP_MIN_PCT
    assert profile["exit_param_sweep_max_pct"] == config.EXIT_PARAM_SWEEP_MAX_PCT
    assert profile["risk_per_trade_pct"] == config.RISK_PER_TRADE_PCT


def test_swing_profile_is_a_distinct_longer_horizon_profile():
    from src import config

    swing = config.STRATEGY_PROFILES["swing"]
    default = config.STRATEGY_PROFILES["default"]
    # Not required to differ on every field, but the whole point of a
    # second profile is a longer-horizon bias -- assert the two knobs
    # that actually encode that (wider ATR stop/target multipliers,
    # looser exit threshold), not just "the dict is different".
    assert swing["stop_loss_atr_multiplier"] > default["stop_loss_atr_multiplier"]
    assert swing["take_profit_atr_multiplier"] > default["take_profit_atr_multiplier"]
    assert swing["exit_score_threshold"] < default["exit_score_threshold"]


def test_every_profile_enforces_exit_threshold_hysteresis(monkeypatch):
    monkeypatch.setenv("SWING_EXIT_SCORE_THRESHOLD", "999")  # >= min_opportunity_score -> invalid
    with pytest.raises(ValueError, match="exit_score_threshold must be < min_opportunity_score"):
        importlib.reload(importlib.import_module("src.config"))
    # Restore the real module state for every test after this one in the
    # same process (pytest reuses the interpreter across test files).
    monkeypatch.delenv("SWING_EXIT_SCORE_THRESHOLD", raising=False)
    importlib.reload(importlib.import_module("src.config"))
