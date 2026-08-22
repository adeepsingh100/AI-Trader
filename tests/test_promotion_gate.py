from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.learning.promotion_gate import (
    _backtest_evidence,
    _champion_improvement_gate,
    _complexity_delta,
    _cooldown_gate,
    _dedup_by_snapshot_time,
    _execution_quality_component,
    _paired_champion_comparison,
    _pct_improvement,
    _promotion_score,
    _regime_robustness_gate,
    _sample_size_gates,
    _symbol_robustness_gate,
    evaluate_promotion,
)


def _candidate(days_ago=30, params_json=None, version_id=10):
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": version_id,
        "created_at": created.isoformat(),
        "params_json": params_json if params_json is not None else {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
    }


def _champion(version_id=1, params_json=None):
    return {"id": version_id, "params_json": params_json or {"stop_loss_pct": 0.03, "take_profit_pct": 0.03}}


def _trade(pnl, day, symbol="BTCINR", regime="strong_bull"):
    return {"pnl": pnl, "closed_at": f"2026-01-{day:02d}T00:00:00Z", "symbol": symbol, "market_regime": regime}


def _diverse_trades(n, win_pnl=50, loss_pnl=-20, symbols=("BTCINR", "ETHINR", "SOLINR"), regimes=("strong_bull", "sideways", "weak_bear")):
    trades = []
    for i in range(n):
        day = (i % 28) + 1
        symbol = symbols[i % len(symbols)]
        regime = regimes[i % len(regimes)]
        pnl = win_pnl if i % 3 != 0 else loss_pnl  # ~67% win rate
        trades.append(_trade(pnl, day, symbol, regime))
    return trades


def _snap(offset_minutes, equity):
    return {"snapshot_time": datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=offset_minutes), "equity": equity}


_EMPTY_BACKTEST_EVIDENCE = {
    "backtest_trades_count": None,
    "walk_forward_folds": None,
    "walk_forward_trades_count": None,
    "overfitting_report": None,
    "champion_challenger": None,
    "paired_snapshot_count": None,
    "paired_return_observations": None,
}

_GOOD_PAIRED_COMPARISON = {
    "paired_snapshot_count": 250,
    "paired_return_observations": 249,
    "mean_difference": 12.5,
    "median_difference": 10.0,
    "std_difference": 3.0,
    "p25_difference": 8.0,
    "p75_difference": 15.0,
    "p95_difference": 20.0,
    "bootstrap_probability_candidate_better_pct": 98.0,
    "bootstrap_method": "moving_block",
    "bootstrap_block_length": 5,
    "bootstrap_iterations": 1000,
    "confidence_threshold_pct": 95.0,
    "statistical_gate_status": "PASS",
    "statistical_gate_reason": "bootstrap_probability_candidate_better_pct=98.0 >= threshold 95.0",
    "seed": 42,
    "significant": True,
}


def _no_prior_promotions(mock_models):
    mock_models.get_latest_promotion_audit.return_value = None
    mock_models.get_closed_trades.return_value = []  # champion's own trades, for regime robustness


# --- pure helpers ---


def test_cooldown_gate_passes_with_no_prior_promotion():
    with patch("src.learning.promotion_gate.models") as mock_models:
        mock_models.get_latest_promotion_audit.return_value = None
        assert _cooldown_gate("paper")["passed"] is True


def test_cooldown_gate_blocks_within_window():
    with patch("src.learning.promotion_gate.models") as mock_models, patch(
        "src.learning.promotion_gate.PROMOTION_COOLDOWN_DAYS", 7
    ):
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        mock_models.get_latest_promotion_audit.return_value = {"created_at": recent.isoformat()}
        gate = _cooldown_gate("paper")
        assert gate["passed"] is False
        assert "cooldown active" in gate["detail"]


def test_cooldown_gate_clears_after_window():
    with patch("src.learning.promotion_gate.models") as mock_models, patch(
        "src.learning.promotion_gate.PROMOTION_COOLDOWN_DAYS", 7
    ):
        old = datetime.now(timezone.utc) - timedelta(days=10)
        mock_models.get_latest_promotion_audit.return_value = {"created_at": old.isoformat()}
        assert _cooldown_gate("paper")["passed"] is True


def test_sample_size_gates_below_floor_is_false_not_none():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        gates = _sample_size_gates(50, None, None, None, None, True)
        assert gates["paper_trades"]["passed"] is False
        assert gates["backtest_trades"]["passed"] is None
        assert gates["walk_forward_trades"]["passed"] is None
        assert gates["paired_snapshots"]["passed"] is None
        assert gates["paired_return_observations"]["passed"] is None


def test_sample_size_gates_all_clear():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PAIRED_SNAPSHOTS", 200
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS", 200):
        gates = _sample_size_gates(500, 1200, 400, 250, 249, True)
        assert gates["paper_trades"]["passed"] is True
        assert gates["backtest_trades"]["passed"] is True
        assert gates["walk_forward_trades"]["passed"] is True
        assert gates["paired_snapshots"]["passed"] is True
        assert gates["paired_return_observations"]["passed"] is True


def test_sample_size_gates_paired_gates_not_applicable_without_champion():
    # Fix 9: no champion -> nothing to pair against -> NOT_APPLICABLE
    # (passed=True) for BOTH paired gates, never a perpetual None deadlock.
    gates = _sample_size_gates(500, 1200, 400, None, None, False)
    assert gates["paired_snapshots"]["passed"] is True
    assert gates["paired_snapshots"]["status"] == "NOT_APPLICABLE"
    assert gates["paired_return_observations"]["passed"] is True
    assert gates["paired_return_observations"]["status"] == "NOT_APPLICABLE"


# --- backtest sample size actually enforced ---


def test_backtest_trades_below_minimum_extends_validation():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 999, None, None, None, True)
        assert gates["backtest_trades"]["passed"] is False
        assert gates["backtest_trades"]["count"] == 999


def test_backtest_trades_at_minimum_passes():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 1000, None, None, None, True)
        assert gates["backtest_trades"]["passed"] is True


# --- Fix 2: paired SNAPSHOT count vs paired RETURN observation count ---


def test_paired_return_observations_below_minimum_extends_validation():
    # TEST 4: 199 return observations, minimum 200 -> not a pass, even
    # though the snapshot count (200) alone would clear its own floor.
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_SNAPSHOTS", 200), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS", 200
    ):
        gates = _sample_size_gates(500, 1000, 400, 200, 199, True)
        assert gates["paired_snapshots"]["passed"] is True
        assert gates["paired_return_observations"]["passed"] is False
        assert gates["paired_return_observations"]["count"] == 199


def test_paired_return_observations_at_minimum_passes():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS", 200):
        gates = _sample_size_gates(500, 1000, 400, 201, 200, True)
        assert gates["paired_return_observations"]["passed"] is True


def test_paired_snapshots_below_minimum_extends_validation():
    # TEST 7: only 120 matched snapshots, minimum 200 -> not a pass,
    # regardless of how many trades either side has independently.
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_SNAPSHOTS", 200):
        gates = _sample_size_gates(500, 1000, 400, 120, 119, True)
        assert gates["paired_snapshots"]["passed"] is False
        assert gates["paired_snapshots"]["count"] == 120


def test_symbol_robustness_single_symbol_fails_concentration():
    with patch("src.learning.promotion_gate.PROMOTION_MAX_SYMBOL_PROFIT_CONCENTRATION_PCT", 60), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2
    ):
        trades = [_trade(50, d, symbol="DOGEINR") for d in range(1, 10)]
        gate = _symbol_robustness_gate(trades)
        assert gate["passed"] is False
        assert gate["profitable_symbols_count"] == 1


def test_symbol_robustness_diversified_passes():
    with patch("src.learning.promotion_gate.PROMOTION_MAX_SYMBOL_PROFIT_CONCENTRATION_PCT", 60), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2
    ):
        trades = [_trade(50, (i % 28) + 1, symbol=s) for i, s in enumerate(["BTCINR", "ETHINR", "SOLINR"] * 5)]
        gate = _symbol_robustness_gate(trades)
        assert gate["passed"] is True
        assert gate["profitable_symbols_count"] == 3


def test_regime_robustness_no_champion_data_vacuously_passes():
    with patch("src.learning.promotion_gate.models") as mock_models, patch(
        "src.learning.promotion_gate.RECOMMENDATION_MIN_SAMPLE_SIZE", 3
    ):
        mock_models.get_closed_trades.return_value = []
        trades = [_trade(-50, d, regime="strong_bear") for d in range(1, 10)]
        gate = _regime_robustness_gate("paper", trades, champion_id=None, capital_to_use=10000)
        assert gate["passed"] is True


def test_regime_robustness_degraded_vs_champion_fails():
    with patch("src.learning.promotion_gate.models") as mock_models, patch(
        "src.learning.promotion_gate.RECOMMENDATION_MIN_SAMPLE_SIZE", 3
    ), patch("src.learning.promotion_gate.PROMOTION_MAX_REGIME_DEGRADATION_PCT", 30):
        # champion made good money in strong_bear; candidate loses money there
        mock_models.get_closed_trades.return_value = [_trade(100, d, regime="strong_bear") for d in range(1, 10)]
        candidate_trades = [_trade(-50, d, regime="strong_bear") for d in range(1, 10)]
        gate = _regime_robustness_gate("paper", candidate_trades, champion_id=1, capital_to_use=10000)
        assert gate["passed"] is False
        assert "strong_bear" in gate["degraded_regimes"]


def test_complexity_delta_counts_changed_keys():
    candidate = {"params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.05}}
    champion = {"params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04}}
    assert _complexity_delta(candidate, champion) == 1  # only take_profit_pct differs


def test_complexity_delta_no_champion_counts_all_keys():
    candidate = {"params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04}}
    assert _complexity_delta(candidate, None) == 2


def test_pct_improvement_basic():
    assert _pct_improvement(100, 110) == pytest.approx(10.0)
    assert _pct_improvement(None, 110) is None
    assert _pct_improvement(0, 110) is None


# --- Fix 3: matched by verified timestamp, never by array index ---


def test_dedup_by_snapshot_time_drops_ambiguous_duplicates():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshots = [
        {"snapshot_time": t0, "equity": 100.0},
        {"snapshot_time": t0, "equity": 105.0},  # duplicate identifier -> ambiguous, dropped
        {"snapshot_time": t0 + timedelta(minutes=10), "equity": 110.0},
    ]
    result = _dedup_by_snapshot_time(snapshots)
    assert t0 not in result
    assert result[t0 + timedelta(minutes=10)] == 110.0


def test_paired_champion_comparison_matches_by_identifier_not_index():
    # Deliberately different-length, offset snapshot lists — a blind
    # zip(champion, candidate) would silently mispair every observation.
    champ_snaps = [_snap(i * 10, 10000 + i * 5) for i in range(10)]
    cand_snaps = [_snap(i * 10, 10000 + i * 8) for i in range(2, 14)]  # shifted +2, 12 points
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result is not None
    assert result["paired_snapshot_count"] == 8  # only offsets 20..90 overlap
    assert result["paired_return_observations"] == 7  # TEST 3: one fewer than snapshots


def test_paired_champion_comparison_drops_duplicate_snapshot_timestamps():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    champ_snaps = [
        {"snapshot_time": t0, "equity": 10000},
        {"snapshot_time": t0, "equity": 10050},  # ambiguous duplicate on champion's own side
        {"snapshot_time": t0 + timedelta(minutes=10), "equity": 10100},
        {"snapshot_time": t0 + timedelta(minutes=20), "equity": 10150},
        {"snapshot_time": t0 + timedelta(minutes=30), "equity": 10200},
    ]
    cand_snaps = [
        {"snapshot_time": t0, "equity": 10000},
        {"snapshot_time": t0 + timedelta(minutes=10), "equity": 10120},
        {"snapshot_time": t0 + timedelta(minutes=20), "equity": 10200},
        {"snapshot_time": t0 + timedelta(minutes=30), "equity": 10300},
    ]
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["paired_snapshot_count"] == 3  # t0 excluded on both sides, not silently kept


def test_paired_champion_comparison_count_independent_of_trade_counts():
    # TEST 7: champion and challenger each have 500 backtest-replay
    # snapshots, but only 120 share a decision-cycle timestamp. The
    # matched count must be 120 — NOT any function of 500 or 500 (never
    # min(champion_trades, challenger_trades)).
    champ_snaps = [_snap(i * 10, 10000 + i) for i in range(500)]
    shared = [_snap(i * 10, 10000 + i * 2) for i in range(120)]
    disjoint = [_snap(100_000 + i * 10, 10000 + i) for i in range(380)]
    cand_snaps = shared + disjoint
    assert len(cand_snaps) == 500
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["paired_snapshot_count"] == 120


def test_paired_champion_comparison_none_below_three_shared_observations():
    champ_snaps = [_snap(0, 10000), _snap(10, 10010)]
    cand_snaps = [_snap(0, 10000), _snap(10, 10020)]
    assert _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps}) is None


def test_paired_champion_comparison_none_when_no_shared_timestamps():
    champ_snaps = [_snap(i * 10, 10000) for i in range(5)]
    cand_snaps = [_snap(1_000_000 + i * 10, 10000) for i in range(5)]  # entirely disjoint clock
    assert _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps}) is None


# --- Fix 1/4: time-series-aware (block bootstrap) explicit statistical test ---


def test_paired_champion_comparison_candidate_consistently_ahead_is_significant():
    champ_snaps = [_snap(i * 10, 10000 + i * 2) for i in range(30)]
    cand_snaps = [_snap(i * 10, 10000 + i * 10) for i in range(30)]  # consistently outgrows champion
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["paired_snapshot_count"] == 30
    assert result["paired_return_observations"] == 29
    assert result["mean_difference"] > 0
    assert result["bootstrap_probability_candidate_better_pct"] == 100.0
    assert result["significant"] is True
    assert result["statistical_gate_status"] == "PASS"


def test_paired_champion_comparison_reports_mean_median_std_percentiles():
    champ_snaps = [_snap(i * 10, 10000 + i * 3) for i in range(40)]
    cand_snaps = [_snap(i * 10, 10000 + i * 7) for i in range(40)]
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["mean_difference"] is not None
    assert result["median_difference"] is not None
    assert result["std_difference"] is not None
    assert result["p25_difference"] <= result["median_difference"] <= result["p75_difference"] <= result["p95_difference"]


def test_paired_champion_comparison_records_bootstrap_metadata():
    # TEST 12: bootstrap block size changes -> result metadata records it.
    champ_snaps = [_snap(i * 10, 10000 + i * 2) for i in range(30)]
    cand_snaps = [_snap(i * 10, 10000 + i * 10) for i in range(30)]
    with patch("src.learning.promotion_gate.PROMOTION_BOOTSTRAP_BLOCK_LENGTH", 4):
        result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["bootstrap_method"] == "moving_block"
    assert result["bootstrap_block_length"] == 4
    assert result["bootstrap_iterations"] > 0
    assert result["seed"] is not None


def test_paired_champion_comparison_deterministic_given_same_seed():
    # Fix 10: same data + same seed -> identical result.
    champ_snaps = [_snap(i * 10, 10000 + (i % 5) * 3 - i) for i in range(40)]
    cand_snaps = [_snap(i * 10, 10000 + (i % 7) * 4 - i) for i in range(40)]
    with patch("src.learning.promotion_gate.BACKTEST_RANDOM_SEED", 42):
        r1 = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
        r2 = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert r1 == r2


# --- champion improvement gate ---


def test_champion_improvement_gate_no_champion_vacuously_passes():
    gate = _champion_improvement_gate(None, _EMPTY_BACKTEST_EVIDENCE)
    assert gate["passed"] is True
    assert gate["status"] == "NOT_APPLICABLE"


def test_champion_improvement_gate_missing_backtest_data_is_pending():
    gate = _champion_improvement_gate(_champion(), _EMPTY_BACKTEST_EVIDENCE)
    assert gate["passed"] is None
    assert gate["status"] == "UNAVAILABLE"


def test_champion_improvement_gate_missing_sharpe_is_pending_not_pass():
    # A missing Sharpe improvement is never a silent pass.
    evidence = {
        "champion_challenger": {
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": None, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": None, "max_drawdown_pct": 5.0},
        }
    }
    gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is None
    assert gate["detail"] == "Insufficient Sharpe evidence"


def test_champion_improvement_gate_significant_and_meets_minimums_passes():
    evidence = {
        "champion_challenger": {
            "paired_comparison": _GOOD_PAIRED_COMPARISON,
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    with patch("src.learning.promotion_gate.PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT", 5), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT", 10
    ), patch("src.learning.promotion_gate.PROMOTION_MAX_DRAWDOWN_INCREASE_PCT", 0):
        gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is True
    assert gate["status"] == "AVAILABLE"
    assert gate["result"] == "candidate_significantly_better"
    assert gate["bootstrap_probability_candidate_better_pct"] == 98.0
    assert gate["bootstrap_method"] == "moving_block"
    assert gate["bootstrap_block_length"] == 5
    assert gate["paired_snapshot_count"] == 250
    assert gate["paired_return_observations"] == 249


# --- FIX 1/5/7: no weak statistical fallback, missing evidence never promotes ---


def test_champion_improvement_gate_missing_paired_comparison_never_falls_back():
    # TEST 8: paired comparison unavailable -> UNAVAILABLE/None. No other
    # statistical result exists in this gate's evidence to fall back to.
    evidence = {
        "champion_challenger": {
            "paired_comparison": None,
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is None
    assert gate["status"] == "UNAVAILABLE"
    assert "paired" in gate["detail"]


def test_champion_improvement_gate_low_bootstrap_probability_rejects():
    # TEST 5: 200 valid paired return observations, bootstrap probability
    # 94% < minimum 95% -> gate fails (no promotion).
    evidence = {
        "champion_challenger": {
            "paired_comparison": {
                "paired_snapshot_count": 201,
                "paired_return_observations": 200,
                "mean_difference": 1.5,
                "median_difference": 1.2,
                "std_difference": 0.5,
                "p25_difference": 1.0,
                "p75_difference": 2.0,
                "p95_difference": 3.0,
                "bootstrap_probability_candidate_better_pct": 94.0,
                "bootstrap_method": "moving_block",
                "bootstrap_block_length": 5,
                "bootstrap_iterations": 1000,
                "confidence_threshold_pct": 95.0,
                "statistical_gate_status": "FAIL",
                "statistical_gate_reason": "bootstrap_probability_candidate_better_pct=94.0 < threshold 95.0",
                "seed": 42,
                "significant": False,
            },
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.24, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},  # sharpe +12%
        }
    }
    with patch("src.learning.promotion_gate.PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT", 5), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT", 10
    ):
        gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is False
    assert gate["bootstrap_probability_candidate_better_pct"] == 94.0
    assert gate["statistical_gate_status"] == "FAIL"
    assert gate["result"] == "not_significant"


# --- FIX 5: old unpaired comparison is structurally unreachable ---


def test_promotion_gate_module_does_not_import_old_unpaired_comparison():
    import src.learning.promotion_gate as pg

    assert not hasattr(pg, "compare_strategies")
    assert not hasattr(pg, "compare")
    assert "strategy_comparison" not in pg.__dict__


@patch("src.learning.promotion_gate._has_historical_candles", return_value=True)
@patch("src.learning.promotion_gate.run_walk_forward", return_value=[])
@patch("src.backtest.performance_analyzer.analyze")
@patch("src.backtest.engine.BacktestEngine")
@patch("src.backtest.strategy_comparison.compare")
def test_backtest_evidence_never_calls_old_unpaired_comparison(
    mock_compare, mock_engine_cls, mock_analyze, mock_walk_forward, mock_has_candles
):
    # TEST 9: even with a real champion present (the only branch that used
    # to call the old unpaired comparison), the promotion pipeline never
    # invokes it.
    mock_engine = mock_engine_cls.return_value
    mock_engine.run.return_value = {"closed_trades": [], "snapshots": [_snap(i * 10, 10000 + i) for i in range(10)]}
    mock_analyze.return_value = {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0}

    candidate = _candidate()
    champion = _champion()
    trades = [_trade(10, d) for d in range(1, 10)]
    _backtest_evidence(candidate, champion, trades, symbol_to_pair={"BTCINR": "I-BTCINR"})

    mock_compare.assert_not_called()


def test_promotion_score_renormalizes_among_available_components():
    gates = {
        "paper_trades": {"passed": True},
        "walk_forward_trades": {"passed": None},
        "paper_days_pnl_drawdown": {"passed": True, "detail": {"max_drawdown_pct": 5.0}},
        "monte_carlo": {"passed": None},
        "bootstrap_ci": {"detail": None},
        "regime_robustness": {"passed": True, "degraded_regimes": [], "detail": {}},
    }
    score, components = _promotion_score(gates, {"passed": True}, None, 0, [])
    assert score is not None
    assert 0 <= score <= 100


# --- execution_quality: real data or None, never a neutral 50 (TEST 10 spirit) ---


def test_execution_quality_unavailable_returns_none_and_redistributes():
    trades_no_slippage_data = [{"pnl": 10}, {"pnl": -5}]
    assert _execution_quality_component(trades_no_slippage_data) is None

    gates = {
        "paper_trades": {"passed": True},
        "walk_forward_trades": {"passed": None},
        "paper_days_pnl_drawdown": {"passed": True, "detail": {"max_drawdown_pct": 5.0}},
        "monte_carlo": {"passed": None},
        "bootstrap_ci": {"detail": None},
        "regime_robustness": {"passed": True, "degraded_regimes": [], "detail": {}},
    }
    score, components = _promotion_score(gates, {"passed": True}, None, 0, trades_no_slippage_data)
    assert components["execution_quality"] is None
    assert score is not None  # still computed, renormalized among what's available


def test_execution_quality_uses_real_slippage_when_available():
    with patch("src.learning.promotion_gate.SLIPPAGE_BPS", 5.0):
        low_slippage = [{"entry_slippage_pct": 0.01}, {"entry_slippage_pct": -0.01}]
        high_slippage = [{"entry_slippage_pct": 0.5}, {"entry_slippage_pct": -0.5}]
        low = _execution_quality_component(low_slippage)
        high = _execution_quality_component(high_slippage)
        assert low is not None and high is not None
        assert low > high


# --- evaluate_promotion: end-to-end decision scenarios ---


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_weak_candidate_rejected_on_drawdown(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350, win_pnl=10, loss_pnl=-900)  # catastrophic drawdown
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "REJECT"
    assert decision.gates["paper_days_pnl_drawdown"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_insufficient_paper_trades_extends_validation(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(20)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "EXTEND_VALIDATION"
    assert decision.gates["paper_trades"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_missing_historical_data_extends_validation_not_reject(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "EXTEND_VALIDATION"
    assert decision.gates["overfitting"]["passed"] is None
    assert any("insufficient evidence" in r for r in decision.reasons)


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_single_symbol_candidate_rejected(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = [_trade(50, (i % 28) + 1, symbol="DOGEINR") for i in range(350)]
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2
    ):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "REJECT"
    assert decision.gates["symbol_robustness"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_overfit_candidate_rejected_via_walk_forward_verdict(mock_models, mock_evidence):
    from src.backtest.overfitting_detection import OverfittingReport

    _no_prior_promotions(mock_models)
    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["walk_forward_trades_count"] = 400
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=1, walk_forward_failure_rate=80.0,
        in_sample_out_of_sample_gap_pct=90.0, parameter_sensitivity=None, verdict="overfit",
    )
    mock_evidence.return_value = evidence
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300
    ):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "REJECT"
    assert decision.gates["overfitting"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_cooldown_blocks_regardless_of_otherwise_clean_evidence(mock_models, mock_evidence):
    recent = datetime.now(timezone.utc) - timedelta(days=1)
    mock_models.get_latest_promotion_audit.return_value = {"created_at": recent.isoformat()}
    mock_models.get_closed_trades.return_value = []
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_COOLDOWN_DAYS", 7), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300
    ):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "EXTEND_VALIDATION"
    assert decision.gates["cooldown"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_reject_wins_over_extend_when_both_present(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(20, win_pnl=10, loss_pnl=-900)  # few trades AND catastrophic drawdown
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "REJECT"


_UNSET = object()  # sentinel distinct from an explicitly-passed None (== "genuinely unavailable")


def _full_evidence(
    paired_snapshot_count=250,
    paired_return_observations=249,
    paired_comparison=_UNSET,
    backtest_trades_count=1200,
    walk_forward_trades_count=400,
):
    from src.backtest.overfitting_detection import OverfittingReport

    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = backtest_trades_count
    evidence["walk_forward_trades_count"] = walk_forward_trades_count
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    evidence["paired_snapshot_count"] = paired_snapshot_count
    evidence["paired_return_observations"] = paired_return_observations
    evidence["champion_challenger"] = {
        "paired_comparison": dict(_GOOD_PAIRED_COMPARISON) if paired_comparison is _UNSET else paired_comparison,
        "candidate_metrics": {
            "expectancy": 25.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 4.0, "sortino_ratio": 2.5, "profit_factor": 2.1,
        },
        "champion_metrics": {
            "expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0, "sortino_ratio": 1.5, "profit_factor": 1.6,
        },
    }
    return evidence


_GATE_PATCHES = (
    ("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300),
    ("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000),
    ("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300),
    ("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_SNAPSHOTS", 200),
    ("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_RETURN_OBSERVATIONS", 200),
    ("src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2),
    ("src.learning.promotion_gate.PROMOTION_MIN_SCORE", 50),
)


def _apply_gate_patches(stack: ExitStack, *extra):
    for target, value in list(_GATE_PATCHES) + list(extra):
        stack.enter_context(patch(target, value))


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_valid_robust_candidate_promotes(mock_models, mock_evidence):
    # TEST 6: 200 valid paired return observations, bootstrap probability
    # 97% (>= 95 minimum), every other mandatory gate clears -> AUTO-PROMOTE.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_snapshot_count=250,
        paired_return_observations=249,
        paired_comparison={
            **_GOOD_PAIRED_COMPARISON,
            "paired_return_observations": 200,
            "bootstrap_probability_candidate_better_pct": 97.0,
        },
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)  # clean, diversified, low drawdown

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "PROMOTE"
    assert decision.promotion_score is not None
    assert set(decision.gates) >= {
        "cooldown", "paper_trades", "backtest_trades", "walk_forward_trades",
        "paired_snapshots", "paired_return_observations",
        "paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo",
        "regime_robustness", "symbol_robustness", "overfitting", "champion_improvement",
    }
    summary = decision.breakdown["summary"]
    assert summary["final_status"] == "PROMOTE"
    assert summary["backtest_trade_count"] == 1200
    assert summary["paired_snapshot_count"] == 250
    assert summary["paired_return_observations"] == 249
    assert summary["bootstrap_probability_candidate_better_pct"] == 97.0
    assert summary["bootstrap_method"] == "moving_block"
    assert summary["statistical_status"] == "PASS"
    assert summary["champion_comparison_status"] == "AVAILABLE"
    assert summary["champion_comparison_result"] == "candidate_significantly_better"
    assert summary["failed_gates"] == []


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_missing_paired_comparison_extends_validation(mock_models, mock_evidence):
    # TEST 8: paired comparison unavailable -> EXTEND_VALIDATION, no
    # promotion, regardless of everything else looking clean. No fallback.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_snapshot_count=None, paired_return_observations=None, paired_comparison=None,
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "EXTEND_VALIDATION"
    assert decision.gates["champion_improvement"]["passed"] is None
    assert decision.gates["paired_snapshots"]["passed"] is None
    assert decision.gates["paired_return_observations"]["passed"] is None


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_low_bootstrap_probability_blocks_promotion(mock_models, mock_evidence):
    # TEST 5 (end-to-end): 200 valid paired return observations, bootstrap
    # probability 94% < minimum 95% -> no promotion.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_snapshot_count=201,
        paired_return_observations=200,
        paired_comparison={
            **_GOOD_PAIRED_COMPARISON,
            "paired_snapshot_count": 201,
            "paired_return_observations": 200,
            "bootstrap_probability_candidate_better_pct": 94.0,
            "statistical_gate_status": "FAIL",
            "significant": False,
        },
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision != "PROMOTE"
    assert decision.gates["champion_improvement"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_no_champion_first_deployment_promotes_with_not_applicable_comparison(mock_models, mock_evidence):
    # TEST 10: no current champion, every applicable first-deployment gate
    # passes -> AUTO-PROMOTE, champion comparison NOT_APPLICABLE, all
    # other required gates were still enforced (backtest/walk-forward/
    # paper/risk/regime/symbol/overfitting all had to clear independently).
    _no_prior_promotions(mock_models)
    from src.backtest.overfitting_detection import OverfittingReport

    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = 1200
    evidence["walk_forward_trades_count"] = 400
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    # champion=None -> real _backtest_evidence would never populate these:
    evidence["champion_challenger"] = None
    mock_evidence.return_value = evidence
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)

    assert decision.decision == "PROMOTE"
    assert decision.gates["champion_improvement"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_snapshots"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_return_observations"]["status"] == "NOT_APPLICABLE"
    assert decision.breakdown["summary"]["champion_comparison_status"] == "NOT_APPLICABLE"


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_no_champion_paper_validation_failure_blocks_promotion(mock_models, mock_evidence):
    # No current champion, but paper validation itself fails (catastrophic
    # drawdown) -> no promotion. Champion absence never bypasses this.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350, win_pnl=10, loss_pnl=-900)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision != "PROMOTE"
    assert decision.gates["paper_days_pnl_drawdown"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_score_does_not_override_failed_paired_snapshot_gate(mock_models, mock_evidence):
    # Every other signal (feeding a high promotion score) looks great, but
    # the mandatory paired-SNAPSHOT sample gate itself fails (70 < 200) ->
    # no promotion, score is irrelevant.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_snapshot_count=70,
        paired_return_observations=69,
        paired_comparison={
            **_GOOD_PAIRED_COMPARISON,
            "paired_snapshot_count": 70, "paired_return_observations": 69,
            "bootstrap_probability_candidate_better_pct": 99.0,
        },
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision != "PROMOTE"
    assert decision.gates["paired_snapshots"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_score_does_not_override_missing_paired_statistical_evidence(mock_models, mock_evidence):
    # TEST 9 (score angle) / Fix 12: paired sample counts clear the floor,
    # but the statistical test itself couldn't be computed -> no
    # promotion, regardless of an otherwise-high score.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_snapshot_count=250, paired_return_observations=249, paired_comparison=None,
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision != "PROMOTE"
    assert decision.gates["champion_improvement"]["passed"] is None


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_promotion_score_does_not_override_failed_mandatory_risk_gate(mock_models, mock_evidence):
    # TEST 11: score = high (great win rate/expectancy, statistical gate
    # would also pass), but a genuinely mandatory gate (drawdown) fails ->
    # REJECT regardless of the score.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence()
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10) + [_trade(-50000, 15)]

    with ExitStack() as stack:
        _apply_gate_patches(stack, ("src.learning.promotion_gate.PROMOTION_MIN_SCORE", 0))
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "REJECT"
    assert decision.gates["paper_days_pnl_drawdown"]["passed"] is False
    assert decision.gates["champion_improvement"]["passed"] is True  # stat gate itself was fine


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_first_ever_promotion_no_champion_not_blocked_by_champion_gate(mock_models, mock_evidence):
    # champion=None must vacuously pass the champion-improvement gate AND
    # both paired sample gates — the decision here is still
    # EXTEND_VALIDATION (walk-forward/backtest data missing), but NOT
    # because of any champion-related gate.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.gates["champion_improvement"]["passed"] is True
    assert decision.gates["champion_improvement"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_snapshots"]["passed"] is True
    assert decision.gates["paired_snapshots"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_return_observations"]["passed"] is True
    assert decision.gates["paired_return_observations"]["status"] == "NOT_APPLICABLE"
    assert decision.decision == "EXTEND_VALIDATION"  # backtest/walk-forward still missing
