from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from src.learning.promotion_gate import (
    _bootstrap_probability_of_profit,
    _champion_improvement_gate,
    _complexity_delta,
    _cooldown_gate,
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
    "paired_observation_count": None,
}

_GOOD_PAIRED_COMPARISON = {
    "paired_observation_count": 250,
    "mean_difference": 12.5,
    "median_difference": 10.0,
    "bootstrap_probability_candidate_better_pct": 98.0,
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
        gates = _sample_size_gates(50, None, None, None, True)
        assert gates["paper_trades"]["passed"] is False
        assert gates["backtest_trades"]["passed"] is None
        assert gates["walk_forward_trades"]["passed"] is None
        assert gates["paired_observations"]["passed"] is None


def test_sample_size_gates_all_clear():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_PAIRED_OBSERVATIONS", 200
    ):
        gates = _sample_size_gates(500, 1200, 400, 250, True)
        assert gates["paper_trades"]["passed"] is True
        assert gates["backtest_trades"]["passed"] is True
        assert gates["walk_forward_trades"]["passed"] is True
        assert gates["paired_observations"]["passed"] is True


def test_sample_size_gates_paired_observations_not_applicable_without_champion():
    # Fix 3: no champion -> nothing to pair against -> NOT_APPLICABLE
    # (passed=True), never a perpetual None/EXTEND_VALIDATION deadlock.
    gates = _sample_size_gates(500, 1200, 400, None, False)
    assert gates["paired_observations"]["passed"] is True
    assert gates["paired_observations"]["status"] == "NOT_APPLICABLE"


# --- Issue 1: backtest sample size actually enforced ---


def test_backtest_trades_below_minimum_extends_validation():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 999, None, None, True)
        assert gates["backtest_trades"]["passed"] is False
        assert gates["backtest_trades"]["count"] == 999


def test_backtest_trades_at_minimum_passes():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 1000, None, None, True)
        assert gates["backtest_trades"]["passed"] is True


# --- TEST 2: paired-observation sample-size gate ---


def test_paired_observations_below_minimum_extends_validation():
    # TEST 2: 199 paired observations, minimum 200 -> not a pass.
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_OBSERVATIONS", 200):
        gates = _sample_size_gates(500, 1000, 400, 199, True)
        assert gates["paired_observations"]["passed"] is False
        assert gates["paired_observations"]["count"] == 199


def test_paired_observations_at_minimum_passes():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_OBSERVATIONS", 200):
        gates = _sample_size_gates(500, 1000, 400, 200, True)
        assert gates["paired_observations"]["passed"] is True


def test_bootstrap_probability_of_profit_all_positive_pnls():
    assert _bootstrap_probability_of_profit([10, 20, 30], iterations=200) == 100.0


def test_bootstrap_probability_of_profit_all_negative_pnls():
    assert _bootstrap_probability_of_profit([-10, -20, -30], iterations=200) == 0.0


def test_bootstrap_probability_of_profit_none_below_two_trades():
    assert _bootstrap_probability_of_profit([10]) is None


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


# --- Fix 2: true paired observations, matched by identifier, never by index ---


def test_paired_champion_comparison_matches_by_identifier_not_index():
    # Deliberately different-length, offset snapshot lists — a blind
    # zip(champion, candidate) would silently mispair every observation.
    champ_snaps = [_snap(i * 10, 10000 + i * 5) for i in range(10)]
    cand_snaps = [_snap(i * 10, 10000 + i * 8) for i in range(2, 14)]  # shifted +2, 12 points
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result is not None
    assert result["paired_observation_count"] == 8  # only offsets 20..90 overlap


def test_paired_champion_comparison_count_independent_of_trade_counts():
    # TEST 5: champion and challenger each have 500 backtest-replay
    # snapshots, but only 125 share a decision-cycle timestamp. The
    # matched count must be 125 — NOT any function of 500 or 500 (never
    # min(champion_trades, challenger_trades)).
    champ_snaps = [_snap(i * 10, 10000 + i) for i in range(500)]
    shared = [_snap(i * 10, 10000 + i * 2) for i in range(125)]
    disjoint = [_snap(100_000 + i * 10, 10000 + i) for i in range(375)]
    cand_snaps = shared + disjoint
    assert len(cand_snaps) == 500
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["paired_observation_count"] == 125


def test_paired_champion_comparison_none_below_three_shared_observations():
    champ_snaps = [_snap(0, 10000), _snap(10, 10010)]
    cand_snaps = [_snap(0, 10000), _snap(10, 10020)]
    assert _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps}) is None


def test_paired_champion_comparison_none_when_no_shared_timestamps():
    champ_snaps = [_snap(i * 10, 10000) for i in range(5)]
    cand_snaps = [_snap(1_000_000 + i * 10, 10000) for i in range(5)]  # entirely disjoint clock
    assert _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps}) is None


def test_paired_champion_comparison_candidate_consistently_ahead_is_significant():
    champ_snaps = [_snap(i * 10, 10000 + i * 2) for i in range(30)]
    cand_snaps = [_snap(i * 10, 10000 + i * 10) for i in range(30)]  # consistently outgrows champion
    result = _paired_champion_comparison({"snapshots": champ_snaps}, {"snapshots": cand_snaps})
    assert result["paired_observation_count"] == 30
    assert result["mean_difference"] > 0
    assert result["bootstrap_probability_candidate_better_pct"] == 100.0
    assert result["significant"] is True


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
            "comparison": {"winner": "b"},
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
            "comparison": {"winner": "b"},
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
    assert gate["paired_observation_count"] == 250


# --- FIX 1: no weak statistical fallback ---


def test_champion_improvement_gate_missing_paired_comparison_never_falls_back():
    # TEST 1: paired comparison unavailable -> UNAVAILABLE/None, even
    # though the OLD unpaired `comparison` says winner="b" (would have
    # been treated as a pass by the retired fallback). Never inferred.
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": "b"},  # deliberately "favorable" — must NOT be used
            "paired_comparison": None,
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is None
    assert gate["status"] == "UNAVAILABLE"
    assert "paired" in gate["detail"]


def test_champion_improvement_gate_low_confidence_rejects():
    # TEST 3: paired observations=200 (real, computed), statistical
    # evidence=92% < minimum 95% -> gate fails (REJECT, not a silent pass).
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": "b"},
            "paired_comparison": {
                "paired_observation_count": 200,
                "mean_difference": 1.5,
                "median_difference": 1.2,
                "bootstrap_probability_candidate_better_pct": 92.0,
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
    assert gate["bootstrap_probability_candidate_better_pct"] == 92.0
    assert gate["result"] == "not_significant"


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


# --- Issue 4 / "Execution Quality" fix: real data or None, never a neutral 50 ---


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


def _full_evidence(paired_observation_count=250, paired_comparison=_UNSET, backtest_trades_count=1200, walk_forward_trades_count=400):
    from src.backtest.overfitting_detection import OverfittingReport

    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = backtest_trades_count
    evidence["walk_forward_trades_count"] = walk_forward_trades_count
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    evidence["paired_observation_count"] = paired_observation_count
    evidence["champion_challenger"] = {
        "comparison": {"winner": "b"},
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
    ("src.learning.promotion_gate.PROMOTION_MIN_PAIRED_OBSERVATIONS", 200),
    ("src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2),
    ("src.learning.promotion_gate.PROMOTION_MIN_SCORE", 50),
)


def _apply_gate_patches(stack: ExitStack, *extra):
    for target, value in list(_GATE_PATCHES) + list(extra):
        stack.enter_context(patch(target, value))


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_valid_robust_candidate_promotes(mock_models, mock_evidence):
    # TEST 4: paired observations=200(+), statistical evidence=97% (>= 95
    # minimum), every other mandatory gate clears -> AUTO-PROMOTE.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_observation_count=250,
        paired_comparison={
            "paired_observation_count": 250, "mean_difference": 15.0, "median_difference": 12.0,
            "bootstrap_probability_candidate_better_pct": 97.0, "significant": True,
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
        "cooldown", "paper_trades", "backtest_trades", "walk_forward_trades", "paired_observations",
        "paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo",
        "regime_robustness", "symbol_robustness", "overfitting", "champion_improvement",
    }
    summary = decision.breakdown["summary"]
    assert summary["promotion_status"] == "PROMOTE"
    assert summary["backtest_trade_count"] == 1200
    assert summary["paired_observation_count"] == 250
    assert summary["bootstrap_probability_candidate_better_pct"] == 97.0
    assert summary["champion_comparison_status"] == "AVAILABLE"
    assert summary["champion_comparison_result"] == "candidate_significantly_better"
    assert summary["failed_gates"] == []


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_missing_paired_comparison_extends_validation(mock_models, mock_evidence):
    # TEST 1: paired comparison unavailable -> EXTEND_VALIDATION, no
    # promotion, regardless of everything else looking clean.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(paired_observation_count=None, paired_comparison=None)
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "EXTEND_VALIDATION"
    assert decision.gates["champion_improvement"]["passed"] is None
    assert decision.gates["paired_observations"]["passed"] is None


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_low_statistical_confidence_rejects(mock_models, mock_evidence):
    # TEST 3: paired observations=200 (real), statistical evidence=92% <
    # minimum 95% -> no promotion.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_observation_count=200,
        paired_comparison={
            "paired_observation_count": 200, "mean_difference": 1.0, "median_difference": 0.8,
            "bootstrap_probability_candidate_better_pct": 92.0, "significant": False,
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
    # TEST 6: no current champion, every applicable first-deployment gate
    # passes -> AUTO-PROMOTE, champion comparison NOT_APPLICABLE.
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
    evidence["paired_observation_count"] = None
    mock_evidence.return_value = evidence
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)

    assert decision.decision == "PROMOTE"
    assert decision.gates["champion_improvement"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_observations"]["status"] == "NOT_APPLICABLE"
    assert decision.breakdown["summary"]["champion_comparison_status"] == "NOT_APPLICABLE"


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_no_champion_paper_validation_failure_blocks_promotion(mock_models, mock_evidence):
    # TEST 7: no current champion, but paper validation itself fails
    # (catastrophic drawdown) -> no promotion.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350, win_pnl=10, loss_pnl=-900)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision != "PROMOTE"
    assert decision.gates["paper_days_pnl_drawdown"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_score_does_not_override_failed_paired_sample_gate(mock_models, mock_evidence):
    # TEST 8: every other signal (feeding a high promotion score) looks
    # great, but the mandatory paired-observation SAMPLE gate itself fails
    # (70 < 200) -> no promotion, score is irrelevant.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(
        paired_observation_count=70,  # below PROMOTION_MIN_PAIRED_OBSERVATIONS=200
        paired_comparison={
            "paired_observation_count": 70, "mean_difference": 20.0, "median_difference": 18.0,
            "bootstrap_probability_candidate_better_pct": 99.0, "significant": True,
        },
    )
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)

    with ExitStack() as stack:
        _apply_gate_patches(stack)
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision != "PROMOTE"
    assert decision.gates["paired_observations"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_score_does_not_override_missing_paired_statistical_evidence(mock_models, mock_evidence):
    # TEST 9: paired sample count clears the floor, but the statistical
    # test itself couldn't be computed -> no promotion, regardless of an
    # otherwise-high score.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = _full_evidence(paired_observation_count=250, paired_comparison=None)
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
    # Score = high (great win rate/expectancy), but a genuinely mandatory
    # gate (drawdown) fails -> REJECT regardless of the score.
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


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_first_ever_promotion_no_champion_not_blocked_by_champion_gate(mock_models, mock_evidence):
    # champion=None must vacuously pass the champion-improvement gate AND
    # the paired-observations sample gate — the decision here is still
    # EXTEND_VALIDATION (walk-forward/backtest data missing), but NOT
    # because of either champion-related gate.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.gates["champion_improvement"]["passed"] is True
    assert decision.gates["champion_improvement"]["status"] == "NOT_APPLICABLE"
    assert decision.gates["paired_observations"]["passed"] is True
    assert decision.gates["paired_observations"]["status"] == "NOT_APPLICABLE"
    assert decision.decision == "EXTEND_VALIDATION"  # backtest/walk-forward still missing
