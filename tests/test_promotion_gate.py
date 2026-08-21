from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from src.learning.promotion_gate import (
    PromotionDecision,
    _bootstrap_probability_of_profit,
    _champion_improvement_gate,
    _complexity_delta,
    _cooldown_gate,
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


_EMPTY_BACKTEST_EVIDENCE = {
    "backtest_trades_count": None,
    "walk_forward_folds": None,
    "walk_forward_trades_count": None,
    "overfitting_report": None,
    "champion_challenger": None,
    "champion_challenger_trades_count": None,
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
        gates = _sample_size_gates(50, None, None, None)
        assert gates["paper_trades"]["passed"] is False
        assert gates["backtest_trades"]["passed"] is None
        assert gates["walk_forward_trades"]["passed"] is None
        assert gates["champion_challenger_trades"]["passed"] is None


def test_sample_size_gates_all_clear():
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_CHAMPION_CHALLENGER_TRADES", 200
    ):
        gates = _sample_size_gates(500, 1200, 400, 250)
        assert gates["paper_trades"]["passed"] is True
        assert gates["backtest_trades"]["passed"] is True
        assert gates["walk_forward_trades"]["passed"] is True
        assert gates["champion_challenger_trades"]["passed"] is True


# --- Issue 1: backtest sample size actually enforced ---


def test_backtest_trades_below_minimum_extends_validation():
    # TEST 1: backtest_trades=999, minimum=1000 -> not a pass (routes to
    # EXTEND_VALIDATION via evaluate_promotion, never REJECT — see the
    # end-to-end test further down).
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 999, None, None)
        assert gates["backtest_trades"]["passed"] is False
        assert gates["backtest_trades"]["count"] == 999


def test_backtest_trades_at_minimum_passes():
    # TEST 2: backtest_trades=1000, minimum=1000 -> gate passes.
    with patch("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000):
        gates = _sample_size_gates(500, 1000, None, None)
        assert gates["backtest_trades"]["passed"] is True


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


def test_champion_improvement_gate_no_champion_vacuously_passes():
    gate = _champion_improvement_gate(None, _EMPTY_BACKTEST_EVIDENCE)
    assert gate["passed"] is True


def test_champion_improvement_gate_missing_backtest_data_is_pending():
    gate = _champion_improvement_gate(_champion(), _EMPTY_BACKTEST_EVIDENCE)
    assert gate["passed"] is None


def test_champion_improvement_gate_significant_and_meets_minimums_passes():
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": "b"},
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    with patch("src.learning.promotion_gate.PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT", 5), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT", 10
    ), patch("src.learning.promotion_gate.PROMOTION_MAX_DRAWDOWN_INCREASE_PCT", 0):
        gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is True


def test_champion_improvement_gate_missing_sharpe_is_pending_not_pass():
    # TEST 3: Sharpe improvement = null -> never a silent pass, always
    # "insufficient evidence" (EXTEND_VALIDATION at the decision level).
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


def test_champion_improvement_gate_low_confidence_rejects():
    # TEST 4: Sharpe improvement clears its own minimum, but the paired
    # champion-vs-challenger confidence (82%) is below PROMOTION_MIN_
    # CONFIDENCE_PCT (95%) -> not significant -> gate fails.
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": "b"},
            "paired_comparison": {"significant": False, "statistical_confidence_pct": 82.0},
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.24, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},  # sharpe +12%
        }
    }
    with patch("src.learning.promotion_gate.PROMOTION_MIN_EXPECTANCY_IMPROVEMENT_PCT", 5), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SHARPE_IMPROVEMENT_PCT", 10
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_CONFIDENCE_PCT", 95):
        gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is False
    assert gate["statistical_confidence_pct"] == 82.0


def test_champion_improvement_gate_paired_comparison_not_significant_rejects():
    # TEST 5: candidate is profitable, but the paired test says it's NOT
    # statistically better than champion -> gate fails (never a promotion
    # on "profitable" alone).
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": "b"},
            "paired_comparison": {"significant": False, "statistical_confidence_pct": 55.0},
            "candidate_metrics": {"expectancy": 11.0, "sharpe_ratio": 1.05, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is False


def test_champion_improvement_gate_not_significant_fails():
    evidence = {
        "champion_challenger": {
            "comparison": {"winner": None},  # not statistically significant
            "candidate_metrics": {"expectancy": 20.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 5.0},
            "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
        }
    }
    gate = _champion_improvement_gate(_champion(), evidence)
    assert gate["passed"] is False


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


# --- Issue 4: execution_quality is real data or None, never a neutral 50 ---


def test_execution_quality_unavailable_returns_none_and_redistributes():
    from src.learning.promotion_gate import _execution_quality_component

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
    from src.learning.promotion_gate import _execution_quality_component

    with patch("src.learning.promotion_gate.SLIPPAGE_BPS", 5.0):
        low_slippage = [{"entry_slippage_pct": 0.01}, {"entry_slippage_pct": -0.01}]
        high_slippage = [{"entry_slippage_pct": 0.5}, {"entry_slippage_pct": -0.5}]
        low = _execution_quality_component(low_slippage)
        high = _execution_quality_component(high_slippage)
        assert low is not None and high is not None
        assert low > high


# --- evaluate_promotion: end-to-end decision scenarios (Phase 23) ---


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
    # Plenty of clean, profitable, diversified paper trades — every gate
    # that CAN run passes — but no historical data exists yet for walk-
    # forward/champion-challenger, so the decision must be EXTEND_VALIDATION,
    # never a silent promotion and never a REJECT.
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
def test_statistically_insignificant_champion_comparison_rejected(mock_models, mock_evidence):
    _no_prior_promotions(mock_models)
    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["champion_challenger_trades_count"] = 250
    evidence["champion_challenger"] = {
        "comparison": {"winner": None},  # no statistically significant difference
        "candidate_metrics": {"expectancy": 11.0, "sharpe_ratio": 1.05, "max_drawdown_pct": 5.0},
        "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
    }
    mock_evidence.return_value = evidence
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_CHAMPION_CHALLENGER_TRADES", 200
    ):
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )
    assert decision.decision == "REJECT"
    assert decision.gates["champion_improvement"]["passed"] is False


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
def test_valid_robust_candidate_promotes(mock_models, mock_evidence):
    # TEST 6: candidate is statistically better than champion (paired
    # comparison significant), every mandatory gate clears -> AUTO-PROMOTE.
    _no_prior_promotions(mock_models)
    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = 1200  # TEST 2: clears PROMOTION_MIN_BACKTEST_TRADES
    evidence["walk_forward_trades_count"] = 400
    from src.backtest.overfitting_detection import OverfittingReport

    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    evidence["champion_challenger_trades_count"] = 250
    evidence["champion_challenger"] = {
        "comparison": {"winner": "b"},
        "paired_comparison": {"significant": True, "statistical_confidence_pct": 98.0},
        "candidate_metrics": {"expectancy": 25.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 4.0, "sortino_ratio": 2.5, "profit_factor": 2.1},
        "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0, "sortino_ratio": 1.5, "profit_factor": 1.6},
    }
    mock_evidence.return_value = evidence
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10)  # clean, diversified, low drawdown

    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_CHAMPION_CHALLENGER_TRADES", 200
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SCORE", 50
    ):
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "PROMOTE"
    assert decision.promotion_score is not None
    # audit completeness: every gate this run touched is present regardless of decision
    assert set(decision.gates) >= {
        "cooldown", "paper_trades", "backtest_trades", "walk_forward_trades", "champion_challenger_trades",
        "paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo",
        "regime_robustness", "symbol_robustness", "overfitting", "champion_improvement",
    }
    # Issue 6: explicit named-field decision record, persisted into breakdown.
    summary = decision.breakdown["summary"]
    assert summary["decision"] == "PROMOTE"
    assert summary["backtest_trades"] == 1200
    assert summary["statistical_confidence_pct"] == 98.0
    assert summary["failed_gates"] == []


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_high_promotion_score_does_not_override_failed_mandatory_gate(mock_models, mock_evidence):
    # TEST 9: every soft signal (score-eligible components) looks great,
    # but ONE mandatory gate — drawdown — genuinely fails -> REJECT
    # regardless of how high the promotion score would otherwise be.
    _no_prior_promotions(mock_models)
    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = 1200
    evidence["walk_forward_trades_count"] = 400
    from src.backtest.overfitting_detection import OverfittingReport

    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    evidence["champion_challenger_trades_count"] = 250
    evidence["champion_challenger"] = {
        "comparison": {"winner": "b"},
        "paired_comparison": {"significant": True, "statistical_confidence_pct": 99.0},
        "candidate_metrics": {"expectancy": 25.0, "sharpe_ratio": 2.0, "max_drawdown_pct": 4.0},
        "champion_metrics": {"expectancy": 10.0, "sharpe_ratio": 1.0, "max_drawdown_pct": 5.0},
    }
    mock_evidence.return_value = evidence
    # Excellent win rate/expectancy (drives the score up) but a single
    # catastrophic loss blows past PROMOTION_MAX_DRAWDOWN_PCT.
    trades = _diverse_trades(350, win_pnl=60, loss_pnl=-10) + [_trade(-50000, 15)]

    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_CHAMPION_CHALLENGER_TRADES", 200
    ), patch("src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2), patch(
        "src.learning.promotion_gate.PROMOTION_MIN_SCORE", 0
    ):  # score floor disabled so only the hard gate is under test
        decision = evaluate_promotion(
            "paper", _candidate(), trades, 10000, champion=_champion(), symbol_to_pair={"BTCINR": "I-BTCINR"}
        )

    assert decision.decision == "REJECT"
    assert decision.gates["paper_days_pnl_drawdown"]["passed"] is False


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_first_ever_promotion_no_champion_not_blocked_by_champion_gate(mock_models, mock_evidence):
    # champion=None must vacuously pass the champion-improvement gate —
    # the decision here is still EXTEND_VALIDATION (walk-forward/
    # overfitting data missing), but NOT because of the champion gate.
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(350)
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.gates["champion_improvement"]["passed"] is True
    assert decision.gates["champion_improvement"]["status"] == "NOT_APPLICABLE"  # TEST 8
    assert decision.decision == "EXTEND_VALIDATION"  # other mandatory gates (backtest/walk-forward) still active


@patch("src.learning.promotion_gate._backtest_evidence")
@patch("src.learning.promotion_gate.models")
def test_reject_wins_over_extend_when_both_present(mock_models, mock_evidence):
    # A genuine hard failure (drawdown) must produce REJECT even when
    # sample-size floors are ALSO unmet elsewhere — real evidence of
    # failure outranks "not enough evidence yet".
    _no_prior_promotions(mock_models)
    mock_evidence.return_value = dict(_EMPTY_BACKTEST_EVIDENCE)
    trades = _diverse_trades(20, win_pnl=10, loss_pnl=-900)  # few trades AND catastrophic drawdown
    with patch("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300):
        decision = evaluate_promotion("paper", _candidate(), trades, 10000, champion=None, symbol_to_pair=None)
    assert decision.decision == "REJECT"
