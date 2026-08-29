"""Phase 1 integration tests: the FIRST-EVER production promotion,
end to end — no existing champion. Unlike tests/test_promotion_gate.py's
unit tests (which cover evaluate_promotion() in isolation), these drive
the REAL evolution_agent.run_evolution() -> promotion_gate.evaluate_
promotion() -> models.promote_version()/insert_promotion_audit() chain
with evaluate_promotion() NOT mocked, so the actual gate cascade runs.
Only genuine I/O boundaries are mocked: both modules' `models` (Supabase),
promotion_gate.build_symbol_to_pair (network), promotion_gate.
_backtest_evidence (BacktestEngine/walk-forward, which need historical
candle data this suite never has), and learning_status.compute_
learning_status (an unrelated hourly side computation). Same "no real
network/DB in this suite" discipline as every other test file — see
CLAUDE.md."""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from src.agents.evolution_agent import run_evolution
from src.backtest.overfitting_detection import OverfittingReport
from src.learning.learning_status import LearningStatus


def _status():
    return LearningStatus(
        stage="HYPOTHESIS", trades_collected=350, rejected_trades=0, winning_trades=0,
        losing_trades=0, evidence={}, evidence_readiness_pct=0.0, data_sufficiency_pct=0.0,
        recommendations_count=0, simulations_count=0, candidates_count=0, promotion_eligible=False,
        next_stage=None, trades_to_next_stage=0, evidence_gaps=[], current_activity="", reason="",
    )


def _version(days_ago=20):
    created = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": 7,
        "version_number": 4,
        "created_at": created.isoformat(),
        "promoted_to_real": False,
        "promotion_eligible": False,
        "params_json": {"stop_loss_pct": 0.02, "take_profit_pct": 0.04},
    }


def _trade(pnl, day, symbol="BTCINR", regime="strong_bull"):
    return {"pnl": pnl, "closed_at": f"2026-01-{day:02d}T00:00:00Z", "symbol": symbol, "market_regime": regime}


def _diverse_trades(n, win_pnl=60, loss_pnl=-10):
    symbols = ("BTCINR", "ETHINR", "SOLINR")
    regimes = ("strong_bull", "sideways", "weak_bear")
    trades = []
    for i in range(n):
        pnl = win_pnl if i % 3 != 0 else loss_pnl  # ~67% win rate
        trades.append(_trade(pnl, (i % 28) + 1, symbols[i % len(symbols)], regimes[i % len(regimes)]))
    return trades


_EMPTY_BACKTEST_EVIDENCE = {
    "backtest_trades_count": None,
    "walk_forward_folds": None,
    "walk_forward_trades_count": None,
    "overfitting_report": None,
    "champion_challenger": None,
    "paired_snapshot_count": None,
    "paired_return_observations": None,
}


def _full_first_promotion_evidence():
    """What the real _backtest_evidence() would return for a first-ever
    promotion (champion=None) once historical data + walk-forward both
    clear their floors: champion_challenger stays None by construction
    (no champion to replay against), everything else present and robust."""
    evidence = dict(_EMPTY_BACKTEST_EVIDENCE)
    evidence["backtest_trades_count"] = 1200
    evidence["walk_forward_trades_count"] = 400
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=5, walk_forward_failure_rate=0.0,
        in_sample_out_of_sample_gap_pct=5.0, parameter_sensitivity=None, verdict="robust",
    )
    return evidence


_GATE_PATCHES = (
    ("src.learning.promotion_gate.PROMOTION_MIN_PAPER_TRADES", 300),
    ("src.learning.promotion_gate.PROMOTION_MIN_BACKTEST_TRADES", 1000),
    ("src.learning.promotion_gate.PROMOTION_MIN_WALK_FORWARD_TRADES", 300),
    ("src.learning.promotion_gate.PROMOTION_MIN_PROFITABLE_SYMBOLS", 2),
    ("src.learning.promotion_gate.PROMOTION_MIN_SCORE", 50),
)


def _run(mock_evolution_models, mock_gate_models, trades, evidence, version=None, extra_patches=()):
    """Runs the REAL run_evolution() -> evaluate_promotion() chain. Both
    modules' `models` name bindings are separately mocked (each does its
    own `from src.db import models`); build_symbol_to_pair and
    _backtest_evidence are the only other boundaries stubbed."""
    mock_evolution_models.get_capital_config.return_value = {"capital_to_use": 10000}
    mock_evolution_models.get_active_strategy_types.return_value = ["default"]
    mock_evolution_models.get_latest_version.return_value = version or _version()
    mock_evolution_models.get_closed_trades.return_value = trades
    mock_evolution_models.get_latest_promoted_version.return_value = None  # no champion

    mock_gate_models.get_latest_promotion_audit.return_value = None  # cooldown: no prior promotion
    mock_gate_models.get_closed_trades.return_value = []  # champion-side regime lookup (unreached, no champion)

    patches = [
        patch("src.learning.promotion_gate.build_symbol_to_pair", return_value=None),
        patch("src.learning.promotion_gate._backtest_evidence", return_value=evidence),
        patch("src.learning.learning_status.compute_learning_status", return_value=_status()),
    ]
    for target, value in list(_GATE_PATCHES) + list(extra_patches):
        patches.append(patch(target, value))

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        # run_evolution now returns {strategy_type: {...}} — unwrap to the
        # single "default" type's result so every existing assertion in
        # this file (result["promoted"], etc.) keeps working unchanged.
        return run_evolution(mode="paper")["default"]


# --- Phase 1: positive first-ever-promotion end-to-end ---


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_ever_promotion_auto_promotes_end_to_end(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(350)
    result = _run(mock_evolution_models, mock_gate_models, trades, _full_first_promotion_evidence())

    # 1. Strategy becomes Production
    assert result["promoted"] is True
    assert result["promotion_decision"] == "PROMOTE"
    assert result["promotion_eligible"] is True

    # 2. Version is persisted (promote_version actually called, on the
    # right id — "Strategy becomes Production" is this call, not a guess)
    mock_evolution_models.promote_version.assert_called_once_with(7)
    mock_evolution_models.set_strategy_version_promotion_eligible.assert_called_once_with(7, True)

    # 3. Previous Champion remains null/none + 4. Audit log is created
    mock_evolution_models.insert_promotion_audit.assert_called_once()
    audit = mock_evolution_models.insert_promotion_audit.call_args.kwargs
    assert audit["event_type"] == "promotion"
    assert audit["decision"] == "PROMOTE"
    assert audit["candidate_version_id"] == 7
    assert audit["previous_champion_id"] is None
    assert audit["new_champion_id"] == 7

    # 5. No EXTEND_VALIDATION remains because of missing Champion — the
    # champion-related gates are NOT_APPLICABLE (vacuously satisfied),
    # never the reason anything stayed pending.
    gates = audit["gates"]
    assert gates["champion_improvement"]["status"] == "NOT_APPLICABLE"
    assert gates["champion_improvement"]["passed"] is True
    assert gates["paired_snapshots"]["status"] == "NOT_APPLICABLE"
    assert gates["paired_return_observations"]["status"] == "NOT_APPLICABLE"

    # 6. No unrelated gate is bypassed — every gate that could run did,
    # and none of them silently failed/were skipped.
    assert audit["breakdown"]["summary"]["failed_gates"] == []
    for name in (
        "cooldown", "paper_trades", "backtest_trades", "walk_forward_trades",
        "paper_days_pnl_drawdown", "bootstrap_ci", "fitness_floor", "monte_carlo",
        "regime_robustness", "symbol_robustness", "overfitting",
    ):
        assert gates[name]["passed"] is True, f"{name} did not genuinely pass: {gates[name]}"


# --- Phase 1: first-promotion negative scenarios, through the full wiring ---


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_insufficient_backtest_trades_blocks_promotion(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(350)
    evidence = _full_first_promotion_evidence()
    evidence["backtest_trades_count"] = 500  # below the 1000 floor patched in
    result = _run(mock_evolution_models, mock_gate_models, trades, evidence)

    assert result["promoted"] is False
    assert result["promotion_decision"] == "EXTEND_VALIDATION"
    mock_evolution_models.promote_version.assert_not_called()


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_insufficient_paper_trades_blocks_promotion(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(50)  # below the 300 floor
    result = _run(mock_evolution_models, mock_gate_models, trades, _full_first_promotion_evidence())

    assert result["promoted"] is False
    assert result["promotion_decision"] == "EXTEND_VALIDATION"
    mock_evolution_models.promote_version.assert_not_called()


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_monte_carlo_failure_blocks_promotion(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(350)
    result = _run(
        mock_evolution_models, mock_gate_models, trades, _full_first_promotion_evidence(),
        # Impossible bar -> the Monte Carlo profitability check fails.
        extra_patches=(("src.learning.promotion_gate.PROMOTION_MC_MIN_PROFITABLE_PCT", 100.1),),
    )

    assert result["promoted"] is False
    assert result["promotion_decision"] == "REJECT"
    mock_evolution_models.promote_version.assert_not_called()


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_risk_failure_blocks_promotion(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(350, win_pnl=10, loss_pnl=-900)  # catastrophic drawdown
    result = _run(mock_evolution_models, mock_gate_models, trades, _full_first_promotion_evidence())

    assert result["promoted"] is False
    assert result["promotion_decision"] == "REJECT"
    mock_evolution_models.promote_version.assert_not_called()


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_overfit_failure_blocks_promotion(mock_evolution_models, mock_gate_models):
    trades = _diverse_trades(350)
    evidence = _full_first_promotion_evidence()
    evidence["overfitting_report"] = OverfittingReport(
        n_folds=5, n_passed=1, walk_forward_failure_rate=80.0,
        in_sample_out_of_sample_gap_pct=90.0, parameter_sensitivity=None, verdict="overfit",
    )
    result = _run(mock_evolution_models, mock_gate_models, trades, evidence)

    assert result["promoted"] is False
    assert result["promotion_decision"] == "REJECT"
    mock_evolution_models.promote_version.assert_not_called()


@patch("src.learning.promotion_gate.models")
@patch("src.agents.evolution_agent.models")
def test_first_promotion_missing_required_evidence_blocks_promotion(mock_evolution_models, mock_gate_models):
    # No historical candles ingested yet -> _backtest_evidence's real
    # implementation would return the all-None shape. Missing evidence
    # must extend validation, never promote on what IS present alone.
    trades = _diverse_trades(350)
    result = _run(mock_evolution_models, mock_gate_models, trades, dict(_EMPTY_BACKTEST_EVIDENCE))

    assert result["promoted"] is False
    assert result["promotion_decision"] == "EXTEND_VALIDATION"
    mock_evolution_models.promote_version.assert_not_called()
