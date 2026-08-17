from unittest.mock import patch

from src.learning.learning_status import compute_learning_status

_FULL_COVERAGE_EVIDENCE = {
    "closed_trades": 0,
    "winning_trades": 0,
    "losing_trades": 0,
    "exit_reasons_seen": 0,
    "strategy_versions_seen": 0,
    "rejected_opportunities": 0,
    "candidate_opportunities": 0,
    "symbols_covered": 0,
    "market_regimes_covered": 0,
    "trading_hours_covered": 0,
    "feature_coverage_pct": 0.0,
    "confidence_coverage_pct": 0.0,
    "learning_coverage_pct": 0.0,
    "symbols_rarely_qualifying": [],
    "regimes_with_no_candidates": [],
}


def _evidence(**overrides) -> dict:
    return {**_FULL_COVERAGE_EVIDENCE, **overrides}


def _status(mock_models, mock_engine, mock_readiness, trades_collected, evidence_overrides=None, readiness_pct=0.0, version=None):
    evidence = _evidence(closed_trades=trades_collected, **(evidence_overrides or {}))
    mock_engine.return_value.collect.return_value = evidence
    mock_readiness.return_value = {"evidence_readiness_pct": readiness_pct, "components": {"trade_coverage": min(100.0, trades_collected / 500 * 100)}}
    mock_models.get_recommendations.return_value = []
    mock_models.get_strategy_simulations.return_value = []
    mock_models.get_adaptive_strategy_versions.return_value = []
    mock_models.get_latest_version.return_value = version
    return compute_learning_status("paper")


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_stage_bootstrap_when_no_dimension_clears_a_bar(mock_models, mock_engine, mock_readiness):
    status = _status(mock_models, mock_engine, mock_readiness, trades_collected=10)
    assert status.stage == "BOOTSTRAP"
    assert status.next_stage == "OBSERVATION"
    assert status.trades_to_next_stage == 15
    assert any("closed trades" in gap for gap in status.evidence_gaps)


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_observation_unlocks_via_trade_count_path(mock_models, mock_engine, mock_readiness):
    status = _status(mock_models, mock_engine, mock_readiness, trades_collected=25)
    assert status.stage == "OBSERVATION"
    assert status.next_stage == "HYPOTHESIS"
    assert status.trades_to_next_stage == 75


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_observation_unlocks_via_rejected_opportunities_alone(mock_models, mock_engine, mock_readiness):
    """Evidence-Driven Learning Progression: only 3 closed trades, but 500+
    rejected candidates is on its own enough to reach OBSERVATION — the
    whole point of replacing a single trade-count axis."""
    status = _status(mock_models, mock_engine, mock_readiness, trades_collected=3, evidence_overrides={"rejected_opportunities": 600})
    assert status.stage == "OBSERVATION"


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_observation_unlocks_via_regime_coverage_alone(mock_models, mock_engine, mock_readiness):
    status = _status(mock_models, mock_engine, mock_readiness, trades_collected=2, evidence_overrides={"market_regimes_covered": 6})
    assert status.stage == "OBSERVATION"


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_observation_unlocks_via_evidence_readiness_pct_alone(mock_models, mock_engine, mock_readiness):
    status = _status(mock_models, mock_engine, mock_readiness, trades_collected=1, readiness_pct=55.0)
    assert status.stage == "OBSERVATION"


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_hypothesis_simulation_validation_stay_trade_count_gated(mock_models, mock_engine, mock_readiness):
    """Deeper stages are NOT evidence-substitutable — strong coverage with
    too few trades stays short of HYPOTHESIS (statistically irreducible)."""
    status = _status(
        mock_models, mock_engine, mock_readiness, trades_collected=40,
        evidence_overrides={"rejected_opportunities": 5000, "symbols_covered": 50, "market_regimes_covered": 6},
        readiness_pct=90.0,
    )
    assert status.stage == "OBSERVATION"  # unlocked via evidence
    assert status.can_generate_hypotheses() is False  # but hypotheses still need real trade count

    status_100 = _status(mock_models, mock_engine, mock_readiness, trades_collected=100)
    assert status_100.stage == "HYPOTHESIS"
    assert status_100.can_generate_hypotheses() is True
    assert status_100.can_simulate() is False

    status_250 = _status(mock_models, mock_engine, mock_readiness, trades_collected=250)
    assert status_250.stage == "SIMULATION"
    assert status_250.can_simulate() is True
    assert status_250.can_create_candidate() is False
    assert status_250.can_validate() is False

    status_500 = _status(mock_models, mock_engine, mock_readiness, trades_collected=500)
    assert status_500.stage == "VALIDATION"
    assert status_500.can_validate() is True
    assert status_500.can_create_candidate() is True
    assert status_500.next_stage is None


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_can_promote_reads_promotion_eligible_without_recomputing(mock_models, mock_engine, mock_readiness):
    status_true = _status(mock_models, mock_engine, mock_readiness, trades_collected=500, version={"promotion_eligible": True})
    assert status_true.can_promote() is True

    status_false = _status(mock_models, mock_engine, mock_readiness, trades_collected=500, version=None)
    assert status_false.can_promote() is False


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_fields_reflect_evidence_wins_losses_rejected(mock_models, mock_engine, mock_readiness):
    status = _status(
        mock_models, mock_engine, mock_readiness, trades_collected=30,
        evidence_overrides={"winning_trades": 12, "losing_trades": 18, "rejected_opportunities": 88},
    )
    assert status.trades_collected == 30
    assert status.winning_trades == 12
    assert status.losing_trades == 18
    assert status.rejected_trades == 88
    assert status.evidence["winning_trades"] == 12


@patch("src.learning.learning_status.compute_evidence_readiness")
@patch("src.learning.learning_status.EvidenceEngine")
@patch("src.learning.learning_status.models")
def test_current_activity_and_reason_are_nonempty_strings(mock_models, mock_engine, mock_readiness):
    for n in (0, 25, 100, 250, 500):
        status = _status(mock_models, mock_engine, mock_readiness, trades_collected=n)
        assert status.current_activity
        assert status.reason
