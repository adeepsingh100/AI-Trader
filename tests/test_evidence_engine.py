from unittest.mock import patch

import pytest

from src.learning.evidence_engine import EvidenceEngine, compute_evidence_readiness


def _trade(trade_id, pnl, exit_reason="ai_exit", version_id=1):
    return {"id": trade_id, "pnl": pnl, "exit_reason": exit_reason, "version_id": version_id}


def _evaluation(symbol, final_decision, market_regime=None, timestamp="2026-01-01T10:00:00+00:00", llm_decision=None):
    return {
        "symbol": symbol,
        "final_decision": final_decision,
        "market_regime": market_regime,
        "timestamp": timestamp,
        "llm_decision": llm_decision,
    }


@patch("src.learning.evidence_engine.models")
def test_collect_counts_closed_winning_losing(mock_models):
    mock_models.get_recently_closed_trades.return_value = [
        _trade(1, 100), _trade(2, -50), _trade(3, 30),
    ]
    mock_models.get_opportunity_evaluations_for_trail.return_value = []
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    assert evidence["closed_trades"] == 3
    assert evidence["winning_trades"] == 2
    assert evidence["losing_trades"] == 1


@patch("src.learning.evidence_engine.models")
def test_collect_symbols_regimes_hours_from_all_decisions_not_just_holds(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    mock_models.get_opportunity_evaluations_for_trail.return_value = [
        _evaluation("BTCINR", "hold", market_regime="sideways", timestamp="2026-01-01T04:30:00+00:00"),
        _evaluation("ETHINR", "buy", market_regime="strong_bull", timestamp="2026-01-01T09:00:00+00:00"),
        _evaluation("ETHINR", "buy", market_regime="strong_bull", timestamp="2026-01-01T09:00:00+00:00"),
    ]
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    assert evidence["symbols_covered"] == 2
    assert evidence["market_regimes_covered"] == 2
    assert evidence["rejected_opportunities"] == 1
    assert evidence["candidate_opportunities"] == 3
    # regime seen on a buy -> not "no candidates"; sideways only ever held
    assert evidence["regimes_with_no_candidates"] == ["sideways"]


@patch("src.learning.evidence_engine.models")
def test_collect_confidence_coverage_is_fraction_reaching_llm(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    mock_models.get_opportunity_evaluations_for_trail.return_value = [
        _evaluation("BTCINR", "hold", llm_decision="reject"),
        _evaluation("BTCINR", "hold", llm_decision=None),
        _evaluation("BTCINR", "hold", llm_decision=None),
        _evaluation("BTCINR", "hold", llm_decision=None),
    ]
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    assert evidence["confidence_coverage_pct"] == 25.0


@patch("src.learning.evidence_engine.models")
def test_collect_confidence_coverage_zero_when_no_evaluations(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    mock_models.get_opportunity_evaluations_for_trail.return_value = []
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    assert evidence["confidence_coverage_pct"] == 0.0


@patch("src.learning.evidence_engine.models")
def test_collect_symbols_rarely_qualifying_excludes_accepted_and_thin_samples(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    # DOGEINR: seen 20 times, always rejected, never bought -> flagged.
    # BTCINR: seen 20 times, rejected AND bought at least once -> not flagged.
    # LTCINR: seen only 3 times (below RECOMMENDATION_MIN_SAMPLE_SIZE) -> not flagged.
    evaluations = (
        [_evaluation("DOGEINR", "hold") for _ in range(20)]
        + [_evaluation("BTCINR", "hold") for _ in range(19)]
        + [_evaluation("BTCINR", "buy")]
        + [_evaluation("LTCINR", "hold") for _ in range(3)]
    )
    mock_models.get_opportunity_evaluations_for_trail.return_value = evaluations
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    flagged = {r["symbol"] for r in evidence["symbols_rarely_qualifying"]}
    assert flagged == {"DOGEINR"}


@patch("src.learning.evidence_engine.models")
def test_collect_feature_coverage_excludes_blended_sentinel(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    mock_models.get_opportunity_evaluations_for_trail.return_value = []
    mock_models.get_feature_importance.return_value = [
        {"feature_name": "rsi", "timeframe": "1h"},
        {"feature_name": "adx", "timeframe": "1h"},
        {"feature_name": "trend_score", "timeframe": "blended"},  # excluded, sub-score correlation not a raw feature
    ]
    mock_models.get_learning_statistics.return_value = []

    evidence = EvidenceEngine().collect("paper")

    from src.features.feature_engine import FEATURE_KEYS

    # exactly 2 raw feature keys counted, not 3 (blended sentinel excluded)
    assert evidence["feature_coverage_pct"] == pytest.approx(2 / len(FEATURE_KEYS) * 100)


@patch("src.learning.evidence_engine.models")
def test_collect_learning_coverage_pct_from_dimension_types_present(mock_models):
    mock_models.get_recently_closed_trades.return_value = []
    mock_models.get_opportunity_evaluations_for_trail.return_value = []
    mock_models.get_feature_importance.return_value = []
    mock_models.get_learning_statistics.return_value = [
        {"dimension_type": "symbol", "dimension_value": "BTCINR"},
        {"dimension_type": "market_regime", "dimension_value": "sideways"},
    ]

    evidence = EvidenceEngine().collect("paper")

    assert evidence["learning_coverage_pct"] == 25.0  # 2 of 8 tracked dimension types


# --- compute_evidence_readiness ---


def _evidence(**overrides):
    base = {
        "closed_trades": 0,
        "market_regimes_covered": 0,
        "symbols_covered": 0,
        "feature_coverage_pct": 0.0,
        "trading_hours_covered": 0,
        "confidence_coverage_pct": 0.0,
        "rejected_opportunities": 0,
    }
    base.update(overrides)
    return base


def test_compute_evidence_readiness_zero_when_nothing_collected():
    result = compute_evidence_readiness(_evidence())
    assert result["evidence_readiness_pct"] == 0.0


def test_compute_evidence_readiness_full_marks_when_every_dimension_maxed():
    result = compute_evidence_readiness(
        _evidence(
            closed_trades=500,
            market_regimes_covered=6,
            symbols_covered=20,
            feature_coverage_pct=100.0,
            trading_hours_covered=24,
            confidence_coverage_pct=100.0,
            rejected_opportunities=500,
        )
    )
    assert result["evidence_readiness_pct"] == 100.0


def test_compute_evidence_readiness_weights_each_component_by_its_configured_share():
    # Only rejected_opportunities has any signal; every component function
    # always returns a concrete float (never None, unlike fitness_score's
    # inputs), so weighted_average has nothing to renormalize away here —
    # this is a plain weighted sum: 100 * EVIDENCE_WEIGHT_REJECTION_EVIDENCE.
    result = compute_evidence_readiness(_evidence(rejected_opportunities=500))
    assert result["components"]["rejection_evidence"] == 100.0
    assert result["evidence_readiness_pct"] == 10.0
