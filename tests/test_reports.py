from unittest.mock import patch

from src.learning.reports import generate_adaptive_strategy_report_html


@patch("src.learning.reports.compute_learning_status")
@patch("src.learning.reports.rejection_breakdown")
@patch("src.learning.reports.identify_weaknesses")
@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_smoke(mock_models, mock_weaknesses, mock_rejections, mock_learning_status):
    mock_learning_status.return_value = {
        "stage": "HYPOTHESIS",
        "trades_collected": 120,
        "rejected_trades": 400,
        "winning_trades": 60,
        "losing_trades": 60,
        "data_sufficiency_pct": 24.0,
        "recommendations_count": 3,
        "simulations_count": 2,
        "candidates_count": 1,
        "promotion_eligible": False,
        "next_stage": "SIMULATION",
        "trades_to_next_stage": 130,
        "current_activity": "Generating hypotheses.",
        "reason": "130 more closed trade(s) needed to reach SIMULATION (requires 250).",
    }
    mock_models.get_recommendations.return_value = [
        {
            "category": "weight",
            "metric_name": "OPPORTUNITY_WEIGHT_TREND",
            "current_value": 0.3,
            "recommended_value": 0.4,
            "confidence": 87.5,
            "sample_size": 30,
            "rationale": "<script>alert(1)</script>",
            "status": "pending",
        },
        {
            "category": "threshold",
            "metric_name": "MIN_OPPORTUNITY_SCORE",
            "current_value": 60,
            "recommended_value": 70,
            "confidence": None,
            "sample_size": 25,
            "rationale": "clear improvement",
            "status": "approved",
        },
        {
            "category": "symbol",
            "metric_name": "avoid_symbol:DOGEINR",
            "current_value": 1.0,
            "recommended_value": 0.0,
            "confidence": 90.0,
            "sample_size": 20,
            "rationale": "underperforms",
            "status": "dismissed",
        },
    ]
    mock_models.get_strategy_simulations.return_value = [
        {"id": 1, "created_at": "2026-01-01T00:00:00Z", "passed": True, "p_value": 0.01, "research_note": "Decision: Promoted"},
        {"id": 2, "created_at": "2026-01-02T00:00:00Z", "passed": False, "p_value": 0.5, "research_note": "Decision: Rejected"},
    ]
    mock_models.get_adaptive_strategy_versions.return_value = [
        {"version_number": 1, "status": "candidate", "created_at": "2026-01-01T00:00:00Z", "source_simulation_id": 1, "fitness_score": 72.3},
    ]
    mock_weaknesses.return_value = {
        "worst_by_dimension": {"market_regime": {"value": "high_volatility", "expectancy": -25.0, "trades_count": 22}}
    }
    mock_rejections.return_value = [{"reason": "block_concentration_limit", "count": 12, "pct_of_rejections": 60.0}]

    out = generate_adaptive_strategy_report_html("paper")

    assert "OPPORTUNITY_WEIGHT_TREND" in out
    assert "avoid_symbol:DOGEINR" in out
    assert "<script>alert(1)</script>" not in out  # XSS-escaped
    assert "&lt;script&gt;" in out
    assert "PASSED" in out
    assert "0.0100" in out  # p-value formatted
    assert "72.3" in out  # fitness score
    assert "high_volatility" in out  # weakness
    assert "block_concentration_limit" in out  # rejection reason
    assert "HYPOTHESIS" in out  # learning status stage


@patch("src.learning.reports.compute_learning_status")
@patch("src.learning.reports.rejection_breakdown")
@patch("src.learning.reports.identify_weaknesses")
@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_handles_empty_data(mock_models, mock_weaknesses, mock_rejections, mock_learning_status):
    mock_learning_status.return_value = {
        "stage": "BOOTSTRAP",
        "trades_collected": 3,
        "rejected_trades": 40,
        "winning_trades": 1,
        "losing_trades": 2,
        "data_sufficiency_pct": 0.6,
        "recommendations_count": 0,
        "simulations_count": 0,
        "candidates_count": 0,
        "promotion_eligible": False,
        "next_stage": "OBSERVATION",
        "trades_to_next_stage": 22,
        "current_activity": "Collecting trade data only.",
        "reason": "22 more closed trade(s) needed to reach OBSERVATION (requires 25).",
    }
    mock_models.get_recommendations.return_value = []
    mock_models.get_strategy_simulations.return_value = []
    mock_models.get_adaptive_strategy_versions.return_value = []
    mock_weaknesses.return_value = {"worst_by_dimension": {}}
    mock_rejections.return_value = []

    out = generate_adaptive_strategy_report_html("paper")

    assert "No recommendations yet." in out
    assert "No simulations run yet." in out
    assert "No adaptive strategy candidates yet." in out
    assert "Not enough data yet." in out
    assert "No rejected candidates logged yet." in out
