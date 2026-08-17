from unittest.mock import patch

from src.learning.reports import generate_adaptive_strategy_report_html


@patch("src.learning.reports.rejection_breakdown")
@patch("src.learning.reports.identify_weaknesses")
@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_smoke(mock_models, mock_weaknesses, mock_rejections):
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


@patch("src.learning.reports.rejection_breakdown")
@patch("src.learning.reports.identify_weaknesses")
@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_handles_empty_data(mock_models, mock_weaknesses, mock_rejections):
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
