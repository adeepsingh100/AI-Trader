from unittest.mock import patch

from src.learning.reports import generate_adaptive_strategy_report_html


@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_smoke(mock_models):
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
        {"created_at": "2026-01-01T00:00:00Z", "passed": True, "p_value": 0.01},
        {"created_at": "2026-01-02T00:00:00Z", "passed": False, "p_value": 0.5},
    ]
    mock_models.get_adaptive_strategy_versions.return_value = [
        {"version_number": 1, "status": "candidate", "created_at": "2026-01-01T00:00:00Z"},
    ]

    out = generate_adaptive_strategy_report_html("paper")

    assert "OPPORTUNITY_WEIGHT_TREND" in out
    assert "avoid_symbol:DOGEINR" in out
    assert "<script>alert(1)</script>" not in out  # XSS-escaped
    assert "&lt;script&gt;" in out
    assert "PASSED" in out
    assert "0.0100" in out  # p-value formatted


@patch("src.learning.reports.models")
def test_generate_adaptive_strategy_report_html_handles_empty_data(mock_models):
    mock_models.get_recommendations.return_value = []
    mock_models.get_strategy_simulations.return_value = []
    mock_models.get_adaptive_strategy_versions.return_value = []

    out = generate_adaptive_strategy_report_html("paper")

    assert "No recommendations yet." in out
    assert "No simulations run yet." in out
    assert "No adaptive strategy candidates yet." in out
