from datetime import datetime, timezone
from unittest.mock import patch

from src.learning.rejection_analysis import rejection_breakdown


def _row(reason=None, risk_manager_result=None):
    return {"reason": reason, "risk_manager_result": risk_manager_result}


def test_rejection_breakdown_ranks_by_count_descending():
    rows = [
        _row(risk_manager_result="block_concentration_limit"),
        _row(risk_manager_result="block_concentration_limit"),
        _row(reason="not_a_candidate"),
        _row(reason="confidence gated: 40.0 < 50"),
    ]
    with patch("src.learning.rejection_analysis.models") as mock_models:
        mock_models.get_hold_evaluations_since.return_value = rows
        result = rejection_breakdown("paper", since=datetime.now(timezone.utc))

    assert result[0]["reason"] == "block_concentration_limit"
    assert result[0]["count"] == 2
    assert result[0]["pct_of_rejections"] == 50.0


def test_rejection_breakdown_prefers_risk_manager_result_over_reason():
    rows = [_row(reason="risk_manager blocked: block_max_positions", risk_manager_result="block_max_positions")]
    with patch("src.learning.rejection_analysis.models") as mock_models:
        mock_models.get_hold_evaluations_since.return_value = rows
        result = rejection_breakdown("paper")

    assert result[0]["reason"] == "block_max_positions"


def test_rejection_breakdown_empty_when_no_rows():
    with patch("src.learning.rejection_analysis.models") as mock_models:
        mock_models.get_hold_evaluations_since.return_value = []
        assert rejection_breakdown("paper") == []


def test_rejection_breakdown_unknown_label_when_both_missing():
    with patch("src.learning.rejection_analysis.models") as mock_models:
        mock_models.get_hold_evaluations_since.return_value = [_row()]
        result = rejection_breakdown("paper")

    assert result[0]["reason"] == "unknown"
