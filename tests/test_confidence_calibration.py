from unittest.mock import patch

import pytest

from src.learning.confidence_calibration import calibrate_confidence


def test_pure_ai_fallback_when_no_historical_confidence():
    result = calibrate_confidence(ai_confidence=70, historical_confidence=None, sample_size=0)
    assert result["final_confidence"] == 70
    assert result["ai_weight_used"] == 1.0
    assert result["historical_weight_used"] == 0.0


@patch("src.learning.confidence_calibration.MIN_SIMILAR_TRADES", 5)
def test_pure_ai_fallback_when_sample_size_too_small():
    result = calibrate_confidence(ai_confidence=70, historical_confidence=90, sample_size=2)
    assert result["final_confidence"] == 70
    assert result["historical_weight_used"] == 0.0


@patch("src.learning.confidence_calibration.MIN_SIMILAR_TRADES", 5)
@patch("src.learning.confidence_calibration.CONFIDENCE_AI_WEIGHT", 0.6)
@patch("src.learning.confidence_calibration.CONFIDENCE_HISTORICAL_WEIGHT", 0.4)
def test_blended_confidence_known_value():
    result = calibrate_confidence(ai_confidence=90, historical_confidence=50, sample_size=10)
    assert result["final_confidence"] == pytest.approx(90 * 0.6 + 50 * 0.4)
    assert result["ai_weight_used"] == pytest.approx(0.6)
    assert result["historical_weight_used"] == pytest.approx(0.4)


@patch("src.learning.confidence_calibration.MIN_SIMILAR_TRADES", 5)
@patch("src.learning.confidence_calibration.CONFIDENCE_AI_WEIGHT", 3.0)
@patch("src.learning.confidence_calibration.CONFIDENCE_HISTORICAL_WEIGHT", 1.0)
def test_weights_renormalized_when_not_summing_to_one():
    result = calibrate_confidence(ai_confidence=100, historical_confidence=0, sample_size=10)
    assert result["ai_weight_used"] == pytest.approx(0.75)
    assert result["historical_weight_used"] == pytest.approx(0.25)


@patch("src.learning.confidence_calibration.MIN_SIMILAR_TRADES", 5)
def test_pure_historical_when_ai_confidence_missing():
    result = calibrate_confidence(ai_confidence=None, historical_confidence=80, sample_size=10)
    assert result["final_confidence"] == 80
    assert result["ai_weight_used"] == 0.0
    assert result["historical_weight_used"] == 1.0
