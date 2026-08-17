from unittest.mock import patch

from src.learning.weakness_detection import identify_weaknesses


def _stat_row(dimension_type, dimension_value, expectancy, trades_count=25):
    return {
        "dimension_type": dimension_type,
        "dimension_value": dimension_value,
        "expectancy": expectancy,
        "trades_count": trades_count,
    }


def _importance_row(feature_name, timeframe, correlation_score, sample_count=25):
    return {
        "feature_name": feature_name,
        "timeframe": timeframe,
        "correlation_score": correlation_score,
        "sample_count": sample_count,
    }


def test_identify_weaknesses_ranks_worst_and_best_by_expectancy():
    rows = [
        _stat_row("market_regime", "strong_bull", expectancy=50.0),
        _stat_row("market_regime", "high_volatility", expectancy=-30.0),
        _stat_row("symbol", "BTCINR", expectancy=10.0),
    ]
    with patch("src.learning.weakness_detection.models") as mock_models:
        mock_models.get_learning_statistics.return_value = rows
        mock_models.get_feature_importance.return_value = []
        result = identify_weaknesses("paper")

    assert result["worst_by_dimension"]["market_regime"]["value"] == "high_volatility"
    assert result["best_by_dimension"]["market_regime"]["value"] == "strong_bull"
    assert result["worst_by_dimension"]["symbol"]["value"] == "BTCINR"


def test_identify_weaknesses_excludes_below_sample_floor(monkeypatch):
    monkeypatch.setattr("src.learning.weakness_detection.LEARNING_STAGE_OBSERVATION_MIN_TRADES", 20)
    rows = [_stat_row("symbol", "THINCOIN", expectancy=-100.0, trades_count=3)]
    with patch("src.learning.weakness_detection.models") as mock_models:
        mock_models.get_learning_statistics.return_value = rows
        mock_models.get_feature_importance.return_value = []
        result = identify_weaknesses("paper")

    assert "symbol" not in result["worst_by_dimension"]


def test_identify_weaknesses_ranks_indicators_excluding_blended():
    rows = [
        _importance_row("rsi", "1h", correlation_score=0.35),
        _importance_row("adx", "1h", correlation_score=-0.20),
        _importance_row("trend_score", "blended", correlation_score=0.9),
    ]
    with patch("src.learning.weakness_detection.models") as mock_models:
        mock_models.get_learning_statistics.return_value = []
        mock_models.get_feature_importance.return_value = rows
        result = identify_weaknesses("paper")

    assert result["best_indicator"]["feature_name"] == "rsi"
    assert result["worst_indicator"]["feature_name"] == "adx"


def test_identify_weaknesses_empty_when_nothing_stored():
    with patch("src.learning.weakness_detection.models") as mock_models:
        mock_models.get_learning_statistics.return_value = []
        mock_models.get_feature_importance.return_value = []
        result = identify_weaknesses("paper")

    assert result == {
        "worst_by_dimension": {},
        "best_by_dimension": {},
        "worst_indicator": None,
        "best_indicator": None,
    }
