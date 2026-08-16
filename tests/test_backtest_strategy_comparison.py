from datetime import datetime, timezone
from unittest.mock import patch

from src.backtest.portfolio_manager import ClosedTrade
from src.backtest.strategy_comparison import compare


def _trades(pnls):
    return [
        ClosedTrade(
            symbol="TESTINR", side="buy", qty=1.0, entry_price=100.0, exit_price=100.0 + p,
            entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
            pnl=p, fees=0.1, slippage_cost=0.1, exit_reason="ai_exit", confidence=None,
            opportunity_score=70.0, market_regime=None, mfe_pct=0.0, mae_pct=0.0,
        )
        for p in pnls
    ]


@patch("src.backtest.strategy_comparison.SIGNIFICANCE_THRESHOLD", 0.05)
def test_compare_picks_b_when_significantly_better_expectancy():
    trades_a = _trades([-5] * 25 + [1] * 5)
    trades_b = _trades([10] * 25 + [-1] * 5)
    metrics_a = {"win_rate": 5 / 30, "expectancy": -3.0}
    metrics_b = {"win_rate": 25 / 30, "expectancy": 8.0}

    result = compare(trades_a, trades_b, metrics_a, metrics_b)

    assert result["winner"] == "b"
    assert result["promotion_recommended"] is True


@patch("src.backtest.strategy_comparison.SIGNIFICANCE_THRESHOLD", 0.05)
def test_compare_no_winner_when_not_significant():
    trades_a = _trades([1, 2, -1, 2, 1])
    trades_b = _trades([1, 1, 2, -1, 2])
    metrics_a = {"win_rate": 0.8, "expectancy": 1.0}
    metrics_b = {"win_rate": 0.8, "expectancy": 1.1}

    result = compare(trades_a, trades_b, metrics_a, metrics_b)

    assert result["winner"] is None
    assert result["promotion_recommended"] is False


def test_compare_returns_p_values_dict():
    trades_a = _trades([1, 2, 3])
    trades_b = _trades([4, 5, 6])
    result = compare(trades_a, trades_b, {"win_rate": 1.0, "expectancy": 2.0}, {"win_rate": 1.0, "expectancy": 5.0})
    assert set(result["p_values"].keys()) == {"win_rate", "expectancy"}
