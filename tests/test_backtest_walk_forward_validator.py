from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

from src.backtest.portfolio_manager import ClosedTrade
from src.backtest.walk_forward_validator import run_walk_forward


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


@patch("src.backtest.walk_forward_validator.RECOMMENDATION_MIN_SAMPLE_SIZE", 5)
@patch("src.backtest.walk_forward_validator.analyze")
@patch("src.backtest.walk_forward_validator.BacktestEngine")
def test_run_walk_forward_produces_one_fold_per_window_that_fits(mock_engine_cls, mock_analyze):
    fake_engine = MagicMock()
    fake_engine.run.return_value = {"closed_trades": _trades([1] * 10), "snapshots": [], "open_positions": [], "execution_history": []}
    fake_engine.portfolio.starting_capital = 10_000
    mock_engine_cls.return_value = fake_engine
    mock_analyze.return_value = {"trades_count": 10, "expectancy": 1.0}

    folds = run_walk_forward(
        ["TESTINR"], {"TESTINR": "I-TEST_INR"},
        overall_start=date(2024, 1, 1), overall_end=date(2024, 5, 1),
        params_json={}, n_folds=2, train_days=30, test_days=15,
    )

    assert len(folds) == 2
    assert folds[0].fold_number == 1
    assert folds[0].train_window_start == date(2024, 1, 1)
    assert folds[0].train_window_end == date(2024, 1, 31)
    assert folds[0].test_window_start == date(2024, 1, 31)
    assert folds[0].test_window_end == date(2024, 2, 15)
    # non-overlapping: fold 2 starts where fold 1's test window started + test_days
    assert folds[1].train_window_start == date(2024, 1, 16)


@patch("src.backtest.walk_forward_validator.RECOMMENDATION_MIN_SAMPLE_SIZE", 100)
@patch("src.backtest.walk_forward_validator.analyze")
@patch("src.backtest.walk_forward_validator.BacktestEngine")
def test_run_walk_forward_insufficient_sample_reports_none_pvalue(mock_engine_cls, mock_analyze):
    fake_engine = MagicMock()
    fake_engine.run.return_value = {"closed_trades": _trades([1] * 3), "snapshots": [], "open_positions": [], "execution_history": []}
    fake_engine.portfolio.starting_capital = 10_000
    mock_engine_cls.return_value = fake_engine
    mock_analyze.return_value = {"trades_count": 3, "expectancy": 1.0}

    folds = run_walk_forward(
        ["TESTINR"], {"TESTINR": "I-TEST_INR"},
        overall_start=date(2024, 1, 1), overall_end=date(2024, 3, 1),
        params_json={}, n_folds=1, train_days=30, test_days=15,
    )

    assert folds[0].p_value is None
    assert folds[0].passed is None


@patch("src.backtest.walk_forward_validator.SIGNIFICANCE_THRESHOLD", 0.05)
@patch("src.backtest.walk_forward_validator.RECOMMENDATION_MIN_SAMPLE_SIZE", 5)
@patch("src.backtest.walk_forward_validator.analyze")
@patch("src.backtest.walk_forward_validator.BacktestEngine")
def test_run_walk_forward_passes_when_out_of_sample_beats_in_sample(mock_engine_cls, mock_analyze):
    fake_engine = MagicMock()
    fake_engine.portfolio.starting_capital = 10_000
    mock_engine_cls.return_value = fake_engine

    train_trades = _trades([1, 2, 1, 2, 1, 2])
    test_trades = _trades([10, 12, 9, 11, 10, 13])

    def fake_run():
        return {"closed_trades": fake_engine.run.call_args, "snapshots": [], "open_positions": [], "execution_history": []}

    # first call -> train window, second -> test window
    fake_engine.run.side_effect = [
        {"closed_trades": train_trades, "snapshots": [], "open_positions": [], "execution_history": []},
        {"closed_trades": test_trades, "snapshots": [], "open_positions": [], "execution_history": []},
    ]
    mock_analyze.side_effect = [
        {"trades_count": 6, "expectancy": 1.5},
        {"trades_count": 6, "expectancy": 10.8},
    ]

    folds = run_walk_forward(
        ["TESTINR"], {"TESTINR": "I-TEST_INR"},
        overall_start=date(2024, 1, 1), overall_end=date(2024, 3, 1),
        params_json={}, n_folds=1, train_days=30, test_days=15,
    )

    assert folds[0].passed is True
