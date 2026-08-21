from datetime import date, datetime, timezone
from unittest.mock import ANY, MagicMock, patch

import pytest

from src.backtest.engine import SURVIVORSHIP_BIAS_NOTE, BacktestEngine, _date_to_ms, run_and_persist
from src.backtest.portfolio_manager import ClosedTrade


def _engine(**overrides):
    with patch("src.backtest.data_provider.models") as mock_models:
        mock_models.get_historical_candles.return_value = []
        kwargs = dict(
            symbols=["TESTINR"],
            symbol_to_pair={"TESTINR": "I-TEST_INR"},
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 2),
            warmup_buffer_days=0,
            starting_capital=10_000,
            position_size_pct=10,
            max_concurrent_positions=5,
        )
        kwargs.update(overrides)
        return BacktestEngine(**kwargs)


# --- warm-up buffer: widens the CandleStore range, never the clock ticks ---


def test_warmup_buffer_widens_candle_store_range_only():
    with patch("src.backtest.data_provider.models") as mock_models:
        mock_models.get_historical_candles.return_value = []
        engine = BacktestEngine(
            symbols=["TESTINR"],
            symbol_to_pair={"TESTINR": "I-TEST_INR"},
            start_date=date(2024, 1, 10),
            end_date=date(2024, 1, 11),
            warmup_buffer_days=5,
        )
        requested_start_ms = mock_models.get_historical_candles.call_args_list[0].args[2]
        assert requested_start_ms == _date_to_ms(date(2024, 1, 5))

        # the clock only ticks the REQUESTED window, not the warm-up period
        ticks = list(engine.clock.ticks())
        assert min(ticks) == _date_to_ms(date(2024, 1, 10))


# --- circuit breaker: all three checkpoints ---


def test_risk_check_pass_flattens_when_already_triggered_before_sweep():
    engine = _engine()
    engine.daily_pnl["circuit_breaker_triggered"] = True
    engine._flatten_all = MagicMock()

    engine._risk_check_pass(engine.clock.now_ms, {"TESTINR": 100.0})

    engine._flatten_all.assert_called_once()


def test_risk_check_pass_stop_loss_trips_breaker_post_sweep():
    engine = _engine()
    engine.capital_config["max_daily_loss"] = 1.0
    engine.params_json = {"stop_loss_pct": 0.05}
    engine._bar_volume = MagicMock(return_value=1_000_000)  # empty CandleStore -> avoid a 0-volume reject
    t0 = datetime.fromtimestamp(engine.clock.now_ms / 1000, tz=timezone.utc)
    # qty*price well above BACKTEST_MIN_NOTIONAL_INR so the exit fill isn't
    # itself rejected for being a dust-sized order.
    engine.portfolio.open_position("TESTINR", qty=1.0, fill_price=1000.0, entry_time=t0, fees=0.0)

    engine._risk_check_pass(engine.clock.now_ms, {"TESTINR": 940.0})  # -6% -> stop_loss (5%) hit

    assert "TESTINR" not in engine.portfolio.positions
    assert engine.daily_pnl["trades_count"] == 1
    assert engine.daily_pnl["circuit_breaker_triggered"] is True


def test_decision_pass_top_of_pass_flattens_when_already_triggered():
    engine = _engine()
    engine.daily_pnl["circuit_breaker_triggered"] = True
    engine._flatten_all = MagicMock()

    engine._decision_pass(engine.clock.now_ms, {"TESTINR": 100.0})

    engine._flatten_all.assert_called_once()


def test_decision_pass_stops_before_later_candidate_once_breaker_trips_mid_loop():
    engine = _engine(symbols=["AINR", "BINR"], symbol_to_pair={"AINR": "I-A_INR", "BINR": "I-B_INR"})
    engine.capital_config["max_daily_loss"] = 1.0
    t0 = datetime.fromtimestamp(engine.clock.now_ms / 1000, tz=timezone.utc)
    # AINR is held at a big loss and will be exit-scored below the exit
    # threshold; BINR is a fresh, high-scoring entry candidate that should
    # never actually get processed once AINR's exit trips the breaker.
    engine.portfolio.open_position("AINR", qty=1.0, fill_price=1000.0, entry_time=t0, fees=0.0)

    def fake_score(symbol, as_of_ms, price):
        base = {
            "features_by_tf": {},
            "trend_score": None,
            "momentum_score": None,
            "volume_score": None,
            "volatility_score": None,
            "risk_score": None,
            "market_regime": None,
        }
        if symbol == "AINR":
            # -50% vs. entry (1000) -- big enough loss to trip the breaker,
            # notional (qty 1.0 * 500) still well above BACKTEST_MIN_NOTIONAL_INR.
            return {"symbol": "AINR", "last_price": 500.0, "opportunity_score": 1.0, **base}
        return {"symbol": "BINR", "last_price": 100.0, "opportunity_score": 99.0, **base}

    engine._score_symbol = fake_score
    engine._bar_volume = MagicMock(return_value=1_000_000)

    engine._decision_pass(engine.clock.now_ms, {"AINR": 1.0, "BINR": 100.0})

    assert "AINR" not in engine.portfolio.positions  # exited at a huge loss
    assert "BINR" not in engine.portfolio.positions  # never reached
    assert engine.daily_pnl["circuit_breaker_triggered"] is True


# --- decision cadence produces a real entry when quant-only accepts ---


def test_decision_pass_opens_position_for_qualifying_candidate():
    engine = _engine()
    engine._score_symbol = MagicMock(
        return_value={
            "symbol": "TESTINR",
            "last_price": 100.0,
            "features_by_tf": {},
            "trend_score": 90.0,
            "momentum_score": 90.0,
            "volume_score": 90.0,
            "volatility_score": 90.0,
            "risk_score": 90.0,
            "opportunity_score": 90.0,
            "market_regime": "strong_bull",
        }
    )
    engine._bar_volume = MagicMock(return_value=1_000_000)

    engine._decision_pass(engine.clock.now_ms, {"TESTINR": 100.0})

    assert "TESTINR" in engine.portfolio.positions
    pos = engine.portfolio.positions["TESTINR"]
    assert pos.opportunity_score == 90.0
    assert pos.market_regime == "strong_bull"
    assert pos.confidence == 90.0  # quant-only: confidence is the opportunity_score itself


def test_decision_pass_blocks_oversized_candidate_via_concentration_gate():
    """Backtest/live parity (PROJECT_SPEC.md §3d): risk_manager.evaluate()'s
    concentration cap, already live in real/paper trading, must also gate
    backtest entries now that engine.py passes symbol/portfolio_positions/
    price_history — mirrors test_risk_manager.py's own oversized-candidate
    case (position_size_pct=50 of a 20-slot book's ~5% fair share)."""
    engine = _engine(position_size_pct=50, max_concurrent_positions=20)
    engine._score_symbol = MagicMock(
        return_value={
            "symbol": "TESTINR",
            "last_price": 100.0,
            "features_by_tf": {},
            "trend_score": 90.0,
            "momentum_score": 90.0,
            "volume_score": 90.0,
            "volatility_score": 90.0,
            "risk_score": 90.0,
            "opportunity_score": 90.0,
            "market_regime": "strong_bull",
        }
    )
    engine._bar_volume = MagicMock(return_value=1_000_000)

    engine._decision_pass(engine.clock.now_ms, {"TESTINR": 100.0})

    assert "TESTINR" not in engine.portfolio.positions


# --- day rollover ---


def test_maybe_roll_day_resets_daily_pnl_on_new_calendar_day():
    engine = _engine()
    engine.daily_pnl["realized_pnl"] = -500.0
    engine.daily_pnl["trades_count"] = 3
    engine._current_day = date(2000, 1, 1)  # force a stale "previous" day

    engine._maybe_roll_day()

    assert engine.daily_pnl == {"realized_pnl": 0.0, "trades_count": 0, "circuit_breaker_triggered": False}


# --- run() completes cleanly with no data and surfaces the bias note ---


def test_run_with_no_candle_data_completes_and_returns_bias_note():
    engine = _engine()
    result = engine.run()
    assert result["closed_trades"] == []
    assert result["survivorship_bias_note"] == SURVIVORSHIP_BIAS_NOTE


# --- run_and_persist: persistence wiring ---


@patch("src.backtest.engine.models")
@patch("src.backtest.engine.BacktestEngine")
def test_run_and_persist_writes_run_trades_snapshots_and_metrics(mock_engine_cls, mock_models):
    fake_engine = MagicMock()
    fake_engine.capital_config = {"capital_to_use": 10_000}
    fake_engine.use_llm_signal_agent = False
    fake_engine.portfolio.starting_capital = 10_000
    trade = ClosedTrade(
        symbol="TESTINR",
        side="buy",
        qty=1.0,
        entry_price=100.0,
        exit_price=110.0,
        entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        exit_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        pnl=8.0,
        fees=2.0,
        slippage_cost=0.5,
        exit_reason="ai_exit",
        confidence=None,
        opportunity_score=70.0,
        market_regime="strong_bull",
        mfe_pct=12.0,
        mae_pct=1.0,
    )
    fake_engine.run.return_value = {
        "closed_trades": [trade],
        "open_positions": [],
        "snapshots": [
            {
                "snapshot_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "cash": 9000,
                "equity": 10008,
                "unrealized_pnl": 0,
                "realized_pnl": 8.0,
                "exposure_pct": 0.0,
                "open_positions_count": 0,
            }
        ],
        "execution_history": [{"symbol": "TESTINR", "order_type": "market", "status": "filled"}],
        "circuit_breaker_events": 0,
        "survivorship_bias_note": "note",
    }
    mock_engine_cls.return_value = fake_engine
    mock_models.insert_backtest_run.return_value = {"id": 5}

    run_id = run_and_persist(["TESTINR"], {"TESTINR": "I-TEST_INR"}, date(2024, 1, 1), date(2024, 1, 2))

    assert run_id == 5
    mock_models.insert_backtest_trade.assert_called_once()
    mock_models.insert_backtest_portfolio_snapshots.assert_called_once()
    mock_models.insert_backtest_execution_events.assert_called_once()
    mock_models.insert_backtest_performance_metrics.assert_called_once()
    mock_models.update_backtest_run_status.assert_called_once_with(5, "completed", completed_at=ANY)


@patch("src.backtest.engine.models")
@patch("src.backtest.engine.BacktestEngine")
def test_run_and_persist_marks_failed_on_exception_and_reraises(mock_engine_cls, mock_models):
    fake_engine = MagicMock()
    fake_engine.capital_config = {"capital_to_use": 10_000}
    fake_engine.use_llm_signal_agent = False
    fake_engine.run.side_effect = RuntimeError("boom")
    mock_engine_cls.return_value = fake_engine
    mock_models.insert_backtest_run.return_value = {"id": 9}

    with pytest.raises(RuntimeError):
        run_and_persist(["TESTINR"], {"TESTINR": "I-TEST_INR"}, date(2024, 1, 1), date(2024, 1, 2))

    mock_models.update_backtest_run_status.assert_called_once_with(9, "failed")
