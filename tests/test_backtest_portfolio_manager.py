from datetime import datetime, timezone

from src.backtest.portfolio_manager import PortfolioManager


def test_open_position_deducts_cash_including_fees():
    pm = PortfolioManager(starting_capital=10_000)
    pm.open_position("BTCINR", qty=1.0, fill_price=100.0, entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), fees=5.0)
    assert pm.cash == 10_000 - 100.0 - 5.0
    assert "BTCINR" in pm.positions


def test_close_position_credits_cash_and_computes_pnl():
    pm = PortfolioManager(starting_capital=10_000)
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    pm.open_position("BTCINR", qty=1.0, fill_price=100.0, entry_time=t0, fees=1.0)
    trade = pm.close_position("BTCINR", fill_price=110.0, exit_time=t1, fees=1.0, slippage_cost=0.5)

    assert "BTCINR" not in pm.positions
    assert trade.pnl == (110.0 - 100.0) * 1.0 - 1.0 - 1.0
    assert pm.realized_pnl == trade.pnl
    assert pm.cash == 10_000 - 100.0 - 1.0 + 110.0 - 1.0


def test_committed_capital_uses_entry_basis():
    pm = PortfolioManager(starting_capital=10_000)
    pm.open_position("BTCINR", qty=2.0, fill_price=50.0, entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), fees=0)
    assert pm.committed_capital() == 100.0


def test_update_excursion_tracks_mfe_and_mae():
    pm = PortfolioManager(starting_capital=10_000)
    pm.open_position("BTCINR", qty=1.0, fill_price=100.0, entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), fees=0)

    pm.update_excursion("BTCINR", 110.0)  # +10% favorable
    pm.update_excursion("BTCINR", 95.0)  # -5% adverse
    pm.update_excursion("BTCINR", 105.0)  # back up, doesn't reduce prior MFE/MAE

    pos = pm.positions["BTCINR"]
    assert pos.mfe_pct == 10.0
    assert pos.mae_pct == 5.0


def test_equity_reflects_unrealized_gains():
    pm = PortfolioManager(starting_capital=10_000)
    pm.open_position("BTCINR", qty=1.0, fill_price=100.0, entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), fees=0)
    equity = pm.equity({"BTCINR": 150.0})
    assert equity == (10_000 - 100.0) + 150.0


def test_snapshot_records_exposure_and_open_position_count():
    pm = PortfolioManager(starting_capital=10_000)
    pm.open_position("BTCINR", qty=1.0, fill_price=100.0, entry_time=datetime(2024, 1, 1, tzinfo=timezone.utc), fees=0)
    snap = pm.snapshot(datetime(2024, 1, 1, tzinfo=timezone.utc), {"BTCINR": 100.0})

    assert snap["open_positions_count"] == 1
    assert snap["equity"] == pm.equity({"BTCINR": 100.0})
    assert pm.snapshots == [snap]


def test_buying_power_and_leverage_are_spot_only():
    pm = PortfolioManager(starting_capital=10_000)
    assert pm.buying_power == pm.cash
    assert pm.leverage == 1.0
