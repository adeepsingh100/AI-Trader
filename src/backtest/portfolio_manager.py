"""Cash/equity/position/exposure tracking with a mark-to-market equity
curve — genuinely new capability (the only existing drawdown,
evolution_agent._max_drawdown_pct, walks the trade-pnl sequence, not an
intraday equity curve). Pure in-memory bookkeeping, no DB/network —
BacktestEngine persists snapshots/trades to Supabase separately, at the
decision-cycle cadence, not every tick. Spot-only, like live (CoinDCX spot
has no margin/leverage anywhere in this codebase) — buying_power is just
cash, leverage is always 1.0, no margin machinery is built."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: datetime
    entry_fees: float
    stop_loss_price: float | None = None
    take_profit_price: float | None = None
    confidence: float | None = None
    opportunity_score: float | None = None
    market_regime: str | None = None
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    fees: float
    slippage_cost: float
    exit_reason: str | None
    confidence: float | None
    opportunity_score: float | None
    market_regime: str | None
    mfe_pct: float
    mae_pct: float

    @property
    def holding_duration_seconds(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds()

    @property
    def return_pct(self) -> float | None:
        basis = self.entry_price * self.qty
        return (self.pnl / basis * 100) if basis else None


@dataclass
class PortfolioManager:
    starting_capital: float
    cash: float = field(init=False)
    positions: dict[str, Position] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    realized_pnl: float = 0.0
    snapshots: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.starting_capital

    @property
    def buying_power(self) -> float:
        return self.cash

    @property
    def leverage(self) -> float:
        return 1.0

    def committed_capital(self) -> float:
        """Entry-basis cost of open positions (matches
        risk_manager.committed_capital's convention exactly, for the same
        capital-limit check reused unchanged in engine.py)."""
        return sum(p.qty * p.entry_price for p in self.positions.values())

    def open_position(
        self,
        symbol: str,
        qty: float,
        fill_price: float,
        entry_time: datetime,
        fees: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
        confidence: float | None = None,
        opportunity_score: float | None = None,
        market_regime: str | None = None,
    ) -> Position:
        self.cash -= qty * fill_price + fees
        pos = Position(
            symbol=symbol,
            qty=qty,
            entry_price=fill_price,
            entry_time=entry_time,
            entry_fees=fees,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            confidence=confidence,
            opportunity_score=opportunity_score,
            market_regime=market_regime,
        )
        self.positions[symbol] = pos
        return pos

    def close_position(
        self,
        symbol: str,
        fill_price: float,
        exit_time: datetime,
        fees: float,
        slippage_cost: float,
        exit_reason: str | None = None,
    ) -> ClosedTrade:
        pos = self.positions.pop(symbol)
        self.cash += pos.qty * fill_price - fees
        pnl = (fill_price - pos.entry_price) * pos.qty - fees - pos.entry_fees
        self.realized_pnl += pnl
        trade = ClosedTrade(
            symbol=symbol,
            side="buy",
            qty=pos.qty,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            entry_time=pos.entry_time,
            exit_time=exit_time,
            pnl=pnl,
            fees=fees + pos.entry_fees,
            slippage_cost=slippage_cost,
            exit_reason=exit_reason,
            confidence=pos.confidence,
            opportunity_score=pos.opportunity_score,
            market_regime=pos.market_regime,
            mfe_pct=pos.mfe_pct,
            mae_pct=pos.mae_pct,
        )
        self.closed_trades.append(trade)
        return trade

    def update_excursion(self, symbol: str, price: float) -> None:
        pos = self.positions.get(symbol)
        if pos is None or not pos.entry_price:
            return
        favorable = max(0.0, (price - pos.entry_price) / pos.entry_price * 100)
        adverse = max(0.0, (pos.entry_price - price) / pos.entry_price * 100)
        pos.mfe_pct = max(pos.mfe_pct, favorable)
        pos.mae_pct = max(pos.mae_pct, adverse)

    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, pos in self.positions.items():
            price = prices.get(symbol)
            if price is None:
                continue
            total += (price - pos.entry_price) * pos.qty
        return total

    def market_value(self, prices: dict[str, float]) -> float:
        return sum(pos.qty * prices.get(pos.symbol, pos.entry_price) for pos in self.positions.values())

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.market_value(prices)

    def snapshot(self, time: datetime, prices: dict[str, float]) -> dict:
        market_value = self.market_value(prices)
        equity = self.cash + market_value
        snap = {
            "snapshot_time": time,
            "cash": self.cash,
            "equity": equity,
            "unrealized_pnl": self.unrealized_pnl(prices),
            "realized_pnl": self.realized_pnl,
            "exposure_pct": (market_value / equity * 100) if equity > 0 else 0.0,
            "open_positions_count": len(self.positions),
        }
        self.snapshots.append(snap)
        return snap
