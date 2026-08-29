from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionAgent(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        """Returns a fill: {"fill_price": float, "fees": float}."""

    @abstractmethod
    def flatten_all(self, mode: str, strategy_type: str | None = None) -> list[dict]:
        """Force-close every open position for mode (and, if given,
        strategy_type only — each strategy type's circuit breaker is
        independent, so tripping one must never flatten another's
        positions). Returns closed trades."""
