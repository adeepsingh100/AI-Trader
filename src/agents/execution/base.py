from __future__ import annotations

from abc import ABC, abstractmethod


class ExecutionAgent(ABC):
    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, price: float) -> dict:
        """Returns a fill: {"fill_price": float, "fees": float}."""

    @abstractmethod
    def flatten_all(self, mode: str) -> list[dict]:
        """Force-close every open position for mode. Returns closed trades."""
