"""Ad-hoc CLI to set capital_config for a mode, and to bootstrap
strategy_versions with an initial prompt if none exists yet. Throwaway —
replaced by the dashboard's authenticated Config panel at build step 10."""

from __future__ import annotations

from src.db import models

INITIAL_STRATEGY_PROMPT = (
    "You are a disciplined intraday crypto trading strategist trading INR "
    "pairs on CoinDCX. Analyze the given market snapshot (last price, "
    "recent 1-minute candles) and decide whether to buy, sell, or stay "
    "flat. Prefer high-conviction setups over frequent trading — a flat "
    "call is always acceptable when signals are unclear. Keep reasoning "
    "concise (1-3 sentences) and grounded in the specific data given, not "
    "generic market commentary."
)


def _prompt_float(label: str, default: float | None = None) -> float:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        try:
            return float(raw)
        except ValueError:
            print("enter a number")


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return int(raw) if raw else default


def seed_initial_strategy_version() -> None:
    if models.get_latest_version() is not None:
        return
    models.insert_strategy_version(
        version_number=1,
        prompt_text=INITIAL_STRATEGY_PROMPT,
        params_json={},
        notes="initial baseline",
    )
    print("strategy_versions: seeded version 1 (baseline prompt).")


def main() -> None:
    mode = input("mode (paper/real) [paper]: ").strip() or "paper"
    total_capital = _prompt_float("total_capital (INR)")
    capital_to_use = _prompt_float("capital_to_use (INR)", default=total_capital)
    daily_profit_target = _prompt_float("daily_profit_target (INR)")
    max_daily_loss = _prompt_float("max_daily_loss (INR)")
    position_size_pct = _prompt_float("position_size_pct (%)", default=10)
    max_concurrent_positions = _prompt_int("max_concurrent_positions", default=5)

    models.upsert_capital_config(
        mode=mode,
        total_capital=total_capital,
        capital_to_use=capital_to_use,
        daily_profit_target=daily_profit_target,
        max_daily_loss=max_daily_loss,
        position_size_pct=position_size_pct,
        max_concurrent_positions=max_concurrent_positions,
    )
    print(f"capital_config[{mode}] saved.")

    seed_initial_strategy_version()


if __name__ == "__main__":
    main()
