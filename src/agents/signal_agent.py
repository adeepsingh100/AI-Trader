"""Scores one market snapshot with the LLM, using the active strategy
version's prompt. Falls back to a flat/no-op signal if the model doesn't
return parseable JSON, or if every model in the chain fails outright, so
neither can crash the cycle."""

from __future__ import annotations

import json

from src.groq_client import AllModelsFailedError, ModelUsageEvent, chat
from src.lenient_json import parse_llm_json


def _messages_for(market: dict, strategy_prompt: str) -> list[dict]:
    market_summary = {
        "symbol": market["symbol"],
        "last_price": market["last_price"],
        "recent_candles": market["candles"][:5],
    }
    return [
        {"role": "system", "content": strategy_prompt},
        {
            "role": "user",
            "content": (
                json.dumps(market_summary)
                + '\n\nRespond as JSON only: {"direction": "buy|sell|flat", '
                '"confidence": 0-1, "reasoning": "..."}'
            ),
        },
    ]


def get_signal(market: dict, strategy_prompt: str) -> tuple[dict, list[ModelUsageEvent]]:
    try:
        content, events = chat(_messages_for(market, strategy_prompt))
    except AllModelsFailedError as e:
        return {
            "direction": "flat",
            "confidence": 0.0,
            "reasoning": f"all models failed: {e}",
        }, e.events

    try:
        signal = parse_llm_json(content)
    except (json.JSONDecodeError, TypeError):
        signal = {
            "direction": "flat",
            "confidence": 0.0,
            "reasoning": f"unparseable model response: {content!r}",
        }
    return signal, events
