import json
from unittest.mock import patch

from src.agents.signal_agent import validate_opportunity
from src.groq_client import AllModelsFailedError, ModelUsageEvent


def _summary():
    return {
        "symbol": "BTCINR",
        "last_price": 6_200_000,
        "opportunity_score": 82,
        "sub_scores": {"trend": 90, "momentum": 75, "volume": 80, "volatility": 100, "risk": 65},
        "volatility_label": "medium",
        "support": 6_000_000,
        "resistance": 6_400_000,
        "distance_from_resistance_pct": 3.2,
        "volume_spike": True,
        "adx": 28.5,
        "di_plus": 22.0,
        "di_minus": 12.0,
    }


def test_validate_opportunity_parses_accept_json():
    events = [ModelUsageEvent("model-a", None, 100, True)]
    verdict_json = json.dumps({"decision": "accept", "reasoning": "strong bullish alignment"})
    with patch("src.agents.signal_agent.chat", return_value=(verdict_json, events)):
        verdict, returned_events = validate_opportunity(_summary(), "system prompt", context="entry")

    assert verdict["decision"] == "accept"
    assert verdict["reasoning"] == "strong bullish alignment"
    assert returned_events == events


def test_validate_opportunity_falls_back_to_reject_on_bad_json():
    with patch("src.agents.signal_agent.chat", return_value=("not json", [])):
        verdict, _ = validate_opportunity(_summary(), "system prompt", context="entry")

    assert verdict["decision"] == "reject"


def test_validate_opportunity_falls_back_to_reject_when_all_models_fail():
    # a total LLM outage must fail CLOSED (reject), not silently let a
    # trade through — the LLM is a gate now, not the primary decision maker
    failure_events = [ModelUsageEvent("model-a", None, 50, False)]
    with patch(
        "src.agents.signal_agent.chat",
        side_effect=AllModelsFailedError("all models in chain failed: [...]", failure_events),
    ):
        verdict, returned_events = validate_opportunity(_summary(), "system prompt", context="exit")

    assert verdict["decision"] == "reject"
    assert "all models failed" in verdict["reasoning"]
    assert returned_events == failure_events


def test_validate_opportunity_message_never_contains_raw_candles():
    # the core requirement of the refactor: the LLM sees a curated
    # summary, never raw candle data
    captured = {}

    def _fake_chat(messages, *args, **kwargs):
        captured["messages"] = messages
        return json.dumps({"decision": "accept", "reasoning": "ok"}), []

    with patch("src.agents.signal_agent.chat", side_effect=_fake_chat):
        validate_opportunity(_summary(), "system prompt", context="entry")

    full_text = json.dumps(captured["messages"])
    assert "candles" not in full_text
    assert "recent_candles" not in full_text


def test_validate_opportunity_entry_vs_exit_ask_different_questions():
    captured = []

    def _fake_chat(messages, *args, **kwargs):
        captured.append(messages[1]["content"])
        return json.dumps({"decision": "accept", "reasoning": "ok"}), []

    with patch("src.agents.signal_agent.chat", side_effect=_fake_chat):
        validate_opportunity(_summary(), "system prompt", context="entry")
        validate_opportunity(_summary(), "system prompt", context="exit")

    assert "accepted" in captured[0]
    assert "closed" in captured[1]
    assert captured[0] != captured[1]
