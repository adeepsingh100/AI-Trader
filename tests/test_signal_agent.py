from unittest.mock import patch

from src.agents.signal_agent import get_signal
from src.groq_client import ModelUsageEvent


def _market():
    return {"symbol": "BTCINR", "last_price": 6200000, "candles": [{"close": 6200000}]}


def test_get_signal_parses_json_response():
    events = [ModelUsageEvent("model-a", None, 100, True)]
    with patch(
        "src.agents.signal_agent.chat",
        return_value=('{"direction": "buy", "confidence": 0.8, "reasoning": "uptrend"}', events),
    ):
        signal, returned_events = get_signal(_market(), "system prompt")

    assert signal == {"direction": "buy", "confidence": 0.8, "reasoning": "uptrend"}
    assert returned_events == events


def test_get_signal_falls_back_to_flat_on_bad_json():
    with patch("src.agents.signal_agent.chat", return_value=("not json", [])):
        signal, _ = get_signal(_market(), "system prompt")

    assert signal["direction"] == "flat"
    assert signal["confidence"] == 0.0
