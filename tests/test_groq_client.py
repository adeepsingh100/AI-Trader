from unittest.mock import Mock

import pytest

from src.groq_client import AllModelsFailedError, chat


def _resp(text):
    return Mock(choices=[Mock(message=Mock(content=text))])


def test_fallback_to_next_model_on_failure(monkeypatch):
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)
    client = Mock()
    client.chat.completions.create.side_effect = [
        RuntimeError("429 rate limited"),
        RuntimeError("429 rate limited"),
        RuntimeError("429 rate limited"),
        _resp("hello from fallback model"),
    ]

    content, events = chat(
        [{"role": "user", "content": "hi"}],
        model_chain=["model-a", "model-b"],
        client=client,
    )

    assert content == "hello from fallback model"
    assert [e.model_used for e in events] == ["model-a", "model-a", "model-a", "model-b"]
    assert [e.success for e in events] == [False, False, False, True]
    assert "model-a failed" in events[-1].fallback_reason


def test_all_models_failed_raises(monkeypatch):
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("boom")

    with pytest.raises(AllModelsFailedError):
        chat([{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client)
