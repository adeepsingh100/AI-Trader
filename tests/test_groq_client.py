from unittest.mock import Mock, patch

import pytest
import requests

from src.groq_client import AllModelsFailedError, ModelUsageEvent, chat


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


def test_chat_bounds_output_with_max_tokens_by_default():
    client = Mock()
    client.chat.completions.create.return_value = _resp("ok")

    chat([{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client)

    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 1024


def test_chat_max_tokens_is_overridable():
    client = Mock()
    client.chat.completions.create.return_value = _resp("ok")

    chat([{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client, max_tokens=2048)

    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 2048


# --- Ollama provider ---


def _ollama_resp(text):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"message": {"content": text}}
    return resp


def test_ollama_provider_posts_expected_shape(monkeypatch):
    monkeypatch.setattr("src.groq_client.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("src.groq_client.OLLAMA_BASE_URL", "https://ollama.example")
    monkeypatch.setattr("src.groq_client.OLLAMA_API_KEY", "test-key")

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _ollama_resp("hi from ollama")

        content, events = chat(
            [{"role": "user", "content": "hi"}], model_chain=["gpt-oss:120b-cloud"], max_tokens=512
        )

    assert content == "hi from ollama"
    assert events == [ModelUsageEvent("gpt-oss:120b-cloud", None, events[0].latency_ms, True)]

    args, kwargs = mock_post.call_args
    assert args[0] == "https://ollama.example/api/chat"
    assert kwargs["json"] == {
        "model": "gpt-oss:120b-cloud",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 512},
    }
    assert kwargs["headers"] == {"Authorization": "Bearer test-key"}


def test_ollama_omits_auth_header_without_api_key(monkeypatch):
    monkeypatch.setattr("src.groq_client.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("src.groq_client.OLLAMA_API_KEY", "")

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _ollama_resp("hi")
        chat([{"role": "user", "content": "hi"}], model_chain=["local-model"])

    assert mock_post.call_args.kwargs["headers"] == {}


def test_ollama_uses_its_own_default_model_chain(monkeypatch):
    monkeypatch.setattr("src.groq_client.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("src.groq_client.OLLAMA_MODEL_CHAIN", ["gpt-oss:120b-cloud"])

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _ollama_resp("hi")
        chat([{"role": "user", "content": "hi"}])  # no explicit model_chain

    assert mock_post.call_args.kwargs["json"]["model"] == "gpt-oss:120b-cloud"


def test_ollama_falls_back_to_next_model_on_failure(monkeypatch):
    monkeypatch.setattr("src.groq_client.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.side_effect = [
            requests.ConnectionError("model-a unreachable"),
            requests.ConnectionError("model-a unreachable"),
            requests.ConnectionError("model-a unreachable"),
            _ollama_resp("hello from model-b"),
        ]

        content, events = chat(
            [{"role": "user", "content": "hi"}], model_chain=["model-a", "model-b"]
        )

    assert content == "hello from model-b"
    assert [e.model_used for e in events] == ["model-a", "model-a", "model-a", "model-b"]
    assert [e.success for e in events] == [False, False, False, True]


def test_ollama_all_models_failed_raises(monkeypatch):
    monkeypatch.setattr("src.groq_client.LLM_PROVIDER", "ollama")
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.side_effect = requests.ConnectionError("down")

        with pytest.raises(AllModelsFailedError):
            chat([{"role": "user", "content": "hi"}], model_chain=["model-a"])
