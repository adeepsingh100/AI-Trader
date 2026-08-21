from unittest.mock import Mock, patch

import pytest

from src.groq_client import AllModelsFailedError, _gemini_completion, chat


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


# --- Gemini fallback (auto, when the entire Groq chain fails) ---


def _gemini_resp(text):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = {"candidates": [{"content": {"parts": [{"text": text}]}}]}
    return resp


def test_falls_back_to_gemini_when_groq_chain_fully_fails(monkeypatch):
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)
    monkeypatch.setattr("src.groq_client.GEMINI_MODEL_CHAIN", ["gemini-2.5-flash"])
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("groq quota exhausted")

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _gemini_resp("hi from gemini")

        content, events = chat(
            [{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client
        )

    assert content == "hi from gemini"
    # Groq's 3 failed attempts (2 retries) precede the successful Gemini one.
    assert [e.model_used for e in events] == ["model-a", "model-a", "model-a", "gemini-2.5-flash"]
    assert [e.success for e in events] == [False, False, False, True]
    # The Gemini attempt's fallback_reason records WHY it's being tried —
    # the Groq chain's failure, not a blank slate.
    assert "model-a failed" in events[-1].fallback_reason


def test_all_models_failed_raises_after_both_chains_exhausted(monkeypatch):
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)
    monkeypatch.setattr("src.groq_client.GEMINI_MODEL_CHAIN", [])  # nothing to fall through to
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("boom")

    with pytest.raises(AllModelsFailedError) as exc_info:
        chat([{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client)

    # the real failure reason must survive on the exception, not just the
    # generic "all models failed" — this is what a caller logs for diagnosis
    assert "boom" in str(exc_info.value)
    assert len(exc_info.value.events) == 3
    assert all(e.success is False for e in exc_info.value.events)


def test_gemini_also_fails_raises_with_both_chains_recorded(monkeypatch):
    monkeypatch.setattr("src.groq_client.BACKOFF_BASE_SECONDS", 0)
    monkeypatch.setattr("src.groq_client.GEMINI_MODEL_CHAIN", ["gemini-2.5-flash"])
    client = Mock()
    client.chat.completions.create.side_effect = RuntimeError("groq down")

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.side_effect = RuntimeError("gemini down")

        with pytest.raises(AllModelsFailedError) as exc_info:
            chat([{"role": "user", "content": "hi"}], model_chain=["model-a"], client=client)

    # 3 failed groq attempts + 3 failed gemini attempts (2 retries each)
    assert len(exc_info.value.events) == 6
    assert [e.model_used for e in exc_info.value.events] == ["model-a"] * 3 + ["gemini-2.5-flash"] * 3
    assert "gemini down" in str(exc_info.value)


# --- _gemini_completion request/response shape ---


def test_gemini_completion_posts_expected_shape(monkeypatch):
    monkeypatch.setattr("src.groq_client.GEMINI_API_KEY", "test-key")

    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _gemini_resp("hi from gemini")

        content = _gemini_completion(
            "gemini-2.5-flash",
            [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}],
            512,
        )

    assert content == "hi from gemini"
    args, kwargs = mock_post.call_args
    assert args[0] == "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    assert kwargs["params"] == {"key": "test-key"}
    assert kwargs["json"] == {
        "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
        "generationConfig": {"maxOutputTokens": 512},
        "systemInstruction": {"parts": [{"text": "be terse"}]},
    }


def test_gemini_completion_omits_system_instruction_without_system_message():
    with patch("src.groq_client.requests.post") as mock_post:
        mock_post.return_value = _gemini_resp("hi")
        _gemini_completion("gemini-2.5-flash", [{"role": "user", "content": "hi"}], 512)

    assert "systemInstruction" not in mock_post.call_args.kwargs["json"]
