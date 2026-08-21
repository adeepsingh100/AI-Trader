"""LLM chat wrapper: retries each model in the chain with backoff, then
falls back to the next model. Every attempt is returned as a
ModelUsageEvent for the caller to persist to `model_usage`.

Provider is picked by LLM_PROVIDER — "groq" (default), "ollama" (Ollama
Cloud), or "gemini" (Google AI Studio, a separate free-tier quota from
Groq — useful as a same-day out when Groq's daily token limit is hit).
Same retry/fallback loop either way; only how a single model gets called
differs. Switching providers is an env var change, not a code change —
signal_agent.py and evolution_agent.py never know which one ran."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from groq import Groq

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_CHAIN,
    GROQ_API_KEY,
    GROQ_MODEL_CHAIN,
    LLM_BACKOFF_BASE_SECONDS,
    LLM_MAX_RETRIES_PER_MODEL,
    LLM_PROVIDER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_CHAIN,
)
from src.resilience import backoff_delay

MAX_RETRIES_PER_MODEL = LLM_MAX_RETRIES_PER_MODEL
BACKOFF_BASE_SECONDS = LLM_BACKOFF_BASE_SECONDS
# openai/gpt-oss-120b on Groq caps context at 8K tokens, input+output
# combined. Our prompts are small (a strategy prompt + a market snapshot,
# or a metrics summary), so the real risk is unbounded *output* eating
# the budget — this keeps every call well inside it regardless of chain.
DEFAULT_MAX_TOKENS = 1024
OLLAMA_TIMEOUT_SECONDS = 120  # generous — cloud inference is slower than Groq
GEMINI_TIMEOUT_SECONDS = 60
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class ModelUsageEvent:
    model_used: str
    fallback_reason: str | None
    latency_ms: int
    success: bool


class AllModelsFailedError(RuntimeError):
    """Carries the per-attempt events (with their fallback_reason chain)
    so a caller that catches this to degrade gracefully can still log
    the real failure reason to model_usage instead of losing it."""

    def __init__(self, message: str, events: list[ModelUsageEvent]):
        super().__init__(message)
        self.events = events


def _groq_completion(client: Groq, model: str, messages: list[dict], max_tokens: int) -> str:
    resp = client.chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
    return resp.choices[0].message.content


def _ollama_completion(model: str, messages: list[dict], max_tokens: int) -> str:
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {},
        timeout=OLLAMA_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _gemini_completion(model: str, messages: list[dict], max_tokens: int) -> str:
    # Gemini's REST API uses "contents"/"parts" instead of OpenAI-style
    # chat messages, and a separate systemInstruction field rather than a
    # "system" role entry — translated here so signal_agent.py's messages
    # (built once, OpenAI-shaped) never need to know which provider runs.
    system_text = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    contents = [
        {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
        for m in messages
        if m["role"] != "system"
    ]
    payload: dict = {"contents": contents, "generationConfig": {"maxOutputTokens": max_tokens}}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    resp = requests.post(
        f"{GEMINI_API_BASE}/{model}:generateContent",
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=GEMINI_TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def chat(
    messages: list[dict],
    model_chain: list[str] | None = None,
    client: Groq | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, list[ModelUsageEvent]]:
    if LLM_PROVIDER == "ollama":
        chain = model_chain if model_chain is not None else OLLAMA_MODEL_CHAIN

        def call(model: str) -> str:
            return _ollama_completion(model, messages, max_tokens)
    elif LLM_PROVIDER == "gemini":
        chain = model_chain if model_chain is not None else GEMINI_MODEL_CHAIN

        def call(model: str) -> str:
            return _gemini_completion(model, messages, max_tokens)
    else:
        chain = model_chain if model_chain is not None else GROQ_MODEL_CHAIN
        groq_client = client if client is not None else Groq(api_key=GROQ_API_KEY)

        def call(model: str) -> str:
            return _groq_completion(groq_client, model, messages, max_tokens)

    events: list[ModelUsageEvent] = []
    fallback_reason: str | None = None

    for model in chain:
        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            start = time.monotonic()
            try:
                content = call(model)
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, True))
                return content, events
            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, False))
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(backoff_delay(BACKOFF_BASE_SECONDS, attempt))
                else:
                    fallback_reason = f"{model} failed: {e}"

    raise AllModelsFailedError(
        f"all models in chain failed: {chain} (last error: {fallback_reason})", events
    )
