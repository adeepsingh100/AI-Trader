"""LLM chat wrapper: retries each model in the chain with backoff, then
falls back to the next model. Every attempt is returned as a
ModelUsageEvent for the caller to persist to `model_usage`.

Provider is picked by LLM_PROVIDER — "groq" (default) or "ollama" (Ollama
Cloud). Same retry/fallback loop either way; only how a single model gets
called differs. Switching providers is an env var change, not a code
change — signal_agent.py and evolution_agent.py never know which one ran."""

from __future__ import annotations

import time
from dataclasses import dataclass

import requests
from groq import Groq

from src.config import (
    GROQ_API_KEY,
    GROQ_MODEL_CHAIN,
    LLM_PROVIDER,
    OLLAMA_API_KEY,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_CHAIN,
)

MAX_RETRIES_PER_MODEL = 2
BACKOFF_BASE_SECONDS = 1.0
# openai/gpt-oss-120b on Groq caps context at 8K tokens, input+output
# combined. Our prompts are small (a strategy prompt + a market snapshot,
# or a metrics summary), so the real risk is unbounded *output* eating
# the budget — this keeps every call well inside it regardless of chain.
DEFAULT_MAX_TOKENS = 1024
OLLAMA_TIMEOUT_SECONDS = 120  # generous — cloud inference is slower than Groq


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
                    time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                else:
                    fallback_reason = f"{model} failed: {e}"

    raise AllModelsFailedError(
        f"all models in chain failed: {chain} (last error: {fallback_reason})", events
    )
