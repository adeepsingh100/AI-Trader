"""LLM chat wrapper: retries each model in a chain with backoff, then
falls back to the next model. Every attempt is returned as a
ModelUsageEvent for the caller to persist to `model_usage`.

Auto-fallback across providers, not a provider-select switch: every call
tries the full Groq chain first, then automatically falls through to the
full Gemini chain if every Groq model fails. Groq's free-tier daily token
quota gets exhausted fast at this codebase's call volume (see
PROJECT_SPEC.md §4), and nobody's reliably around to flip a manual
switch when that happens — this keeps trading running same-cycle instead
of going dark until someone notices (the actual incident this replaces).
signal_agent.py and evolution_agent.py never know which provider
answered."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import requests
from groq import Groq

from src.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL_CHAIN,
    GROQ_API_KEY,
    GROQ_MODEL_CHAIN,
    LLM_BACKOFF_BASE_SECONDS,
    LLM_MAX_RETRIES_PER_MODEL,
)
from src.resilience import backoff_delay

MAX_RETRIES_PER_MODEL = LLM_MAX_RETRIES_PER_MODEL
BACKOFF_BASE_SECONDS = LLM_BACKOFF_BASE_SECONDS
# openai/gpt-oss-120b on Groq caps context at 8K tokens, input+output
# combined. Our prompts are small (a strategy prompt + a market snapshot,
# or a metrics summary), so the real risk is unbounded *output* eating
# the budget — this keeps every call well inside it regardless of chain.
DEFAULT_MAX_TOKENS = 1024
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


def _walk_chain(
    call: Callable[[str], str],
    chain: list[str],
    events: list[ModelUsageEvent],
    fallback_reason: str | None = None,
) -> tuple[str | None, str | None]:
    """Tries each model in `chain` (with per-model retry+backoff),
    appending every attempt to `events`. Returns (content, None) on
    success, or (None, last_fallback_reason) if every model in THIS
    chain failed — the caller decides what happens next (e.g. walking a
    different provider's chain, threading the reason through so the next
    chain's first attempt still records *why* it's being tried)."""
    for model in chain:
        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            start = time.monotonic()
            try:
                content = call(model)
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, True))
                return content, None
            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, False))
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(backoff_delay(BACKOFF_BASE_SECONDS, attempt))
                else:
                    fallback_reason = f"{model} failed: {e}"
    return None, fallback_reason


def chat(
    messages: list[dict],
    model_chain: list[str] | None = None,
    client: Groq | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[str, list[ModelUsageEvent]]:
    events: list[ModelUsageEvent] = []

    groq_chain = model_chain if model_chain is not None else GROQ_MODEL_CHAIN
    groq_client = client if client is not None else Groq(api_key=GROQ_API_KEY)
    content, fallback_reason = _walk_chain(
        lambda m: _groq_completion(groq_client, m, messages, max_tokens), groq_chain, events
    )
    if content is not None:
        return content, events

    content, fallback_reason = _walk_chain(
        lambda m: _gemini_completion(m, messages, max_tokens), GEMINI_MODEL_CHAIN, events, fallback_reason
    )
    if content is not None:
        return content, events

    raise AllModelsFailedError(
        f"all models failed — groq chain {groq_chain}, gemini chain {GEMINI_MODEL_CHAIN} "
        f"(last error: {fallback_reason})",
        events,
    )
