"""Groq chat wrapper: retries each model in the chain with backoff, then
falls back to the next model. Every attempt is returned as a
ModelUsageEvent for the caller to persist to `model_usage` (build step 4
adds the DB layer that actually writes these)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from groq import Groq

from src.config import GROQ_API_KEY, GROQ_MODEL_CHAIN

MAX_RETRIES_PER_MODEL = 2
BACKOFF_BASE_SECONDS = 1.0


@dataclass
class ModelUsageEvent:
    model_used: str
    fallback_reason: str | None
    latency_ms: int
    success: bool


class AllModelsFailedError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    model_chain: list[str] | None = None,
    client: Groq | None = None,
) -> tuple[str, list[ModelUsageEvent]]:
    chain = model_chain if model_chain is not None else GROQ_MODEL_CHAIN
    client = client if client is not None else Groq(api_key=GROQ_API_KEY)

    events: list[ModelUsageEvent] = []
    fallback_reason: str | None = None

    for model in chain:
        for attempt in range(MAX_RETRIES_PER_MODEL + 1):
            start = time.monotonic()
            try:
                resp = client.chat.completions.create(model=model, messages=messages)
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, True))
                return resp.choices[0].message.content, events
            except Exception as e:
                latency_ms = int((time.monotonic() - start) * 1000)
                events.append(ModelUsageEvent(model, fallback_reason, latency_ms, False))
                if attempt < MAX_RETRIES_PER_MODEL:
                    time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
                else:
                    fallback_reason = f"{model} failed: {e}"

    raise AllModelsFailedError(f"all models in chain failed: {chain}")
