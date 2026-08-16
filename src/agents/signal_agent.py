"""LLM validation gate — the last step before a quant-selected opportunity
becomes a trade. Never sees raw candles and never picks direction itself:
Feature Engine + OpportunityScorer (src/features/) already did that
deterministically. This module only asks the LLM to accept or reject a
curated summary of an already-ranked candidate.

Falls back to a safe "reject" if the model doesn't return parseable JSON,
or if every model in the chain fails outright — a broken/unreachable LLM
gate must not silently let a trade through (fail-closed, unlike the old
get_signal's "flat" default, since the LLM is a gate now, not the primary
decision-maker)."""

from __future__ import annotations

import json

from src.groq_client import AllModelsFailedError, ModelUsageEvent, chat
from src.lenient_json import parse_llm_json

_ENTRY_QUESTION = (
    "A quantitative opportunity scorer has already ranked this symbol as a "
    "top candidate among everything scanned this cycle. Should this trade "
    "be accepted?"
)
_EXIT_QUESTION = (
    "This position is currently held and its quantitative opportunity "
    "score has dropped below the exit threshold. Should it be closed now?"
)

_RESPONSE_FORMAT = (
    'Respond as JSON only: {"decision": "accept|reject", "confidence": 0-100, '
    '"reasoning": "...", "risks": "...", "confidence_delta": -1 to 1, '
    '"expected_duration": "...", "invalidation_point": "..."}. "confidence" is '
    "your own numeric confidence in this decision (0=no confidence, 100=certain) — "
    "it is blended with this symbol's historical win rate (see historical_context "
    "in the data above, when present) to produce the final trade confidence, so "
    "answer it independently of whatever historical_context already shows."
)


def _messages_for(opportunity_summary: dict, strategy_prompt: str, context: str) -> list[dict]:
    question = _ENTRY_QUESTION if context == "entry" else _EXIT_QUESTION
    return [
        {"role": "system", "content": strategy_prompt},
        {
            "role": "user",
            "content": f"{json.dumps(opportunity_summary)}\n\n{question} {_RESPONSE_FORMAT}",
        },
    ]


def validate_opportunity(
    opportunity_summary: dict, strategy_prompt: str, context: str
) -> tuple[dict, list[ModelUsageEvent]]:
    try:
        content, events = chat(_messages_for(opportunity_summary, strategy_prompt, context))
    except AllModelsFailedError as e:
        return {"decision": "reject", "reasoning": f"all models failed: {e}"}, e.events

    try:
        verdict = parse_llm_json(content)
    except (json.JSONDecodeError, TypeError):
        verdict = {"decision": "reject", "reasoning": f"unparseable model response: {content!r}"}
    return verdict, events
