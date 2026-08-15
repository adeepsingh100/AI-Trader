"""Best-effort cleanup for JSON an LLM claims to have written but didn't
quite: markdown code fences, // and /* */ comments outside strings, and
trailing commas before a closing bracket. json.loads() runs on the
result — genuinely broken JSON still raises the same JSONDecodeError
callers already catch."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _strip_comments(text: str) -> str:
    out = []
    in_string = False
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if c == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if c == '"':
                in_string = False
            i += 1
            continue

        if c == '"':
            in_string = True
            out.append(c)
            i += 1
            continue

        if c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\n\r":
                i += 1
            continue

        if c == "/" and i + 1 < n and text[i + 1] == "*":
            i += 2
            while i + 1 < n and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2
            continue

        out.append(c)
        i += 1
    return "".join(out)


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_llm_json(text: str) -> Any:
    cleaned = _strip_trailing_commas(_strip_comments(_strip_code_fence(text)))
    return json.loads(cleaned)
