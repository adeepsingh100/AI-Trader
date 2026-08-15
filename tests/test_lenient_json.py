import json

import pytest

from src.lenient_json import parse_llm_json


def test_parses_clean_json():
    assert parse_llm_json('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_strips_markdown_code_fence():
    text = '```json\n{"a": 1}\n```'
    assert parse_llm_json(text) == {"a": 1}


def test_strips_line_comments_outside_strings():
    text = '{\n  "stop_loss_pct": 0.005, // 0.5% of entry price\n  "ok": true\n}'
    assert parse_llm_json(text) == {"stop_loss_pct": 0.005, "ok": True}


def test_does_not_strip_double_slash_inside_string_value():
    text = '{"url": "https://example.com/path"}'
    assert parse_llm_json(text) == {"url": "https://example.com/path"}


def test_strips_block_comments():
    text = '{\n  "a": 1, /* explanatory note */\n  "b": 2\n}'
    assert parse_llm_json(text) == {"a": 1, "b": 2}


def test_strips_trailing_comma_before_close_brace():
    assert parse_llm_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_strips_trailing_comma_before_close_bracket():
    assert parse_llm_json('{"a": [1, 2, 3,]}') == {"a": [1, 2, 3]}


def test_genuinely_broken_json_still_raises():
    with pytest.raises(json.JSONDecodeError):
        parse_llm_json("not json at all")


def test_reproduces_actual_v4_evolution_failure():
    # Trimmed reproduction of the real payload that broke evolution's
    # json.loads: inline `//` comments after numeric params_json values.
    text = """{
  "prompt_text": "You are a disciplined intraday crypto strategist.",
  "params_json": {
    "stop_loss_pct": 0.005, // 0.5% of entry price
    "take_profit_pct": 0.015, // 1.5% target (RRR ~ 3:1)
    "confidence_threshold": 0.7,
    "max_daily_exposure_pct": 0.10, // 10% of equity per day
  },
  "notes": "Lowered risk exposure and tightened spread limits."
}"""
    result = parse_llm_json(text)
    assert result["params_json"]["stop_loss_pct"] == 0.005
    assert result["params_json"]["take_profit_pct"] == 0.015
    assert result["params_json"]["max_daily_exposure_pct"] == 0.10
    assert "Lowered risk exposure" in result["notes"]
