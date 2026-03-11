"""Unit tests for saga/utils/parsers.py — strip_state_block, narrative formatting, JSON parsing."""
import os
import tempfile
import pytest
from saga.utils.parsers import (
    strip_state_block,
    format_turn_narrative,
    parse_llm_json,
)


# ============================================================
# strip_state_block
# ============================================================

class TestStripStateBlock:
    def test_removes_state_block(self):
        text = '''이야기 텍스트입니다.

```state
location: 마을
mood: 평온
```'''
        result = strip_state_block(text)
        assert "```state" not in result
        assert "location:" not in result
        assert "이야기 텍스트입니다." in result

    def test_no_block_unchanged(self):
        text = "Just a normal response."
        assert strip_state_block(text) == text

    def test_preserves_text_before_and_after(self):
        text = '''Before block.

```state
location: 숲
```

After block.'''
        result = strip_state_block(text)
        assert "Before block." in result
        assert "After block." in result

    def test_double_backtick_removed(self):
        text = '''텍스트
``state
location: 여기
``'''
        result = strip_state_block(text)
        assert "location" not in result


# ============================================================
# format_turn_narrative
# ============================================================

class TestFormatTurnNarrative:
    def test_basic_formatting(self):
        state = {"location": "던전", "npc_met": ["고블린"], "items_gained": ["검"]}
        result = format_turn_narrative(5, "공격!", "캐릭터가 싸운다.", state)
        assert "Turn 5" in result
        assert "던전" in result
        assert "고블린" in result
        assert "검" in result

    def test_long_response_truncated(self):
        long_response = "A" * 800
        state = {"location": "여기"}
        result = format_turn_narrative(1, "Hi", long_response, state)
        # format_turn_narrative truncates clean_response to 500 then takes [:300] for output
        assert len(result) < 600

    def test_event_trigger_in_header(self):
        state = {"location": "성", "event_trigger": "보스 등장"}
        result = format_turn_narrative(10, "입장", "보스가 나타났다.", state)
        assert "보스 등장" in result

    def test_empty_state_block(self):
        result = format_turn_narrative(1, "Hi", "Hello!", {})
        assert "Turn 1" in result

    def test_state_block_stripped_from_response(self):
        response = '''이야기 내용```state
location: 마을
```'''
        state = {"location": "마을"}
        result = format_turn_narrative(1, "Hi", response, state)
        assert "```state" not in result


# ============================================================
# parse_llm_json
# ============================================================

class TestParseLlmJson:
    def test_direct_json(self):
        """Valid JSON string parses directly."""
        assert parse_llm_json('{"key": "value"}') == {"key": "value"}

    def test_code_block_json(self):
        """JSON inside ```json ... ``` code block is extracted."""
        text = 'Some preamble\n```json\n{"a": 1, "b": 2}\n```\ntrailing text'
        assert parse_llm_json(text) == {"a": 1, "b": 2}

    def test_code_block_no_lang(self):
        """JSON inside ``` ... ``` (no language tag) is extracted."""
        text = '```\n{"x": true}\n```'
        assert parse_llm_json(text) == {"x": True}

    def test_balanced_brace_with_surrounding_text(self):
        """JSON embedded in natural language text is extracted via brace matching."""
        text = 'Here is the result: {"status": "ok", "count": 3} and that is all.'
        assert parse_llm_json(text) == {"status": "ok", "count": 3}

    def test_nested_braces(self):
        """Nested JSON objects are handled correctly by balanced brace matching."""
        text = 'Response: {"outer": {"inner": 42}, "list": [1, 2]}'
        result = parse_llm_json(text)
        assert result == {"outer": {"inner": 42}, "list": [1, 2]}

    def test_empty_input_returns_none(self):
        assert parse_llm_json("") is None
        assert parse_llm_json(None) is None

    def test_no_json_returns_none(self):
        assert parse_llm_json("This is just plain text with no JSON.") is None

    def test_invalid_json_returns_none(self):
        """Malformed JSON returns None instead of raising."""
        assert parse_llm_json("{invalid json}") is None

    def test_array_returns_none(self):
        """Top-level JSON arrays are not returned (only dicts)."""
        assert parse_llm_json('[1, 2, 3]') is None

    def test_greedy_regex_avoided(self):
        """Balanced brace matching picks the first valid object."""
        text = '{"a": 1} some text {"b": 2}'
        result = parse_llm_json(text)
        assert result == {"a": 1}
