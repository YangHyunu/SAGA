"""Unit tests for saga/utils/parsers.py — state block parsing, list parsing, narrative formatting."""
import os
import tempfile
import pytest
from saga.utils.parsers import (
    parse_state_block,
    strip_state_block,
    format_turn_narrative,
    parse_llm_json,
    _parse_list,
    _parse_transfer_list,
    _parse_relationship_changes,
)


# ============================================================
# parse_state_block
# ============================================================

class TestParseStateBlock:
    def test_normal_triple_backtick(self):
        text = '''Some narrative text here...
```state
location: 던전 입구
location_moved: true
hp_change: -5
items_gained: [검, 방패]
items_lost: []
items_transferred: []
npc_met: [고블린 왕]
npc_separated: []
relationship_changes: []
mood: 긴장
event_trigger: null
notes: 보스 방 근처
```'''
        result = parse_state_block(text)
        assert result is not None
        assert result["location"] == "던전 입구"
        assert result["location_moved"] is True
        assert result["hp_change"] == -5
        assert result["items_gained"] == ["검", "방패"]
        assert result["npc_met"] == ["고블린 왕"]
        assert result["mood"] == "긴장"
        assert result["event_trigger"] is None
        assert result["notes"] == "보스 방 근처"

    def test_double_backtick_variant(self):
        text = '''Response text
``state
location: 마을
mood: 평온
``'''
        result = parse_state_block(text)
        assert result is not None
        assert result["location"] == "마을"
        assert result["mood"] == "평온"

    def test_missing_block_returns_none(self):
        text = "Just a normal response without any state block."
        assert parse_state_block(text) is None

    def test_empty_block_returns_none(self):
        text = '''```state
```'''
        assert parse_state_block(text) is None

    def test_partial_fields(self):
        text = '''```state
location: 숲
mood: 호기심
```'''
        result = parse_state_block(text)
        assert result is not None
        assert result["location"] == "숲"
        assert result["mood"] == "호기심"
        assert "hp_change" not in result

    def test_boolean_true_variants(self):
        for val in ("true", "yes", "1"):
            text = f'''```state
location_moved: {val}
location: 어딘가
```'''
            result = parse_state_block(text)
            assert result["location_moved"] is True, f"Failed for {val}"

    def test_boolean_false_variants(self):
        for val in ("false", "no", "0"):
            text = f'''```state
location_moved: {val}
location: 어딘가
```'''
            result = parse_state_block(text)
            assert result["location_moved"] is False, f"Failed for {val}"

    def test_hp_change_non_numeric(self):
        text = '''```state
hp_change: 약간
location: 여기
```'''
        result = parse_state_block(text)
        assert result["hp_change"] == 0

    def test_event_trigger_null_variants(self):
        for val in ("null", "none", ""):
            text = f'''```state
event_trigger: {val}
location: 여기
```'''
            result = parse_state_block(text)
            assert result["event_trigger"] is None, f"Failed for '{val}'"

    def test_event_trigger_with_value(self):
        text = '''```state
event_trigger: 보스 등장
location: 성
```'''
        result = parse_state_block(text)
        assert result["event_trigger"] == "보스 등장"

    def test_state_block_in_middle_of_text(self):
        text = '''긴 서사 텍스트가 여기에...

캐릭터가 움직였다.

```state
location: 새로운 장소
mood: 놀람
```

이 텍스트는 state 블록 뒤에 있다.'''
        result = parse_state_block(text)
        assert result is not None
        assert result["location"] == "새로운 장소"


# ============================================================
# _parse_list
# ============================================================

class TestParseList:
    def test_bracketed_items(self):
        assert _parse_list("[검, 방패]") == ["검", "방패"]

    def test_empty_string(self):
        assert _parse_list("") == []

    def test_none_value(self):
        assert _parse_list("none") == []

    def test_korean_none(self):
        assert _parse_list("없음") == []

    def test_empty_brackets(self):
        assert _parse_list("[]") == []

    def test_quoted_items(self):
        assert _parse_list('"검", "방패"') == ["검", "방패"]

    def test_single_item(self):
        assert _parse_list("검") == ["검"]

    def test_whitespace_handling(self):
        assert _parse_list("  검 , 방패  ,  활  ") == ["검", "방패", "활"]


# ============================================================
# _parse_transfer_list
# ============================================================

class TestParseTransferList:
    def test_json_format(self):
        result = _parse_transfer_list('[{"item": "검", "to": "고블린"}]')
        assert len(result) == 1
        assert result[0]["item"] == "검"
        assert result[0]["to"] == "고블린"

    def test_regex_fallback(self):
        result = _parse_transfer_list("{item: 검, to: 고블린}")
        assert len(result) == 1
        assert result[0]["item"] == "검"
        assert result[0]["to"] == "고블린"

    def test_invalid_returns_empty(self):
        assert _parse_transfer_list("not valid") == []

    def test_empty_string(self):
        assert _parse_transfer_list("") == []

    def test_multiple_transfers(self):
        result = _parse_transfer_list('[{"item": "검", "to": "A"}, {"item": "활", "to": "B"}]')
        assert len(result) == 2


# ============================================================
# _parse_relationship_changes
# ============================================================

class TestParseRelationshipChanges:
    def test_valid_json(self):
        result = _parse_relationship_changes('[{"from": "주인공", "to": "엘프", "type": "trust", "delta": 10}]')
        assert len(result) == 1
        assert result[0]["from"] == "주인공"
        assert result[0]["delta"] == 10

    def test_invalid_returns_empty(self):
        assert _parse_relationship_changes("not json") == []

    def test_empty_returns_empty(self):
        assert _parse_relationship_changes("") == []

    def test_single_quotes_converted(self):
        result = _parse_relationship_changes("[{'from': 'A', 'to': 'B', 'type': 'ally', 'delta': 5}]")
        assert len(result) == 1


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
