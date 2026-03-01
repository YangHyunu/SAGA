"""Unit tests for saga/utils/parsers.py — state block parsing, list parsing, narrative formatting."""
import os
import tempfile
import pytest
from saga.utils.parsers import (
    parse_state_block,
    strip_state_block,
    format_turn_narrative,
    parse_characters_md,
    parse_lorebook_md,
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
# parse_characters_md
# ============================================================

class TestParseCharactersMd:
    def test_basic_character(self, tmp_path):
        md = tmp_path / "CHARACTERS.md"
        md.write_text("## 아린\n- player: true\n- hp: 85\n- max_hp: 100\n- location: 마을\n- traits: 용감, 친절\n- mood: 기쁨\n", encoding="utf-8")
        result = parse_characters_md(str(md))
        assert len(result) == 1
        c = result[0]
        assert c["name"] == "아린"
        assert c["is_player"] is True
        assert c["hp"] == 85
        assert c["max_hp"] == 100
        assert c["location"] == "마을"
        assert c["traits"] == ["용감", "친절"]
        assert c["mood"] == "기쁨"

    def test_multiple_characters(self, tmp_path):
        md = tmp_path / "CHARACTERS.md"
        md.write_text("## 주인공\n- player: true\n\n## 고블린\n- hp: 50\n- location: 동굴\n", encoding="utf-8")
        result = parse_characters_md(str(md))
        assert len(result) == 2
        assert result[0]["name"] == "주인공"
        assert result[1]["name"] == "고블린"

    def test_file_not_found(self):
        assert parse_characters_md("/nonexistent/path.md") == []

    def test_korean_keys(self, tmp_path):
        md = tmp_path / "CHARACTERS.md"
        md.write_text("## NPC\n- 플레이어: 예\n- 위치: 숲\n- 성격: 조용, 차분\n- 기분: 슬픔\n- 최대hp: 200\n", encoding="utf-8")
        result = parse_characters_md(str(md))
        c = result[0]
        assert c["is_player"] is True
        assert c["location"] == "숲"
        assert c["traits"] == ["조용", "차분"]
        assert c["mood"] == "슬픔"
        assert c["max_hp"] == 200

    def test_custom_fields(self, tmp_path):
        md = tmp_path / "CHARACTERS.md"
        md.write_text("## 캐릭터\n- 종족: 엘프\n- 직업: 마법사\n", encoding="utf-8")
        result = parse_characters_md(str(md))
        assert result[0]["custom"]["종족"] == "엘프"
        assert result[0]["custom"]["직업"] == "마법사"

    def test_defaults(self, tmp_path):
        md = tmp_path / "CHARACTERS.md"
        md.write_text("## 기본캐릭터\n", encoding="utf-8")
        result = parse_characters_md(str(md))
        c = result[0]
        assert c["is_player"] is False
        assert c["hp"] == 100
        assert c["max_hp"] == 100
        assert c["location"] == "unknown"
        assert c["mood"] == "neutral"


# ============================================================
# parse_lorebook_md
# ============================================================

class TestParseLorebookMd:
    def test_basic_entry(self, tmp_path):
        md = tmp_path / "LOREBOOK.md"
        md.write_text("## 엘프 왕국\n- type: 장소\n- layer: A2\n- tags: 엘프, 왕국\n\n오래된 엘프 왕국이다.\n", encoding="utf-8")
        result = parse_lorebook_md(str(md))
        assert len(result) == 1
        e = result[0]
        assert e["name"] == "엘프 왕국"
        assert e["type"] == "장소"
        assert e["layer"] == "A2"
        assert e["tags"] == ["엘프", "왕국"]
        assert "오래된 엘프 왕국이다." in e["text"]

    def test_multiple_entries(self, tmp_path):
        md = tmp_path / "LOREBOOK.md"
        md.write_text("## 마법\n설명1\n\n## 검술\n설명2\n", encoding="utf-8")
        result = parse_lorebook_md(str(md))
        assert len(result) == 2
        assert result[0]["name"] == "마법"
        assert result[1]["name"] == "검술"

    def test_file_not_found(self):
        assert parse_lorebook_md("/nonexistent/path.md") == []

    def test_korean_keys(self, tmp_path):
        md = tmp_path / "LOREBOOK.md"
        md.write_text("## 항목\n- 타입: 인물\n- 레이어: A3\n- 태그: 전사, 영웅\n", encoding="utf-8")
        result = parse_lorebook_md(str(md))
        e = result[0]
        assert e["type"] == "인물"
        assert e["layer"] == "A3"
        assert e["tags"] == ["전사", "영웅"]

    def test_defaults(self, tmp_path):
        md = tmp_path / "LOREBOOK.md"
        md.write_text("## 기본항목\n내용만 있음\n", encoding="utf-8")
        result = parse_lorebook_md(str(md))
        e = result[0]
        assert e["type"] == "lore"
        assert e["layer"] == "A1"
        assert e["tags"] == []
