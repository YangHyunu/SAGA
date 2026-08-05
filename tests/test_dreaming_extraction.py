"""B-1+B-2 단일 구조화 출력 (스펙 §3.2) — 스키마·프롬프트·파서."""
import pytest

from dreaming.dreamer import build_dream_prompt, parse_extraction
from dreaming.records import Actor, Fact

_RAW = [{"turn_number": 0, "user_text": "포션 얼마야?",
         "assistant_text": "50골드다.", "user_hash": "u0", "assistant_hash": "a0"}]


def test_parse_plain_json():
    ext = parse_extraction('{"facts": [{"claim": "포션은 50골드다", "evidence_turn": 0}]}')
    assert ext.facts[0].claim == "포션은 50골드다"
    assert ext.facts[0].action == "ADD"          # 기본값
    assert ext.episodes == [] and ext.commits == [] and ext.actors == []


def test_parse_fenced_json():
    ext = parse_extraction('```json\n{"commits": [{"slot": "소지금", "op": "set", '
                           '"value": 450, "turn": 0}]}\n```')
    assert ext.commits[0].slot == "소지금"
    assert ext.commits[0].value == 450


def test_parse_garbage_raises():
    with pytest.raises(Exception):
        parse_extraction("죄송합니다, JSON을 만들 수 없습니다.")


def test_prompt_contains_turns_existing_ids_and_is_deterministic():
    fact = Fact(claim="리사는 상인이다", status="confirmed")
    actor = Actor(names=["리사"], tier="main")
    args = (_RAW, [fact], {"소지금": 450}, [actor])
    system, user = build_dream_prompt(*args)
    assert "JSON" in system                       # 출력 형식 지시
    assert "포션 얼마야?" in user                  # 원문 턴
    assert fact.id in user                        # UPDATE/DELETE 타겟팅용 id 노출
    assert "소지금" in user and "리사" in user
    assert build_dream_prompt(*args) == (system, user)   # 결정론
