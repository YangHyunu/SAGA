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


# ------------------------------------------------------------------ #
# B-3: 검증·적용
# ------------------------------------------------------------------ #

from dreaming.dreamer import (DreamExtraction, ExtractedNumber,
                              apply_extraction, verify_numbers)
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

_RAW_BY_TURN = {0: {"turn_number": 0, "user_text": "포션 얼마야?",
                    "assistant_text": "50골드다. 잔액은 1,450골드.",
                    "user_hash": "u0", "assistant_hash": "a0"}}


def _store(tmp_path):
    return MemoryStore(JsonDirStorage(tmp_path), "sess1")


def test_verify_numbers_literal_match_with_comma():
    text = "50골드다. 잔액은 1,450골드."
    assert verify_numbers([ExtractedNumber(name="가격", value=50)], text)
    assert verify_numbers([ExtractedNumber(name="잔액", value=1450)], text)
    assert not verify_numbers([ExtractedNumber(name="가격", value=999)], text)


def test_verify_numbers_korean_numerals():
    # 실카드 실측: 원문이 한글 수사면 아라비아 검증이 실패해 과잉 격리됐다
    assert verify_numbers([ExtractedNumber(name="개수", value=3)],
                          "육포 세 개를 건넸다")
    assert verify_numbers([ExtractedNumber(name="나이", value=27)],
                          "저는 스물일곱입니다")
    assert verify_numbers([ExtractedNumber(name="소지금", value=300)],
                          "삼백 푼이 전부요")
    assert not verify_numbers([ExtractedNumber(name="가격", value=40)],
                              "쉰 골드다")


def test_add_verified_fact_becomes_confirmed(tmp_path):
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 0,
         "numbers": [{"name": "가격", "value": 50, "unit": "골드"}]}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    f = store.list_facts()[0]
    assert f.status == "confirmed"
    assert f.evidence[0].pair_hash == "u0"


def test_add_unverified_number_stays_provisional(tmp_path):
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 999골드다", "evidence_turn": 0,
         "numbers": [{"name": "가격", "value": 999}]}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    assert store.list_facts()[0].status == "provisional"


def test_update_builds_version_chain(tmp_path):
    store = _store(tmp_path)
    old = Fact(claim="포션은 60골드다", status="confirmed")
    store.save_fact(old)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 0, "action": "UPDATE",
         "target_fact_id": old.id,
         "numbers": [{"name": "가격", "value": 50}]}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    assert store.get_fact(old.id).status == "superseded"
    live = store.list_facts()          # superseded 제외
    assert len(live) == 1
    assert live[0].supersedes == old.id


def test_user_edited_target_is_protected(tmp_path):
    store = _store(tmp_path)
    edited = Fact(claim="포션은 40골드다 (유저 수정)", status="confirmed",
                  user_edited=True)
    store.save_fact(edited)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 0, "action": "UPDATE",
         "target_fact_id": edited.id, "numbers": [{"name": "가격", "value": 50}]},
        {"claim": "", "evidence_turn": 0, "action": "DELETE",
         "target_fact_id": edited.id}]})
    report = apply_extraction(store, ext, _RAW_BY_TURN)
    assert store.get_fact(edited.id).status == "confirmed"   # 원본 무사
    kinds = {f.status for f in store.list_facts()}
    assert "pending_contradiction" in kinds                  # 모순 관찰로 기록
    assert report["blocked"] == 2


def test_commit_verified_applies_unverified_quarantined(tmp_path):
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"commits": [
        {"slot": "소지금", "op": "set", "value": 1450, "turn": 0},
        {"slot": "소지금", "op": "add", "value": -777, "turn": 0}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    assert store.current_state() == {"소지금": 1450.0}   # -777은 원문에 없음 → 격리


def test_actor_upsert_merges_aliases(tmp_path):
    store = _store(tmp_path)
    store.save_actor(Actor(names=["리사"], profile="시장 상인", tier="support"))
    ext = DreamExtraction.model_validate({"actors": [
        {"names": ["리사", "Lisa"], "profile": "시장 상인, 밀수 연루", "tier": "main"}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    actors = store.list_actors()
    assert len(actors) == 1
    assert set(actors[0].names) == {"리사", "Lisa"}
    assert actors[0].tier == "main"


def test_garbage_target_fact_id_degrades_to_add(tmp_path):
    # 실카드 실측: Flash가 target_fact_id에 id 대신 문장을 넣음 —
    # 사이클을 죽이지 말고 ADD로 강등해야 한다 (fail-open)
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "개선문은 석조 건축물이다", "evidence_turn": 0,
         "action": "UPDATE",
         "target_fact_id": "개선문은 하늘을 가로지르는 거대한 석조 건축물이다."},
        {"claim": "지울 수 없는 대상", "evidence_turn": 0,
         "action": "DELETE", "target_fact_id": "존재하지/않는 아이디"}]})
    report = apply_extraction(store, ext, _RAW_BY_TURN)   # 예외 없이 통과
    assert report["facts"] == 1
    assert store.list_facts()[0].claim == "개선문은 석조 건축물이다"


def test_duplicate_add_is_skipped(tmp_path):
    # 실카드 실측: 같은 claim이 confirmed로 공존 (NOOP 위반, HANDOFF §2) —
    # 프롬프트 지시는 무시될 수 있으니 apply가 결정론 멱등 가드로 막는다
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 0,
         "numbers": [{"name": "가격", "value": 50}]}]})
    r1 = apply_extraction(store, ext, _RAW_BY_TURN)
    r2 = apply_extraction(store, ext, _RAW_BY_TURN)
    assert len(store.list_facts()) == 1
    assert r1["facts"] == 1 and r2["facts"] == 0
    assert r2["deduped"] == 1


def test_scene_fact_is_dropped(tmp_path):
    # 실카드 실측: "날씨는 맑다"·표정 묘사가 [확정 사실]로 영구 주입됐다
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "리사가 씩 웃었다", "evidence_turn": 0, "kind": "scene"},
        {"claim": "날씨는 맑다", "evidence_turn": 0, "kind": "scene"},
        {"claim": "포션은 50골드다", "evidence_turn": 0,
         "numbers": [{"name": "가격", "value": 50}]}]})
    report = apply_extraction(store, ext, _RAW_BY_TURN)
    assert report["scene_dropped"] == 2
    assert [f.claim for f in store.list_facts()] == ["포션은 50골드다"]


def test_update_target_miss_logs_warning(tmp_path, caplog):
    import logging
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 0, "action": "UPDATE",
         "target_fact_id": "0" * 32}]})
    with caplog.at_level(logging.WARNING, logger="dreaming.dreamer"):
        apply_extraction(store, ext, _RAW_BY_TURN)
    assert any("UPDATE target miss" in r.message for r in caplog.records)


def test_prompt_rules_for_quality():
    system, _ = build_dream_prompt(_RAW, [], {}, [])
    assert "kind" in system                 # scene 게이트 스키마
    assert "op=set" in system               # add-델타 선호 규칙 (결함 E)
    assert "NOOP" in system


def test_negative_add_delta_verifies_by_abs(tmp_path):
    # "50을 치렀다" → add -50: 원문엔 양수 50만 있다 — abs로 대조해야
    # add-델타 유도(결함 E 수정)가 격리당하지 않는다
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"commits": [
        {"slot": "소지금", "op": "add", "value": -50, "turn": 0}]})
    raw = {0: {"turn_number": 0, "user_text": "50 골드를 치렀다.",
               "assistant_text": "받았소.", "user_hash": "u0",
               "assistant_hash": "a0"}}
    apply_extraction(store, ext, raw)
    assert store.list_commits()[0].status == "applied"


def test_invalid_evidence_turn_logs_and_stays_provisional(tmp_path, caplog):
    # 실측 4% (HANDOFF §1.2): 스냅샷 밖 evidence_turn — 조용한 누락 금지, 관측만
    import logging
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"facts": [
        {"claim": "포션은 50골드다", "evidence_turn": 99,
         "numbers": [{"name": "가격", "value": 50}]}]})
    with caplog.at_level(logging.WARNING, logger="dreaming.dreamer"):
        apply_extraction(store, ext, _RAW_BY_TURN)
    f = store.list_facts()[0]
    assert f.status == "provisional" and f.evidence == []
    assert any("evidence_turn" in r.message for r in caplog.records)


def test_episode_range_from_raw_hashes(tmp_path):
    store = _store(tmp_path)
    ext = DreamExtraction.model_validate({"episodes": [
        {"start_turn": 0, "end_turn": 0, "title": "포션 흥정",
         "summary": "가격을 물었다.", "open_threads": ["잔액의 출처"]},
        {"start_turn": 5, "end_turn": 9, "title": "없는 턴", "summary": "스킵"}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    eps = store.list_episodes()
    assert len(eps) == 1
    assert eps[0].range_start == "u0" and eps[0].range_end == "u0"
