# Dreaming Plan 4 — 청크 압축(B-4) + Tier 승격 + BP2 + 품질 수정 6건

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 꿈이 확정한 에피소드를 결정론적 청크로 조립해 프리픽스를 압축하고(BP2 마킹 포함, 스펙 §5·§6), 실카드 30턴 벤치에서 실측된 Dreamer 품질 결함 6건을 수정한다.

**Architecture:** 압축은 2단계다 — (1) 꿈 사이클 끝(B-4)에 `build_compression`이 에피소드를 Tier1/Tier2 청크 메시지 리스트로 조립해 `{session}/compression`에 플랜 문서로 저장하고, (2) 동기 경로가 매 요청 그 플랜을 읽어 히스토리 선두 K pair를 청크로 **위치 기반** 치환한다. 플랜은 꿈에서만 바뀌므로(유휴 = TTL 경과 = 캐시 사망 시점, §6.3) 재압축의 한계 캐시 비용은 0이고, 꿈 사이 구간에서는 같은 입력 → 같은 바이트다. 품질 수정은 전부 결정론 로직(한글 수사 사전, commit abs 대조, claim dedup)이거나 프롬프트 규칙이다.

**Tech Stack:** Python 3 표준 라이브러리 + 기존 의존성(pydantic v2, fastapi, httpx)만. 새 의존성 금지.

## Global Constraints

- 실행은 전부 `python3` (시스템에 `python` 없음). 테스트: `python3 -m pytest`.
- 저장은 KV 문서 샤드 단일 — SQL·SQLite·외부 DB 금지 (스펙 §8).
- `saga` 패키지 import는 `hash_text` / `extract_pairs` / `classify` 3개 순수 함수만. `saga/` diff 0 유지.
- 동기 경로 = 턴당 LLM 0콜. 압축은 결정론적 조립 — 같은 입력 → 같은 바이트 (스펙 §6.1).
- 압축 플랜 변경은 꿈 사이클 안에서만. 동기 경로는 플랜을 읽기만 한다 (스펙 §6.3 TTL 창구).
- fail-open: 플랜-히스토리 불일치(짧은 히스토리 등) → 원본 무가공 전달. 예외로 채팅을 막지 않는다 (스펙 §2.6).
- pytest-asyncio 금지 — 비동기 테스트는 `asyncio.run(...)` 패턴.
- API 키 값 출력·커밋 절대 금지. `dreaming_data/`·`.env`는 gitignored 유지.
- `pytest | tail` 같은 파이프는 exit code를 삼킨다 — 테스트 실행과 exit code 확인을 분리할 것.
- **worldstate.replay 비숫자 슬롯 add 크래시 방어는 별도 세션(eloquent-dirac-7fa062) 담당 — 이 플랜에서 구현 금지.** Task 3의 add-델타 유도가 add 커밋을 늘리므로, 실행 시점에 그 수정 머지 여부를 확인할 것.
- **로어북 keyed delta 프리픽스 파손 대책(lore_shift)은 ba42ff 워크트리 담당 — 비스코프.** BP2·압축의 라이브 효과 측정은 lore 대책 머지 후 또는 `--no-keyed-lore` 대조군에서만 의미 있다 (HANDOFF §4: churn 턴 히트 16.9% ↔ 비-churn 97.8% — 노이즈가 압축 효과를 묻는다).
- 비스코프(후속 플랜): 총량 상한 ~30% 예산 계산, keyExcerpts 비압축 보존(§6.2), 유휴 전 예산 임계 중도 압축(§6.3 마지막 줄), Tier3 시놉시스, pinned 승격 지연, 임베딩 검색.

## 실카드 벤치 실측 결함 (이 플랜이 고치는 것)

> **2026-08-06 검증 세션(ba42ff) 감사 반영** — 상세: `docs/dreaming/HANDOFF-plan4-verification.md`.
> 근거 데이터 2세션 중 card-realm-30은 같은 세션 ID 2회 실행으로 **오염**이었다.
> 정상 세션 재검증 결과 B 기각·C 축소, 나머지 확실. (soyeon dup 1/87, realm dup 10/39 — 본 세션에서 재현 확인함.)

| # | 결함 | 판정 | 수정 |
|---|---|---|---|
| A | 한글 수사 미검증 | **확실** — 정상 세션에서 "스물일곱" 등 provisional 격리 | Task 1: 수사 사전 |
| B | evidence_turn 몰림 | **기각** — 증거 체인 건강 (30/30 턴 커버, dangling 0; "전 레코드 None"은 Evidence 모델에 없는 필드를 읽은 감사 아티팩트). 실결함은 무효 evidence_turn 시 조용한 `evidence=[]` 4%뿐 | Task 2: 재탐색 폐기 → abs 대조 + 관측 로그로 축소 |
| C | fact 중복 re-ADD | **축소** — 34%는 오염 아티팩트, 정상 세션 ~1%. 단 같은 claim 둘 다 confirmed 공존 사례 실존 (NOOP 위반) | Task 3: 싼 멱등 가드로 존치, 화력은 D로 |
| D | 장면 묘사가 [확정 사실] 주입 | **확실** — "날씨는 맑다"·"온도 -3°C"·"제의는 흰색" confirmed. exact-match가 못 잡는 준중복 연쇄("창백해졌다/더욱 창백해졌다/식은 땀…")도 이 결함의 증상 | Task 3: kind=scene 게이트 |
| E | commit set-계산값 | **확실** — 정상 세션에서 `set 250` 격리 → stale `set 300` 재적용 재현 | Task 3: add-델타 프롬프트 + Task 2의 abs 검증 |
| F | 리뷰어 🟡 3건 | 코드 리뷰 (데이터 무관) | Task 3(로그)·Task 4 |

## File Structure

- Create: `dreaming/numerals.py` — 한국어 수사 표기 생성 (순수 함수, 의존 없음)
- Create: `dreaming/chunks.py` — Tier1/Tier2 템플릿 조립 + 압축 플랜 build/apply (스펙 §6)
- Modify: `dreaming/dreamer.py` — verify_numbers 수사 대조, commit abs 대조·evidence 관측 로그, dedup·scene 게이트, 프롬프트 규칙, 락 원자화, B-4 배선
- Modify: `dreaming/records.py` — Episode에 `start_turn`/`end_turn` 추가
- Modify: `dreaming/marking.py` — `bp2_index` 파라미터 (BP2 마킹)
- Modify: `dreaming/sync.py` — SyncPath에 압축 적용 배선 + demote_after 압축 무효화
- Modify: `dreaming/upstream.py` — to_wire 카피 일관성
- Create: `tests/test_dreaming_numerals.py`, `tests/test_dreaming_chunks.py`
- Modify: `tests/test_dreaming_extraction.py`, `tests/test_dreaming_dreamer.py`, `tests/test_dreaming_demote.py`, `tests/test_dreaming_proxy.py`, `tests/test_dreaming_upstream.py`

---

### Task 1: 한글 수사 검증 — `dreaming/numerals.py`

**Files:**
- Create: `dreaming/numerals.py`
- Modify: `dreaming/dreamer.py` (`verify_numbers`, 148~156행 부근)
- Test: `tests/test_dreaming_numerals.py`, `tests/test_dreaming_extraction.py`

**Interfaces:**
- Consumes: 없음 (순수 함수).
- Produces: `korean_spellings(value: int) -> List[str]` — 1~9999 정수의 한글 표기 목록(한자어 전 구간 + 고유어 1~99), 범위 밖은 `[]`. `verify_numbers`는 아라비아 표기 실패 시 이 목록으로 재대조한다.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_numerals.py` 신규:

```python
"""한국어 수사 표기 생성 — B-3 숫자 검증 보조 (실카드 실측 결함 A)."""
from dreaming.numerals import korean_spellings


def test_native_small_numbers():
    assert {"셋", "세"} <= set(korean_spellings(3))
    assert {"스물", "스무"} <= set(korean_spellings(20))
    assert "스물일곱" in korean_spellings(27)
    assert "쉰" in korean_spellings(50)


def test_sino_numbers():
    assert "삼" in korean_spellings(3)
    assert "이십칠" in korean_spellings(27)
    assert "삼백" in korean_spellings(300)
    assert "천이백삼십사" in korean_spellings(1234)


def test_out_of_range_is_empty():
    assert korean_spellings(0) == []
    assert korean_spellings(-3) == []
    assert korean_spellings(10000) == []
```

그리고 `tests/test_dreaming_extraction.py`의 `test_verify_numbers_literal_match_with_comma` 아래에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_numerals.py tests/test_dreaming_extraction.py -q`
Expected: FAIL — `ModuleNotFoundError: dreaming.numerals` 및 한글 수사 테스트 실패.

- [ ] **Step 3: 구현** — `dreaming/numerals.py` 신규:

```python
"""dreaming/numerals.py — 한국어 수사 표기 생성 (스펙 §3.2 B-3 검증 보조).

실카드 실측: "세 개"·"스물일곱"·"삼백"처럼 원문이 한글 수사면 아라비아
숫자 문자열 검증이 실패해 사실이 과잉 격리된다. 1~9999 정수의 한자어
표기(전 구간)와 고유어 표기(1~99)를 결정론적으로 생성한다.
"""

from __future__ import annotations

from typing import List

_NATIVE_ONES = {1: ["하나", "한"], 2: ["둘", "두"], 3: ["셋", "세"],
                4: ["넷", "네"], 5: ["다섯"], 6: ["여섯"], 7: ["일곱"],
                8: ["여덟"], 9: ["아홉"]}
_NATIVE_TENS = {1: ["열"], 2: ["스물", "스무"], 3: ["서른"], 4: ["마흔"],
                5: ["쉰"], 6: ["예순"], 7: ["일흔"], 8: ["여든"], 9: ["아흔"]}
_SINO_ONES = {1: "일", 2: "이", 3: "삼", 4: "사", 5: "오",
              6: "육", 7: "칠", 8: "팔", 9: "구"}


def _native(n: int) -> List[str]:
    tens, ones = divmod(n, 10)
    if tens == 0:
        return list(_NATIVE_ONES[ones])
    if ones == 0:
        return list(_NATIVE_TENS[tens])
    # 결합형은 기본형만: 스물일곱 (관형형 "스무"는 단독 20에서만)
    return [_NATIVE_TENS[tens][0] + o for o in _NATIVE_ONES[ones]]


def _sino(n: int) -> str:
    s = ""
    for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
        d, n = divmod(n, unit)
        if d:
            s += ("" if d == 1 else _SINO_ONES[d]) + name
    if n:
        s += _SINO_ONES[n]
    return s


def korean_spellings(value: int) -> List[str]:
    if not 1 <= value <= 9999:
        return []
    out = [_sino(value)]
    if value <= 99:
        out += _native(value)
    return out
```

`dreaming/dreamer.py`의 `verify_numbers`를 교체 (import에 `from dreaming.numerals import korean_spellings` 추가):

```python
def verify_numbers(numbers: List[ExtractedNumber], text: str) -> bool:
    """숫자 재검증 (스펙 §3.2 B-3): 원문에 아라비아 표기 또는
    한글 수사(정수 1~9999)로 문자 그대로 있어야 한다."""
    plain = text.replace(",", "")
    for n in numbers:
        v = n.value
        s = str(int(v)) if float(v).is_integer() else str(v)
        if s in plain:
            continue
        if float(v).is_integer() and any(
                k in plain for k in korean_spellings(int(v))):
            continue
        return False
    return True
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_numerals.py tests/test_dreaming_extraction.py -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/numerals.py dreaming/dreamer.py tests/test_dreaming_numerals.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): 한글 수사 숫자 검증 — 세 개/스물일곱/삼백 과잉 격리 수정"
```

---

### Task 2: commit abs 대조 + 무효 evidence_turn 관측 로그

> **B 기각 반영 (HANDOFF §1)** — 증거 체인은 건강하다 (3세션 30/30 턴 커버,
> dangling 0). "증거 턴 재탐색" 설계는 폐기. 남는 실결함 둘만 고친다:
> ① add 음수 델타는 abs 대조 없이는 전부 격리 — 결함 E(add-델타 유도)의 전제,
> ② LLM이 스냅샷 밖 evidence_turn을 주면 조용히 `evidence=[]` 저장 (실측 4%) — 관측 로그.

**Files:**
- Modify: `dreaming/dreamer.py` (`_build_fact` 174~185행, commit 검증 216~229행)
- Test: `tests/test_dreaming_extraction.py`

**Interfaces:**
- Consumes: Task 1의 `verify_numbers`.
- Produces: commit 숫자 검증은 `abs(value)`로 주장 턴 원문과 대조 (재탐색·턴 이동 없음). `_build_fact`는 무효 evidence_turn일 때 `logger.warning` 후 기존대로 `evidence=[]`·provisional.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_extraction.py`에 추가:

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -q`
Expected: FAIL — abs 미대조로 add -50이 pending_contradiction, 경고 로그 부재.

- [ ] **Step 3: 구현** — `dreaming/dreamer.py`.

`_build_fact`에 경고 추가 (동작은 불변 — 관측만):

```python
def _build_fact(ef: ExtractedFact, raw_by_turn: Dict[int, Dict]) -> Fact:
    raw = raw_by_turn.get(ef.evidence_turn)
    if raw is None:
        # 실측 4%: LLM이 스냅샷 밖 evidence_turn을 준다 — 조용한 누락 금지
        logger.warning("[dreamer] invalid evidence_turn=%s: %r",
                       ef.evidence_turn, ef.claim[:40])
    verified = raw is not None and verify_numbers(ef.numbers, _turn_text(raw))
    return Fact(
        claim=ef.claim,
        entities=ef.entities,
        numbers=[TypedNumber(name=n.name, value=n.value, unit=n.unit or None)
                 for n in ef.numbers],
        evidence=[Evidence(pair_hash=raw["user_hash"])] if raw else [],
        learned_by=ef.learned_by,
        status="confirmed" if verified else "provisional",
    )
```

`apply_extraction`의 commit 검증에서 probe를 abs로 교체:

```python
    for ec in ext.commits:
        raw = raw_by_turn.get(ec.turn)
        status = "applied"
        if isinstance(ec.value, (int, float)) and not isinstance(ec.value, bool):
            text = _turn_text(raw) if raw else ""
            # add 음수 델타의 원문 표기는 양수("50을 치렀다") — abs로 대조
            probe = ExtractedNumber(name=ec.slot, value=abs(float(ec.value)))
            if not verify_numbers([probe], text):
                status = "pending_contradiction"
        store.append_commit(StateCommit(
            slot=ec.slot, op=ec.op, value=ec.value, turn=ec.turn,
            evidence=Evidence(pair_hash=raw["user_hash"]) if raw else None,
            status=status,
        ))
        report["commits"] += 1
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_dreamer.py -q; echo "exit=$?"`
Expected: PASS, exit=0 (기존 `test_commit_verified_applies_unverified_quarantined`도 유지 — -777은 원문에 없어 여전히 격리).

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py tests/test_dreaming_extraction.py
git commit -m "fix(dreaming): commit abs 대조(add 음수 델타) + 무효 evidence_turn 관측 로그"
```

---

### Task 3: fact dedup + scene 게이트 + 프롬프트 규칙 (NOOP·add-델타)

> **C 축소 반영 (HANDOFF §2)** — 대량 중복은 오염 아티팩트(정상 세션 ~1%),
> dedup은 아래의 싼 멱등 가드면 충분하다 (confirmed 공존 재-ADD 실존이라 존치).
> 화력은 D(scene 게이트)에 — 준중복 연쇄("창백해졌다/더욱 창백해졌다/식은 땀…")와
> "날씨는 맑다"·"온도 -3°C" 류 confirmed 오염이 전부 이 게이트 담당이다.

**Files:**
- Modify: `dreaming/dreamer.py` (`ExtractedFact`, `_SYSTEM`, `apply_extraction`)
- Test: `tests/test_dreaming_extraction.py`, `tests/test_dreaming_dreamer.py`

**Interfaces:**
- Consumes: Task 2의 fact 루프 구조.
- Produces: `ExtractedFact.kind: Literal["persistent", "scene"] = "persistent"`. `apply_extraction`의 report에 `"deduped"`·`"scene_dropped"` 키 추가. scene fact는 저장하지 않고, 실효 ADD(ADD 또는 무타겟 UPDATE 강등)는 살아있는 claim과 문자 동일하면 스킵. UPDATE 무타겟은 `logger.warning`.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_extraction.py`에 추가:

```python
def test_duplicate_add_is_skipped(tmp_path):
    # 실카드 실측: 꿈 사이클마다 같은 claim이 re-ADD돼 fact가 불어난다 —
    # 프롬프트 NOOP 지시는 무시될 수 있으니 apply가 결정론으로 막는다
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
```

`tests/test_dreaming_dreamer.py`의 `test_dream_full_cycle_advances_cursor`에서 report 단언을 키별 비교로 교체 (이후 태스크가 키를 더 추가해도 안정):

```python
    assert report["facts"] == 1 and report["commits"] == 1
    assert report["actors"] == 1 and report["episodes"] == 1
    assert report["blocked"] == 0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -q`
Expected: FAIL — kind 필드 없음(ValidationError 아님 — 미지 필드는 무시되므로 scene_dropped 키 부재로 실패), deduped 키 부재, 로그 부재, 프롬프트 규칙 부재.

- [ ] **Step 3: 구현** — `dreaming/dreamer.py`.

`ExtractedFact`에 필드 추가:

```python
class ExtractedFact(BaseModel):
    claim: str
    entities: List[str] = []
    numbers: List[ExtractedNumber] = []
    evidence_turn: int
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = "ADD"   # mem0 4분류
    target_fact_id: Optional[str] = None
    learned_by: List[str] = []
    kind: Literal["persistent", "scene"] = "persistent"
```

`_SYSTEM` 스키마의 facts 줄을 다음으로 교체:

```
  "facts": [{"claim": str, "kind": "persistent|scene", "entities": [str],
             "numbers": [{"name": str, "value": number, "unit": str}],
             "evidence_turn": int, "action": "ADD|UPDATE|DELETE|NOOP",
             "target_fact_id": str|null, "learned_by": [str]}],
```

`_SYSTEM` 규칙 블록 끝에 추가:

```
- fact는 시간이 지나도 참인 지속 정보만 (신상·관계·약속·소유·세계 설정).
  일회성 장면 묘사·순간 동작·감정 표현·표정 변화, 그리고 날씨·온도 같은
  일시적 환경 상태는 kind="scene"으로 표시하라.
- [기존 사실]에 이미 있는 내용을 다시 ADD 하지 마라 — action=NOOP.
- commits는 op=add 우선: 원문에 증감량이 명시되면 그 숫자를 그대로 쓴다
  (지출·차감은 음수). op=set은 원문에 최종값 자체가 적힌 경우만.
  계산해서 얻은 값을 넣지 마라 — 누적 계산은 리플레이가 한다.
```

`apply_extraction`의 fact 루프 교체 (report 초기화 포함):

```python
    report = {"facts": 0, "blocked": 0, "deduped": 0, "scene_dropped": 0,
              "commits": 0, "actors": 0, "episodes": 0}

    live_claims = {f.claim.strip() for f in store.list_facts()}
    for ef in ext.facts:
        if ef.action == "NOOP":
            continue
        if ef.kind == "scene":
            # 일회성 묘사는 에피소드 summary의 재료 — fact로 저장하지 않는다
            report["scene_dropped"] += 1
            continue
        target = _lookup_target(store, ef.target_fact_id)
        if ef.action == "DELETE":
            if target is None:
                continue
            if not dreamer_can_modify(target):
                report["blocked"] += 1
                continue
            store.save_fact(target.model_copy(update={"status": "superseded"}))
            continue
        if ef.action == "UPDATE" and target is None and ef.target_fact_id:
            logger.warning("[dreamer] UPDATE target miss: %r — ADD 강등",
                           ef.target_fact_id[:60])
        if target is None and ef.claim.strip() in live_claims:
            # 실카드 실측: NOOP 지시 무시로 매 사이클 re-ADD — 결정론 백스톱
            report["deduped"] += 1
            continue
        new = _build_fact(ef, raw_by_turn)
        if ef.action == "UPDATE" and target is not None:
            if dreamer_can_modify(target):
                old2, new = supersede(target, new)
                store.save_fact(old2)
            else:
                # 유저 편집이 ground truth (스펙 §2.7) — 모순으로 관찰만
                new = new.model_copy(update={"status": "pending_contradiction"})
                report["blocked"] += 1
        store.save_fact(new)
        live_claims.add(new.claim.strip())
        report["facts"] += 1
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_dreamer.py tests/test_dreaming_proxy.py -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py tests/test_dreaming_extraction.py tests/test_dreaming_dreamer.py
git commit -m "fix(dreaming): fact dedup 백스톱 + scene 게이트 + add-델타 프롬프트 규칙"
```

---

### Task 4: 락 원자화 + to_wire 카피 일관성 (리뷰어 🟡)

**Files:**
- Modify: `dreaming/dreamer.py` (`Dreamer.__init__`, `dream`)
- Modify: `dreaming/upstream.py` (`to_wire`)
- Test: `tests/test_dreaming_dreamer.py`, `tests/test_dreaming_upstream.py`

**Interfaces:**
- Consumes: 없음.
- Produces: `Dreamer._active: set[str]` — `asyncio.Lock` 대체 (체크와 점유 사이 await 없음 = 원자). 외부 동작 불변: 꿈꾸는 중 재진입 → `None`. `to_wire`는 무마킹 메시지도 얕은 사본을 반환.

- [ ] **Step 1: 테스트 수정·추가** — `tests/test_dreaming_dreamer.py`의 `test_concurrent_dream_skips`를 내부 구조 변경에 맞춰 교체:

```python
def test_concurrent_dream_skips(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))

    async def scenario():
        d._active.add("sess1")                # 꿈꾸는 중 시뮬레이션
        try:
            return await d.dream("sess1")
        finally:
            d._active.discard("sess1")

    assert asyncio.run(scenario()) is None
    assert asyncio.run(d.dream("sess1")) is not None   # 해제 후 정상 진행
```

`tests/test_dreaming_upstream.py`에 추가:

```python
def test_to_wire_returns_copies_for_unmarked_messages():
    src = [{"role": "user", "content": "hi"}]
    wired = to_wire(src)
    wired[0]["content"] = "changed"
    assert src[0]["content"] == "hi"          # 원본 별칭 반환 금지
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_dreamer.py tests/test_dreaming_upstream.py -q`
Expected: FAIL — `_active` 속성 없음(AttributeError), to_wire 별칭 오염.

- [ ] **Step 3: 구현**

`dreaming/dreamer.py` — `Dreamer.__init__`의 `self._locks: Dict[str, asyncio.Lock] = {}`를 `self._active: set = set()`으로 교체하고 `dream`을 교체 (import에서 `asyncio` 사용처가 사라지면 import도 제거):

```python
    async def dream(self, session: str) -> Optional[Dict]:
        # 체크와 점유 사이 await 없음 — 이벤트 루프 단일 스레드에서 원자
        if session in self._active:
            return None
        self._active.add(session)
        try:
            return await self._cycle(session)
        except Exception:
            # 사이클 폐기, 커서 불변 → 다음 유휴에 재시도 (스펙 §2.6, §3.2)
            logger.exception("[dreamer] cycle discarded: %s", session)
            return None
        finally:
            self._active.discard(session)
```

`dreaming/upstream.py` — `to_wire`의 무마킹 분기를 사본 반환으로:

```python
        if cc is None:
            out.append(dict(m))
            continue
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_dreamer.py tests/test_dreaming_upstream.py tests/test_dreaming_proxy.py -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py dreaming/upstream.py tests/test_dreaming_dreamer.py tests/test_dreaming_upstream.py
git commit -m "fix(dreaming): 꿈 락 원자화(set 기반) + to_wire 무마킹 메시지 사본 반환"
```

---

### Task 5: Episode 턴 범위 + Tier1/Tier2 조립 템플릿

**Files:**
- Modify: `dreaming/records.py` (`Episode`)
- Modify: `dreaming/dreamer.py` (`apply_extraction`의 episode 저장)
- Create: `dreaming/chunks.py` (조립 함수만 — 플랜 build/apply는 Task 6·7)
- Test: `tests/test_dreaming_chunks.py`, `tests/test_dreaming_extraction.py`

**Interfaces:**
- Consumes: `Episode` 레코드 (records.py).
- Produces: `Episode.start_turn: Optional[int] = None`, `Episode.end_turn: Optional[int] = None` (구버전 데이터는 None → 압축 제외). `assemble_tier1(ep: Episode) -> str`, `assemble_tier2(episodes: List[Episode]) -> str` — 결정론 템플릿, 같은 입력 → 같은 바이트.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_chunks.py` 신규:

```python
"""청크 조립 — 결정론 템플릿 (스펙 §6.1) + Tier 계층 (§6.2)."""
from dreaming.chunks import assemble_tier1, assemble_tier2
from dreaming.records import Episode

_EP = Episode(range_start="u0", range_end="u3", start_turn=0, end_turn=3,
              title="포션 흥정", summary="리사와 가격을 흥정해 50골드에 샀다.",
              open_threads=["잔액의 출처"])


def test_tier1_template_is_deterministic():
    text = assemble_tier1(_EP)
    assert text == assemble_tier1(_EP.model_copy())     # 같은 입력 → 같은 바이트
    assert "포션 흥정" in text and "50골드" in text
    assert "잔액의 출처" in text                          # open_threads 포함


def test_tier1_without_threads_has_no_thread_line():
    ep = _EP.model_copy(update={"open_threads": []})
    assert "실마리" not in assemble_tier1(ep)


def test_tier2_is_one_line_per_episode():
    ep2 = _EP.model_copy(update={"title": "여관 투숙",
                                 "summary": "방을 80골드에\n잡았다."})
    text = assemble_tier2([_EP, ep2])
    lines = text.splitlines()
    assert len(lines) == 3                               # 헤더 + 에피소드 2
    assert "여관 투숙" in lines[2] and "\n" not in lines[2]
```

`tests/test_dreaming_extraction.py`의 `test_episode_range_from_raw_hashes`에 단언 추가:

```python
    assert eps[0].start_turn == 0 and eps[0].end_turn == 0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_extraction.py -q`
Expected: FAIL — `dreaming.chunks` 없음, Episode에 start_turn 없음(ValidationError).

- [ ] **Step 3: 구현**

`dreaming/records.py` — `Episode`에 필드 추가 (`open_threads` 아래):

```python
    start_turn: Optional[int] = None   # 압축 선택용 턴 범위 (Plan 4) —
    end_turn: Optional[int] = None     # 구버전 레코드는 None → 압축 제외
```

`dreaming/dreamer.py` — `apply_extraction`의 episode 저장에 턴 전달:

```python
        store.save_episode(Episode(
            range_start=start["user_hash"], range_end=end["user_hash"],
            start_turn=ep.start_turn, end_turn=ep.end_turn,
            title=ep.title, summary=ep.summary, open_threads=ep.open_threads,
        ))
```

`dreaming/chunks.py` 신규:

```python
"""dreaming/chunks.py — 청크 압축 (스펙 §6).

압축은 결정론적 템플릿 조립이다 — 이해(요약문)는 꿈이 이미 끝냈고,
여기서는 같은 입력에 같은 바이트만 만든다 (§6.1). LLM 0콜.
"""

from __future__ import annotations

from typing import List

from dreaming.records import Episode


def _one_line(text: str) -> str:
    return " ".join(text.split())


def assemble_tier1(ep: Episode) -> str:
    """에피소드 청크 (~70% 압축): 제목 + 요약 + 미회수 복선."""
    lines = [f"[지난 이야기 · {ep.title}]", ep.summary.strip()]
    if ep.open_threads:
        lines.append("남은 실마리: " + " / ".join(ep.open_threads))
    return "\n".join(lines)


def assemble_tier2(episodes: List[Episode]) -> str:
    """챕터 청크 (~90% 압축): 에피소드당 한 줄."""
    lines = ["[지난 장 요약]"]
    for ep in episodes:
        lines.append(f"- {ep.title}: {_one_line(ep.summary)[:100]}")
    return "\n".join(lines)
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_extraction.py tests/test_dreaming_dreamer.py -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/records.py dreaming/dreamer.py dreaming/chunks.py tests/test_dreaming_chunks.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): Episode 턴 범위 + Tier1/Tier2 결정론 조립 템플릿 (스펙 §6.1~6.2)"
```

---

### Task 6: 압축 플랜 빌드 + Dreamer B-4 배선

**Files:**
- Modify: `dreaming/chunks.py`
- Modify: `dreaming/dreamer.py` (`_cycle`)
- Test: `tests/test_dreaming_chunks.py`, `tests/test_dreaming_dreamer.py`

**Interfaces:**
- Consumes: Task 5의 조립 함수, `MemoryStore.list_episodes()`, `Storage.put`.
- Produces: `build_compression(store: MemoryStore, last_turn: int) -> Optional[Dict]` — 플랜 문서 `{"covers_until_turn": int, "messages": [{"role": "assistant", "content": str}, ...]}` 또는 None(압축할 것 없음). 상수 `TAIL_KEEP = 6`(원문 꼬리 pair), `T1_MAX = 8`, `CHAPTER_SIZE = 5`. Dreamer `_cycle`은 커서 전진 후 플랜을 `{session}/compression` / key `"plan"`에 저장하고 report에 `"chunks"` 키를 넣는다. **플랜 갱신은 꿈 안에서만** — 유휴 = TTL 경과 = 캐시 사망 시점이라 재압축 비용 0 (스펙 §6.3; `Settings.idle_seconds` 기본 300s = TTL "5m" 정렬).

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_chunks.py`에 추가:

```python
import json

from dreaming.chunks import TAIL_KEEP, build_compression
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore


def _ep(start, end, title="에피"):
    return Episode(range_start=f"u{start}", range_end=f"u{end}",
                   start_turn=start, end_turn=end, title=f"{title}{start}",
                   summary=f"{start}~{end} 요약.")


def _store_with(tmp_path, episodes):
    store = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    for e in episodes:
        store.save_episode(e)
    return store


def test_contiguous_prefix_outside_tail_is_chunked(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
    plan = build_compression(store, last_turn=7 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 8
    assert [m["role"] for m in plan["messages"]] == ["assistant", "assistant"]
    assert "에피0" in plan["messages"][0]["content"]


def test_tail_and_gap_stop_chunking(tmp_path):
    # 꼬리 안 에피소드는 미압축, 턴 갭(4 누락)에서 중단
    store = _store_with(tmp_path, [_ep(0, 3), _ep(5, 6)])
    plan = build_compression(store, last_turn=6 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 4                 # 갭 앞까지만
    store2 = _store_with(tmp_path / "b", [_ep(0, 3)])
    assert build_compression(store2, last_turn=3) is None  # 전부 꼬리 안


def test_overlapping_redream_episode_is_skipped(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(0, 2, title="중복"),
                                   _ep(4, 5)])
    plan = build_compression(store, last_turn=5 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 6
    assert len(plan["messages"]) == 2                     # 중복 스킵


def test_legacy_episode_without_turns_is_ignored(tmp_path):
    legacy = Episode(range_start="u0", range_end="u3", title="구버전",
                     summary="턴 없음")
    store = _store_with(tmp_path, [legacy])
    assert build_compression(store, last_turn=99) is None


def test_tier2_promotion_is_stable(tmp_path):
    # T1_MAX(8) 초과 시 오래된 것부터 CHAPTER_SIZE(5) 고정 블록으로 승격 —
    # 에피소드가 늘어도 기존 챕터 바이트는 불변 (프리픽스 안정)
    eps = [_ep(i * 2, i * 2 + 1) for i in range(10)]      # 10개 → 챕터 1 + T1 5개
    store = _store_with(tmp_path, eps)
    plan = build_compression(store, last_turn=19 + TAIL_KEEP)
    assert len(plan["messages"]) == 1 + 5
    assert plan["messages"][0]["content"].startswith("[지난 장 요약]")

    store.save_episode(_ep(20, 21))                       # 11개 → 그룹 경계 불변
    plan2 = build_compression(store, last_turn=21 + TAIL_KEEP)
    assert plan2["messages"][0] == plan["messages"][0]


def test_plan_is_deterministic(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
    a = build_compression(store, last_turn=20)
    b = build_compression(store, last_turn=20)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
```

`tests/test_dreaming_dreamer.py`의 `test_dream_full_cycle_advances_cursor`에 단언 추가 (에피소드가 전부 꼬리 안이라 플랜 없음):

```python
    assert report["chunks"] == 0
    assert storage.get("sess1/compression", "plan") is None
```

그리고 같은 파일에 B-4 배선 테스트 추가:

```python
def test_dream_writes_compression_plan(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=10)
    ext = json.dumps({"episodes": [
        {"start_turn": 0, "end_turn": 3, "title": "초반",
         "summary": "만남과 흥정.", "open_threads": []}]}, ensure_ascii=False)
    report = asyncio.run(Dreamer(storage, FakeLLM(ext)).dream("sess1"))
    assert report["chunks"] == 1
    plan = storage.get("sess1/compression", "plan")
    assert plan["covers_until_turn"] == 4
    assert "초반" in plan["messages"][0]["content"]
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_dreamer.py -q`
Expected: FAIL — `build_compression` 없음, report에 `chunks` 키 없음.

- [ ] **Step 3: 구현**

`dreaming/chunks.py`에 추가 (import에 `Optional`, `Dict`, `MemoryStore` 추가):

```python
TAIL_KEEP = 6      # 원문 꼬리로 남길 최근 pair 수 (스펙 §5)
T1_MAX = 8         # Tier1 청크 상한 — 초과분은 챕터로 승격 (§6.2)
CHAPTER_SIZE = 5   # 챕터 1개로 묶을 에피소드 수 — 고정 블록이라 승격이 안정


def build_compression(store: MemoryStore, last_turn: int) -> Optional[Dict]:
    """에피소드 → 압축 플랜 (B-4, 꿈 안에서만 호출 — §6.3 TTL 창구).

    턴 0부터의 연속 구간만 압축한다 (치환이 위치 기반이라 프리픽스 연속성이
    전제). 갭·꼬리(최근 TAIL_KEEP pair)에서 중단, 재드림 중복 구간은 스킵.
    """
    eps = [e for e in store.list_episodes()
           if e.start_turn is not None and e.end_turn is not None]
    eps.sort(key=lambda e: (e.start_turn, e.recorded_at))
    cutoff = last_turn - TAIL_KEEP
    chain: List[Episode] = []
    next_turn = 0
    for e in eps:
        if e.start_turn < next_turn:
            continue                      # 이미 덮인 구간 (재드림 중복)
        if e.start_turn > next_turn or e.end_turn > cutoff:
            break                         # 갭 또는 꼬리 진입
        chain.append(e)
        next_turn = e.end_turn + 1
    if not chain:
        return None

    n_chapters = 0
    if len(chain) > T1_MAX:
        n_chapters = -(-(len(chain) - T1_MAX) // CHAPTER_SIZE)  # ceil
    messages: List[Dict] = []
    idx = 0
    for _ in range(n_chapters):
        group = chain[idx: idx + CHAPTER_SIZE]
        messages.append({"role": "assistant", "content": assemble_tier2(group)})
        idx += len(group)
    for e in chain[idx:]:
        messages.append({"role": "assistant", "content": assemble_tier1(e)})
    return {"covers_until_turn": next_turn, "messages": messages}
```

`dreaming/dreamer.py` — import에 `from dreaming.chunks import build_compression` 추가, `_cycle` 끝을 교체:

```python
        report = apply_extraction(store, ext, raw_by_turn)       # B-3
        self._storage.put(f"{session}/dreamer", "cursor",
                          {"next_turn": raw_turns[-1]["turn_number"] + 1})
        plan = build_compression(                                # B-4 (§6.3)
            store, last_turn=raw_turns[-1]["turn_number"])
        if plan is not None:
            self._storage.put(f"{session}/compression", "plan", plan)
        report["chunks"] = len(plan["messages"]) if plan else 0
        return report
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_dreamer.py tests/test_dreaming_proxy.py -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/chunks.py dreaming/dreamer.py tests/test_dreaming_chunks.py tests/test_dreaming_dreamer.py
git commit -m "feat(dreaming): B-4 압축 플랜 빌드 — 연속 프리픽스 선별 + Tier2 고정 블록 승격"
```

---

### Task 7: 압축 적용 + BP2 마킹 + SyncPath 배선

> **BP1 버그 동시 수정 (HANDOFF §3.1)** — charx `post_history_instructions`는
> RisuAI globalNote로 **메시지 배열 꼬리에 system**으로 붙는다 (`prompt.ts:427`,
> `characterCards.ts:992`). 기존 `mark_cache`는 전체에서 마지막 system을 BP1로
> 잡아 꼬리 PHI에 찍혔다 — 지식 주입이 캐시 span 안에 들어가는 스펙 §3.1 위반.
> BP1 후보를 **선두 연속 system 구간**으로 한정한다. 같은 수정이 ba42ff
> 워크트리에 미커밋으로 존재 — 머지 시 한쪽만 채택 (테스트 이름 동일하게 유지).

**Files:**
- Modify: `dreaming/chunks.py` (`apply_compression`)
- Modify: `dreaming/marking.py` (`mark_cache` — BP1 선두 한정 + `bp2_index`)
- Modify: `dreaming/sync.py` (`SyncPath.process`)
- Test: `tests/test_dreaming_chunks.py`, `tests/test_dreaming_marking.py`, `tests/test_dreaming_proxy.py`

**Interfaces:**
- Consumes: Task 6의 플랜 문서 형식.
- Produces: `apply_compression(messages: List[Dict], plan: Dict) -> Tuple[List[Dict], Optional[int]]` — 선두 system 블록·인사(첫 user 이전 assistant)는 보존, 이후 첫 K pair를 플랜 메시지로 치환. 반환 (치환된 메시지, 첫 청크 인덱스 | None). pair 부족 시 원본 그대로 + None (fail-open). `mark_cache(messages, ttl="5m", bp2_index=None)` — BP1 = **선두 연속 system 구간의 끝** (꼬리 PHI system 제외), bp2_index에 BP2 마킹 (스펙 §3.1: BP2 = 첫 청크 assistant. 새 청크는 첫 청크 **뒤에** 붙으므로 BP2 프리픽스는 꿈 사이에 불변). SyncPath 순서: 판정 → 강등 → 압축 → 주입 → 마킹.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_chunks.py`에 추가:

```python
from dreaming.chunks import apply_compression

_PLAN = {"covers_until_turn": 2,
         "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]}


def _msgs(pairs, greeting=True):
    out = [{"role": "system", "content": "너는 리사다."}]
    if greeting:
        out.append({"role": "assistant", "content": "어서 와요."})
    for i in range(pairs):
        out.append({"role": "user", "content": f"질문{i}"})
        out.append({"role": "assistant", "content": f"답{i}"})
    out.append({"role": "user", "content": "새 질문"})
    return out


def test_apply_replaces_first_k_pairs_keeps_system_and_greeting():
    msgs = _msgs(4)
    out, bp2 = apply_compression(msgs, _PLAN)
    assert out[0]["content"] == "너는 리사다."
    assert out[1]["content"] == "어서 와요."               # 인사 보존
    assert out[2]["content"] == "[지난 이야기 · 초반]"      # 청크
    assert bp2 == 2
    texts = [m["content"] for m in out]
    assert "질문0" not in texts and "질문2" in texts        # 꼬리 보존
    assert out[-1]["content"] == "새 질문"
    assert msgs[2]["content"] == "질문0"                    # 원본 불변


def test_apply_short_history_fails_open():
    msgs = _msgs(1)                                        # pair 1 < K=2
    out, bp2 = apply_compression(msgs, _PLAN)
    assert out is msgs and bp2 is None


def test_mark_cache_bp2():
    from dreaming.marking import mark_cache
    out, bp2 = apply_compression(_msgs(4), _PLAN)
    marked = mark_cache(out, bp2_index=bp2)
    assert marked[0]["cache_control"]["type"] == "ephemeral"   # BP1
    assert marked[bp2]["cache_control"]["type"] == "ephemeral" # BP2
    last_asst = max(i for i, m in enumerate(marked)
                    if m["role"] == "assistant")
    assert marked[last_asst]["cache_control"]["type"] == "ephemeral"  # BP3
    assert sum(1 for m in marked if "cache_control" in m) == 3
```

`tests/test_dreaming_marking.py`에 BP1 회귀 테스트 추가 (ba42ff와 동일 이름 — 머지 시 한쪽 채택):

```python
def test_bp1_stays_in_leading_system_run():
    # charx PHI는 globalNote로 꼬리 system에 붙는다 (prompt.ts:427) —
    # BP1이 거기 찍히면 지식 주입이 캐시 span 안에 들어간다 (스펙 §3.1 위반)
    msgs = [{"role": "system", "content": "본문"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답"},
            {"role": "system", "content": "PHI 꼬리"}]
    marked = mark_cache(msgs)
    assert "cache_control" in marked[0]              # BP1 = 선두 연속 system 끝
    assert "cache_control" not in marked[3]          # 꼬리 system 금지


def test_no_bp_after_injected_last_user():
    msgs = [{"role": "system", "content": "본문"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답"},
            {"role": "user",
             "content": "<dreaming_context>지식</dreaming_context>\n\n새 질문"},
            {"role": "system", "content": "globalNote"}]
    marked = mark_cache(msgs)
    idxs = [i for i, m in enumerate(marked) if "cache_control" in m]
    assert max(idxs) == 2                            # 마지막 user 뒤엔 BP 없음
```

`tests/test_dreaming_proxy.py`에 추가 (SyncPath 경유 확인 — 저장된 플랜이 실제 요청에 적용되고 원본으로 기록되는지):

```python
def test_stored_plan_compresses_outbound_but_records_original(tmp_path):
    storage = JsonDirStorage(tmp_path)
    storage.put("sess1/compression", "plan", {
        "covers_until_turn": 1,
        "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    r = client.post("/v1/chat/completions",
                    json=_body("질문0", "답0", "질문1"),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    sent = up.payloads[0]["messages"]
    joined = json.dumps(sent, ensure_ascii=False)
    assert "[지난 이야기" in joined and "질문0" not in joined
    chunk = sent[1]                                    # system 다음 = 첫 청크
    assert chunk["content"][0]["cache_control"]["type"] == "ephemeral"  # BP2
    raw = storage.get("sess1/raw", "000001")
    assert raw["user_text"] == "질문1"                  # 기록은 원본 기준
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_proxy.py -q`
Expected: FAIL — `apply_compression` 없음, mark_cache에 bp2_index 없음(TypeError), 프록시 미배선.

- [ ] **Step 3: 구현**

`dreaming/chunks.py`에 추가 (import에 `copy`, `Tuple` 추가):

```python
def apply_compression(messages: List[Dict],
                      plan: Dict) -> Tuple[List[Dict], Optional[int]]:
    """히스토리 선두 K pair를 청크로 위치 기반 치환 (스펙 §5 레이아웃).

    선두 system 블록과 인사(첫 user 이전 assistant)는 보존한다.
    히스토리가 플랜보다 짧으면(리롤 직후·낯선 요청) 원본 그대로 — fail-open.
    반환: (메시지, 첫 청크 인덱스 | None).
    """
    k = plan["covers_until_turn"]
    i = 0
    while i < len(messages) and messages[i].get("role") != "user":
        i += 1                             # 첫 user 앞(system·인사)은 보존
    pairs, j = 0, i
    while j < len(messages) and pairs < k:
        if messages[j].get("role") == "user":
            if (j + 1 < len(messages)
                    and messages[j + 1].get("role") == "assistant"):
                j += 2
                pairs += 1
            else:
                break                      # 미완 pair(현재 턴) — 압축 불가
        else:
            j += 1
    if pairs < k:
        return messages, None
    out = messages[:i] + copy.deepcopy(plan["messages"]) + messages[j:]
    return out, i
```

`dreaming/marking.py` — `mark_cache` 전체 교체 (import에 `Optional` 추가). BP1은 선두 연속 system 구간으로 한정 — Anthropic 변환도 선두 밖 system은 user로 강등한다 (`anthropic.ts:226-238`):

```python
def mark_cache(messages: List[Dict], ttl: str = "5m",
               bp2_index: Optional[int] = None) -> List[Dict]:
    out = [copy.deepcopy(m) for m in messages]
    for m in out:
        m.pop("cache_control", None)

    last_system = None
    for i, m in enumerate(out):          # BP1 후보는 선두 연속 system만 —
        if m.get("role") != "system":    # 꼬리 PHI(globalNote)는 캐시 밖
            break
        last_system = i
    last_assistant = None
    for i, m in enumerate(out):
        if m.get("role") == "assistant":
            last_assistant = i

    mark = {"type": "ephemeral", "ttl": ttl}
    if last_system is not None:
        out[last_system]["cache_control"] = dict(mark)     # BP1
    if last_assistant is not None:
        out[last_assistant]["cache_control"] = dict(mark)  # BP3
    if bp2_index is not None and 0 <= bp2_index < len(out):
        out[bp2_index]["cache_control"] = dict(mark)       # BP2 = 첫 청크
    return out
```

`dreaming/sync.py` — import에 `from dreaming.chunks import apply_compression` 추가, `SyncPath.process`의 조립부 교체:

```python
        knowledge = clip_knowledge(render_knowledge(self._store))
        out, bp2 = messages, None
        plan = self._storage.get(f"{self._session}/compression", "plan")
        if plan is not None:
            out, bp2 = apply_compression(out, plan)
        out = inject_knowledge(out, knowledge)
        out = mark_cache(out, bp2_index=bp2)
        return out, verdict
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_marking.py tests/test_dreaming_proxy.py tests/ -q; echo "exit=$?"`
Expected: PASS, exit=0 (전체 스위트 — 기존 마킹 테스트는 선두 system 구성이라 BP1 한정과 충돌 없음, 확인함).

- [ ] **Step 5: 커밋**

```bash
git add dreaming/chunks.py dreaming/marking.py dreaming/sync.py tests/test_dreaming_chunks.py tests/test_dreaming_marking.py tests/test_dreaming_proxy.py
git commit -m "feat(dreaming): 압축 적용 + BP2 + BP1 선두 system 한정 + 동기 경로 배선 (스펙 §5, §3.1)"
```

---

### Task 8: 리롤/분기 시 압축·에피소드 무효화

**Files:**
- Modify: `dreaming/sync.py` (`demote_after`)
- Test: `tests/test_dreaming_demote.py`

**Interfaces:**
- Consumes: Task 6의 플랜 문서, `Episode.end_turn`, `Storage.delete`.
- Produces: `demote_after`가 분기점이 압축 구간 안이면(`covers_until_turn > from_turn`) 플랜을 **삭제**하고(다음 꿈이 재조립 — TTL 창구라 공짜), `end_turn >= from_turn`인 에피소드를 삭제한다. 흔한 케이스(마지막 턴 리롤)는 압축 구간 밖이라 아무것도 안 건드린다. 부분 트림은 하지 않는다 — 깊은 분기는 드물고, 한 번의 cold miss 후 재조립이 단순함을 이긴다.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_demote.py`에 추가 (기존 헬퍼·import 재사용, `Episode`는 `dreaming.records`에서):

```python
def test_deep_divergence_invalidates_plan_and_stale_episodes(tmp_path):
    storage = JsonDirStorage(tmp_path)
    store = MemoryStore(storage, "sess1")
    storage.put("sess1/compression", "plan",
                {"covers_until_turn": 4, "messages": [
                    {"role": "assistant", "content": "청크"}]})
    store.save_episode(Episode(range_start="u0", range_end="u1",
                               start_turn=0, end_turn=1,
                               title="보존", summary="분기 전"))
    store.save_episode(Episode(range_start="u2", range_end="u3",
                               start_turn=2, end_turn=3,
                               title="무효", summary="분기 걸침"))
    demote_after(storage, "sess1", from_turn=2)
    assert storage.get("sess1/compression", "plan") is None
    assert [e.title for e in store.list_episodes()] == ["보존"]


def test_late_reroll_keeps_plan_and_episodes(tmp_path):
    # 흔한 케이스: 마지막 턴 리롤 — 압축 구간(0~3) 밖이라 무손상
    storage = JsonDirStorage(tmp_path)
    store = MemoryStore(storage, "sess1")
    plan = {"covers_until_turn": 4,
            "messages": [{"role": "assistant", "content": "청크"}]}
    storage.put("sess1/compression", "plan", plan)
    store.save_episode(Episode(range_start="u0", range_end="u3",
                               start_turn=0, end_turn=3,
                               title="보존", summary="압축 구간"))
    demote_after(storage, "sess1", from_turn=9)
    assert storage.get("sess1/compression", "plan") == plan
    assert len(store.list_episodes()) == 1
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_demote.py -q`
Expected: FAIL — 플랜·에피소드가 그대로 남아 있음.

- [ ] **Step 3: 구현** — `dreaming/sync.py`의 `demote_after` 끝(커서 되감기 앞)에 추가:

```python
    # 분기점이 압축 구간 안이면 플랜 폐기 + 걸친 에피소드 삭제 —
    # 다음 꿈이 재조립한다 (TTL 창구라 캐시 비용 0, 스펙 §6.3)
    plan = storage.get(f"{session}/compression", "plan")
    if plan is not None and plan["covers_until_turn"] > from_turn:
        storage.delete(f"{session}/compression", "plan")
    for e in store.list_episodes():
        if e.end_turn is not None and e.end_turn >= from_turn:
            storage.delete(f"{session}/episodes", e.id)
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_demote.py tests/ -q; echo "exit=$?"`
Expected: PASS, exit=0

- [ ] **Step 5: 커밋**

```bash
git add dreaming/sync.py tests/test_dreaming_demote.py
git commit -m "feat(dreaming): 분기 시 압축 플랜·걸친 에피소드 무효화 — 다음 꿈이 재조립"
```

---

### Task 9: e2e 통합 테스트 + 전체 회귀 + push

**Files:**
- Test: `tests/test_dreaming_proxy.py`
- 수정 없음 (배선은 Task 1~8에서 완료 — 여기서 깨지면 해당 태스크로 돌아가 수정)

**Interfaces:**
- Consumes: 전부.
- Produces: 프록시 레벨 풀 루프 검증 — 캐치업 꿈이 에피소드를 만들고, 플랜이 저장되고, **다음 요청**의 아웃바운드 프리픽스가 압축된다 (스펙 §3.2 캐치업: 첫 요청은 기존 프리픽스로 즉시 통과, 2턴째부터 새 프리픽스).

- [ ] **Step 1: e2e 테스트 작성** — `tests/test_dreaming_proxy.py`에 추가:

```python
_E2E_EXTRACTION = json.dumps({"episodes": [
    {"start_turn": 0, "end_turn": 3, "title": "포션 흥정",
     "summary": "리사와 가격을 흥정했다.", "open_threads": []}]},
    ensure_ascii=False)


def test_full_loop_dream_then_compressed_prefix(tmp_path):
    storage = JsonDirStorage(tmp_path)
    for t in range(10):
        storage.put("sess1/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"질문{t}",
            "assistant_text": f"답{t}", "user_hash": f"u{t}",
            "assistant_hash": f"a{t}"})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up,
                     dream_llm=FakeLLM(_E2E_EXTRACTION))
    history = []
    for t in range(10):
        history += [f"질문{t}", f"답{t}"]

    with TestClient(app) as client:
        r = client.post("/v1/chat/completions",
                        json=_body(*history, "새 질문"),
                        headers={"x-dreaming-session-id": "sess1"})
        assert r.status_code == 200                    # 첫 요청 즉시 통과
        first = json.dumps(up.payloads[0]["messages"], ensure_ascii=False)
        assert "질문0" in first                        # 꿈 전엔 무압축
        for _ in range(100):                           # 캐치업 꿈 대기
            if storage.get("sess1/compression", "plan"):
                break
            time.sleep(0.02)
        r2 = client.post("/v1/chat/completions",
                         json=_body(*history, "새 질문", "50골드다.", "다음 질문"),
                         headers={"x-dreaming-session-id": "sess1"})
        assert r2.status_code == 200

    plan = storage.get("sess1/compression", "plan")
    assert plan["covers_until_turn"] == 4
    sent = up.payloads[1]["messages"]
    joined = json.dumps(sent, ensure_ascii=False)
    assert "포션 흥정" in joined                       # 청크 등장
    assert "질문0" not in joined and "질문4" in joined  # 선두 치환, 꼬리 보존
    marks = sum(1 for m in sent
                if isinstance(m.get("content"), list)
                and "cache_control" in m["content"][0])
    assert marks == 3                                  # BP1 + BP2 + BP3
```

- [ ] **Step 2: 실행 (배선이 완성됐으면 바로 통과해야 정상)**

Run: `python3 -m pytest tests/test_dreaming_proxy.py -q; echo "exit=$?"`
Expected: PASS, exit=0. 실패하면 원인 태스크로 돌아가 고친 후 재실행 (테스트를 약화시키지 말 것).

- [ ] **Step 3: 전체 회귀**

Run: `python3 -m pytest tests/ -q; echo "exit=$?"`
Expected: 전체 PASS, exit=0. `git diff --stat main -- saga/` 가 빈 출력인지도 확인 (saga diff 0 제약).

- [ ] **Step 4: 커밋 + push + PR**

```bash
git add tests/test_dreaming_proxy.py
git commit -m "test(dreaming): 풀 루프 e2e — 캐치업 꿈 → 압축 프리픽스 + 3-BP 검증"
git push -u origin dreaming/spec
```

PR 생성 (`gh pr create`) — 제목: `feat(dreaming): Phase 1 청크 압축 — B-4 조립·BP2·TTL 창구 + 실카드 품질 수정 (스펙 §5~6)`. 본문에 Task 목록과 실카드 결함 A~F 매핑 요약.

---

## Self-Review 결과 (작성 시 + 2026-08-06 ba42ff 감사 반영 후 재수행)

1. **스코프 커버리지**: §6.1 조립(T5), §6.2 Tier1/2(T5·6), §6.3 TTL 창구(T6 — 꿈 안에서만 갱신), §5 레이아웃·BP2·BP1 선두 한정(T7), §3.1 리롤 연동(T8), 결함 A(T1)·B 잔여분=관측 로그(T2)·C·D·E(T3, E는 T2의 abs 대조와 합작)·F(T3 로그 + T4). §6.2의 총량 30%·keyExcerpts·pinned 지연, §6.3의 예산 임계 중도 압축, Tier3, replay 크래시(별도 세션), lore_shift(ba42ff)는 Global Constraints에 명시.
2. **자리표시자 없음**: 전 태스크 실코드·실테스트 포함.
3. **타입 일관성**: 플랜 문서 형식 `{"covers_until_turn": int, "messages": [...]}`이 T6(생산)·T7(소비)·T8(무효화)·T9(검증)에서 동일. `korean_spellings`/`apply_compression`/`mark_cache(bp2_index=)` 시그니처 상호 일치 확인. `test_dream_full_cycle_advances_cursor`의 report 단언은 T3에서 키별 비교로 바꿔 T6의 `chunks` 키 추가와 충돌 없음. T2 축소 후 `_resolve_evidence` 참조는 플랜 어디에도 남아 있지 않음.

## 실행 후 확인 (플랜 밖, 선택)

- 라이브 검증은 ba42ff의 `benchmarks/cardsim/` 하네스로 — 단 **record/replay(대본 고정)가 들어간 뒤에야 전/후 비교가 통제된다** (HANDOFF §5: 유저 발화를 매번 LLM이 새로 만들어 churn 턴 수가 실행마다 갈림).
- BP2·압축 효과 측정은 반드시 `--no-keyed-lore` 대조군에서 (HANDOFF §4: keyed 로어 churn이 프리픽스 index 1을 매 턴 흔들어 히트율 16.9%↔97.8% 노이즈 — lore_shift 머지 전엔 압축 효과가 묻힌다).
- 프로브의 진짜 시험: 압축 후 프리픽스에서 원문이 빠졌을 때 정답이 청크+지식 주입에서 나오는가. PAUSES idle 단축은 테스트용 — 프로덕션 기본(idle 300s = TTL 5m)에서는 §6.3대로 캐시 사망 후 재압축이다.
