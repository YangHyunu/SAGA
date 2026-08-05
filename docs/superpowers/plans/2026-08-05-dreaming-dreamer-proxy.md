# Dreaming Plan 3/6 — Dreamer(유휴 이해 사이클) + HTTP 프록시 배선

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 유휴 시 LLM 1콜로 원문을 이해(추출·검증·적용)하는 Dreamer와, RisuAI가 바라볼 OpenAI 호환 리버스 프록시를 배선한다 — 동기 경로(Plan 2) 위에 얹어 end-to-end로 돌아가는 소프트웨어를 만든다.

**Architecture:** Dreamer는 `{session}/raw`의 미처리 턴을 스냅샷(B-0)해 단일 구조화 출력 1콜(B-1 경계판정 + B-2 추출 병합)을 부르고, 숫자 정규식 재검증·mem0 4분류·user_edited 보호(B-3)를 거쳐 MemoryStore에 적용한다. B-4 재압축(청크)은 Plan 4 — **Plan 3의 꿈은 지식 계층만 쓰므로 프리픽스가 불변**이고, 꿈 도중 요청이 와도 캐시 충돌이 없다. 프록시는 FastAPI 단일 라우트(`/v1/chat/completions`)로 SyncPath 주입·마킹 후 OpenAI 호환 업스트림(기본 OpenRouter)에 전달한다. cache_control은 content part로 변환하므로 스트리밍은 바이트 passthrough다.

**Tech Stack:** Python 3, pydantic v2, FastAPI, httpx (전부 기설치). 새 의존성 없음.

**스펙:** `docs/dreaming/SPEC.md` — §3.2 (Dreamer), §3.1 (reroll/diverged 강등), §8 (프록시/Storage/LLMClient 추상화). 근거 애매하면 스펙 §0.1~0.3 자료 확인 후 진행 (추측 구현 금지).

## Global Constraints

- 인터프리터는 `python3` (`python` 없음). 테스트: `python3 -m pytest tests/ -x -q`
- 저장은 KV 문서 샤드 단일 — SQL·SQLite·외부 DB(PostgreSQL/Honcho/Mem0류) 금지 (스펙 §8)
- saga import 허용 범위: `saga.services.pair_ledger`의 `hash_text`/`extract_pairs`/`classify` 3개뿐. saga/ 코드 수정 금지 (diff 0)
- 동기 경로는 턴당 LLM 0콜 유지 (스펙 §2.2) — LLM 호출은 Dreamer 내부에만 존재
- fail-open (스펙 §2.6): dreaming 오류로 채팅 요청이 실패하면 안 된다 — 프록시는 원본 그대로 전달
- Dreamer 사이클 = LLM 1콜 (스펙 §3.2) — B-1과 B-2를 단일 구조화 출력으로 병합
- 비동기 테스트는 pytest-asyncio 없이 `asyncio.run(...)` 패턴 (플러그인 미설치)
- 새 pip 의존성 추가 금지
- Episode의 `embedding`/`causes`는 이번 플랜에서 기록하지 않는다 (검색 고도화·인과 링크는 후속 플랜) — 필드는 Plan 1에서 이미 존재, None/빈 리스트로 둔다
- 커밋 메시지 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## File Structure

| 파일 | 책임 |
|---|---|
| `dreaming/llm.py` (신규) | `LLMClient` Protocol + `OpenAICompatLLM` (httpx, 스펙 §8 LLMClient 추상화) |
| `dreaming/dreamer.py` (신규) | 추출 스키마 / 프롬프트 / 파서 / B-3 검증·적용 / `Dreamer` 사이클 |
| `dreaming/idle.py` (신규) | `IdleWatcher` — 세션 유휴 타이머 (스펙 §3.2 IdleTrigger 추상화) |
| `dreaming/upstream.py` (신규) | `to_wire` (cache_control→content part) + `OpenAIUpstream` (complete/stream) |
| `dreaming/proxy.py` (신규) | `Settings` + `create_app` — FastAPI 라우트, fail-open, 캐치업 드림 |
| `dreaming/__main__.py` (신규) | uvicorn 실행 진입점 |
| `dreaming/store.py` (수정) | `update_commit_status` 1메서드 추가 (강등용) |
| `dreaming/sync.py` (수정) | `demote_after` + SyncPath reroll/diverged 배선 (스펙 §3.1) |

---

### Task 1: LLM 클라이언트 (`dreaming/llm.py`)

**Files:**
- Create: `dreaming/llm.py`
- Test: `tests/test_dreaming_llm.py`

**Interfaces:**
- Consumes: 없음 (독립)
- Produces: `LLMClient` Protocol — `async def complete(self, system: str, user: str) -> str`; `OpenAICompatLLM(base_url, api_key, model, timeout=120.0, client=None)`. Task 4의 Dreamer와 Task 8의 create_app이 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""LLMClient 추상화 (스펙 §8) — Dreamer 전용. 동기 경로는 LLM 0콜."""
import asyncio
import json

import httpx
import pytest

from dreaming.llm import OpenAICompatLLM


def _client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             base_url="http://fake")


def test_complete_returns_message_content():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"facts": []}'}}]})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    assert asyncio.run(llm.complete("sys", "usr")) == '{"facts": []}'


def test_sends_model_messages_temperature_zero():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    asyncio.run(llm.complete("sys", "usr"))
    assert seen["model"] == "flash"
    assert seen["temperature"] == 0
    assert seen["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "usr"}]


def test_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(llm.complete("sys", "usr"))
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.llm'`

- [ ] **Step 3: 구현**

```python
"""dreaming/llm.py — LLMClient 추상화 (스펙 §8).

Dreamer 전용. 동기 경로는 LLM 0콜 (스펙 §2.2).
OpenAI 호환 chat/completions 하나로 OpenRouter·Gemini(OpenAI compat) 전부 커버.
"""

from __future__ import annotations

from typing import Optional, Protocol

import httpx


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        r = await self._client.post("/chat/completions", json={
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_llm.py -v`
Expected: 3 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/llm.py tests/test_dreaming_llm.py
git commit -m "feat(dreaming): LLMClient 추상화 + OpenAI 호환 구현 (스펙 §8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: 추출 스키마 + 프롬프트 + 파서 (`dreaming/dreamer.py` 1부)

**Files:**
- Create: `dreaming/dreamer.py`
- Test: `tests/test_dreaming_extraction.py`

**Interfaces:**
- Consumes: `dreaming.records`의 `Fact`, `Actor` (프롬프트 재료)
- Produces: pydantic 모델 `DreamExtraction`(`episodes/facts/commits/actors`), `ExtractedNumber(name, value: float, unit: str = "")`, `ExtractedFact(claim, entities, numbers, evidence_turn: int, action, target_fact_id, learned_by)`, `ExtractedCommit(slot, op, value: Union[float, str], turn: int)`, `ExtractedActor(names, profile, tier)`, `ExtractedEpisode(start_turn, end_turn, title, summary, open_threads)`; 함수 `parse_extraction(text: str) -> DreamExtraction`, `build_dream_prompt(raw_turns: List[Dict], facts: List[Fact], state: Dict, actors: List[Actor]) -> Tuple[str, str]`. Task 3~4가 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
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
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.dreamer'`

- [ ] **Step 3: 구현**

`dreaming/dreamer.py` 신규 작성 (이 태스크 범위는 스키마/파서/프롬프트까지 — 검증·적용·사이클은 Task 3~4에서 같은 파일에 추가):

```python
"""dreaming/dreamer.py — 유휴 이해 사이클 B-0~B-3 (스펙 §3.2).

B-1(에피소드 경계)과 B-2(추출)는 단일 구조화 출력 1콜로 병합한다 —
스펙의 "사이클 = LLM 1콜 + 임베딩"에서 임베딩은 후속 플랜.
B-4 재압축(청크 조립)은 Plan 4 — 이 모듈의 꿈은 지식 계층만 쓰므로
프리픽스가 불변이고, 꿈 도중 요청이 와도 캐시 충돌이 없다.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from dreaming.records import Actor, Fact

_MAX_PROMPT_FACTS = 50


# ------------------------------------------------------------------ #
# 추출 스키마 — 단일 구조화 출력 (B-1 + B-2)
# ------------------------------------------------------------------ #

class ExtractedNumber(BaseModel):
    name: str
    value: float
    unit: str = ""


class ExtractedFact(BaseModel):
    claim: str
    entities: List[str] = []
    numbers: List[ExtractedNumber] = []
    evidence_turn: int
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = "ADD"   # mem0 4분류
    target_fact_id: Optional[str] = None
    learned_by: List[str] = []


class ExtractedCommit(BaseModel):
    slot: str
    op: Literal["set", "add"]
    value: Union[float, str]
    turn: int


class ExtractedActor(BaseModel):
    names: List[str]
    profile: str = ""
    tier: Literal["main", "support", "extra"] = "support"


class ExtractedEpisode(BaseModel):
    start_turn: int
    end_turn: int
    title: str
    summary: str
    open_threads: List[str] = []


class DreamExtraction(BaseModel):
    episodes: List[ExtractedEpisode] = []
    facts: List[ExtractedFact] = []
    commits: List[ExtractedCommit] = []
    actors: List[ExtractedActor] = []


_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n?|\n?```\s*$")


def parse_extraction(text: str) -> DreamExtraction:
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"dream output is not JSON: {e}") from e
    return DreamExtraction.model_validate(data)


# ------------------------------------------------------------------ #
# 프롬프트 — 결정론적 조립
# ------------------------------------------------------------------ #

_SYSTEM = """너는 RP 세션 로그에서 구조화 지식을 추출하는 분석기다.
반드시 JSON 객체 하나만 출력한다. 마크다운 펜스·설명·사과 금지.

스키마:
{
  "episodes": [{"start_turn": int, "end_turn": int, "title": str, "summary": str, "open_threads": [str]}],
  "facts": [{"claim": str, "entities": [str],
             "numbers": [{"name": str, "value": number, "unit": str}],
             "evidence_turn": int, "action": "ADD|UPDATE|DELETE|NOOP",
             "target_fact_id": str|null, "learned_by": [str]}],
  "commits": [{"slot": str, "op": "set|add", "value": number|str, "turn": int}],
  "actors": [{"names": [str], "profile": str, "tier": "main|support|extra"}]
}

규칙:
- fact 1개 = 원자 명제 1개. 복합 문장 금지.
- numbers는 원문에 명시된 숫자만. 추측·계산 금지.
- 기존 사실을 갱신하면 action=UPDATE + target_fact_id(기존 지식의 id).
  기존 사실이 무효가 됐으면 DELETE. 변화 없으면 NOOP. 새 사실은 ADD.
- commits는 수치·상태 변화만 (소지금, 위치, 시각 등). 서술은 fact로.
- episodes는 장면 경계로 나눈다. open_threads엔 미회수 복선만.
- learned_by는 그 사실을 알게 된 인물 이름 목록."""


def build_dream_prompt(raw_turns: List[Dict], facts: List[Fact],
                       state: Dict, actors: List[Actor]) -> Tuple[str, str]:
    known: List[str] = []
    if state:
        known.append("현재 상태:\n" + "\n".join(
            f"- {slot}: {value}" for slot, value in sorted(state.items())))
    recent = sorted(facts, key=lambda f: f.recorded_at)[-_MAX_PROMPT_FACTS:]
    if recent:
        known.append("기존 사실:\n" + "\n".join(
            f"- [{f.id}] {f.claim} ({f.status})" for f in recent))
    if actors:
        known.append("인물:\n" + "\n".join(
            f"- {'/'.join(a.names)} ({a.tier})"
            for a in sorted(actors, key=lambda a: a.names[0])))

    turns = "\n\n".join(
        f"[턴 {r['turn_number']}]\nuser: {r['user_text']}\nassistant: {r['assistant_text']}"
        for r in raw_turns)

    user = ("[기존 지식]\n" + ("\n\n".join(known) if known else "없음")
            + "\n\n[미처리 원문]\n" + turns)
    return _SYSTEM, user
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): 꿈 추출 스키마·프롬프트·파서 — B-1+B-2 단일 1콜 (스펙 §3.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: B-3 검증·적용 (`dreaming/dreamer.py` 2부)

**Files:**
- Modify: `dreaming/dreamer.py` (Task 2 파일에 추가)
- Test: `tests/test_dreaming_extraction.py` (추가)

**Interfaces:**
- Consumes: Task 2의 `DreamExtraction` 계열; `dreaming.store.MemoryStore`; `dreaming.facts.supersede`, `dreamer_can_modify`; `dreaming.records`의 `Fact/TypedNumber/Evidence/StateCommit/Actor/Episode`
- Produces: `verify_numbers(numbers: List[ExtractedNumber], text: str) -> bool`, `apply_extraction(store: MemoryStore, ext: DreamExtraction, raw_by_turn: Dict[int, Dict]) -> Dict[str, int]` (report: `facts/blocked/commits/actors/episodes`). Task 4가 사용.

동작 규칙 (스펙 §3.2 B-3 + §4.1):
- 숫자 재검증: fact/commit의 숫자는 evidence 턴 원문(user+assistant, 콤마 제거)에 문자 그대로 존재해야 함. 통과 → fact `confirmed` / commit `applied`. 실패 → fact `provisional` / commit `pending_contradiction`(리플레이 제외).
- mem0 4분류: `NOOP` 무시. `DELETE`는 대상 fact를 `superseded`로 (invalidate-and-append, 하드 삭제 없음). `UPDATE`는 `supersede` 버전 체인.
- `user_edited` 보호: `dreamer_can_modify`가 False인 대상에 UPDATE → 새 fact를 `pending_contradiction`으로 기록(관찰), DELETE → 차단. (스펙 §2.7)
- Actor upsert: 기존 actor와 이름 교집합 있으면 같은 id로 병합(이름 합집합, 새 profile 우선), 없으면 신규.
- Episode: start/end 턴의 raw `user_hash`를 range로. 없는 턴 참조 시 해당 에피소드 스킵.

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_dreaming_extraction.py`에 추가)

```python
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
    from dreaming.records import Fact
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
    from dreaming.records import Fact
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
    from dreaming.records import Actor
    store.save_actor(Actor(names=["리사"], profile="시장 상인", tier="support"))
    ext = DreamExtraction.model_validate({"actors": [
        {"names": ["리사", "Lisa"], "profile": "시장 상인, 밀수 연루", "tier": "main"}]})
    apply_extraction(store, ext, _RAW_BY_TURN)
    actors = store.list_actors()
    assert len(actors) == 1
    assert set(actors[0].names) == {"리사", "Lisa"}
    assert actors[0].tier == "main"


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
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -v`
Expected: 기존 4개 PASS, 신규 8개 FAIL — `ImportError: cannot import name 'apply_extraction'`

- [ ] **Step 3: 구현** (`dreaming/dreamer.py`에 추가)

파일 상단 import에 추가:

```python
from dreaming.facts import dreamer_can_modify, supersede
from dreaming.records import Episode, Evidence, StateCommit, TypedNumber
from dreaming.store import MemoryStore
```

본문에 추가:

```python
# ------------------------------------------------------------------ #
# B-3: 검증·적용
# ------------------------------------------------------------------ #

def verify_numbers(numbers: List[ExtractedNumber], text: str) -> bool:
    """숫자 정규식 재검증 (스펙 §3.2 B-3): 원문에 문자 그대로 있어야 한다."""
    plain = text.replace(",", "")
    for n in numbers:
        v = n.value
        s = str(int(v)) if float(v).is_integer() else str(v)
        if s not in plain:
            return False
    return True


def _turn_text(raw: Dict) -> str:
    return raw["user_text"] + "\n" + raw["assistant_text"]


def _build_fact(ef: ExtractedFact, raw_by_turn: Dict[int, Dict]) -> Fact:
    raw = raw_by_turn.get(ef.evidence_turn)
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


def apply_extraction(store: MemoryStore, ext: DreamExtraction,
                     raw_by_turn: Dict[int, Dict]) -> Dict[str, int]:
    report = {"facts": 0, "blocked": 0, "commits": 0, "actors": 0, "episodes": 0}

    for ef in ext.facts:
        if ef.action == "NOOP":
            continue
        target = store.get_fact(ef.target_fact_id) if ef.target_fact_id else None
        if ef.action == "DELETE":
            if target is None:
                continue
            if not dreamer_can_modify(target):
                report["blocked"] += 1
                continue
            store.save_fact(target.model_copy(update={"status": "superseded"}))
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
        report["facts"] += 1

    for ec in ext.commits:
        raw = raw_by_turn.get(ec.turn)
        status = "applied"
        if isinstance(ec.value, (int, float)) and not isinstance(ec.value, bool):
            text = _turn_text(raw) if raw else ""
            probe = ExtractedNumber(name=ec.slot, value=float(ec.value))
            if not verify_numbers([probe], text):
                status = "pending_contradiction"
        store.append_commit(StateCommit(
            slot=ec.slot, op=ec.op, value=ec.value, turn=ec.turn,
            evidence=Evidence(pair_hash=raw["user_hash"]) if raw else None,
            status=status,
        ))
        report["commits"] += 1

    existing = store.list_actors()
    for ea in ext.actors:
        match = next((a for a in existing if set(a.names) & set(ea.names)), None)
        if match is not None:
            names = list(match.names) + [n for n in ea.names if n not in match.names]
            store.save_actor(match.model_copy(update={
                "names": names,
                "profile": ea.profile or match.profile,
                "tier": ea.tier,
            }))
        else:
            store.save_actor(Actor(names=ea.names, profile=ea.profile, tier=ea.tier))
        report["actors"] += 1

    for ep in ext.episodes:
        start = raw_by_turn.get(ep.start_turn)
        end = raw_by_turn.get(ep.end_turn)
        if start is None or end is None:
            continue
        store.save_episode(Episode(
            range_start=start["user_hash"], range_end=end["user_hash"],
            title=ep.title, summary=ep.summary, open_threads=ep.open_threads,
        ))
        report["episodes"] += 1

    return report
```

주의: `verify_numbers`에서 음수 `add`(-777)는 `str(int(-777.0))` = `"-777"`로 비교된다 — 원문에 "-777"이 없으면 격리. 의도된 동작.

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py -v`
Expected: 12 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): B-3 검증·적용 — 숫자 재검증 + mem0 4분류 + 유저편집 보호 (스펙 §3.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Dreamer 사이클 (`dreaming/dreamer.py` 3부)

**Files:**
- Modify: `dreaming/dreamer.py`
- Test: `tests/test_dreaming_dreamer.py`

**Interfaces:**
- Consumes: Task 1 `LLMClient`, Task 2~3의 파서·적용, `dreaming.storage.Storage`, `dreaming.store.MemoryStore`
- Produces: `Dreamer(storage: Storage, llm: LLMClient)` — `async def dream(session: str) -> Optional[Dict]` (report 또는 None), `def has_backlog(session: str) -> bool`, `def snapshot(session: str) -> List[Dict]`. 커서 문서: ns `f"{session}/dreamer"`, key `"cursor"`, 값 `{"next_turn": int}`. Task 5(커서 되감기)와 Task 8(프록시)이 사용.

동작 규칙:
- B-0: 커서(`next_turn`, 기본 0) 이후의 `{session}/raw` 턴을 스냅샷. 비면 no-op.
- 세션당 `asyncio.Lock` — 이미 꿈꾸는 중이면 즉시 None (중복 사이클 방지). 크래시 안전은 커서가 담당: 적용 성공 후에만 전진하므로 실패·중단 시 다음 유휴에 같은 구간 재시도 = 캐치업 (스펙 §3.2).
- 어떤 예외든 사이클 폐기 + 커서 불변 (fail-open, 스펙 §2.6). 적용 단계엔 await가 없어 단일 프로세스 asyncio에서 원자적이다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""Dreamer 사이클 B-0~B-3 (스펙 §3.2) — FakeLLM으로 1콜 검증."""
import asyncio
import json

import pytest

from dreaming.dreamer import Dreamer
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

_EXTRACTION = json.dumps({
    "episodes": [{"start_turn": 0, "end_turn": 0, "title": "포션 흥정",
                  "summary": "가격을 물었다.", "open_threads": []}],
    "facts": [{"claim": "포션은 50골드다", "evidence_turn": 0,
               "numbers": [{"name": "가격", "value": 50, "unit": "골드"}]}],
    "commits": [{"slot": "소지금", "op": "set", "value": 1450, "turn": 0}],
    "actors": [{"names": ["리사"], "profile": "시장 상인", "tier": "main"}],
}, ensure_ascii=False)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete(self, system, user):
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _seed_raw(storage, session="sess1", turns=1):
    for t in range(turns):
        storage.put(f"{session}/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"포션 얼마야? ({t})",
            "assistant_text": "50골드다. 잔액은 1,450골드.",
            "user_hash": f"u{t}", "assistant_hash": f"a{t}"})


def test_dream_full_cycle_advances_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    llm = FakeLLM(_EXTRACTION)
    report = asyncio.run(Dreamer(storage, llm).dream("sess1"))
    assert report == {"facts": 1, "blocked": 0, "commits": 1,
                      "actors": 1, "episodes": 1}
    assert len(llm.calls) == 1                                    # 사이클당 1콜
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}
    store = MemoryStore(storage, "sess1")
    assert store.list_facts()[0].status == "confirmed"
    assert store.current_state() == {"소지금": 1450.0}


def test_dream_without_backlog_is_noop(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    llm = FakeLLM(_EXTRACTION)
    d = Dreamer(storage, llm)
    asyncio.run(d.dream("sess1"))
    assert asyncio.run(d.dream("sess1")) is None                 # 두 번째: 잔량 없음
    assert len(llm.calls) == 1


def test_llm_failure_discards_cycle_keeps_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(RuntimeError("api down")))
    assert asyncio.run(d.dream("sess1")) is None                 # fail-open
    assert storage.get("sess1/dreamer", "cursor") is None        # 커서 불변
    assert d.has_backlog("sess1")                                # 다음 유휴에 재시도


def test_garbage_json_discards_cycle(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM("JSON 아님"))
    assert asyncio.run(d.dream("sess1")) is None
    assert storage.get("sess1/dreamer", "cursor") is None


def test_concurrent_dream_skips(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))

    async def scenario():
        lock = d._locks.setdefault("sess1", asyncio.Lock())
        async with lock:                      # 꿈꾸는 중 시뮬레이션
            return await d.dream("sess1")

    assert asyncio.run(scenario()) is None


def test_has_backlog_and_snapshot_respect_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=3)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))
    storage.put("sess1/dreamer", "cursor", {"next_turn": 2})
    snap = d.snapshot("sess1")
    assert [r["turn_number"] for r in snap] == [2]
    assert d.has_backlog("sess1")
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_dreamer.py -v`
Expected: FAIL — `ImportError: cannot import name 'Dreamer'`

- [ ] **Step 3: 구현** (`dreaming/dreamer.py`에 추가)

상단 import에 추가:

```python
import asyncio
import logging

from dreaming.llm import LLMClient
from dreaming.storage import Storage
```

모듈 상단에 `logger = logging.getLogger(__name__)` 추가 후 본문 끝에:

```python
# ------------------------------------------------------------------ #
# 사이클 오케스트레이터 (B-0 스냅샷 → 1콜 → B-3 적용 → 커서 전진)
# ------------------------------------------------------------------ #

class Dreamer:
    def __init__(self, storage: Storage, llm: LLMClient) -> None:
        self._storage = storage
        self._llm = llm
        self._locks: Dict[str, asyncio.Lock] = {}

    def _cursor(self, session: str) -> int:
        doc = self._storage.get(f"{session}/dreamer", "cursor")
        return doc["next_turn"] if doc else 0

    def snapshot(self, session: str) -> List[Dict]:
        cur = self._cursor(session)
        rows = [row for _, row in self._storage.scan(f"{session}/raw")
                if row["turn_number"] >= cur]
        return sorted(rows, key=lambda r: r["turn_number"])

    def has_backlog(self, session: str) -> bool:
        return bool(self.snapshot(session))

    async def dream(self, session: str) -> Optional[Dict]:
        lock = self._locks.setdefault(session, asyncio.Lock())
        if lock.locked():
            return None
        async with lock:
            try:
                return await self._cycle(session)
            except Exception:
                # 사이클 폐기, 커서 불변 → 다음 유휴에 재시도 (스펙 §2.6, §3.2)
                logger.exception("[dreamer] cycle discarded: %s", session)
                return None

    async def _cycle(self, session: str) -> Optional[Dict]:
        raw_turns = self.snapshot(session)                       # B-0
        if not raw_turns:
            return None
        store = MemoryStore(self._storage, session)
        system, user = build_dream_prompt(
            raw_turns, store.list_facts(), store.current_state(),
            store.list_actors())
        ext = parse_extraction(await self._llm.complete(system, user))  # B-1+B-2
        raw_by_turn = {r["turn_number"]: r for r in raw_turns}
        report = apply_extraction(store, ext, raw_by_turn)       # B-3
        self._storage.put(f"{session}/dreamer", "cursor",
                          {"next_turn": raw_turns[-1]["turn_number"] + 1})
        return report
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_dreamer.py tests/test_dreaming_extraction.py -v`
Expected: 18 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/dreamer.py tests/test_dreaming_dreamer.py
git commit -m "feat(dreaming): Dreamer 사이클 — B-0 스냅샷·락·커서, fail-open 폐기 (스펙 §3.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: 리롤·분기 강등 + 커서 되감기 (`dreaming/sync.py`, `dreaming/store.py` 수정)

**Files:**
- Modify: `dreaming/store.py` (`update_commit_status` 추가)
- Modify: `dreaming/sync.py` (`demote_after` + SyncPath 배선)
- Test: `tests/test_dreaming_demote.py`

**Interfaces:**
- Consumes: `MemoryStore`, `Storage`, `SyncPath`(Plan 2), Task 4 커서 문서(`{session}/dreamer`/`cursor`)
- Produces: `MemoryStore.update_commit_status(commit_id: str, status: str) -> None`; `demote_after(storage: Storage, session: str, from_turn: int) -> None` (sync.py 모듈 함수). `SyncPath.process`가 `verdict.kind in ("reroll", "diverged")`일 때 자동 호출.

동작 규칙 (스펙 §3.1):
- reroll = "이전 턴 기록을 잠정 무효화(삭제 아님)", diverged = "분기점 이후 Fact를 provisional로 강등".
- `from_turn` 이후 턴에서 배운 것들: Fact(evidence pair_hash가 해당 구간 user_hash) → `provisional` (단, `user_edited`·`superseded` 제외), applied commit(turn >= from_turn) → `pending_contradiction` (리플레이 제외).
- **꿈 커서 되감기**: 커서가 이미 지나갔다면 `from_turn`으로 되감아 다음 꿈이 대체 응답을 재추출한다. (record_turn이 같은 turn_number의 raw를 덮어쓰므로 최신 원문이 재료가 된다.)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""리롤/분기 시 지식 강등 + 꿈 커서 되감기 (스펙 §3.1)."""
from dreaming.records import Evidence, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, demote_after


def _seed(storage, session="sess1"):
    for t, (uh, ah) in enumerate([("u0", "a0"), ("u1", "a1")]):
        storage.put(f"{session}/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"u{t}", "assistant_text": f"a{t}",
            "user_hash": uh, "assistant_hash": ah})
    store = MemoryStore(storage, session)
    store.save_fact(Fact(claim="턴0에서 배움", status="confirmed",
                         evidence=[Evidence(pair_hash="u0")]))
    store.save_fact(Fact(claim="턴1에서 배움", status="confirmed",
                         evidence=[Evidence(pair_hash="u1")]))
    store.save_fact(Fact(claim="유저가 고정", status="confirmed", user_edited=True,
                         evidence=[Evidence(pair_hash="u1")]))
    store.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=0))
    store.append_commit(StateCommit(slot="소지금", op="set", value=400, turn=1))
    storage.put(f"{session}/dreamer", "cursor", {"next_turn": 2})
    return store


def test_demote_after_turn1(tmp_path):
    storage = JsonDirStorage(tmp_path)
    store = _seed(storage)
    demote_after(storage, "sess1", 1)
    by_claim = {f.claim: f for f in store.list_facts()}
    assert by_claim["턴0에서 배움"].status == "confirmed"       # 분기점 이전 무사
    assert by_claim["턴1에서 배움"].status == "provisional"     # 강등
    assert by_claim["유저가 고정"].status == "confirmed"        # 유저 편집 보호
    assert store.current_state() == {"소지금": 450.0}           # 턴1 커밋 격리
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}  # 되감기


def test_demote_keeps_earlier_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed(storage)
    storage.put("sess1/dreamer", "cursor", {"next_turn": 0})   # 아직 안 꿈꿈
    demote_after(storage, "sess1", 1)
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 0}  # 그대로


def test_syncpath_reroll_triggers_demotion(tmp_path):
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "sess1")

    def msgs(*texts):
        roles = ["user", "assistant"]
        out = [{"role": "system", "content": "너는 상인 리사다."}]
        out += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
        return out

    m1 = msgs("안녕")
    _, v1 = sp.process(m1)
    sp.record_response(v1, m1, "어서 와.")
    m2 = msgs("안녕", "어서 와.", "포션 얼마야?")
    _, v2 = sp.process(m2)
    sp.record_response(v2, m2, "50골드다.")

    # 꿈이 턴1까지 처리했고 턴1에서 fact를 배웠다고 시뮬레이션
    store = MemoryStore(storage, "sess1")
    raw1 = storage.get("sess1/raw", "000001")
    store.save_fact(Fact(claim="포션은 50골드다", status="confirmed",
                         evidence=[Evidence(pair_hash=raw1["user_hash"])]))
    storage.put("sess1/dreamer", "cursor", {"next_turn": 2})

    # 리롤: 같은 요청 재전송
    _, v3 = sp.process(msgs("안녕", "어서 와.", "포션 얼마야?"))
    assert v3.kind == "reroll"
    assert store.list_facts()[0].status == "provisional"
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_demote.py -v`
Expected: FAIL — `ImportError: cannot import name 'demote_after'`

- [ ] **Step 3: 구현**

`dreaming/store.py` — Actor 섹션 앞(WorldState 섹션 끝)에 추가:

```python
    def update_commit_status(self, commit_id: str, status: str) -> None:
        """status는 판정 메타 — append-only 원장에서 유일하게 in-place 갱신 허용."""
        data = self._storage.get(self._ns("commits"), commit_id)
        if data is not None:
            data["status"] = status
            self._storage.put(self._ns("commits"), commit_id, data)
```

`dreaming/sync.py` — import에 `from dreaming.records import ...` 없음 주의, 모듈 함수 추가 (`render_knowledge` 아래):

```python
def demote_after(storage: Storage, session: str, from_turn: int) -> None:
    """리롤/분기 시점 이후에서 배운 지식을 잠정화한다 (스펙 §3.1).

    Fact → provisional (user_edited 보호), 해당 구간 applied commit →
    pending_contradiction, 꿈 커서 되감기 → 다음 꿈이 대체 응답을 재추출.
    """
    stale_hashes = {row["user_hash"] for _, row in storage.scan(f"{session}/raw")
                    if row["turn_number"] >= from_turn}
    store = MemoryStore(storage, session)
    for f in store.list_facts():
        if f.user_edited or f.status == "provisional":
            continue
        if any(e.pair_hash in stale_hashes for e in f.evidence):
            store.save_fact(f.model_copy(update={"status": "provisional"}))
    for c in store.list_commits():
        if c.turn >= from_turn and c.status == "applied":
            store.update_commit_status(c.id, "pending_contradiction")
    cursor = storage.get(f"{session}/dreamer", "cursor")
    if cursor is not None and cursor["next_turn"] > from_turn:
        storage.put(f"{session}/dreamer", "cursor", {"next_turn": from_turn})
```

`SyncPath.process` 수정 — verdict 직후에 배선:

```python
    def process(self, messages: List[Dict]) -> Tuple[List[Dict], Verdict]:
        pairs, last_user_hash = extract_pairs(messages)
        verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
        if (verdict.kind in ("reroll", "diverged")
                and verdict.reroll_turn_number is not None):
            demote_after(self._storage, self._session, verdict.reroll_turn_number)
        knowledge = clip_knowledge(render_knowledge(self._store))
        out = inject_knowledge(messages, knowledge)
        out = mark_cache(out)
        return out, verdict
```

- [ ] **Step 4: 통과 확인 (기존 sync 테스트 회귀 포함)**

Run: `python3 -m pytest tests/test_dreaming_demote.py tests/test_dreaming_sync.py tests/test_dreaming_store.py -v`
Expected: 신규 3개 포함 전부 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/store.py dreaming/sync.py tests/test_dreaming_demote.py
git commit -m "feat(dreaming): 리롤·분기 강등 — fact 잠정화 + 커밋 격리 + 꿈 커서 되감기 (스펙 §3.1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: IdleWatcher (`dreaming/idle.py`)

**Files:**
- Create: `dreaming/idle.py`
- Test: `tests/test_dreaming_idle.py`

**Interfaces:**
- Consumes: 없음 (asyncio만)
- Produces: `IdleWatcher(idle_seconds: float, on_idle: Callable[[str], Awaitable[None]])` — `touch(session: str)` (타이머 리셋, 반드시 실행 중인 이벤트 루프 안에서 호출), `cancel_all()`. Task 8이 사용.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""세션 유휴 타이머 (스펙 §3.2) — 유휴 = 캐시 TTL 경과 = 꿈 트리거."""
import asyncio

from dreaming.idle import IdleWatcher


def _run(coro):
    return asyncio.run(coro)


def test_fires_once_after_idle():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.03, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.1)

    _run(scenario())
    assert fired == ["sess1"]


def test_touch_resets_timer():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.05, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.03)
        w.touch("sess1")                 # 리셋 — 아직 안 울려야 함
        await asyncio.sleep(0.03)
        assert fired == []
        await asyncio.sleep(0.05)

    _run(scenario())
    assert fired == ["sess1"]


def test_on_idle_exception_swallowed():
    async def scenario():
        async def on_idle(s):
            raise RuntimeError("dream failed")
        w = IdleWatcher(0.01, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.05)        # 예외가 새어나오면 여기서 터짐

    _run(scenario())                     # fail-open: 통과하면 성공


def test_cancel_all():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.02, on_idle)
        w.touch("a")
        w.touch("b")
        w.cancel_all()
        await asyncio.sleep(0.06)

    _run(scenario())
    assert fired == []
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_idle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.idle'`

- [ ] **Step 3: 구현**

```python
"""dreaming/idle.py — IdleTrigger 추상화 (스펙 §3.2, §8).

유휴 기준 = 캐시 TTL 경과(기본 5m) — 캐시가 이미 죽은 시점이라
꿈(과 Plan 4의 재압축)이 공짜인 창구다. cron 아님, 세션별 타이머.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict

logger = logging.getLogger(__name__)


class IdleWatcher:
    def __init__(self, idle_seconds: float,
                 on_idle: Callable[[str], Awaitable[None]]) -> None:
        self._idle = idle_seconds
        self._on_idle = on_idle
        self._tasks: Dict[str, asyncio.Task] = {}

    def touch(self, session: str) -> None:
        prev = self._tasks.pop(session, None)
        if prev is not None:
            prev.cancel()
        self._tasks[session] = asyncio.create_task(self._wait(session))

    async def _wait(self, session: str) -> None:
        try:
            await asyncio.sleep(self._idle)
        except asyncio.CancelledError:
            return
        self._tasks.pop(session, None)
        try:
            await self._on_idle(session)
        except Exception:
            logger.exception("[idle] on_idle failed: %s", session)   # fail-open

    def cancel_all(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_idle.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/idle.py tests/test_dreaming_idle.py
git commit -m "feat(dreaming): IdleWatcher — 세션 유휴 타이머, TTL 창구 트리거 (스펙 §3.2)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: 업스트림 어댑터 (`dreaming/upstream.py`)

**Files:**
- Create: `dreaming/upstream.py`
- Test: `tests/test_dreaming_upstream.py`

**Interfaces:**
- Consumes: Plan 2 `mark_cache`가 만드는 메시지 레벨 `cache_control`
- Produces: `to_wire(messages: List[Dict]) -> List[Dict]` (메시지 레벨 cache_control → OpenAI content part 형식, OpenRouter 캐싱 규약); `OpenAIUpstream(base_url, api_key, timeout=300.0, client=None)` — `async def complete(payload: Dict) -> Dict`, `def stream(payload: Dict) -> AsyncIterator[bytes]` (SSE raw bytes). Task 8이 사용.

배경: OpenRouter는 Anthropic 프롬프트 캐싱을 OpenAI 형식으로 받되 `cache_control`을 **content part 안에** 요구한다 — `content: [{"type": "text", "text": ..., "cache_control": {...}}]`. 우리 `mark_cache`는 메시지 레벨에 마킹하므로 전송 직전 여기서 변환한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""OpenAI 호환 업스트림 — cache_control content part 변환 + passthrough."""
import asyncio
import json

import httpx

from dreaming.upstream import OpenAIUpstream, to_wire


def test_to_wire_moves_cache_control_into_content_part():
    msgs = [
        {"role": "system", "content": "봇 정의",
         "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        {"role": "user", "content": "안녕"},
    ]
    wire = to_wire(msgs)
    assert wire[0]["content"] == [{
        "type": "text", "text": "봇 정의",
        "cache_control": {"type": "ephemeral", "ttl": "5m"}}]
    assert "cache_control" not in wire[0]          # 메시지 레벨에선 제거
    assert wire[1] == {"role": "user", "content": "안녕"}   # 무마킹은 그대로
    assert "cache_control" in msgs[0]              # 원본 불변


def _upstream(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://up")
    return OpenAIUpstream("http://up", "k", client=client)


def test_complete_posts_payload_and_returns_json():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "어서 와."}}]})

    up = _upstream(handler)
    resp = asyncio.run(up.complete({"model": "m", "messages": []}))
    assert seen["model"] == "m"
    assert resp["choices"][0]["message"]["content"] == "어서 와."


def test_complete_raises_on_http_error():
    import pytest
    def handler(request):
        return httpx.Response(429, json={"error": "rate"})
    up = _upstream(handler)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(up.complete({"model": "m", "messages": []}))


def test_stream_yields_raw_bytes():
    body = b'data: {"choices":[{"delta":{"content":"어서"}}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, content=body)

    up = _upstream(handler)

    async def collect():
        return b"".join([c async for c in up.stream({"model": "m"})])

    assert asyncio.run(collect()) == body
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_upstream.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.upstream'`

- [ ] **Step 3: 구현**

```python
"""dreaming/upstream.py — OpenAI 호환 업스트림 (기본 OpenRouter).

cache_control은 OpenRouter 규약대로 content part 안에 넣는다.
업스트림이 OpenAI 호환이므로 스트리밍은 SSE 바이트 passthrough다.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

import httpx


def to_wire(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        cc = m.get("cache_control")
        if cc is None:
            out.append(m)
            continue
        mm = {k: v for k, v in m.items() if k != "cache_control"}
        mm["content"] = [{"type": "text", "text": m["content"],
                          "cache_control": cc}]
        out.append(mm)
    return out


class OpenAIUpstream:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(self, payload: Dict) -> Dict:
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    async def stream(self, payload: Dict) -> AsyncIterator[bytes]:
        async with self._client.stream(
                "POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_upstream.py -v`
Expected: 4 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/upstream.py tests/test_dreaming_upstream.py
git commit -m "feat(dreaming): OpenAI 호환 업스트림 — cache_control content part 변환 + SSE passthrough

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: 프록시 앱 (`dreaming/proxy.py`, `dreaming/__main__.py`)

**Files:**
- Create: `dreaming/proxy.py`
- Create: `dreaming/__main__.py`
- Test: `tests/test_dreaming_proxy.py`

**Interfaces:**
- Consumes: Plan 2 `SyncPath`, Task 4 `Dreamer`, Task 6 `IdleWatcher`, Task 7 `to_wire`/`OpenAIUpstream`, Task 1 `OpenAICompatLLM`
- Produces: `Settings` (pydantic, `from_env()`), `create_app(settings, *, upstream=None, dream_llm=None) -> FastAPI`. 라우트: `POST /v1/chat/completions` (stream/non-stream), `GET /health`.

동작 규칙:
- 세션 ID: 헤더 `x-dreaming-session-id` → body `user` 필드 → `"default"`. Storage 세그먼트 규칙(`^[A-Za-z0-9._-]+$`)에 맞게 sanitize (`.`/`..`은 storage가 이미 거부하지만 sanitize에서 빈 결과·순수 특수문자는 `default`로).
- fail-open (스펙 §2.6): `SyncPath.process` 실패 → 원본 메시지 그대로 전달, 기록 스킵. 업스트림 실패만 502.
- `record_response`는 **주입 전 원본 messages**로 호출한다 (주입문이 들어가면 user hash가 달라져 다음 턴 판정이 깨진다).
- 스트리밍: 업스트림 SSE를 그대로 흘리면서 `choices[0].delta.content`를 누적 → 종료 시 기록.
- 응답 후: `IdleWatcher.touch(session)` (유휴 타이머 리셋 → TTL 경과 시 꿈).
- 캐치업 드림: 프로세스 기동 후 세션 첫 요청 시, **이번 턴을 기록하기 전에** backlog를 확인 (기록 후에 보면 방금 쓴 턴 때문에 신규 세션도 오탐) → 있으면 백그라운드 꿈. 첫 요청은 즉시 통과, 2턴째부터 새 지식 적용 (스펙 §3.2).
- `dream_llm`/`dream_model` 미설정이면 Dreamer 비활성 (프록시 기능만 — fail-open 사상).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""프록시 end-to-end — 주입·마킹·기록·fail-open·캐치업 (스펙 §3.1~3.2, §8)."""
import json
import time

from fastapi.testclient import TestClient

from dreaming.proxy import Settings, create_app
from dreaming.records import StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

_EXTRACTION = json.dumps({
    "facts": [{"claim": "포션은 50골드다", "evidence_turn": 0,
               "numbers": [{"name": "가격", "value": 50}]}],
}, ensure_ascii=False)


class FakeUpstream:
    def __init__(self):
        self.payloads = []

    async def complete(self, payload):
        self.payloads.append(payload)
        return {"choices": [{"message": {"content": "50골드다."}}]}

    async def stream(self, payload):
        self.payloads.append(payload)
        for piece in ["50골드", "다."]:
            data = json.dumps({"choices": [{"delta": {"content": piece}}]},
                              ensure_ascii=False)
            yield f"data: {data}\n\n".encode()
        yield b"data: [DONE]\n\n"


class FakeLLM:
    def __init__(self, response):
        self.response = response

    async def complete(self, system, user):
        return self.response


def _settings(tmp_path, idle=300.0):
    return Settings(data_dir=str(tmp_path), upstream_base_url="http://up",
                    upstream_api_key="k", idle_seconds=idle)


def _body(*texts, stream=False):
    roles = ["user", "assistant"]
    msgs = [{"role": "system", "content": "너는 상인 리사다."}]
    msgs += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
    return {"model": "anthropic/claude-sonnet-4.5", "messages": msgs,
            "stream": stream}


def test_non_stream_injects_marks_and_records(tmp_path):
    up = FakeUpstream()
    storage = JsonDirStorage(tmp_path)
    MemoryStore(storage, "sess1").append_commit(
        StateCommit(slot="소지금", op="set", value=450, turn=0))
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)

    r = client.post("/v1/chat/completions", json=_body("포션 얼마야?"),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "50골드다."

    sent = up.payloads[0]["messages"]
    sys_part = sent[0]["content"][0]                 # BP1 → content part 변환
    assert sys_part["cache_control"]["type"] == "ephemeral"
    assert "<dreaming_context>" in sent[-1]["content"]
    assert "소지금: 450" in sent[-1]["content"]

    raw = storage.get("sess1/raw", "000000")         # 원본 기준으로 기록됨
    assert raw["user_text"] == "포션 얼마야?"
    assert raw["assistant_text"] == "50골드다."


def test_stream_passthrough_accumulates_and_records(tmp_path):
    up = FakeUpstream()
    storage = JsonDirStorage(tmp_path)
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)

    r = client.post("/v1/chat/completions", json=_body("포션 얼마야?", stream=True),
                    headers={"x-dreaming-session-id": "sess1"})
    assert r.status_code == 200
    assert b"data:" in r.content and b"[DONE]" in r.content
    raw = storage.get("sess1/raw", "000000")
    assert raw["assistant_text"] == "50골드다."


def test_fail_open_on_sync_error(tmp_path, monkeypatch):
    import dreaming.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod.SyncPath, "process",
                        lambda self, m: (_ for _ in ()).throw(RuntimeError("boom")))
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_body("안녕"))
    assert r.status_code == 200                       # 채팅은 안 죽는다
    assert up.payloads[0]["messages"][-1]["content"] == "안녕"   # 원본 무가공


def test_upstream_error_returns_502(tmp_path):
    class DeadUpstream:
        async def complete(self, payload):
            raise RuntimeError("connection refused")
    app = create_app(_settings(tmp_path), upstream=DeadUpstream())
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json=_body("안녕"))
    assert r.status_code == 502


def test_sessions_are_isolated(tmp_path):
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up)
    client = TestClient(app)
    client.post("/v1/chat/completions", json=_body("안녕"),
                headers={"x-dreaming-session-id": "sess-a"})
    client.post("/v1/chat/completions", json=_body("반가워"),
                headers={"x-dreaming-session-id": "sess-b"})
    storage = JsonDirStorage(tmp_path)
    assert storage.get("sess-a/raw", "000000")["user_text"] == "안녕"
    assert storage.get("sess-b/raw", "000000")["user_text"] == "반가워"


def test_catchup_dream_runs_in_background(tmp_path):
    storage = JsonDirStorage(tmp_path)
    storage.put("sess1/raw", "000000", {          # 이전 기동에서 밀린 턴
        "turn_number": 0, "user_text": "포션 얼마야?",
        "assistant_text": "50골드다.", "user_hash": "u0", "assistant_hash": "a0"})
    up = FakeUpstream()
    app = create_app(_settings(tmp_path), upstream=up,
                     dream_llm=FakeLLM(_EXTRACTION))
    with TestClient(app) as client:               # with = 루프 유지
        r = client.post("/v1/chat/completions", json=_body("다음 질문"),
                        headers={"x-dreaming-session-id": "sess1"})
        assert r.status_code == 200               # 첫 요청은 즉시 통과
        for _ in range(100):                      # 백그라운드 꿈 완료 대기
            if storage.get("sess1/dreamer", "cursor"):
                break
            time.sleep(0.02)
    assert storage.get("sess1/dreamer", "cursor") is not None
    facts = MemoryStore(storage, "sess1").list_facts()
    assert any(f.claim == "포션은 50골드다" for f in facts)


def test_health(tmp_path):
    app = create_app(_settings(tmp_path), upstream=FakeUpstream())
    assert TestClient(app).get("/health").json() == {"ok": True}
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_proxy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.proxy'`

- [ ] **Step 3: 구현**

`dreaming/proxy.py`:

```python
"""dreaming/proxy.py — Phase 1 리버스 프록시 (스펙 §8).

RisuAI(OpenAI 호환 커스텀 URL) → 여기 → OpenAI 호환 업스트림(기본 OpenRouter).
동기 경로(SyncPath)로 주입·마킹하고, 응답 후 원장 기록·유휴 타이머·캐치업 드림.
fail-open: dreaming 오류는 절대 채팅을 막지 않는다 (스펙 §2.6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from dreaming.dreamer import Dreamer
from dreaming.idle import IdleWatcher
from dreaming.llm import LLMClient, OpenAICompatLLM
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath
from dreaming.upstream import OpenAIUpstream, to_wire

logger = logging.getLogger(__name__)

_SESSION_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class Settings(BaseModel):
    data_dir: str
    upstream_base_url: str
    upstream_api_key: str = ""
    idle_seconds: float = 300.0          # 캐시 TTL 5m와 동기 (스펙 §3.2)
    dream_base_url: str = ""
    dream_api_key: str = ""
    dream_model: str = ""                # 비면 Dreamer 비활성

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=os.environ.get("DREAMING_DATA_DIR", "./dreaming_data"),
            upstream_base_url=os.environ.get(
                "DREAMING_UPSTREAM_BASE", "https://openrouter.ai/api/v1"),
            upstream_api_key=os.environ.get("DREAMING_UPSTREAM_KEY", ""),
            idle_seconds=float(os.environ.get("DREAMING_IDLE_SECONDS", "300")),
            dream_base_url=os.environ.get("DREAMING_DREAM_BASE", ""),
            dream_api_key=os.environ.get("DREAMING_DREAM_KEY", ""),
            dream_model=os.environ.get("DREAMING_DREAM_MODEL", ""),
        )


def _session_of(request: Request, body: Dict) -> str:
    raw = request.headers.get("x-dreaming-session-id") or str(body.get("user") or "")
    s = _SESSION_SANITIZE_RE.sub("-", raw).strip("-")
    if not s or s in (".", ".."):
        return "default"
    return s


def _assistant_text(resp: Dict) -> str:
    try:
        content = resp["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def create_app(settings: Settings, *,
               upstream=None,
               dream_llm: Optional[LLMClient] = None) -> FastAPI:
    storage = JsonDirStorage(Path(settings.data_dir))
    up = upstream or OpenAIUpstream(
        settings.upstream_base_url, settings.upstream_api_key)

    llm = dream_llm
    if llm is None and settings.dream_model:
        llm = OpenAICompatLLM(
            settings.dream_base_url or settings.upstream_base_url,
            settings.dream_api_key or settings.upstream_api_key,
            settings.dream_model)
    dreamer = Dreamer(storage, llm) if llm is not None else None

    async def _on_idle(session: str) -> None:
        if dreamer is not None:
            await dreamer.dream(session)

    watcher = IdleWatcher(settings.idle_seconds, _on_idle)
    syncpaths: Dict[str, SyncPath] = {}
    seen_sessions: set = set()

    app = FastAPI(title="Dreaming Proxy")
    app.state.storage = storage
    app.state.dreamer = dreamer
    app.state.watcher = watcher

    def _sync(session: str) -> SyncPath:
        if session not in syncpaths:
            syncpaths[session] = SyncPath(storage, session)
        return syncpaths[session]

    def _finish(session: str, verdict, original_messages: List[Dict],
                assistant_text: str) -> None:
        """응답 완료 후: 원장 기록 → 유휴 타이머. 전부 fail-open."""
        if verdict is not None and assistant_text:
            try:
                _sync(session).record_response(
                    verdict, original_messages, assistant_text)
            except Exception:
                logger.exception("[proxy] record failed: %s", session)
        watcher.touch(session)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        original_messages = body.get("messages") or []
        session = _session_of(request, body)

        # 캐치업 드림 (스펙 §3.2): 이번 요청을 기록하기 *전에* backlog를 봐야
        # 신규 세션 오탐이 없다. 첫 요청은 즉시 통과, 꿈은 백그라운드.
        if (dreamer is not None and session not in seen_sessions
                and dreamer.has_backlog(session)):
            asyncio.create_task(dreamer.dream(session))
        seen_sessions.add(session)

        try:
            out, verdict = _sync(session).process(original_messages)
        except Exception:
            logger.exception("[proxy] sync path failed (fail-open): %s", session)
            out, verdict = original_messages, None

        payload = dict(body)
        payload["messages"] = to_wire(out)

        if body.get("stream"):
            async def relay():
                parts: List[str] = []
                buf = b""
                try:
                    async for chunk in up.stream(payload):
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            s = line.decode("utf-8", "ignore").strip()
                            if not s.startswith("data:"):
                                continue
                            data = s[len("data:"):].strip()
                            if data == "[DONE]":
                                continue
                            try:
                                delta = (json.loads(data)["choices"][0]
                                         ["delta"].get("content"))
                            except (ValueError, KeyError, IndexError, TypeError):
                                delta = None
                            if delta:
                                parts.append(delta)
                        yield chunk
                finally:
                    _finish(session, verdict, original_messages, "".join(parts))
            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            resp = await up.complete(payload)
        except Exception:
            logger.exception("[proxy] upstream failed")
            return JSONResponse(status_code=502, content={
                "error": "upstream_error", "message": "upstream request failed"})
        _finish(session, verdict, original_messages, _assistant_text(resp))
        return JSONResponse(resp)

    return app
```

`dreaming/__main__.py`:

```python
"""python3 -m dreaming — Phase 1 프록시 실행 (스펙 §8)."""
import uvicorn

from dreaming.proxy import Settings, create_app


def main() -> None:
    uvicorn.run(create_app(Settings.from_env()), host="127.0.0.1", port=8787)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_proxy.py -v`
Expected: 7 passed

- [ ] **Step 5: 커밋**

```bash
git add dreaming/proxy.py dreaming/__main__.py tests/test_dreaming_proxy.py
git commit -m "feat(dreaming): Phase 1 프록시 — 주입·마킹·기록·유휴 꿈·캐치업 배선 (스펙 §3, §8)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: 전체 회귀 + push

**Files:** 없음 (검증만)

- [ ] **Step 1: dreaming 전체 + 기존 스위트 회귀**

Run: `python3 -m pytest tests/ -q`
Expected: 전부 passed (Plan 2 종료 시점 414개 + 이번 플랜 신규 ~39개), saga/ diff 0 확인:

```bash
git diff --stat main -- saga/
```

Expected: 출력 없음

- [ ] **Step 2: push**

```bash
git push origin dreaming/spec
```

---

## Self-Review (작성 시 수행 완료)

- 스펙 §3.2 커버리지: B-0(Task 4 snapshot/lock/cursor), B-1+B-2(Task 2 단일 1콜 병합 — 스펙 "1콜" 문구 준수), B-3(Task 3 숫자 재검증·pending_contradiction·mem0 4분류), 캐치업 드림(Task 8 + Task 4 커서 재시도), 유휴 트리거(Task 6). **B-4 재압축·Tier·BP2는 Plan 4**, 임베딩·인과 링크는 후속 플랜 (Global Constraints에 명시).
- 스펙 §3.1 잔여분: reroll/diverged Fact 강등 + 커밋 격리 + 커서 되감기(Task 5) — Plan 2에서 못 했던 부분.
- 타입 일관성: `Verdict.reroll_turn_number`(identity.py), `MemoryStore` 시그니처, `Fact/StateCommit/Actor/Episode` 필드명 — 현행 코드와 대조 완료. `ExtractedNumber.unit=""` → `TypedNumber.unit=None` 변환은 `n.unit or None`으로 처리.
- 판정 주의: `test_full_turn_cycle_then_reroll`(기존)과 Task 5 신규 테스트는 같은 세션 흐름 사용 — SyncPath.process 수정 후에도 기존 sync 테스트 7개가 깨지면 안 됨 (Task 5 Step 4에서 회귀 확인).
