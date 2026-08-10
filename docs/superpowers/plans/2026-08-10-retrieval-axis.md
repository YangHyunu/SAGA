# 회수 축 복원 (지식 검색 + keyExcerpts) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 스펙 §3.1-2(지식 검색)와 §6.2(keyExcerpts)의 미구현 회수 축을 복원한다 — 기억 실패 8건 중 검색실패 3건·활용실패 1건이 이 축의 부재 탓이다 (docs/FINDINGS-2026-08-10-fix-run.md §4).

**Architecture:** 두 갈래. ① 지식 블록의 fact 선별을 "pinned+최신순 스택"에서 "어휘 랭킹(IDF+문자 bigram)"으로 교체 — 쿼리는 최근 장면(마지막 비-system 메시지들), LLM 0콜, 인덱스 없이 요청 시 즉석 계산. ② Dreamer가 에피소드를 접을 때 결정적 원문 인용(keyExcerpts, 유닛당 ≤3개 ≤400자)을 원문 대조 검증 후 Episode에 보존하고, Tier1 청크 조립에 포함 — 캐시되는 프리픽스에 살므로 턴당 추가 비용 ~0.

**Tech Stack:** Python 3.13, pydantic v2, pytest. 신규 의존성 없음.

## Global Constraints

- **Track B 파일 표면 수정 금지**: `benchmarks/eval/{prompts.py, run2.py, quality.py, lucid.py}`, `benchmarks/eval/prompts/` — 다른 세션이 진행 중 (FINDINGS §5 트랙 분리). 신규 파일 추가는 허용.
- **동기 경로 LLM 0콜** (스펙 §2): 검색·랭킹에 LLM/임베딩 호출 금지.
- **덴스·하이브리드 검색 도입 금지**: fix-drm-r0 오프라인 실측에서 기각됨 (FINDINGS §4 — 덴스가 어휘를 깎음, T29 1위→9위). 어휘 단독.
- **청크 byte-stable** (스펙 §6.1): 기존 에피소드(key_excerpts 없음)의 조립 바이트가 변하면 안 된다 — 필드 기본값 `[]`일 때 조립 결과는 현재와 동일해야 함.
- 테스트 실행: `python3 -m pytest` (이 머신에 `python` 없음).
- 커밋 메시지는 저장소 관례(한국어, `feat(dreaming):` 등) 유지. 각 커밋 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `dreaming_data/`는 로컬 전용(gitignore) — Task 3의 랩은 데이터 없으면 skip되게.

## 배경 (실행자가 알아야 할 최소 컨텍스트)

- Dreaming = RisuAI 리버스 프록시. 매 요청 프롬프트를 [system → 청크(접힌 과거, 캐시) → 원문 꼬리 → 지식 블록(캐시 밖) → 유저 입력]으로 조립한다.
- 지식 블록은 `dreaming/sync.py:render_knowledge()`가 만든다. 현재 fact를 **pinned 우선+최신순**으로 예산(6000자)까지 채운다 — 관련성 개념이 없다. 스펙 §3.1-2는 "현재 장면 기준으로 관련분만 선별"을 요구한다. 이 갭이 이 플랜의 본체.
- 실측 데이터 `dreaming_data/fix-drm-r0/`: 100턴 런의 산출물. `facts/` 353개(confirmed 329), `raw/` 99턴 원문(`user_text`/`assistant_text`/`turn_number`), 프로브 결과는 `dreaming_data/eval/v2-fix-drm-r0-run0.json`.
- 알려진 한계(고치지 않음): T59류 패러프레이즈 질문("그때 그것 덕분에 몸이 따뜻해졌던")은 어휘로 못 잡는다 — 실측에서 덴스도 못 잡았다(순위 152/329). 이 플랜의 목표는 검색실패 3건 중 어휘로 잡히는 것(T29·T39)+활용실패(T99)의 회복이지 전건 회복이 아니다.

## File Structure

| 파일 | 책임 |
|---|---|
| `dreaming/retrieval.py` (신규) | 어휘 피처 추출 + fact 랭킹. 순수 함수, 저장소·LLM 무의존 |
| `dreaming/sync.py` (수정) | `render_knowledge`가 랭킹 순서로 fact를 채움. 쿼리 텍스트 추출 헬퍼 |
| `dreaming/records.py` (수정) | `Episode.key_excerpts` 필드 추가 |
| `dreaming/dreamer.py` (수정) | 추출 스키마·프롬프트에 key_excerpts 추가 + 원문 대조 검증 |
| `dreaming/chunks.py` (수정) | Tier1 조립에 발췌 포함 (Tier2는 제외) |
| `benchmarks/eval/retrieval_lab.py` (신규) | fix-drm-r0 대상 오프라인 순위 측정 (라이브 런 없이 $0 반복) |
| `docs/dreaming/SPEC.md` (수정) | §6.3 프로바이더별 TTL 주석, §5 예산 실측 각주 |
| 테스트 | `tests/test_dreaming_retrieval.py` (신규), `tests/test_dreaming_sync.py`, `tests/test_dreaming_records.py`, `tests/test_dreaming_extraction.py`, `tests/test_dreaming_chunks.py` (수정) |

---

### Task 1: 어휘 랭킹 모듈 `dreaming/retrieval.py`

**Files:**
- Create: `dreaming/retrieval.py`
- Test: `tests/test_dreaming_retrieval.py`

**Interfaces:**
- Consumes: `dreaming.records.Fact` (필드: `claim: str`, `entities: List[str]`, `pinned: bool`, `recorded_at: str`)
- Produces: `features(text: str) -> Set[str]`, `rank_facts(facts: List[Fact], query: str) -> List[Fact]` — Task 2가 이 시그니처 그대로 사용

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""tests/test_dreaming_retrieval.py — 어휘 랭킹 (스펙 §3.1-2)."""
from dreaming.records import Fact
from dreaming.retrieval import features, rank_facts


def _fact(claim, **kw):
    kw.setdefault("status", "confirmed")
    return Fact(claim=claim, **kw)


def test_features_토큰과_bigram():
    fs = features("잿빛 강돌은 열쇠다")
    assert "강돌은" in fs          # 토큰
    assert "강돌" in fs            # bigram — 조사 변형 흡수의 핵심
    assert "잿빛" in fs


def test_쿼리_고유명사가_최신성을_이긴다():
    old = _fact("잿빛 강돌은 돌 관의 십자 표식 중앙에 끼워 넣는 열쇠 역할을 한다",
                recorded_at="2026-01-01T00:00:00+00:00")
    new = _fact("위지소연은 마을 사람들이 바치는 공물로 생활을 유지한다",
                recorded_at="2026-06-01T00:00:00+00:00")
    ranked = rank_facts([new, old], "그 강돌을 관에 끼우면 어떻게 되는 거야?")
    assert ranked[0] is old


def test_조사_변형은_bigram으로_매칭된다():
    f1 = _fact("설한초의 독은 일반 약초로 해독되지 않는다")
    f2 = _fact("보자기 속 강돌은 짐승의 뼈에서 떼어낸 것이다")
    ranked = rank_facts([f2, f1], "설한초를 먹으면 어떻게 돼?")  # "설한초의"≠"설한초를"
    assert ranked[0] is f1


def test_entities도_매칭에_참여():
    f1 = _fact("그 검은 왕가의 유물이다", entities=["은검", "유리"])
    f2 = _fact("마을 축제는 보름마다 열린다")
    ranked = rank_facts([f2, f1], "은검 얘기 좀 해줘")
    assert ranked[0] is f1


def test_빈_쿼리는_최신순_폴백():
    old = _fact("옛 사실", recorded_at="2026-01-01T00:00:00+00:00")
    new = _fact("새 사실", recorded_at="2026-06-01T00:00:00+00:00")
    assert rank_facts([old, new], "")[0] is new


def test_pinned는_점수_무관_선두():
    pin = _fact("핀 사실", pinned=True, recorded_at="2026-01-01T00:00:00+00:00")
    hit = _fact("강돌은 열쇠다", recorded_at="2026-06-01T00:00:00+00:00")
    assert rank_facts([hit, pin], "강돌 어디 씀?")[0] is pin


def test_동점은_최신순():
    a = _fact("무관한 사실 하나", recorded_at="2026-01-01T00:00:00+00:00")
    b = _fact("무관한 사실 둘", recorded_at="2026-06-01T00:00:00+00:00")
    ranked = rank_facts([a, b], "강돌")   # 둘 다 점수 0
    assert ranked[0] is b
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.retrieval'`

- [ ] **Step 3: 구현**

```python
"""dreaming/retrieval.py — 지식 검색: 어휘 랭킹 (스펙 §3.1-2).

fix-drm-r0 오프라인 실측(docs/FINDINGS-2026-08-10-fix-run.md §4)에서
어휘(IDF+bigram)가 덴스(bge-m3)·RRF 하이브리드를 이겼다 — 한국어 RP는
고유명사·아이템이 리터럴로 등장하고, 덴스는 어휘 상위를 오히려 깎았다.
그래서 어휘 단독이다. LLM 0콜, 영속 인덱스 없음 — fact 수백 개 규모에선
요청 시 즉석 계산이 충분히 싸다 (329개 기준 ~ms).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set

from dreaming.records import Fact

_RUN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def features(text: str) -> Set[str]:
    """토큰 + 문자 bigram. 한국어는 조사가 붙어 토큰 일치가 잘 안 되므로
    bigram이 주력이고, 토큰 자체는 영단어·숫자 리터럴 일치를 담당한다."""
    out: Set[str] = set()
    for run in _RUN_RE.findall(text):
        run = run.lower()
        out.add(run)
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
    return out


def rank_facts(facts: List[Fact], query: str) -> List[Fact]:
    """pinned 먼저, 그 안에서 어휘 점수 내림차순, 동점은 최신순.

    점수 0인 사실은 자동으로 최신순 꼬리가 된다 — 예산이 남으면 기존
    최신순 동작이 바닥값으로 유지된다 (쿼리가 빈약한 초반 턴 방어).
    빈 쿼리면 계산 없이 기존 동작(핀 우선+최신순) 그대로.
    """
    if not query.strip():
        return sorted(facts, key=lambda f: (f.pinned, f.recorded_at),
                      reverse=True)
    q = features(query)
    n = len(facts) or 1
    df: Dict[str, int] = {}
    docs: List[Set[str]] = []
    for f in facts:
        fs = features(f.claim + " " + " ".join(f.entities))
        docs.append(fs)
        for feat in fs:
            df[feat] = df.get(feat, 0) + 1
    scores = [sum(math.log(1.0 + n / (1 + df[feat])) for feat in (q & docs[i]))
              for i in range(len(facts))]
    order = sorted(range(len(facts)),
                   key=lambda i: (facts[i].pinned, scores[i],
                                  facts[i].recorded_at),
                   reverse=True)
    return [facts[i] for i in order]
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_retrieval.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add dreaming/retrieval.py tests/test_dreaming_retrieval.py
git commit -m "feat(dreaming): 어휘 랭킹 모듈 — 지식 검색의 랭커 (스펙 §3.1-2)

fix-drm-r0 실측(FINDINGS §4)에서 이긴 IDF+문자 bigram 단독. 덴스·하이브리드는
실측 기각으로 미도입. LLM 0콜, 인덱스 없이 요청 시 즉석 계산.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: `render_knowledge` 랭킹 배선 + 쿼리 추출

**Files:**
- Modify: `dreaming/sync.py` — `render_knowledge()` (33행 부근), `process_request()`의 `clip_knowledge(render_knowledge(self._store))` 호출부 (182행 부근)
- Test: `tests/test_dreaming_sync.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: Task 1의 `rank_facts(facts, query)`
- Produces: `render_knowledge(store, query: str = "") -> str` (예산 파라미터·블록 구조는 불변), `scene_query(messages: List[Dict]) -> str` — 이후 태스크는 이를 몰라도 됨

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_dreaming_sync.py`의 "render_knowledge" 섹션에 추가. 파일 상단 import에 `scene_query` 추가 필요. 기존 fixture 패턴은 `MemoryStore(JsonDirStorage(tmp_path), "sess1")` (같은 파일 `test_render_includes_state_pinned_facts_main_actors` 참조):

```python
def test_render_knowledge_쿼리_관련_사실이_최신순을_이긴다(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.save_fact(Fact(claim="잿빛 강돌은 돌 관의 십자 표식에 끼우는 열쇠다",
                      status="confirmed",
                      recorded_at="2026-01-01T00:00:00+00:00"))
    for i in range(400):   # 예산 6000자 초과 유도 — 최신순이면 강돌이 잘린다
        ms.save_fact(Fact(claim=f"무관한 최신 사실 {i:03d} " + "채움" * 10,
                          status="confirmed",
                          recorded_at=f"2026-06-01T00:{i // 60:02d}:{i % 60:02d}+00:00"))
    text = render_knowledge(ms, query="그 강돌을 관에 끼우면 어떻게 되지?")
    assert "강돌" in text          # 쿼리 없던 시절엔 예산에 밀려 탈락하던 사실


def test_render_knowledge_쿼리_없으면_최신순_유지(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.save_fact(Fact(claim="옛 사실", status="confirmed",
                      recorded_at="2026-01-01T00:00:00+00:00"))
    ms.save_fact(Fact(claim="새 사실", status="confirmed",
                      recorded_at="2026-06-01T00:00:00+00:00"))
    text = render_knowledge(ms)    # query 기본값 "" — 기존 동작 보존
    assert text.index("새 사실") < text.index("옛 사실")


def test_scene_query_system_제외_최근_메시지만():
    msgs = [{"role": "system", "content": "카드"},
            {"role": "user", "content": "옛말 " * 3000},   # 캡 검증용 장문
            {"role": "assistant", "content": "응답A"},
            {"role": "user", "content": "강돌 얘기"}]
    q = scene_query(msgs)
    assert "카드" not in q
    assert "강돌 얘기" in q
    assert len(q) <= 2400
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_sync.py -q -k "render_knowledge_쿼리 or scene_query"`
Expected: FAIL — `scene_query` 미정의 / 쿼리 파라미터 없음

- [ ] **Step 3: 구현** — `dreaming/sync.py`:

```python
from dreaming.retrieval import rank_facts   # 파일 상단 import에 추가

_QUERY_CHAR_CAP = 2400   # 최근 장면 쿼리 상한 — 마지막 user + 직전 응답이면 충분


def scene_query(messages: List[Dict]) -> str:
    """지식 검색 쿼리 = 현재 장면 (스펙 §3.1-2 "현재 장면 기준").

    마지막 비-system 메시지들에서 뒤에서부터 모은다. 유저 질문만이 아니라
    직전 응답까지 포함해야 대명사 질문("그거 어디 씀?")의 선행사가 잡힌다.
    """
    parts: List[str] = []
    total = 0
    for m in reversed(messages):
        if m.get("role") == "system":
            continue
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        parts.append(c)
        total += len(c)
        if len(parts) >= 4 or total >= _QUERY_CHAR_CAP:
            break
    return "\n".join(reversed(parts))[-_QUERY_CHAR_CAP:]
```

`render_knowledge` 수정 — 시그니처에 `query: str = ""` 추가, fact 정렬 한 줄 교체:

```python
def render_knowledge(store: MemoryStore, query: str = "",
                     budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    ...  # state_block, actor_block 그대로
    facts = [f for f in store.list_facts()
             if f.pinned or f.status == "confirmed"]
    facts = rank_facts(facts, query)      # ← 기존 facts.sort(...) 대체
    ...  # room 루프 그대로 (랭킹 순서로 예산까지 채움)
```

기존 docstring의 "pinned 우선, 그 안에서 최신순" 문장은 "pinned 우선, 그 안에서 장면 관련도순(어휘 랭킹, 빈 쿼리면 최신순)"으로 갱신. 호출부(182행 부근):

```python
knowledge = clip_knowledge(render_knowledge(self._store,
                                            query=scene_query(messages)))
```

- [ ] **Step 4: 통과 확인 + 전체 회귀**

Run: `python3 -m pytest tests/test_dreaming_sync.py tests/test_dreaming_retrieval.py -q` 후 `python3 -m pytest tests/ -q`
Expected: 전부 PASS (기존 render_knowledge 테스트 중 최신순을 가정한 게 깨지면 — 그 테스트가 빈 쿼리 경로인지 확인. 빈 쿼리면 동작 불변이어야 하므로 구현 버그다. 쿼리 있는 경로를 가정한 기존 테스트는 없다.)

- [ ] **Step 5: Commit**

```bash
git add dreaming/sync.py tests/test_dreaming_sync.py
git commit -m "feat(dreaming): 지식 블록 fact 선별을 장면-어휘 랭킹으로 — 스펙 §3.1-2 복원

최신순 스택이 예산(6000자)에서 잘리며 정답 사실 58%를 버렸다(FINDINGS §4
검색실패 3건). 이제 최근 장면을 쿼리로 관련분부터 채운다. 빈 쿼리는 기존
동작(핀+최신순) 그대로 — 초반 턴 안전.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: 오프라인 검색 랩 — fix-drm-r0 실측 재현

**Files:**
- Create: `benchmarks/eval/retrieval_lab.py`
- Test: `tests/test_dreaming_retrieval.py`에 스모크 1개 추가

**Interfaces:**
- Consumes: Task 1 `rank_facts`, `dreaming_data/fix-drm-r0/{facts,raw}/*.json`, `dreaming_data/eval/v2-fix-drm-r0-run0.json`
- Produces: 실행 스크립트 (라이브러리 아님). 성공 기준의 측정 도구

- [ ] **Step 1: 데이터 구조 (검증 완료 — 탐색 불필요)**

`v2-fix-drm-r0-run0.json`은 dict이고 `probes` 키에 레코드 9개. 각 레코드 필드 (2026-08-10 실물 확인): `turn`(int), `ptype`("recent"/"false"/…), `fact`(정답 명제), `value`(정답 값 문자열), `question`(프로브 질문 전문), `miss_cause`, `oracle`/`judge`. 프로브 턴은 19·29·39·49·59·69·79·89·99 — 이 중 검색 관심 대상은 **29·39·59·99** (나머지는 추출실패로 스토어에 정답이 없어 검색 무의미. FINDINGS §4).

- [ ] **Step 2: 랩 스크립트 작성**

```python
"""benchmarks/eval/retrieval_lab.py — 검색 랭킹 오프라인 실측 ($0, LLM 0콜).

fix-drm-r0 스토어(353 fact) + 프로브에 대해 rank_facts의 정답 순위를 잰다.
FINDINGS §4의 어휘 실측(T29=1, T39=30, T59=328, T99=2)을 재현·개선 확인.

실행: python3 -m benchmarks.eval.retrieval_lab
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

from dreaming.records import Fact
from dreaming.retrieval import rank_facts

DATA = Path("dreaming_data/fix-drm-r0")
RUN = Path("dreaming_data/eval/v2-fix-drm-r0-run0.json")
BUDGET_CHARS = 6000            # HOT_ZONE_CHAR_BUDGET와 동일


def load_facts() -> list[Fact]:
    return [Fact.model_validate(json.load(open(p)))
            for p in glob.glob(str(DATA / "facts" / "*.json"))]


def load_raw() -> dict[int, dict]:
    rows = [json.load(open(p)) for p in glob.glob(str(DATA / "raw" / "*.json"))]
    return {r["turn_number"]: r for r in rows}


def load_probes() -> list[dict]:
    """[{turn, question, answer, ptype}] — 구조는 Task 3 Step 1 참조."""
    d = json.load(open(RUN))
    return [{"turn": p["turn"], "question": p["question"],
             "answer": p["value"], "ptype": p["ptype"]}
            for p in d["probes"]]


def budget_cut(ranked: list[Fact]) -> int:
    """예산 안에 들어가는 fact 개수 (render_knowledge의 room 루프 근사)."""
    room, count = BUDGET_CHARS, 0
    for f in ranked:
        line = len(f"- {f.claim}") + 1
        if line > room:
            break
        room -= line
        count += 1
    return count


def main() -> int:
    facts, raw = load_facts(), load_raw()
    print(f"facts={len(facts)}")
    for p in load_probes():
        # 쿼리 = 질문 + 프로브 직전 턴 원문 (라이브의 scene_query 근사)
        prev = raw.get(p["turn"] - 1)
        query = (prev["assistant_text"][-1200:] + "\n" if prev else "") + p["question"]
        ranked = rank_facts(facts, query)
        cut = budget_cut(ranked)
        hit = [i for i, f in enumerate(ranked) if p["answer"] in f.claim]
        rank = (hit[0] + 1) if hit else None
        status = "∅스토어에없음" if rank is None else (
            "IN " if rank <= cut else "OUT") + f" rank={rank}/{len(facts)}"
        print(f"T{p['turn']:>3} [{p['ptype']:>7}] {status} "
              f"(budget_cut={cut}) {p['question'][:40]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: 스모크 테스트 추가** — `tests/test_dreaming_retrieval.py`:

```python
import pytest
from pathlib import Path


@pytest.mark.skipif(not Path("dreaming_data/fix-drm-r0/facts").is_dir(),
                    reason="로컬 실측 데이터 없음")
def test_retrieval_lab_실데이터_로드():
    from benchmarks.eval.retrieval_lab import load_facts
    assert len(load_facts()) > 300
```

- [ ] **Step 4: 실행 + 결과 기록**

Run: `python3 -m benchmarks.eval.retrieval_lab`
Expected: T29·T39·T99가 `IN`(예산 내), T59는 `OUT` 허용 (알려진 한계). **결과 숫자를 이 플랜 파일 하단 "실행 후기"에 기록.** T29나 T39가 `OUT`이면 쿼리 조립(직전 턴 포함 범위)을 조정하고 재실행 — 그래도 안 되면 멈추고 결과와 함께 보고.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/retrieval_lab.py tests/test_dreaming_retrieval.py
git commit -m "feat(eval): 검색 랭킹 오프라인 랩 — fix-drm-r0 프로브 순위 실측

라이브 런 없이 rank_facts 반복 측정. FINDINGS §4 어휘 실측 재현 도구.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: keyExcerpts 추출 — 스키마·프롬프트·원문 대조 검증

**Files:**
- Modify: `dreaming/records.py` — `Episode`에 필드 추가 (60행 부근)
- Modify: `dreaming/dreamer.py` — `ExtractedEpisode`(66행 부근), `_SYSTEM` 스키마·규칙(102행·118행 부근), `apply_extraction`의 에피소드 루프(278행 부근), 검증 헬퍼 신규
- Test: `tests/test_dreaming_extraction.py`, `tests/test_dreaming_records.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: `apply_extraction(store, ext, raw_by_turn)` 기존 시그니처 (raw_by_turn: `{turn: {"user_text", "assistant_text", ...}}`)
- Produces: `Episode.key_excerpts: List[str]`, `clean_excerpts(excerpts: List[str], source_text: str) -> List[str]` — Task 5가 `Episode.key_excerpts`를 소비

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_dreaming_records.py`에 추가:
```python
def test_episode_key_excerpts_기본값_빈리스트():
    ep = Episode(range_start="a", range_end="b", title="t", summary="s")
    assert ep.key_excerpts == []
```

`tests/test_dreaming_extraction.py`에 추가 (기존 fixture 패턴 준수):
```python
from dreaming.dreamer import clean_excerpts


def test_clean_excerpts_원문에_있으면_통과():
    src = "소연이 말했다. 생강과 꿀을 우린 차는 몸을 데운다. 그리고 떠났다."
    assert clean_excerpts(["생강과 꿀을 우린 차는 몸을 데운다."], src) == \
        ["생강과 꿀을 우린 차는 몸을 데운다."]


def test_clean_excerpts_지어낸_인용은_폐기():
    assert clean_excerpts(["원문에 없는 문장"], "실제 원문 텍스트") == []


def test_clean_excerpts_공백_정규화_후_대조():
    src = "약속은  보름달이\n뜨는 밤이다"
    assert clean_excerpts(["약속은 보름달이 뜨는 밤이다"], src) \
        == ["약속은 보름달이 뜨는 밤이다"]


def test_clean_excerpts_최대_3개_각_400자():
    src = "가" * 2000
    out = clean_excerpts(["가" * 500] * 5, src)
    assert len(out) == 3 and all(len(x) <= 400 for x in out)


def test_apply_extraction_에피소드에_검증된_발췌만_저장(tmp_path):
    # store/raw_by_turn 준비는 같은 파일의 기존 apply_extraction 테스트와 동일 방식
    ms = MemoryStore(JsonDirStorage(tmp_path), "s")
    raw_by_turn = {1: {"turn_number": 1, "user_hash": "h1",
                       "user_text": "차 좀 줘",
                       "assistant_text": "생강과 꿀을 우린 차를 내밀었다"}}
    ext = DreamExtraction(episodes=[ExtractedEpisode(
        start_turn=1, end_turn=1, title="차", summary="차를 마셨다",
        key_excerpts=["생강과 꿀을 우린 차를 내밀었다", "지어낸 인용문"])])
    apply_extraction(ms, ext, raw_by_turn)
    (ep,) = ms.list_episodes()
    assert ep.key_excerpts == ["생강과 꿀을 우린 차를 내밀었다"]
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_records.py -q`
Expected: FAIL — `key_excerpts` 필드 없음 / `clean_excerpts` 미정의

- [ ] **Step 3: 구현**

`records.py` — Episode에 한 줄 (embedding 위):
```python
    key_excerpts: List[str] = []  # 결정적 원문 비압축 보존 (스펙 §6.2 keyExcerpts)
```

`dreamer.py` — `ExtractedEpisode`에 `key_excerpts: List[str] = []`. `_SYSTEM`의 episodes 스키마 줄을:
```
"episodes": [{"start_turn": int, "end_turn": int, "title": str, "summary": str,
              "open_threads": [str], "key_excerpts": [str]}],
```
규칙 블록에 추가:
```
- key_excerpts: 그 에피소드에서 나중에 정확히 되짚어야 할 결정적 원문 문장을
  **원문 그대로 복사** (최대 3개, 각 400자 이내). 수치·약속·비법·해독법·아이템
  전달처럼 요약하면 디테일이 죽는 문장만. 원문에 없는 문장은 폐기된다.
```
검증 헬퍼 (모듈 레벨, WygLore 파라미터 — 스펙 §6.2):
```python
_WS_RE = re.compile(r"\s+")
EXCERPT_MAX = 3        # 유닛당 3개 (스펙 §6.2, WygLore)
EXCERPT_CHAR_CAP = 400  # 400자 게이트


def clean_excerpts(excerpts: List[str], source_text: str) -> List[str]:
    """원문 substring 검증 — 지어낸 인용 폐기 (공백 정규화 후 대조).

    verify_numbers와 같은 철학: LLM 출력은 원문 대조를 통과해야 저장된다.
    """
    src = _WS_RE.sub(" ", source_text)
    out: List[str] = []
    for ex in excerpts[:EXCERPT_MAX]:
        ex = _WS_RE.sub(" ", ex).strip()[:EXCERPT_CHAR_CAP]
        if ex and ex in src:
            out.append(ex)
    return out
```
`apply_extraction` 에피소드 루프 — `save_episode` 직전에:
```python
        span = [raw_by_turn[t] for t in range(ep.start_turn, ep.end_turn + 1)
                if t in raw_by_turn]
        source = " ".join(r["user_text"] + " " + r["assistant_text"]
                          for r in span)
        excerpts = clean_excerpts(ep.key_excerpts, source)
```
그리고 `Episode(...)` 생성자에 `key_excerpts=excerpts` 추가.

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_records.py tests/test_dreaming_dreamer.py -q`
Expected: PASS (기존 dreamer 테스트의 스키마 파싱도 기본값 `[]`로 무해)

- [ ] **Step 5: Commit**

```bash
git add dreaming/records.py dreaming/dreamer.py tests/test_dreaming_records.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): keyExcerpts 추출 — 결정적 원문 보존 (스펙 §6.2)

에피소드 접힘 = 디테일 사망이던 것을 유닛당 3개·400자 원문 인용으로 보상.
인용은 원문 substring 검증 통과분만 저장 (verify_numbers와 같은 철학).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Tier1 청크 조립에 발췌 포함

**Files:**
- Modify: `dreaming/chunks.py` — `assemble_tier1()` (39행 부근)
- Test: `tests/test_dreaming_chunks.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: Task 4의 `Episode.key_excerpts`
- Produces: 없음 (말단)

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_dreaming_chunks.py`에 추가:

```python
def test_tier1_발췌_포함():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.",
                 key_excerpts=["은검을 300골드에 팔았다"])
    out = assemble_tier1(ep)
    assert '원문: "은검을 300골드에 팔았다"' in out


def test_tier1_발췌_없으면_기존_바이트_그대로():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.")
    assert "원문:" not in assemble_tier1(ep)   # byte-stable — 구 에피소드 불변


def test_tier2_발췌_미포함():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.",
                 key_excerpts=["은검을 300골드에 팔았다"])
    assert "원문:" not in assemble_tier2([ep])
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py -q -k 발췌`
Expected: FAIL — tier1에 원문 줄 없음

- [ ] **Step 3: 구현** — `assemble_tier1`:

```python
def assemble_tier1(ep: Episode) -> str:
    """에피소드 청크 (~70% 압축): 제목 + 요약 + 결정적 원문 + 미회수 복선.

    발췌는 캐시되는 프리픽스에 살므로 턴당 추가 비용은 read 0.1×뿐이다.
    Tier2 승격 시 발췌는 버려진다 — 먼 과거의 디테일 회수는 지식 검색 몫.
    """
    lines = [f"[지난 이야기 · {ep.title}]", ep.summary.strip()]
    for ex in ep.key_excerpts:
        lines.append(f'원문: "{ex}"')
    if ep.open_threads:
        lines.append("남은 실마리: " + " / ".join(ep.open_threads))
    return "\n".join(lines)
```

- [ ] **Step 4: 통과 확인 + 전체 회귀**

Run: `python3 -m pytest tests/test_dreaming_chunks.py -q` 후 `python3 -m pytest tests/ -q`
Expected: 전부 PASS

- [ ] **Step 5: Commit**

```bash
git add dreaming/chunks.py tests/test_dreaming_chunks.py
git commit -m "feat(dreaming): Tier1 청크에 keyExcerpts 렌더 — 접힌 과거의 디테일 보존

발췌 없는 구 에피소드는 조립 바이트 불변 (캐시 안전).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: SPEC 개정 — 실측이 뒤집은 2곳

**Files:**
- Modify: `docs/dreaming/SPEC.md` — §6.3 (261행 부근), §5 (221행 부근)

**Interfaces:** 없음 (문서)

- [ ] **Step 1: §6.3에 프로바이더 주석 추가** — 섹션 끝에:

```markdown
- **프로바이더 한정 주의** (fix-drm-r0 실측, FINDINGS §2): "유휴 재압축 = 공짜"는
  Anthropic처럼 TTL 만료로 캐시가 소멸하는 프로바이더에서만 성립한다. DeepSeek의
  자동 프리픽스 캐싱은 유휴와 무관하게 바이트가 바뀌면 그대로 miss — 재압축 비용이
  0이 아니다. 비-Anthropic에서는 재압축 빈도 자체를 낮추는 것(BOUNDARY_STEP)이 방어선.
```

- [ ] **Step 2: §5 주입 예산 각주** — 레이아웃 코드블록 아래에:

```markdown
- 주입 예산 실측 (fix-drm-r0, 100턴): confirmed fact 329개 = 약 14K자(≈7~9K tok).
  "선별 ≤2K"는 비용 상한이지 물리 한계가 아니다 — 예산은 `HOT_ZONE_CHAR_BUDGET`
  파라미터로 두고, 상향 여부는 retrieval_lab의 예산 내 적중률로 결정한다.
```

- [ ] **Step 3: Commit**

```bash
git add docs/dreaming/SPEC.md
git commit -m "docs(dreaming): SPEC 실측 반영 — §6.3 TTL 트릭 프로바이더 한정, §5 예산 각주

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 명시적 Out of Scope

- **Track A** (Dreamer 추출실패 4건 — T19·69·79·89): fact가 스토어에 없어 검색으로 회수 불가. 별도 플랜.
- **벤치 고정 건초더미** (4변형 동일 히스토리·동일 질문): 별도 플랜. 이 플랜 완료 전 야간 본런 재실행 금지.
- 덴스/임베딩 검색, Actor `knows[]` POV 게이팅, Tier3 시놉시스, pinned 승격 지연: 스펙엔 있으나 이번 범위 아님.

## 완료 기준

1. `python3 -m pytest tests/ -q` 전부 PASS.
2. retrieval_lab: T29·T39·T99 예산 내 `IN` (T59는 `OUT` 허용 — 알려진 한계로 기록).
3. 구 에피소드(발췌 없음)의 Tier1 조립 바이트 불변.
4. 실행 후기(플랜 결함·결과 숫자)를 이 파일 하단에 기록 — CLAUDE.md 플랜 후기 규약.
