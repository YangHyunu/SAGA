# 회수 축 복원 (지식 검색 + keyExcerpts) Implementation Plan — v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **v2 개정 이력**: v1을 Sonnet5(구현 정합성)·Opus5(설계 리스크) 이중 리뷰 후 개정.
> 반영: 랩 turn 좌표계 PAD 버그, 데이터 경로 env화, 게이트를 T39 단독+예산 곡선으로
> 재정의, scene_query를 유저턴+직전응답600자로 교체, 태스크 순서 1→랩→배선,
> keyExcerpts Tier2 병합 5개(스펙 §6.2 원안, 유저 확정), clean_excerpts 검증-후-컷,
> ExtractedEpisode import, 첫 요청 프리필 가드. 리뷰가 안전 판정한 것: 재드림
> byte-stability, 주입 재귀 오염 없음, 리롤 랭킹 요동 없음, 랭킹 변동의 캐시 영향 0.

**Goal:** 스펙 §3.1-2(지식 검색)와 §6.2(keyExcerpts)의 미구현 회수 축을 복원한다. 실측 기대 효과는 정직하게: 검색실패 3건 중 어휘로 잡히는 **T39 회복**이 게이트고, T29·T99는 매칭 모호/이미 주입 중이라 진단만, T59(패러프레이즈)는 예산 곡선으로 절단 근거를 만든다 (docs/FINDINGS-2026-08-10-fix-run.md §4).

**Architecture:** 두 갈래. ① 지식 블록의 fact 선별을 "pinned+최신순 스택"에서 "어휘 랭킹(IDF+문자 bigram)"으로 교체 — 쿼리는 마지막 유저 턴+직전 응답 600자 캡, LLM 0콜, 인덱스 없이 요청 시 즉석 계산. ② Dreamer가 에피소드를 접을 때 결정적 원문 인용(keyExcerpts, 유닛당 ≤3개 ≤400자, 원문 substring 검증)을 Episode에 보존하고 Tier1 청크에 렌더, **Tier2 병합 시 5개 캡으로 생존**(스펙 §6.2 원안) — 캐시되는 프리픽스에 살므로 턴당 추가 비용은 read 0.1×.

**Tech Stack:** Python 3.13, pydantic v2, pytest. 신규 의존성 없음.

## Global Constraints

- **Track B 파일 표면 수정 금지**: `benchmarks/eval/` 전체 — 다른 세션 진행 중 (FINDINGS §5). 랩은 `benchmarks/retrieval_lab.py`(eval/ 밖 신규 파일)에 둔다.
- **동기 경로 LLM 0콜** (스펙 §2): 검색·랭킹에 LLM/임베딩 호출 금지.
- **덴스·하이브리드 검색 도입 금지**: fix-drm-r0 실측 기각 (FINDINGS §4 — 덴스가 어휘를 깎음, T29 1위→9위). 어휘 단독.
- **청크 byte-stable** (스펙 §6.1): 기존 에피소드(`key_excerpts=[]`)의 Tier1·Tier2 조립 바이트 불변. (리뷰 검증: 재드림은 새 uuid Episode를 추가할 뿐이고 `build_compression`은 **최초 기록**을 채택하므로(`start_turn < next_turn` 스킵) 기존 청크는 안 바뀜 — 발췌는 크기만 키우고 새 불안정 클래스를 만들지 않는다.)
- 테스트 실행: `python3 -m pytest` (이 머신에 `python` 없음).
- 커밋 메시지는 저장소 관례(한국어, `feat(dreaming):` 등) 유지. 각 커밋 끝에 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- `dreaming_data/`는 gitignore — **메인 체크아웃(`/Users/yanghyeon-u/Desktop/RISU_ENE/dreaming_data`)에만 실존**하고 워크트리엔 없다. 랩·스모크는 `DREAMING_DATA_DIR` env로 경로를 받고, 없으면 명시적으로 skip.

## 배경 (실행자가 알아야 할 최소 컨텍스트)

- Dreaming = RisuAI 리버스 프록시. 매 요청 프롬프트를 [system → 청크(접힌 과거, 캐시) → 원문 꼬리 → 지식 블록(캐시 밖) → 유저 입력]으로 조립한다.
- 지식 블록은 `dreaming/sync.py:render_knowledge()`가 만든다. 현재 fact를 **pinned 우선+최신순**으로 예산(6000자)까지 채운다 — 관련성 개념이 없다. 스펙 §3.1-2는 "현재 장면 기준 관련분 선별"을 요구한다. 이 갭이 본체.
- 실측 데이터 `$DREAMING_DATA_DIR/fix-drm-r0/`: 100턴 런 산출물. `facts/` 353개(pinned∪confirmed 329), `raw/` 99턴 원문. **주의: raw의 `turn_number`는 1025~1123** (`_BASELINE_PAD=1024`, dreaming/identity.py) — 프로브의 `turn`(19~99)과 좌표계가 다르다.
- 프로브: `$DREAMING_DATA_DIR/eval/v2-fix-drm-r0-run0.json`의 `probes` 리스트 9개. 필드: `turn`, `ptype`, `fact`(정답 명제), `value`(정답 값), `question`(전문), `miss_cause` 등 (실물 확인 완료).
- **게이트가 T39 단독인 이유** (리뷰 실측): T29 정답값 `렌`은 fact 48개에 부분문자열 매칭 → 순위 무의미. T99 `십자 표식`은 현행 최신순으로도 이미 주입됨(rank 2) → 랭킹 무관. T59는 패러프레이즈라 어휘 한계(순위 ~247/329, 덴스도 152위 실패). **T39 `산 아래 마을`은 매칭 fact 정확히 1개 + 최신순 328/329위로 잘리던 케이스** — 랭킹이 실제로 살리는 유일한 비모호 표본.
- 알려진 비용 (리뷰 실측): 발췌 3×400자 상주 시 Tier1 청크 4~5배 팽창 → 리빌드 상각 **+~860자/턴 uncached (+11%)**. 감수하는 트레이드다 — 대안(hot zone 주입)은 턴당 전액 과금이라 더 비싸다.

## File Structure

| 파일 | 책임 |
|---|---|
| `dreaming/retrieval.py` (신규) | 어휘 피처·fact 랭킹·장면 쿼리 조립. 순수 함수, 저장소·LLM 무의존 |
| `benchmarks/retrieval_lab.py` (신규) | fix-drm-r0 대상 오프라인 측정 — 실제 `render_knowledge` 경로로 IN/OUT + 예산 곡선 |
| `dreaming/sync.py` (수정) | `render_knowledge`에 쿼리 랭킹 배선 + 호출부 (첫 요청 프리필 가드 포함) |
| `dreaming/records.py` (수정) | `Episode.key_excerpts` 필드 |
| `dreaming/dreamer.py` (수정) | 추출 스키마·프롬프트 + `clean_excerpts` 검증 |
| `dreaming/chunks.py` (수정) | Tier1 발췌 렌더 + Tier2 병합 5개 캡 |
| `docs/dreaming/SPEC.md` (수정) | §6.3 프로바이더 주석, §5 예산 각주 |
| 테스트 | `tests/test_dreaming_retrieval.py` (신규), `tests/test_dreaming_sync.py`, `tests/test_dreaming_records.py`, `tests/test_dreaming_extraction.py`, `tests/test_dreaming_chunks.py` (수정) |

---

### Task 1: 어휘 랭킹 + 장면 쿼리 모듈 `dreaming/retrieval.py`

**Files:**
- Create: `dreaming/retrieval.py`
- Test: `tests/test_dreaming_retrieval.py`

**Interfaces:**
- Consumes: `dreaming.records.Fact` (필드: `claim: str`, `entities: List[str]`, `pinned: bool`, `recorded_at: str`)
- Produces: `features(text: str) -> Set[str]`, `rank_facts(facts: List[Fact], query: str) -> List[Fact]`, `scene_query(messages: List[Dict]) -> str` — Task 2(랩)와 Task 3(배선)이 이 시그니처 그대로 사용

- [x] **Step 1: 실패하는 테스트 작성**

```python
"""tests/test_dreaming_retrieval.py — 어휘 랭킹 + 장면 쿼리 (스펙 §3.1-2)."""
from dreaming.records import Fact
from dreaming.retrieval import features, rank_facts, scene_query


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


def test_scene_query_마지막유저_플러스_직전응답600():
    msgs = [{"role": "system", "content": "카드"},
            {"role": "user", "content": "이전 질문"},
            {"role": "assistant", "content": "긴 응답 " * 300},   # 1500자+
            {"role": "user", "content": "강돌 얘기"}]
    q = scene_query(msgs)
    assert "카드" not in q and "이전 질문" not in q
    assert q.endswith("강돌 얘기")
    assert len(q) <= 600 + 1 + len("강돌 얘기")   # 직전 응답은 600자 캡


def test_scene_query_유저턴만_있으면_그것만():
    assert scene_query([{"role": "user", "content": "안녕"}]) == "안녕"


def test_scene_query_비문자열_content_무시():
    msgs = [{"role": "user", "content": [{"type": "text"}]},
            {"role": "user", "content": "진짜 질문"}]
    assert scene_query(msgs) == "진짜 질문"
```

- [x] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_retrieval.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.retrieval'`

- [x] **Step 3: 구현**

```python
"""dreaming/retrieval.py — 지식 검색: 어휘 랭킹 + 장면 쿼리 (스펙 §3.1-2).

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

# 직전 assistant 응답의 쿼리 참여 상한. 플랜 리뷰 실측: 유저 턴 평균 206자
# vs 어시스턴트 1,925자라 장면 전체(2400자)를 쿼리로 쓰면 ~90%가 모델
# 산문이 되어 정답 순위가 오히려 밀렸다 (T39 1위→33위). 유저턴+직전응답
# 600자 캡이 4개 프로브 중 3개에서 최선이었다.
_PREV_ASSISTANT_CAP = 600


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


def scene_query(messages: List[Dict]) -> str:
    """지식 검색 쿼리 = 마지막 유저 턴 + 직전 assistant 응답 600자 캡.

    직전 응답을 붙이는 이유: 대명사 질문("그거 어디 씀?")의 선행사가
    직전 응답에 있다. 600자 캡인 이유: 위 _PREV_ASSISTANT_CAP 주석.
    """
    cur = ""
    prev_a = ""
    for m in reversed(messages):
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        role = m.get("role")
        if not cur:
            if role == "user":
                cur = c
            continue
        if role == "assistant":
            prev_a = c[-_PREV_ASSISTANT_CAP:]
            break
    return (prev_a + "\n" + cur) if prev_a else cur
```

- [x] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_retrieval.py -q`
Expected: 10 passed

- [x] **Step 5: Commit**

```bash
git add dreaming/retrieval.py tests/test_dreaming_retrieval.py
git commit -m "feat(dreaming): 어휘 랭킹 + 장면 쿼리 — 지식 검색의 랭커 (스펙 §3.1-2)

fix-drm-r0 실측(FINDINGS §4)에서 이긴 IDF+문자 bigram 단독. 덴스·하이브리드는
실측 기각으로 미도입. 쿼리는 유저턴+직전응답 600자 캡 — 장면 전체는 모델
산문이 쿼리를 희석해 실측 열등. LLM 0콜, 인덱스 없이 즉석 계산.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: 오프라인 검색 랩 — 실제 렌더 경로로 측정

**Files:**
- Create: `benchmarks/retrieval_lab.py` (eval/ 밖 — Track B 표면 회피)
- Test: `tests/test_dreaming_retrieval.py`에 스모크 1개 추가

**Interfaces:**
- Consumes: Task 1 `rank_facts`·`scene_query`, `dreaming.sync.render_knowledge`(현행 시그니처 — Task 3 배선 전엔 query 파라미터가 없으므로 **이 태스크는 Task 3 이후 재실행이 본측정**이고, 배선 전 실행은 baseline 측정이다), `dreaming.store.MemoryStore`, `dreaming.storage.JsonDirStorage`
- Produces: 실행 스크립트. 완료 기준 #2의 측정 도구

- [x] **Step 1: 랩 스크립트 작성**

```python
"""benchmarks/retrieval_lab.py — 검색 오프라인 측정 ($0, LLM 0콜).

fix-drm-r0 스토어에 대해 **실제 render_knowledge 경로**로 프로브 정답의
IN/OUT과 필요 예산(budget_needed)을 잰다. FINDINGS §4 어휘 실측의 방향성
근사 — 정확한 순위 재현이 아니라 예산 결정의 근거 산출이 목적이다.

게이트 해석 주의 (플랜 리뷰 실측):
- T39만 비모호 (value 매칭 fact 정확히 1개 + 최신순에서 잘리던 케이스)
- T29는 value '렌'이 fact 48개에 걸려 순위 무의미 → 진단용
- T99는 현행 최신순으로도 이미 IN → 랭킹 무관
- T19·49·69·79·89는 추출실패 — 스토어에 정답 없음 (Track A 몫)

실행: DREAMING_DATA_DIR=/path/to/dreaming_data python3 -m benchmarks.retrieval_lab
      (메인 체크아웃에서는 env 생략 가능 — 기본값 ./dreaming_data)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

from dreaming.retrieval import scene_query
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import render_knowledge

DATA_ROOT = Path(os.environ.get("DREAMING_DATA_DIR", "dreaming_data"))
SESSION = "fix-drm-r0"
RUN = DATA_ROOT / "eval" / "v2-fix-drm-r0-run0.json"
BUDGET_CHARS = 6000            # assembly.HOT_ZONE_CHAR_BUDGET와 동일


def _render(store, query: str, budget: int) -> str:
    """Task 3 배선 전(query 파라미터 없음)에도 baseline 측정이 되게 분기."""
    if "query" in inspect.signature(render_knowledge).parameters:
        return render_knowledge(store, query=query, budget=budget)
    return render_knowledge(store, budget=budget)


def budget_needed(store, query: str, answer: str) -> int | None:
    """정답이 지식 블록에 들어가는 최소 예산 (이분 탐색, 실제 렌더 경로)."""
    lo, hi = 200, 30000
    if answer not in _render(store, query, hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if answer in _render(store, query, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> int:
    if not RUN.is_file():
        print(f"skip: 데이터 없음 ({RUN}) — DREAMING_DATA_DIR로 메인 체크아웃의 "
              "dreaming_data를 지정하라")
        return 0
    store = MemoryStore(JsonDirStorage(DATA_ROOT), SESSION)
    raws = sorted((row for _, row in store._storage.scan(f"{SESSION}/raw")),
                  key=lambda r: r["turn_number"])
    pad = raws[0]["turn_number"] - 1           # _BASELINE_PAD 좌표계 유도
    raw_by_rel = {r["turn_number"] - pad: r for r in raws}
    facts = [f for f in store.list_facts() if f.pinned or f.status == "confirmed"]
    print(f"facts(주입 대상)={len(facts)}  pad={pad}")

    probes = json.load(open(RUN))["probes"]
    for p in probes:
        prev = raw_by_rel.get(p["turn"] - 1)   # 프로브 직전 턴 (상대 좌표)
        msgs = ([{"role": "assistant", "content": prev["assistant_text"]}]
                if prev else []) + [{"role": "user", "content": p["question"]}]
        query = scene_query(msgs)
        text = _render(store, query, BUDGET_CHARS)
        matches = sum(1 for f in facts if p["value"] in f.claim)
        need = budget_needed(store, query, p["value"])
        status = ("∅스토어에없음" if need is None
                  else ("IN " if p["value"] in text else "OUT")
                  + f" need={need}")
        print(f"T{p['turn']:>3} [{p['ptype']:>7}] {status} "
              f"(value매칭 {matches}개) {p['question'][:36]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [x] **Step 2: 스모크 테스트 추가** — `tests/test_dreaming_retrieval.py` 끝에:

```python
import os
from pathlib import Path

import pytest

_DATA = Path(os.environ.get("DREAMING_DATA_DIR", "dreaming_data"))


@pytest.mark.skipif(not (_DATA / "fix-drm-r0" / "facts").is_dir(),
                    reason="로컬 실측 데이터 없음")
def test_retrieval_lab_실데이터_로드():
    from dreaming.storage import JsonDirStorage
    from dreaming.store import MemoryStore
    ms = MemoryStore(JsonDirStorage(_DATA), "fix-drm-r0")
    assert len(ms.list_facts()) > 300
```

- [x] **Step 3: baseline 실행 (배선 전 — 현행 최신순의 성적표)**

Run: `DREAMING_DATA_DIR=/Users/yanghyeon-u/Desktop/RISU_ENE/dreaming_data python3 -m benchmarks.retrieval_lab`
Expected: T39 `OUT` (최신순 328/329로 잘림 — 리뷰 실측), T99 `IN`(이미 주입 중). **출력 전문을 플랜 하단 "실행 후기"에 baseline으로 기록.**

- [x] **Step 4: Commit**

```bash
git add benchmarks/retrieval_lab.py tests/test_dreaming_retrieval.py
git commit -m "feat(bench): 검색 오프라인 랩 — 실렌더 경로 IN/OUT + 필요예산 곡선

라이브 런 없이 \$0 측정. PAD 좌표계 유도로 probe turn(상대)↔raw
turn_number(1024 패딩) 정합. 게이트는 T39 단독 (유일한 비모호 표본).

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `render_knowledge` 랭킹 배선 (+ 첫 요청 프리필 가드)

**Files:**
- Modify: `dreaming/sync.py` — `render_knowledge()` (33행 부근), `process_request()`의 `clip_knowledge(render_knowledge(self._store))` 호출부 (182행 부근)
- Test: `tests/test_dreaming_sync.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: Task 1의 `rank_facts`, `scene_query`
- Produces: `render_knowledge(store, query: str = "", budget: int = HOT_ZONE_CHAR_BUDGET) -> str` — 기존 호출자(`benchmarks/cardsim/bench.py:241`, `benchmarks/longmemeval/run_dreaming.py:120`)는 위치 인자 없이 호출하므로 무해 (리뷰 전수 조사 완료)

- [x] **Step 1: 실패하는 테스트 작성** — `tests/test_dreaming_sync.py`의 render_knowledge 섹션에 추가. fixture 패턴은 같은 파일 `test_render_includes_state_pinned_facts_main_actors` 참조 (`MemoryStore(JsonDirStorage(tmp_path), "sess1")`):

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
```

- [x] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_sync.py -q -k "쿼리"`
Expected: FAIL — `render_knowledge() got an unexpected keyword argument 'query'`

- [x] **Step 3: 구현** — `dreaming/sync.py`:

import에 `from dreaming.retrieval import rank_facts, scene_query` 추가. `render_knowledge` 수정 — 시그니처에 `query: str = ""` 추가(2번째 파라미터), fact 정렬 한 줄 교체:

```python
def render_knowledge(store: MemoryStore, query: str = "",
                     budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    ...  # state_block, actor_block 그대로
    facts = [f for f in store.list_facts()
             if f.pinned or f.status == "confirmed"]
    facts = rank_facts(facts, query)      # ← 기존 facts.sort(...) 대체
    ...  # room 루프 그대로 (랭킹 순서로 예산까지 채움)
```

docstring의 "pinned 우선, 그 안에서 최신순"은 "pinned 우선, 그 안에서 장면 관련도순(어휘 랭킹, 빈 쿼리면 최신순)"으로 갱신. 호출부(182행 부근) — **첫 요청 가드 포함**:

```python
        # 첫 요청은 tail_fp를 못 배워 프리셋 프리필이 messages에 섞여 있다
        # (위 first_request 계산과 동일 조건) — 쿼리가 영문 보일러플레이트로
        # 오염되므로 빈 쿼리(=최신순 폴백)로 렌더한다.
        query = "" if first_request else scene_query(messages)
        knowledge = clip_knowledge(render_knowledge(self._store, query=query))
```

주의: `first_request` 변수는 이미 같은 함수 상단에 있다 (`first_request = not state.get("prev_fp")`). `scene_query`가 받는 `messages`는 `scaffold.split` 이후(꼬리 제거본)이다.

- [x] **Step 4: 통과 확인 + 전체 회귀**

Run: `python3 -m pytest tests/ -q`
Expected: 전부 PASS

- [x] **Step 5: 본측정 — 랩 재실행**

Run: `DREAMING_DATA_DIR=/Users/yanghyeon-u/Desktop/RISU_ENE/dreaming_data python3 -m benchmarks.retrieval_lab`
Expected (게이트): **T39 `IN`** (need ≤ 6000). 추가 산출: 전 프로브 `need` 값 = 예산 곡선 — T59의 need(리뷰 예측 ~8,250)를 확인해 "6000 유지 vs 상향" 결정 근거로 실행 후기에 기록. T29·T99는 진단 참고만 (게이트 아님 — 매칭 모호/이미 IN).
게이트 실패 시: 쿼리 조립(`_PREV_ASSISTANT_CAP`)을 조정해 볼 수 있으나, 2회 안에 안 되면 멈추고 baseline·본측정 출력과 함께 보고.

- [x] **Step 6: Commit**

```bash
git add dreaming/sync.py tests/test_dreaming_sync.py
git commit -m "feat(dreaming): 지식 블록 fact 선별을 장면-어휘 랭킹으로 — 스펙 §3.1-2 복원

최신순 스택이 예산(6000자)에서 잘리며 정답 사실을 버렸다(FINDINGS §4,
T39 실측 328/329위). 이제 유저턴+직전응답 쿼리로 관련분부터 채운다.
빈 쿼리·첫 요청(프리필 미분리)은 기존 최신순 폴백.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: keyExcerpts 추출 — 스키마·프롬프트·원문 대조 검증

**Files:**
- Modify: `dreaming/records.py` — `Episode`에 필드 추가 (60행 부근)
- Modify: `dreaming/dreamer.py` — `ExtractedEpisode`(66행 부근), `_SYSTEM` 스키마·규칙(102행·118행 부근), `apply_extraction`의 에피소드 루프(278행 부근), 검증 헬퍼 신규
- Test: `tests/test_dreaming_extraction.py`, `tests/test_dreaming_records.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: `apply_extraction(store, ext, raw_by_turn)` 기존 시그니처 (raw_by_turn: `{turn: {"user_text", "assistant_text", "user_hash", ...}}`)
- Produces: `Episode.key_excerpts: List[str]`, `clean_excerpts(excerpts: List[str], source_text: str) -> List[str]` — Task 5가 `Episode.key_excerpts`를 소비

- [x] **Step 1: 실패하는 테스트 작성**

`tests/test_dreaming_records.py`에 추가:
```python
def test_episode_key_excerpts_기본값_빈리스트():
    ep = Episode(range_start="a", range_end="b", title="t", summary="s")
    assert ep.key_excerpts == []
```

`tests/test_dreaming_extraction.py`에 추가. **주의: 이 파일의 기존 import(46행 부근)는 `DreamExtraction, ExtractedNumber, apply_extraction, verify_numbers`뿐 — `ExtractedEpisode`와 `clean_excerpts`를 import에 추가해야 한다** (리뷰 지적: 누락 시 NameError):

```python
from dreaming.dreamer import ExtractedEpisode, clean_excerpts   # 기존 import에 병합


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


def test_clean_excerpts_검증_후_컷_그리고_중복제거():
    # 리뷰 지적 반영: [:3]을 검증 전에 하면 유효 인용이 인덱스 3 이후일 때 유실
    src = "진짜 하나. 진짜 둘. 진짜 셋."
    out = clean_excerpts(["가짜", "가짜", "가짜", "진짜 하나.", "진짜 하나.",
                          "진짜 둘.", "진짜 셋."], src)
    assert out == ["진짜 하나.", "진짜 둘.", "진짜 셋."]   # 검증 통과분에서 3개, 중복 1회


def test_clean_excerpts_400자_캡():
    src = "가" * 2000
    out = clean_excerpts(["가" * 500], src)
    assert out == ["가" * 400]


def test_apply_extraction_에피소드에_검증된_발췌만_저장(tmp_path):
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

- [x] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_records.py -q`
Expected: FAIL — `key_excerpts` 필드 없음 / `clean_excerpts` import 불가

- [x] **Step 3: 구현**

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
검증 헬퍼 (모듈 레벨, WygLore 파라미터 — 스펙 §6.2). **검증 통과분에서 3개를 뽑는다 (검증 전 컷 금지) + 중복 제거**:
```python
_WS_RE = re.compile(r"\s+")
EXCERPT_MAX = 3          # 유닛당 3개 (스펙 §6.2, WygLore). 병합 시 5개 캡은
                         # chunks.EXCERPT_MERGE_MAX (dreamer→chunks 순환 방지)
EXCERPT_CHAR_CAP = 400   # 400자 게이트


def clean_excerpts(excerpts: List[str], source_text: str) -> List[str]:
    """원문 substring 검증 — 지어낸 인용 폐기 (공백 정규화 후 대조).

    verify_numbers와 같은 철학: LLM 출력은 원문 대조를 통과해야 저장된다.
    컷은 검증 뒤에 한다 — 앞에서 자르면 유효 인용이 인덱스 3 이후일 때 유실.
    """
    src = _WS_RE.sub(" ", source_text)
    out: List[str] = []
    for ex in excerpts:
        ex = _WS_RE.sub(" ", ex).strip()[:EXCERPT_CHAR_CAP]
        if ex and ex in src and ex not in out:
            out.append(ex)
        if len(out) >= EXCERPT_MAX:
            break
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

- [x] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_extraction.py tests/test_dreaming_records.py tests/test_dreaming_dreamer.py -q`
Expected: PASS (기존 dreamer 테스트의 스키마 파싱도 기본값 `[]`로 무해)

- [x] **Step 5: Commit**

```bash
git add dreaming/records.py dreaming/dreamer.py tests/test_dreaming_records.py tests/test_dreaming_extraction.py
git commit -m "feat(dreaming): keyExcerpts 추출 — 결정적 원문 보존 (스펙 §6.2)

에피소드 접힘 = 디테일 사망이던 것을 유닛당 3개·400자 원문 인용으로 보상.
인용은 원문 substring 검증 통과분만 저장 (verify_numbers와 같은 철학),
검증 후 컷 + 중복 제거.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: 청크 조립 — Tier1 발췌 렌더 + Tier2 병합 5개

**Files:**
- Modify: `dreaming/chunks.py` — `assemble_tier1()`·`assemble_tier2()` (39·46행 부근)
- Test: `tests/test_dreaming_chunks.py` (기존 파일에 추가)

**Interfaces:**
- Consumes: Task 4의 `Episode.key_excerpts`
- Produces: `chunks.EXCERPT_MERGE_MAX = 5` (모듈 로컬 상수 — dreamer가 chunks를 이미 import하므로 역방향은 순환)

- [x] **Step 1: 실패하는 테스트 작성** — `tests/test_dreaming_chunks.py`에 추가:

```python
def test_tier1_발췌_포함():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.",
                 key_excerpts=["은검을 300골드에 팔았다"])
    out = assemble_tier1(ep)
    assert '원문: "은검을 300골드에 팔았다"' in out


def test_tier1_발췌_없으면_기존_바이트_그대로():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.")
    assert "원문:" not in assemble_tier1(ep)   # byte-stable — 구 에피소드 불변


def test_tier2_병합_발췌_5개_캡():
    # 스펙 §6.2: "유닛당 3, 병합 시 5" — 챕터로 접혀도 발췌 5개는 생존
    eps = [Episode(range_start=f"a{i}", range_end=f"b{i}", title=f"장{i}",
                   summary="요약.", key_excerpts=[f"발췌{i}-1", f"발췌{i}-2"])
           for i in range(4)]                   # 총 발췌 8개
    out = assemble_tier2(eps)
    assert out.count("원문:") == 5              # 연대순 선착 5개
    assert '원문: "발췌0-1"' in out and '원문: "발췌2-1"' in out
    assert "발췌3-2" not in out


def test_tier2_발췌_없으면_기존_바이트_그대로():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.")
    assert "원문:" not in assemble_tier2([ep])
```

- [x] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py -q -k 발췌`
Expected: FAIL — 원문 줄 미생성

- [x] **Step 3: 구현** — `dreaming/chunks.py`. **`dreaming/dreamer.py:18`이 이미 chunks를 import하므로 역방향 import는 순환이다 (확인 완료)** — 상수는 chunks.py 모듈 상단 상수 블록에 둔다:

```python
EXCERPT_MERGE_MAX = 5   # 챕터 병합 시 발췌 캡 (스펙 §6.2 "병합 시 5")


def assemble_tier1(ep: Episode) -> str:
    """에피소드 청크 (~70% 압축): 제목 + 요약 + 결정적 원문 + 미회수 복선.

    발췌는 캐시되는 프리픽스에 살므로 턴당 추가 비용은 read 0.1×뿐이다.
    상주 비용 실측(플랜 리뷰): 리빌드 상각 +~860자/턴 uncached (+11%) —
    hot zone 주입(턴당 전액)보다 싸서 감수한다.
    """
    lines = [f"[지난 이야기 · {ep.title}]", ep.summary.strip()]
    for ex in ep.key_excerpts:
        lines.append(f'원문: "{ex}"')
    if ep.open_threads:
        lines.append("남은 실마리: " + " / ".join(ep.open_threads))
    return "\n".join(lines)


def assemble_tier2(episodes: List[Episode]) -> str:
    """챕터 청크 (~90% 압축): 에피소드당 한 줄 + 병합 발췌 5개 (스펙 §6.2)."""
    lines = ["[지난 장 요약]"]
    for ep in episodes:
        lines.append(f"- {ep.title}: {_one_line(ep.summary)[:100]}")
    merged = [ex for ep in episodes for ex in ep.key_excerpts]
    for ex in merged[:EXCERPT_MERGE_MAX]:      # 병합 시 5 — 연대순 선착
        lines.append(f'원문: "{ex}"')
    return "\n".join(lines)
```

- [x] **Step 4: 통과 확인 + 전체 회귀**

Run: `python3 -m pytest tests/ -q`
Expected: 전부 PASS

- [x] **Step 5: Commit**

```bash
git add dreaming/chunks.py tests/test_dreaming_chunks.py
git commit -m "feat(dreaming): 청크에 keyExcerpts 렌더 — Tier1 유닛당 3, Tier2 병합 5 (스펙 §6.2)

발췌 없는 구 에피소드는 두 티어 모두 조립 바이트 불변 (캐시 안전).
챕터로 접혀도 결정적 원문 5개는 생존 — 먼 과거 디테일의 마지막 방어선.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: SPEC 개정 — 실측이 뒤집은 2곳

**Files:**
- Modify: `docs/dreaming/SPEC.md` — §6.3 (261행 부근), §5 (221행 부근)

**Interfaces:** 없음 (문서)

- [x] **Step 1: §6.3에 프로바이더 주석 추가** — 섹션 끝에:

```markdown
- **프로바이더 한정 주의** (fix-drm-r0 실측, FINDINGS §2): "유휴 재압축 = 공짜"는
  Anthropic처럼 TTL 만료로 캐시가 소멸하는 프로바이더에서만 성립한다. DeepSeek의
  자동 프리픽스 캐싱은 유휴와 무관하게 바이트가 바뀌면 그대로 miss — 재압축 비용이
  0이 아니다. 비-Anthropic에서는 재압축 빈도 자체를 낮추는 것(BOUNDARY_STEP)이 방어선.
```

- [x] **Step 2: §5 주입 예산 각주** — 레이아웃 코드블록 아래에:

```markdown
- 주입 예산 실측 (fix-drm-r0, 100턴): confirmed fact 329개 = 약 14K자(≈7~9K tok).
  "선별 ≤2K"는 비용 상한이지 물리 한계가 아니다 — 예산은 `HOT_ZONE_CHAR_BUDGET`
  파라미터로 두고, 상향 여부는 benchmarks/retrieval_lab.py의 프로브별 필요예산
  곡선으로 결정한다 (T59류 패러프레이즈 포함 여부가 관건).
```

- [x] **Step 3: Commit**

```bash
git add docs/dreaming/SPEC.md
git commit -m "docs(dreaming): SPEC 실측 반영 — §6.3 TTL 트릭 프로바이더 한정, §5 예산 각주

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 명시적 Out of Scope

- **Track A** (Dreamer 추출실패 4건 — T19·69·79·89 + **에피소드가 전부 1턴짜리인 경계 품질 문제**): fact가 스토어에 없거나 에피소드 입도가 잘못된 건 검색·발췌로 못 고침. 별도 플랜. keyExcerpts의 실효 사거리도 에피소드 입도가 고쳐져야 늘어난다.
- **벤치 고정 건초더미** (4변형 동일 히스토리·동일 질문): 별도 플랜. **이 플랜 완료 전 야간 본런 재실행 금지.**
- **T59(패러프레이즈) 회수**: 어휘 한계로 이번 범위에서 명시적 절단. 단, Task 3 Step 5의 예산 곡선(need 값)으로 "예산 상향 시 회수 가능한가"의 숫자를 남긴다 — 절단이 상수 6000의 결과인지 물리 한계인지 구분하기 위해.
- 덴스/임베딩 검색, Actor `knows[]` POV 게이팅, Tier3 시놉시스, pinned 승격 지연.

## 완료 기준

1. `python3 -m pytest tests/ -q` 전부 PASS.
2. retrieval_lab 본측정(Task 3 Step 5): **T39 `IN`** (need ≤ 6000). T29·T99는 진단 참고 (게이트 아님 — 매칭 모호 48개/이미 IN).
3. 구 에피소드(발췌 없음)의 Tier1·Tier2 조립 바이트 불변.
4. 예산 곡선(전 프로브 need 값)이 실행 후기에 기록됨 — 6000 유지/상향 결정의 근거.
5. **효과 범위의 정직한 명시**: 이 플랜의 완료 = 유닛테스트 + 오프라인 순위 개선. 실제 기억 개선 주장은 벤치 재설계(고정 건초더미) 후에만 가능 (FINDINGS §3 — 현행 프로브 벤치는 무효 판정). 벤치 재설계 후에도 개선이 없으면 랭킹 배선(Task 3)은 revert 후보다.
6. 실행 후기(플랜 결함·결과 숫자)를 이 파일 하단에 기록 — CLAUDE.md 플랜 후기 규약.

## 실행 후기 (2026-08-10)

### Task 2 baseline 출력 전문

Task 3(query 배선) 이전 — `render_knowledge`에 query 파라미터가 없어 `_render`의
inspect 분기가 `render_knowledge(store, budget=budget)`으로 폴백한다. 즉 아래는
**현행 최신순의 성적표**(배선 전).

실행: `DREAMING_DATA_DIR=/Users/yanghyeon-u/Desktop/RISU_ENE/dreaming_data python3 -m benchmarks.retrieval_lab`

```
facts(주입 대상)=329  pad=1024
T 19 [relation] OUT need=10591 (value매칭 1개) 따뜻한 차 덕분에 얼어붙었던 몸이 조금씩 풀리는 기분이라 마음이 
T 29 [  false] IN  need=200 (value매칭 48개) 세심하게 챙겨주셔서 정말 감사합니다, 소연 님. 지난번에도 이 방
T 39 [ update] OUT need=14254 (value매칭 1개) 품 안에서 꺼낸 돌멩이를 건네며, 문득 제가 이곳에 당도하기 전의
T 49 [ recent] IN  need=5173 (value매칭 16개) 무거운 발걸음을 묵묵히 따르다 보니, 아까 뼈와 함께 묻혀 있던 
T 59 [ recall] OUT need=14205 (value매칭 1개) 바람이 차서 그런지 목 안쪽이 자꾸만 따끔거리네요. **그때** 
T 69 [relation] ∅스토어에없음 (value매칭 0개) 저도 함께 가겠습니다, 혼자 보내드리기엔 마음이 놓이지 않으니까요
T 79 [  false] ∅스토어에없음 (value매칭 0개) "방금 그 소리 들으셨나요? 소연 씨가 아까 저녁만 먹고 바로 산
T 89 [ update] ∅스토어에없음 (value매칭 0개) "그때 그 무덤터 곁에서 저를 보셨다면, 혹시 제가 들고 있던 것
T 99 [ recent] IN  need=350 (value매칭 8개) 이 돌에 새겨진 경고가 마치 **그때** 보았던 차가운 관 뚜껑
```

게이트(T39) 판정: `OUT need=14254` — Expected("T39 `OUT` — 최신순 328/329로 잘림")와
정확히 일치. T99는 `IN need=350`으로 이미 최신순으로 주입 중임을 확인. 나머지는
매칭 다중(T29·T49) 또는 추출실패로 스토어에 정답이 없음(T69·79·89, Track A 몫).

### Task 3 본측정 전 프로브 need 곡선

실행: `DREAMING_DATA_DIR=/Users/yanghyeon-u/Desktop/RISU_ENE/dreaming_data python3 -m benchmarks.retrieval_lab` (배선 후)

```
facts(주입 대상)=329  pad=1024
T 19 [relation] IN  need=5585 (value매칭 1개) 따뜻한 차 덕분에 얼어붙었던 몸이 조금씩 풀리는 기분이라 마음이 
T 29 [  false] IN  need=200 (value매칭 48개) 세심하게 챙겨주셔서 정말 감사합니다, 소연 님. 지난번에도 이 방
T 39 [ update] IN  need=972 (value매칭 1개) 품 안에서 꺼낸 돌멩이를 건네며, 문득 제가 이곳에 당도하기 전의
T 49 [ recent] IN  need=236 (value매칭 16개) 무거운 발걸음을 묵묵히 따르다 보니, 아까 뼈와 함께 묻혀 있던 
T 59 [ recall] OUT need=8251 (value매칭 1개) 바람이 차서 그런지 목 안쪽이 자꾸만 따끔거리네요. **그때** 
T 69 [relation] ∅스토어에없음 (value매칭 0개) 저도 함께 가겠습니다, 혼자 보내드리기엔 마음이 놓이지 않으니까요
T 79 [  false] ∅스토어에없음 (value매칭 0개) "방금 그 소리 들으셨나요? 소연 씨가 아까 저녁만 먹고 바로 산
T 89 [ update] ∅스토어에없음 (value매칭 0개) "그때 그 무덤터 곁에서 저를 보셨다면, 혹시 제가 들고 있던 것
T 99 [ recent] IN  need=344 (value매칭 8개) 이 돌에 새겨진 경고가 마치 **그때** 보았던 차가운 관 뚜껑
```

핵심: **T39 baseline OUT need=14254 → 본측정 IN need=972** — 게이트 통과, 1차
시도(`_PREV_ASSISTANT_CAP` 조정 불필요). **T59 need=8251** — 예산 6000 아래에서는
여전히 OUT (리뷰 사전 예측 "~8,250"과 오차 1로 일치).

| probe | ptype | baseline need | 본측정 need | 판정 |
|---|---|---|---|---|
| T19 | relation | 10591 (OUT) | 5585 | IN |
| T29 | false | 200 (IN, 매칭 48개 — 진단용) | 200 | IN |
| T39 | update | 14254 (OUT) | **972** | **IN — 게이트 통과** |
| T49 | recent | 5173 (IN) | 236 | IN |
| T59 | recall | 14205 (OUT) | 8251 | OUT |
| T99 | recent | 350 (IN, 진단용) | 344 | IN |

### 예산 결정

**6000 유지** — T59 회수는 need=8251로 상수 6000의 결과이지 물리 한계 아님. 상향은
별도 판단. 8251까지 올리려면 예산을 +37% 늘려야 하고 캐시 안정성·지식 블록 외
다른 프롬프트 예산에도 연쇄 영향을 준다 — 이번 플랜 범위를 넘는 트레이드오프라
현행 6000으로 게이트만 통과시키고 상향은 보류한다.

### 정직성 조항

이 브랜치의 완료 = 유닛테스트(724 passed) + 오프라인 순위 개선. 기억 개선 주장은
벤치 재설계(고정 건초더미) 이후에만 가능. 재설계 후에도 개선 없으면 Task 3(랭킹
배선)은 revert 후보다.

### 발췌 forward-only 명시

keyExcerpts는 재드림 시 최초 기록 채택(build_compression 스킵 로직) 때문에 기존
세션의 이미 접힌 청크에는 소급 적용되지 않는다 — 신규 드림 구간부터만 유효.
+860자/턴(+11%) 비용 추정도 신규 구간 한정.

### 실세션 투입 전 게이트

실LLM 1콜 dream 사이클 스모크 필요 (key_excerpts 스키마가 실출력에서 파싱되는지) —
머지 비차단, 라이브런 전 필수.
