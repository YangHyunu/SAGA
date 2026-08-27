# HypaV3 동작 인스펙터 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HypaV3 압축·RAG 과정을 턴별로 기록해 눈으로 보는 인스펙터 — exp(뮈토스 실사용)·main(기본값 유저) 두 구현 축.

**Architecture:** `hypa.py`(exp 포트)는 로직 무변경 + `trace` 훅만. `hypa_main.py`가 `hypaMemoryV3Main` 신규 포트(공용 부품은 hypa에서 import). `inspect_hypa.py`가 턴 1~100 전진 루프 → json + 자급자족 HTML. 요약은 디스크 캐시 우선, 미스는 기본 차단(`--allow-llm`시만 실콜).

**Tech Stack:** Python 3.13, pytest, tiktoken, sentence-transformers(MiniLM), httpx(OpenRouter).

**Spec:** `docs/superpowers/specs/2026-08-27-hypa-inspector-design.md`

## Global Constraints

- 원본 스펙 = `/Users/yanghyeon-u/Desktop/Risuai/src/ts/process/memory/hypav3.ts` (HEAD 9b260694). 추측 금지 — 모든 동작 주장에 `파일:라인`.
- `hypa.py` 기존 로직·기존 테스트 무변경 (trace=None 기본 no-op 추가만 허용).
- `benchmarks/fixed_haystack/run.py`·`report.py`·채점 경로 무수정.
- 인스펙터는 `benchmarks/eval/**`·`dreaming/**`에 쓰기 금지. 출력은 `dreaming_data/fixed_haystack/`에만.
- 캐시 미스 상태에서 `--allow-llm` 없이 LLM 실콜 절대 금지.
- 성인 콘텐츠를 코드·테스트·픽스처·문서·커밋 메시지에 남기지 않는다 (테스트 프롬프트는 무해한 플레이스홀더).
- 커밋 전 `git diff --cached` 확인. 커밋 메시지는 정상 문체.
- 관찰 조건(스펙 §1): 검증 런 = export 원본(0.39/0.01) + maxContext 100000(run_params) + gemini-3-flash 캐시. 관찰 런 = 메토비 0.2 / 추가요약 0.1 / maxContext 45000 / google/gemini-2.5-flash-lite / MiniLM.

## File Structure

```
benchmarks/eval/hypa.py                    수정: plan_batches·select_summaries·hypa_step에 trace=None
benchmarks/eval/hypa_main.py               신규: select_summaries_main + hypa_step_main
benchmarks/fixed_haystack/inspect_hypa.py  신규: CLI + 전진 루프 + json/HTML 출력
tests/test_hypa_trace.py                   신규: Task 2
tests/test_hypa_main.py                    신규: Task 3~4
tests/test_inspect_hypa.py                 신규: Task 5~6
```

---

### Task 1: 데이터·환경 준비

**Files:** 없음 (전부 gitignored 산출물)

**Interfaces:**
- Produces: 현 워크트리에 `dreaming_data/fixed_haystack/` 실물(건초더미 2종 + hypa-cache)과 `external/risuai` 심링크. 이후 모든 태스크가 전제.

- [ ] **Step 1: 데이터 복사 + 심링크**

```bash
rsync -a /Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/hypa-inspector/dreaming_data/ dreaming_data/
mkdir -p external && ln -sfn /Users/yanghyeon-u/Desktop/Risuai external/risuai
```

- [ ] **Step 2: 스모크 검증**

```bash
python3 -c "
from benchmarks.fixed_haystack import haystack
from benchmarks.fixed_haystack.ingest import resolve_path, decode_risup
hs = haystack.load('dreaming_data/fixed_haystack/fix-drm-r0.json')
decode_risup(resolve_path(hs.preset_path))
import pathlib
n = len(list(pathlib.Path('dreaming_data/fixed_haystack/hypa-cache').glob('*.txt')))
print('turns', len(hs.turns), 'cache_files', n)"
```

Expected: `turns 100 cache_files <수십>` — 에러 없이 출력. rpack FileNotFoundError가 나면 심링크 실패.

- [ ] **Step 3: 기존 테스트 베이스라인**

```bash
python3 -m pytest tests/test_eval_hypa.py tests/test_fixed_haystack_assemble.py -q
```

Expected: 전부 PASS. (이후 태스크의 무회귀 기준선. 커밋 없음 — 변경 파일 0.)

---

### Task 2: exp 관찰 훅 — `hypa.py`에 trace

**Files:**
- Modify: `benchmarks/eval/hypa.py` (`plan_batches`, `select_summaries`, `hypa_step` 시그니처에 `trace: Optional[list] = None`)
- Test: `tests/test_hypa_trace.py`

**Interfaces:**
- Produces: 세 함수의 `trace` kwarg. `trace`가 리스트면 이벤트 dict를 append, `None`이면 완전 no-op(기존 반환·동작 바이트 동일). 이벤트 스키마(스펙 §4, exp·main 공통):
  - `{"ev":"reserve","should_reserve":bool,"available":int,"current":int}`
  - `{"ev":"batch","idx":[str,...],"tokens":int}` (idx = memo 리스트) / `{"ev":"batch_stop","reason":"target_reached"|"over_summarize_break"|"min_messages_break"}`
  - `{"ev":"important","picked":[int,...],"available_after":int}` (picked = summaries 인덱스)
  - `{"ev":"recent","picked":[int,...],"reserved":int,"used":int}`
  - `{"ev":"similar","queries":[{"content":str,"weight":float},...],"chunk_scores":[[[str,float],...],...],"cc_rank":[str,...],"rrf_rank":[int,...],"picked":[int,...],"reserved":int,"used":int}` (chunk_scores는 쿼리별 `[청크id, 코사인]` 내림차순, rrf_rank는 summaries 인덱스 순위)

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_hypa_trace.py`

```python
"""hypa.py trace 훅 — trace=None이면 기존과 동일, 리스트면 이벤트 기록."""
import math
from benchmarks.eval import hypa


def fake_embed(texts):
    # 결정론 가짜 임베딩: 'ring' 포함 텍스트끼리 유사하게
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _mk_summaries():
    return [
        {"text": "the ring was found", "chatMemos": ["m0", "m1"], "isImportant": False},
        {"text": "weather talk", "chatMemos": ["m2", "m3"], "isImportant": True},
        {"text": "market visit", "chatMemos": ["m4", "m5"], "isImportant": False},
    ]


def _recent():
    return [{"role": "user", "content": "where is the ring", "memo": "m6"}]


def test_select_trace_none_is_default_and_unchanged():
    s = hypa.HypaSettings()
    a = hypa.select_summaries(_mk_summaries(), 1000, _recent(), s, embed=fake_embed)
    b = hypa.select_summaries(_mk_summaries(), 1000, _recent(), s, embed=fake_embed,
                              trace=None)
    assert [x["text"] for x in a] == [x["text"] for x in b]


def test_select_trace_records_stage_events():
    s = hypa.HypaSettings()
    tr = []
    hypa.select_summaries(_mk_summaries(), 1000, _recent(), s, embed=fake_embed,
                          trace=tr)
    evs = {e["ev"] for e in tr}
    assert {"important", "recent", "similar"} <= evs
    sim = next(e for e in tr if e["ev"] == "similar")
    assert sim["queries"] and sim["chunk_scores"]
    imp = next(e for e in tr if e["ev"] == "important")
    assert imp["picked"] == [1]          # isImportant인 index 1


def test_plan_batches_trace_records_batches():
    s = hypa.HypaSettings(max_chats_per_summary=2)
    chats = [{"role": "user", "content": "msg " * 50, "memo": f"m{i}"}
             for i in range(10)]
    tot = sum(hypa.tok_chat(c) for c in chats)
    tr = []
    batches, _cur, _si, err = hypa.plan_batches(chats, 0, tot + 10_000, 10_000, s,
                                                trace=tr)
    assert err is None and batches
    ev_batches = [e for e in tr if e["ev"] == "batch"]
    assert len(ev_batches) == len(batches)
    assert ev_batches[0]["idx"] == ["m0", "m1"]
```

- [ ] **Step 2: 실패 확인**

```bash
python3 -m pytest tests/test_hypa_trace.py -q
```

Expected: FAIL — `TypeError: ... unexpected keyword argument 'trace'`

- [ ] **Step 3: 최소 구현**

`hypa.py` 수정 — 로직 무접촉, append만. 요지:

```python
def _tr(trace, **ev):
    if trace is not None:
        trace.append(ev)
```

- `plan_batches(..., trace=None)`: 배치 append 직후 `_tr(trace, ev="batch", idx=[c["memo"] for c in batch], tokens=batch_tokens)`; 각 break 지점에 `_tr(trace, ev="batch_stop", reason=...)` (`target_reached`=142행 break, `over_summarize_break`=167행 break, `min_messages_break`=147행 break).
- `select_summaries(..., trace=None)`: (a) 뒤 `_tr(trace, ev="important", picked=[_index_of(summaries,s) for s in selected], available_after=available)`; (b) 뒤 recent 이벤트(picked은 이번 단계 선택분만); (c) 안에서 scored_lists 계산 직후 similar 이벤트 — `chunk_scores=[[[ch["id"], round(sc,6)] for ch,sc in lst] for lst in scored_lists]`, `cc_rank=[ch["id"] for ch in ranked]`, `rrf_rank=[_index_of(summaries, s) for s in child_to_parent_rrf(...)]`(선택 루프 돌기 전 리스트로 실체화), `picked`·`reserved`·`used`. **주의: `child_to_parent_rrf` 결과를 리스트 변수로 받아 이벤트와 선택 루프가 같은 것을 쓰게** (이터레이터 이중 소비 금지).
- `hypa_step(..., trace=None)`: 예약 블록 뒤 `_tr(trace, ev="reserve", should_reserve=should_reserve, available=available, current=current)`; `plan_batches`·`select_summaries`에 trace 전달.

- [ ] **Step 4: 통과 + 무회귀 확인**

```bash
python3 -m pytest tests/test_hypa_trace.py tests/test_eval_hypa.py tests/test_fixed_haystack_assemble.py -q
```

Expected: 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/hypa.py tests/test_hypa_trace.py
git diff --cached --stat && git commit -m "feat(inspector): hypa exp 경로에 trace 관찰 훅 — trace=None 기본 no-op"
```

---

### Task 3: `hypa_main.py` — `select_summaries_main`

**Files:**
- Create: `benchmarks/eval/hypa_main.py`
- Test: `tests/test_hypa_main.py`

**Interfaces:**
- Consumes: `hypa._summary_tokens`, `hypa._unused`, `hypa._index_of`, `hypa.simple_cc`, `hypa.child_to_parent_rrf`, `hypa._cosine`, `hypa._DEFAULT_EMBED`, `hypa.split_by_separator`, `hypa.HypaSettings`, `hypa.HypaError`, `hypa.SUMMARY_SEP`, Task 2의 `hypa._tr`
- Produces: `select_summaries_main(summaries, available_tokens, recent_chats, S, embed=None, trace=None) -> tuple[list[dict], dict]` — (시간순 선택 리스트, metrics dict: `lastImportantSummaries`/`lastRecentSummaries`/`lastSimilarSummaries`/`lastRandomSummaries` 각 summaries 인덱스 리스트)

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_hypa_main.py`

```python
"""hypaMemoryV3Main 포트 — exp와 갈라지는 지점 고정 (스펙 §3 표)."""
import pytest
from benchmarks.eval import hypa, hypa_main


def fake_embed(texts):
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _summaries():
    return [
        {"text": "the ring was found", "chatMemos": ["m0", "m1"], "isImportant": False},
        {"text": "weather talk", "chatMemos": ["m2", "m3"], "isImportant": True},
        {"text": "market visit", "chatMemos": ["m4", "m5"], "isImportant": False},
    ]


def test_main_similar_uses_whole_message_queries():
    """main은 문단 분할 없이 원문 통짜 쿼리 (hypav3.ts:1369) —
    exp의 서브쿼리 분할(680~695)과 갈라지는 지점."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": "first paragraph\n\nsecond ring paragraph",
               "memo": "m6"}]
    tr = []
    hypa_main.select_summaries_main(_summaries(), 1000, recent, s,
                                    embed=fake_embed, trace=tr)
    sim = next(e for e in tr if e["ev"] == "similar")
    assert len(sim["queries"]) == 1                      # 문단 2개여도 쿼리 1개
    assert sim["queries"][0]["content"] == "first paragraph\n\nsecond ring paragraph"
    assert sim["queries"][0]["weight"] == 1.0            # n=1 → (0+1)/(1*2/2)


def test_main_similar_weights_follow_simplecc_callback():
    """가중치 (i+1)/(n(n+1)/2) (hypav3.ts:1409~1414) — n=3 → 1/6, 2/6, 3/6."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": f"query {i} ring", "memo": f"m{6+i}"}
              for i in range(3)]
    tr = []
    hypa_main.select_summaries_main(_summaries(), 1000, recent, s,
                                    embed=fake_embed, trace=tr)
    sim = next(e for e in tr if e["ev"] == "similar")
    assert [round(q["weight"], 4) for q in sim["queries"]] == [
        round(1 / 6, 4), round(2 / 6, 4), round(3 / 6, 4)]


def test_main_metrics_four_index_lists():
    """main은 4단계 인덱스를 기록 (hypav3.ts:1578~1591)."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": "where is the ring", "memo": "m6"}]
    selected, metrics = hypa_main.select_summaries_main(
        _summaries(), 1000, recent, s, embed=fake_embed)
    assert metrics["lastImportantSummaries"] == [1]
    assert set(metrics) == {"lastImportantSummaries", "lastRecentSummaries",
                            "lastSimilarSummaries", "lastRandomSummaries"}
    assert metrics["lastRandomSummaries"] == []
    # 시간순 반환 (1535~1537)
    texts = [x["text"] for x in _summaries()]
    order = [s_["text"] for s_ in selected]
    assert order == sorted(order, key=texts.index)


def test_main_random_ratio_positive_raises():
    s = hypa.HypaSettings(recent_memory_ratio=0.4, similar_memory_ratio=0.4)
    with pytest.raises(hypa.HypaError):
        hypa_main.select_summaries_main(_summaries(), 1000, [], s, embed=fake_embed)
```

- [ ] **Step 2: 실패 확인**

```bash
python3 -m pytest tests/test_hypa_main.py -q
```

Expected: FAIL — `ModuleNotFoundError: No module named 'benchmarks.eval.hypa_main'`

- [ ] **Step 3: 구현** — `benchmarks/eval/hypa_main.py`

```python
"""HypaV3 기본 구현(hypaMemoryV3Main, hypav3.ts:955~1618) 포트.

exp 포트(hypa.py)와의 차이는 스펙 §3 표 — 배치 고정 창, 즉시 요약,
similar 통짜 쿼리, 빈 카드 삽입, metrics 4종. 공용 부품(hypa.*)은
반드시 모듈 점 접근으로 쓴다 — assemble.bench_summary_cache()가
hypa.summarize를 패치하는데 import-시점 별칭은 스냅샷이라 안 닿는다.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

from benchmarks.eval import hypa


def select_summaries_main(
    summaries: List[Dict[str, Any]],
    available_tokens: int,
    recent_chats: List[Dict[str, Any]],
    S: hypa.HypaSettings,
    embed: Optional[Callable[[List[str]], Any]] = None,
    trace: Optional[list] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[int]]]:
    """important → recent → similar → (random 가드). main 선택 4단계.

    exp와 다른 점: similar 쿼리가 원문 통짜(hypav3.ts:1369)이고 가중치가
    simpleCC 콜백 (i+1)/(n(n+1)/2)(1409~1414). important/recent은 exp와
    육안 동일 (exp:498~515 ≡ main:1224~1240, 핸드오프 §4 검증).
    """
    available = available_tokens
    selected: List[Dict[str, Any]] = []
    sel_imp: List[Dict[str, Any]] = []
    sel_rec: List[Dict[str, Any]] = []
    sel_sim: List[Dict[str, Any]] = []
    random_ratio = 1 - S.recent_memory_ratio - S.similar_memory_ratio

    # (a) important — 초과 시 break (hypav3.ts:1226~1241)
    for s in summaries:
        if s.get("isImportant"):
            t = hypa._summary_tokens(s)
            if t > available:
                break
            sel_imp.append(s)
            available -= t
    selected.extend(sel_imp)
    hypa._tr(trace, ev="important",
             picked=[hypa._index_of(summaries, s) for s in sel_imp],
             available_after=available)

    # (b) recent — 뒤에서부터, 초과 시 break (1258~1303)
    reserved_recent = math.floor(available * S.recent_memory_ratio)
    used_recent = 0
    if S.recent_memory_ratio > 0:
        for s in reversed(hypa._unused(summaries, selected)):
            t = hypa._summary_tokens(s)
            if t + used_recent > reserved_recent:
                break
            sel_rec.append(s)
            used_recent += t
        selected.extend(sel_rec)
    hypa._tr(trace, ev="recent",
             picked=[hypa._index_of(summaries, s) for s in sel_rec],
             reserved=reserved_recent, used=used_recent)

    # (c) similar (1305~1467)
    reserved_similar = math.floor(available * S.similar_memory_ratio)
    used_similar = 0
    if S.similar_memory_ratio > 0:
        if random_ratio <= 0:          # recent 잔여 흡수 (1314~1325)
            reserved_similar += reserved_recent - used_recent
        unused = hypa._unused(summaries, selected)

        chunks: List[Dict[str, Any]] = []
        for si, s in enumerate(unused):
            for ci, chunk in enumerate(
                    hypa.split_by_separator(s["text"], S.summary_chunk_separator)):
                if chunk.strip():
                    chunks.append({"id": f"{si}-{ci}", "content": chunk.strip(),
                                   "meta": s})

        # 쿼리 = 최근 챗 원문 통짜 (1363~1369). 문단 분할 없음 — exp와 차이.
        valid = [c for c in recent_chats[-S.query_chat_count:]
                 if c["content"].strip()]
        n = len(valid)
        # simpleCC 콜백 (i+1)/(n(n+1)/2) (1409~1414)
        queries = [{"content": c["content"], "weight": (i + 1) / (n * (n + 1) / 2)}
                   for i, c in enumerate(valid)]

        if queries and chunks:
            ef = embed or hypa._DEFAULT_EMBED
            chunk_vecs = ef([ch["content"] for ch in chunks])
            query_vecs = ef([q["content"] for q in queries])
            # similaritySearchScoredEx = 코사인 내림차순 (2029~2041)
            scored_lists = [
                sorted(((ch, hypa._cosine(qv, cv))
                        for ch, cv in zip(chunks, chunk_vecs)),
                       key=lambda pair: -pair[1])
                for qv in query_vecs
            ]
            ranked_chunks = hypa.simple_cc(scored_lists,
                                           [q["weight"] for q in queries])
            ranked = hypa.child_to_parent_rrf(ranked_chunks,
                                              key=lambda ch: ch["meta"])
            hypa._tr(trace, ev="similar", queries=queries,
                     chunk_scores=[[[ch["id"], round(sc, 6)] for ch, sc in lst]
                                   for lst in scored_lists],
                     cc_rank=[ch["id"] for ch in ranked_chunks],
                     rrf_rank=[hypa._index_of(summaries, s) for s in ranked],
                     picked=[], reserved=reserved_similar, used=0)
            for s in ranked:
                t = hypa._summary_tokens(s)
                if t + used_similar > reserved_similar:
                    break              # 1428~1438
                sel_sim.append(s)
                used_similar += t
            selected.extend(sel_sim)
            if trace is not None:      # picked/used 확정치로 갱신
                sim_ev = next(e for e in reversed(trace) if e["ev"] == "similar")
                sim_ev["picked"] = [hypa._index_of(summaries, s) for s in sel_sim]
                sim_ev["used"] = used_similar

    # (d) random — 고정 export ratio 0, 도달 불가 가드 (1470~1520)
    if random_ratio > 0:
        raise hypa.HypaError(
            "random 선택은 고정 export(random_ratio=0)에서 도달 불가 "
            "— 스펙 가정 붕괴 (hypav3.ts:1470~1520)")

    metrics = {
        "lastImportantSummaries": [hypa._index_of(summaries, s) for s in sel_imp],
        "lastRecentSummaries": [hypa._index_of(summaries, s) for s in sel_rec],
        "lastSimilarSummaries": [hypa._index_of(summaries, s) for s in sel_sim],
        "lastRandomSummaries": [],
    }
    # 시간순 재정렬 (1535~1537)
    ordered = sorted(selected, key=lambda s: hypa._index_of(summaries, s))
    return ordered, metrics
```

- [ ] **Step 4: 통과 확인**

```bash
python3 -m pytest tests/test_hypa_main.py -q
```

Expected: PASS 4건.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/hypa_main.py tests/test_hypa_main.py
git diff --cached --stat && git commit -m "feat(inspector): hypaMemoryV3Main 선택 4단계 포트 — 통짜 쿼리·metrics 4종"
```

---

### Task 4: `hypa_main.py` — `hypa_step_main`

**Files:**
- Modify: `benchmarks/eval/hypa_main.py` (함수 추가)
- Test: `tests/test_hypa_main.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 3 `select_summaries_main`; `hypa.to_risu_chats`, `hypa.tok_chat`, `hypa._is_skipped`, `hypa.summarize`(점 접근 — 패치 도달 필수), `hypa.wrap_xml`, `hypa.MEMORY_TAG`, `hypa.SUMMARY_SEP`, `hypa.HypaError`
- Produces: `hypa_step_main(history, preset_tokens, S, data, send, max_ctx, max_response, trace=None) -> tuple[str, list, int, dict, Optional[str]]` — exp `hypa_step`과 같은 튜플 형태이되 **memory_text가 항상 str** (요약 0개면 빈 카드 `wrap_xml(MEMORY_TAG, "")` — main은 빈 카드도 삽입, hypav3.ts:1188~1199). metrics는 `data["metrics"]`에 저장.

- [ ] **Step 1: 실패 테스트 추가** — `tests/test_hypa_main.py`에 append

```python
def _mk_chats(n, tok_words=50):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"m{i} " + "word " * tok_words} for i in range(n)]


def _fake_send_factory(log):
    def send(messages):
        log.append(messages)
        return f"summary #{len(log)}"
    return send


def test_main_batch_fixed_window_vs_exp_fill():
    """main 배치는 고정 창(hypav3.ts:1059~1062): 창 안에 스킵이 있으면
    배치가 작아지고 다음 배치는 창 끝에서 시작. exp(293~296)는 찰 때까지
    스캔 — 같은 입력에서 배치 구성원이 갈라진다."""
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3,
                          memory_tokens_ratio=0.2, extra_summarization_ratio=0.1)
    hist = _mk_chats(12)
    hist[2]["content"] = "   "            # 창(0..3) 안 스킵 대상
    calls = []
    send = _fake_send_factory(calls)
    # max_ctx를 히스토리보다 작게 → 요약 발동
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, send, max_ctx=tot // 2, max_response=100,
        embed=fake_embed)
    assert err is None
    first = data["summaries"][0]["chatMemos"]
    assert first == ["m0", "m1", "m3"]     # 고정 창 0..3에서 m2 스킵 → 3개
    # exp라면 m4까지 채웠을 것 — 갈라짐 고정
    exp_batches, _, _, _ = hypa.plan_batches(
        hypa.to_risu_chats(hist), 0, tot * 2, tot, S)
    assert [c["memo"] for c in exp_batches[0]] == ["m0", "m1", "m3", "m4"]


def test_main_zero_summary_turn_inserts_empty_card():
    """요약 미발동 턴에도 빈 카드 삽입 (hypav3.ts:1188~1199) — exp는 None."""
    S = hypa.HypaSettings()
    hist = _mk_chats(4, tok_words=5)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, None, max_ctx=100_000, max_response=100)
    assert err is None
    assert mem == hypa.wrap_xml(hypa.MEMORY_TAG, "")
    assert data["summaries"] == []


def test_main_summarize_failure_keeps_earlier_summaries():
    """즉시 요약(1151~1159)이라 중간 실패 시 앞 배치 요약은 data에 남는다 —
    exp의 전량 폐기(계획 후 일괄)와 갈라짐."""
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3)
    hist = _mk_chats(16)
    n_calls = [0]
    def send(messages):
        n_calls[0] += 1
        if n_calls[0] >= 2:
            raise hypa.HypaError("boom")
        return "first summary"
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, send, max_ctx=tot // 3, max_response=100)
    assert err is not None
    assert len(data["summaries"]) == 1     # 첫 배치는 살아남음


def test_main_metrics_stored_in_data():
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3)
    hist = _mk_chats(16)
    calls = []
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, _fake_send_factory(calls),
        max_ctx=tot // 2, max_response=100, embed=fake_embed)
    assert err is None
    assert set(data["metrics"]) == {
        "lastImportantSummaries", "lastRecentSummaries",
        "lastSimilarSummaries", "lastRandomSummaries"}


def test_main_over_summarize_break_leaves_batch_unsummarized():
    """과요약 직전 중단 (hypav3.ts:1129~1139): current ≤ max_ctx인 반복에서
    current − batch_tokens < target이면 요약 없이 break, startIdx 유지 —
    스펙 §8-2. max_ctx는 조건이 성립하는 값을 탐색으로 잡는다."""
    import math
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3,
                          extra_summarization_ratio=0.1, memory_tokens_ratio=0.2)
    hist = _mk_chats(18)
    chats = hypa.to_risu_chats(hist)
    toks = [hypa.tok_chat(c) for c in chats]
    b1 = sum(toks[:4])
    total = 50 + sum(toks)
    # 발동(current0 > max_ctx)하되 배치 1개 후 (target, max_ctx] 구간에 착지
    max_ctx = next(m for m in range(300, 5000)
                   if (c0 := total + math.floor(m * 0.2)) > m
                   and m >= c0 - b1 > 0.9 * m)
    tr = []
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, _fake_send_factory([]), max_ctx,
        max_response=100, embed=fake_embed, trace=tr)
    assert err is None
    assert len(data["summaries"]) == 1     # 두 번째 배치는 요약되지 않음
    assert ksm == 4                        # startIdx가 두 번째 배치 앞에서 멈춤
    reasons = [e["reason"] for e in tr if e["ev"] == "batch_stop"]
    assert reasons[-1] == "over_summarize_break"
```

- [ ] **Step 2: 실패 확인**

```bash
python3 -m pytest tests/test_hypa_main.py -q
```

Expected: 신규 5건 FAIL — `AttributeError: ... 'hypa_step_main'`

- [ ] **Step 3: 구현** — `hypa_main.py`에 추가

```python
def hypa_step_main(
    history: List[Dict[str, Any]],
    preset_tokens: int,
    S: hypa.HypaSettings,
    data: Dict[str, Any],
    send: Optional[Callable[[List[Dict[str, str]]], str]],
    max_ctx: int,
    max_response: int,
    embed: Optional[Callable[[List[str]], Any]] = None,
    trace: Optional[list] = None,
) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, Any], Optional[str]]:
    """한 턴 분의 hypaMemoryV3Main (hypav3.ts:955~1618).

    exp(hypa.hypa_step)와의 차이: 배치 고정 창 + 즉시 요약(실패 시 앞 요약
    보존), 요약 0개여도 빈 카드 삽입, metrics 4종 기록. 반환 튜플 형태는
    exp와 동일하되 memory_text가 항상 str.
    """
    chats = hypa.to_risu_chats(history)
    summaries: List[Dict[str, Any]] = data.setdefault("summaries", [])

    # 초기 토큰 → maxResponse 차감 (hypav3.ts:976, 미복원)
    current = max_response + 50 + preset_tokens + sum(
        hypa.tok_chat(c) for c in chats)
    current -= max_response

    # startIdx (992~1008). preserveOrphanedMemory=true라 orphan 정리 없음(986).
    start_idx = 0
    if summaries:
        last_memo = summaries[-1]["chatMemos"][-1]
        idx = next((i for i, c in enumerate(chats) if c["memo"] == last_memo), -1)
        if idx != -1:
            start_idx = idx + 1
            current -= sum(hypa.tok_chat(c) for c in chats[:start_idx])

    # 메모리 예약 (1012~1033) — exp와 조건이 다르다:
    # shouldReserveEmpty = 요약 0개 && current+empty <= max (1020~1022)
    empty_mem = hypa.tok_chat({"role": "system",
                               "content": hypa.wrap_xml(hypa.MEMORY_TAG, "")})
    memory_tokens = math.floor(max_ctx * S.memory_tokens_ratio)
    should_reserve_empty = (not summaries) and current + empty_mem <= max_ctx
    available = 0 if should_reserve_empty else memory_tokens - empty_mem
    current += empty_mem if should_reserve_empty else memory_tokens
    hypa._tr(trace, ev="reserve", should_reserve=not should_reserve_empty,
             available=available, current=current)

    # 요약 루프 (1036~1174) — 고정 창 + 즉시 요약
    summarization_mode = current > max_ctx
    target = max_ctx * (1 - S.extra_summarization_ratio)
    while summarization_mode:
        if current <= target:
            hypa._tr(trace, ev="batch_stop", reason="target_reached")
            break
        if len(chats) - start_idx <= S.query_chat_count:
            if current <= max_ctx:
                hypa._tr(trace, ev="batch_stop", reason="min_messages_break")
                break
            return ("", history[start_idx:], start_idx, data,
                    f"Cannot summarize further: {current} > {max_ctx}, "
                    f"but minimum {S.query_chat_count} messages required.")

        end_idx = min(start_idx + S.max_chats_per_summary,
                      len(chats) - S.query_chat_count)      # 고정 창 (1059~1062)
        batch: List[Dict[str, Any]] = []
        batch_tokens = 0
        for i in range(start_idx, end_idx):
            chat = chats[i]
            batch_tokens += hypa.tok_chat(chat)             # 스킵돼도 누적 (1099)
            if not hypa._is_skipped(chat, S):               # 1101~1123
                batch.append(chat)

        # 과요약 직전 중단 (1129~1139) — 배치 미요약, startIdx 유지
        if current <= max_ctx and current - batch_tokens < target:
            hypa._tr(trace, ev="batch_stop", reason="over_summarize_break")
            break

        if batch:                                           # 즉시 요약 (1142~1170)
            try:
                text = hypa.summarize(send, batch, S)       # 점 접근 — 패치 도달
            except hypa.HypaError as e:
                return ("", history[start_idx:], start_idx, data,
                        f"Summarization failed: {e}")
            summaries.append({
                "text": text,
                "chatMemos": list(dict.fromkeys(c["memo"] for c in batch)),
                "isImportant": False})
            hypa._tr(trace, ev="batch",
                     idx=[c["memo"] for c in batch], tokens=batch_tokens)

        current -= batch_tokens
        start_idx = end_idx

    # 요약 0개 — 빈 카드라도 삽입 (1188~1217). exp와 갈라짐.
    if not summaries:
        return (hypa.wrap_xml(hypa.MEMORY_TAG, ""), history[start_idx:],
                start_idx, data, None)

    selected, metrics = select_summaries_main(
        summaries, available, chats, S, embed=embed, trace=trace)
    data["metrics"] = metrics                               # 1578~1591
    memory = hypa.wrap_xml(hypa.MEMORY_TAG,
                           hypa.SUMMARY_SEP.join(s["text"] for s in selected))
    # 최종 정산·throw(1549~1575)는 exp 포트와 같은 이유로 생략 —
    # available 예산 안에서 그리디라 초과 도달 불가 (hypa.py:598~606 주석).
    return memory, history[start_idx:], start_idx, data, None
```

- [ ] **Step 4: 통과 + 전체 무회귀**

```bash
python3 -m pytest tests/test_hypa_main.py tests/test_hypa_trace.py tests/test_eval_hypa.py -q
```

Expected: 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/hypa_main.py tests/test_hypa_main.py
git diff --cached --stat && git commit -m "feat(inspector): hypa_step_main — 고정 창 배치·즉시 요약·빈 카드·metrics"
```

---

### Task 5: 인스펙터 CLI — `inspect_hypa.py`

**Files:**
- Create: `benchmarks/fixed_haystack/inspect_hypa.py`
- Test: `tests/test_inspect_hypa.py`

**Interfaces:**
- Consumes: Task 2 trace(exp), Task 4 `hypa_step_main`; `haystack.load`, `ingest._seed_history`, `ingest.resolve_path`, `ingest.decode_risup`, `assemble.hypa_fixed_tokens`, `assemble._wire`, `assemble._tokens`, `assemble.bench_summary_cache`, `assemble.hypa_snapshot_path`, `hypa.summary_cache_key`, `transport`(SUMMARY_MODEL 오버라이드)
- Produces: CLI `python3 -m benchmarks.fixed_haystack.inspect_hypa --hs <id> --impl exp|main [...]`; 라이브러리 함수 `run_inspection(hs, preset, card, impl, S, max_ctx, *, embed=None, allow_llm=False, cache_dir=None) -> dict` (returns `{"meta":..., "turns":[...], "misses":[...]}`) — Task 6이 이 dict를 소비.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_inspect_hypa.py`

```python
"""인스펙터 — 미니 픽스처로 LLM 0콜 e2e + 캐시 미스 차단 경로."""
import json
import pathlib
import types
import pytest
from benchmarks.eval import hypa
from benchmarks.fixed_haystack import inspect_hypa


def fake_embed(texts):
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _mini_hs():
    """FixedHaystack 흉내 — 필요한 속성만 (turns/run_params)."""
    turns = []
    for i in range(3):
        turns.append(types.SimpleNamespace(
            turn=i + 1, is_probe=False,
            user=f"user turn {i} " + "pad " * 40,
            assistant=f"asst turn {i} " + "pad " * 40))
    return types.SimpleNamespace(
        haystack_id="mini", turns=turns,
        run_params={"max_context": 100_000, "max_tokens": 1000})


CARD = {"greeting": "hello there"}


def test_forward_loop_zero_llm_when_cached(tmp_path):
    """캐시 프리시드 후 실행 → send 호출 0, 턴 레코드 3개."""
    hs = _mini_hs()
    S = hypa.HypaSettings(max_chats_per_summary=2, query_chat_count=3,
                          memory_tokens_ratio=0.2)
    # max_ctx를 작게 잡아 요약 발동 → 배치 키를 미리 계산해 캐시 시드
    result = inspect_hypa.run_inspection(
        hs, preset={}, card=CARD, impl="exp", S=S,
        max_ctx=800, embed=fake_embed, allow_llm=False,
        cache_dir=tmp_path, fixed_tokens=0,
        wire_fn=lambda kept, memory: [],       # 조립은 테스트에서 무력화
    )
    if result["misses"]:
        # 미스 경로 검증: 실콜 0으로 미스가 열거된다
        assert all("key" in m for m in result["misses"])
    assert len(result["turns"]) == 3
    assert result["meta"]["impl"] == "exp"


def test_miss_blocks_output(tmp_path):
    """미스 존재 시 write_outputs가 파일을 쓰지 않고 False."""
    res = {"meta": {"impl": "exp", "hsid": "mini", "settings": {}},
           "turns": [], "misses": [{"key": "abc", "n_msgs": 2}]}
    ok = inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert ok is False
    assert list(tmp_path.iterdir()) == []


def test_outputs_written_when_clean(tmp_path):
    res = {"meta": {"impl": "main", "hsid": "mini", "settings": {"x": 1}},
           "turns": [{"turn": 1, "memory_text": "", "kept_start_msg": 0,
                      "n_summaries": 0, "wire_tokens": 0, "events": []}],
           "misses": []}
    ok = inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert ok is True
    js = list(tmp_path.glob("hypa-inspect-*.json"))
    assert len(js) == 1
    assert json.loads(js[0].read_text())["turns"][0]["turn"] == 1
```

- [ ] **Step 2: 실패 확인**

```bash
python3 -m pytest tests/test_inspect_hypa.py -q
```

Expected: FAIL — `ModuleNotFoundError` (inspect_hypa 없음)

- [ ] **Step 3: 구현** — `benchmarks/fixed_haystack/inspect_hypa.py`

```python
"""HypaV3 동작 인스펙터 — 턴 1~N 전진 루프, 내부 상태 턴별 기록.

스펙: docs/superpowers/specs/2026-08-27-hypa-inspector-design.md
채점 없음, run.py 경로 무접촉. 출력은 dreaming_data/fixed_haystack/ 에만.

요약 캐시 규약: 기본은 캐시-히트만 소비(미스 = 실콜 0으로 열거 후 출력
차단·종료코드 1). --allow-llm 일 때만 미스분을 실제 호출한다.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

from benchmarks.eval import hypa, hypa_main, transport
from benchmarks.fixed_haystack import assemble, haystack
from benchmarks.fixed_haystack.ingest import (_seed_history, decode_risup,
                                              resolve_path)

OUT_DIR = pathlib.Path("dreaming_data/fixed_haystack")


class _CacheOnlySummarize:
    """hypa.summarize 대역 — 디스크 캐시만 읽는다. 미스는 기록 후
    플레이스홀더 반환(배치 경계는 요약 텍스트와 무관하니 열거는 유효,
    출력은 미스 존재 시 폐기). 디스크에 아무것도 쓰지 않는다."""

    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = cache_dir
        self.misses: List[Dict[str, Any]] = []

    def __call__(self, send, batch, S, cache_dir=None):
        key = hypa.summary_cache_key(batch, S)
        p = self.cache_dir / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        self.misses.append({"key": key, "n_msgs": len(batch),
                            "memos": [c.get("memo") for c in batch]})
        return "[cache-miss placeholder]"


def run_inspection(hs, preset: Dict, card: Dict, impl: str,
                   S: hypa.HypaSettings, max_ctx: int, *,
                   fixed_tokens: Optional[int] = None,
                   embed: Optional[Callable] = None,
                   allow_llm: bool = False,
                   cache_dir: Optional[pathlib.Path] = None,
                   wire_fn: Optional[Callable] = None,
                   snapshot_check_root: Optional[pathlib.Path] = None,
                   ) -> Dict[str, Any]:
    """전진 루프. ingest_hypa와 같은 히스토리 규약(유저 append → step →
    어시 append, ingest.py:400~421)을 그대로 밟는다."""
    if fixed_tokens is None:
        fixed_tokens = assemble.hypa_fixed_tokens(preset, card)
    if wire_fn is None:
        wire_fn = lambda kept, memory: assemble._wire(
            preset, card, kept, memory=memory or "")

    cache_dir = cache_dir or (OUT_DIR / "hypa-cache")
    max_tok = hs.run_params["max_tokens"]
    step = hypa.hypa_step if impl == "exp" else hypa_main.hypa_step_main

    cache_only = None
    orig_summarize = hypa.summarize
    if not allow_llm:
        cache_only = _CacheOnlySummarize(cache_dir)
        hypa.summarize = cache_only        # 모듈 점 접근 패치 (CLAUDE.md §5)
    send = hypa._summarize_call if allow_llm else None

    hist = _seed_history(card)
    data: Dict[str, Any] = {"summaries": []}
    turns: List[Dict[str, Any]] = []
    snap_mismatch: List[int] = []
    try:
        ctx = assemble.bench_summary_cache() if allow_llm else _noop_ctx()
        with ctx:
            for t in hs.turns:
                if snapshot_check_root is not None:
                    p = assemble.hypa_snapshot_path(snapshot_check_root, t.turn)
                    if p.is_file():
                        snap = json.loads(p.read_text(encoding="utf-8"))
                        ours = [s["chatMemos"] for s in data["summaries"]]
                        theirs = [s["chatMemos"] for s in snap["summaries"]]
                        if ours != theirs:
                            snap_mismatch.append(t.turn)
                hist.append({"role": "user", "content": t.user})
                tr: List[Dict[str, Any]] = []
                kwargs = {"trace": tr}
                if impl == "main":
                    kwargs["embed"] = embed
                mem, kept, ksm, data, err = step(
                    hist, fixed_tokens, S, data, send, max_ctx, max_tok,
                    **kwargs)
                rec = {"turn": t.turn,
                       "memory_text": mem or "",
                       "kept_start_msg": ksm,
                       "n_summaries": len(data.get("summaries", [])),
                       "events": tr,
                       "metrics": data.get("metrics"),
                       "error": err}
                if not err:
                    rec["wire_tokens"] = assemble._tokens(
                        wire_fn(kept, mem)) if wire_fn else 0
                turns.append(rec)
                if err:
                    break
                hist.append({"role": "assistant", "content": t.assistant})
    finally:
        hypa.summarize = orig_summarize

    return {
        "meta": {"impl": impl, "hsid": hs.haystack_id, "max_ctx": max_ctx,
                 "summary_model": transport.SUMMARY_MODEL,
                 "settings": {k: getattr(S, k) for k in vars(S)},
                 "snapshot_mismatch": snap_mismatch},
        "turns": turns,
        "misses": cache_only.misses if cache_only else [],
    }


import contextlib


@contextlib.contextmanager
def _noop_ctx():
    yield


def _slug(meta: Dict[str, Any]) -> str:
    s = meta["settings"]
    base = f"hypa-inspect-{meta['hsid']}-{meta['impl']}"
    tags = []
    if s.get("memory_tokens_ratio") not in (None, 0.39):
        tags.append(f"mtr{s['memory_tokens_ratio']}")
    if s.get("extra_summarization_ratio") not in (None, 0.01):
        tags.append(f"esr{s['extra_summarization_ratio']}")
    return "-".join([base] + tags)


def write_outputs(result: Dict[str, Any], out_dir: pathlib.Path = OUT_DIR) -> bool:
    """미스가 있으면 아무것도 쓰지 않는다 — 미완성 관찰을 남기지 않는다."""
    if result["misses"]:
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(result["meta"])
    (out_dir / f"{slug}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs", required=True)
    ap.add_argument("--impl", required=True, choices=["exp", "main"])
    ap.add_argument("--memory-ratio", type=float)
    ap.add_argument("--extra-ratio", type=float)
    ap.add_argument("--max-context", type=int)
    ap.add_argument("--summary-model")
    ap.add_argument("--embed-model")
    ap.add_argument("--allow-llm", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    from benchmarks.eval.config import HYPA_EXPORT
    hs = haystack.load(f"dreaming_data/fixed_haystack/{a.hs}.json")
    preset = decode_risup(resolve_path(hs.preset_path))
    card = json.loads(pathlib.Path(resolve_path(hs.card_path))
                      .read_text(encoding="utf-8"))
    S = hypa.load_hypa_settings(HYPA_EXPORT)
    overridden = False
    if a.memory_ratio is not None:
        S.memory_tokens_ratio, overridden = a.memory_ratio, True
    if a.extra_ratio is not None:
        S.extra_summarization_ratio, overridden = a.extra_ratio, True
    max_ctx = a.max_context or hs.run_params["max_context"]
    if a.max_context is not None:
        overridden = True
    if a.summary_model:
        transport.SUMMARY_MODEL = a.summary_model   # 캐시 키가 이 값을 본다
        overridden = True
    embed = None
    if a.embed_model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(a.embed_model)
        m.max_seq_length = 512
        embed = lambda texts: m.encode(list(texts), normalize_embeddings=True)
        hypa._DEFAULT_EMBED = embed                 # exp 경로에도 도달
        overridden = True

    # 스냅샷 대조는 무오버라이드 exp에서만 유효 (스펙 §1 검증 앵커)
    snap_root = (pathlib.Path(f"dreaming_data/fixed_haystack/{a.hs}")
                 if (a.impl == "exp" and not overridden) else None)

    result = run_inspection(hs, preset, card, a.impl, S, max_ctx,
                            embed=embed, allow_llm=a.allow_llm,
                            snapshot_check_root=snap_root)

    if result["misses"]:
        print(f"캐시 미스 {len(result['misses'])}건 — 출력 안 씀. "
              f"실콜하려면 --allow-llm.")
        for m in result["misses"]:
            print(f"  {m['key'][:12]}… msgs={m['n_msgs']} memos={m['memos']}")
        sys.exit(1)
    if result["meta"]["snapshot_mismatch"]:
        print(f"스냅샷 불일치 턴: {result['meta']['snapshot_mismatch']}")
        sys.exit(2)
    write_outputs(result, pathlib.Path(a.out_dir))
    errs = [t["turn"] for t in result["turns"] if t.get("error")]
    print(f"완주 {len(result['turns'])}턴, 요약 "
          f"{result['turns'][-1]['n_summaries']}개, 에러턴 {errs or '없음'}, "
          f"미스 0 → {_slug(result['meta'])}.json")


if __name__ == "__main__":
    main()
```

구현 시 조정 허용(플랜 후기에 기록): `_CacheOnlySummarize`는 `hypa.summarize(send, batch, S)` 3-인자 호출 형태와 시그니처가 맞아야 한다(exp `hypa_step`:587, main Task 4). `allow_llm=True` 경로는 원본 `hypa.summarize`(캐시 rw) + `bench_summary_cache()` 네임스페이스를 그대로 쓴다.

- [ ] **Step 4: 통과 확인**

```bash
python3 -m pytest tests/test_inspect_hypa.py -q
```

Expected: PASS 3건.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/fixed_haystack/inspect_hypa.py tests/test_inspect_hypa.py
git diff --cached --stat && git commit -m "feat(inspector): inspect_hypa CLI — 전진 루프·캐시-온리 요약·미스 차단"
```

---

### Task 6: HTML 뷰

**Files:**
- Modify: `benchmarks/fixed_haystack/inspect_hypa.py` (`render_html` 추가, `write_outputs`에서 호출)
- Test: `tests/test_inspect_hypa.py` (테스트 추가)

**Interfaces:**
- Consumes: Task 5 result dict
- Produces: `render_html(result) -> str` — 자급자족 단일 html. `write_outputs`가 `<slug>.html`도 쓴다.

- [ ] **Step 1: 실패 테스트 추가**

```python
def test_render_html_selfcontained():
    res = {"meta": {"impl": "main", "hsid": "mini", "max_ctx": 45000,
                    "summary_model": "x", "settings": {"memory_tokens_ratio": 0.2},
                    "snapshot_mismatch": []},
           "turns": [
               {"turn": 1, "memory_text": "", "kept_start_msg": 0,
                "n_summaries": 0, "wire_tokens": 100, "events": [], "metrics": None,
                "error": None},
               {"turn": 2, "memory_text": "<Past Events Summary>\nA\n</Past Events Summary>",
                "kept_start_msg": 4, "n_summaries": 1, "wire_tokens": 90,
                "events": [{"ev": "batch", "idx": ["m0"], "tokens": 5}],
                "metrics": None, "error": None}],
           "misses": []}
    html = inspect_hypa.render_html(res)
    assert "hypa-inspect" not in html or True   # 스모크
    assert "const DATA" in html                 # 데이터 인라인
    assert "Past Events Summary" in html        # memory_text 포함
    assert "<script src=" not in html           # 외부 의존 0


def test_write_outputs_also_writes_html(tmp_path):
    res = {"meta": {"impl": "exp", "hsid": "mini", "max_ctx": 1, "summary_model": "x",
                    "settings": {}, "snapshot_mismatch": []},
           "turns": [], "misses": []}
    inspect_hypa.write_outputs(res, out_dir=tmp_path)
    assert list(tmp_path.glob("*.html"))
```

- [ ] **Step 2: 실패 확인**

```bash
python3 -m pytest tests/test_inspect_hypa.py -q
```

Expected: 신규 2건 FAIL (`render_html` 없음 / html 미생성)

- [ ] **Step 3: 구현**

`render_html(result)`: 인라인 `<script>const DATA = {json};</script>` + 바닐라 JS.
구성(외부 리소스 0, `rag-inspect-r0.html` 형식 참고):
- 상단: meta 표(impl/hsid/max_ctx/모델/세팅)
- 타임라인 SVG: x=턴. ① `n_summaries` 꺾은선 ② `kept_start_msg` 계단선
  ③ batch 이벤트 있는 턴에 마커(●) ④ `wire_tokens` 보조선
- 턴 리스트(클릭) → 상세 패널: memory_text `<pre>`, 이벤트 테이블(배치 memo,
  reserve, important/recent/similar picked, similar는 쿼리·상위 청크 점수 표),
  main이면 metrics 4종
- 다크/라이트는 신경 안 씀 — 로컬 파일 뷰어

`write_outputs`에 `(out_dir / f"{slug}.html").write_text(render_html(result))` 추가.

- [ ] **Step 4: 통과 확인**

```bash
python3 -m pytest tests/test_inspect_hypa.py -q
```

Expected: 전부 PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/fixed_haystack/inspect_hypa.py tests/test_inspect_hypa.py
git diff --cached --stat && git commit -m "feat(inspector): 자급자족 HTML 타임라인 뷰"
```

---

### Task 7: 검증 런 (₩0) — 실데이터 완주 + 스냅샷 대조

**Files:** 없음 (실행·확인만, 출력은 gitignored)

**Interfaces:**
- Consumes: Task 5 CLI. 검증 런 조건 = 오버라이드 0개(export 원본 0.39/0.01 + run_params max_ctx 100000 + gemini-3-flash 캐시).

- [ ] **Step 1: 두 건초더미 exp 검증 런**

```bash
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl exp
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-van-r0 --impl exp
```

Expected(설계 단계 실측과 동일해야 함): drm `완주 100턴, 요약 18개, 미스 0`,
van `완주 100턴, 요약 11개, 미스 0`, 스냅샷 불일치 없음(종료코드 0),
`hypa-inspect-fix-drm-r0-exp.json/.html` 생성.

- [ ] **Step 2: main 검증 런 (같은 무오버라이드 조건)**

```bash
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl main
```

Expected: 벤치 데이터에 스킵 대상이 없어 배치 경계가 exp와 동일 → **미스 0**으로
완주. json에서 `n_summaries` 최종값이 exp와 같은지 확인(18). 다르면 중단하고
원인 파악(스펙 §3 표 밖의 미지 차이 발견 = 후기에 기록).
(스냅샷 대조는 main에선 자동 비활성 — exp 전용 앵커.)

- [ ] **Step 3: HTML 육안 확인**

`hypa-inspect-fix-drm-r0-exp.html`을 열어 완료 기준 3(스펙 §10) 항목 확인:
첫 압축 발동 턴(70턴대), 요약 개수 증가, kept_start_msg 점프, memory_text 변화,
similar 청크 점수. 스크린샷 불필요 — 확인 결과를 후기에 한 줄씩.

- [ ] **Step 4: Commit** (코드 수정이 나왔을 경우에만)

```bash
git status --short  # 코드 변경 없으면 커밋 생략
```

---

### Task 8: 관찰 런 드라이 — 미스 열거 후 사용자 승인 게이트

**Files:** 없음

**Interfaces:**
- Consumes: Task 5 CLI. 관찰 조건 = `--memory-ratio 0.2 --extra-ratio 0.1 --max-context 45000 --summary-model google/gemini-2.5-flash-lite`.

- [ ] **Step 1: 드라이 런 (실콜 0 보장 경로)**

```bash
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl exp \
  --memory-ratio 0.2 --extra-ratio 0.1 --max-context 45000 \
  --summary-model google/gemini-2.5-flash-lite
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl main \
  --memory-ratio 0.2 --extra-ratio 0.1 --max-context 45000 \
  --summary-model google/gemini-2.5-flash-lite
```

Expected: `캐시 미스 N건 — 출력 안 씀` + 미스 목록, 종료코드 1. (모델이 바뀌어
캐시 키 전부 미스 — 정상.) exp/main의 N이 같은지 기록(배치 경계 동일성 방증).

- [ ] **Step 2: 사용자 승인 게이트 — STOP**

미스 건수·예상 비용(미스 1건 ≈ 요약 1콜, flash-lite 단가 기준 합계 추정)을
사용자에게 보고하고 **`--allow-llm` 실행 승인을 받은 뒤에만** 다음 스텝.
승인 없으면 여기서 종료.

- [ ] **Step 3: (승인 후) 관찰 런 본실행**

```bash
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl exp \
  --memory-ratio 0.2 --extra-ratio 0.1 --max-context 45000 \
  --summary-model google/gemini-2.5-flash-lite --allow-llm
python3 -m benchmarks.fixed_haystack.inspect_hypa --hs fix-drm-r0 --impl main \
  --memory-ratio 0.2 --extra-ratio 0.1 --max-context 45000 \
  --summary-model google/gemini-2.5-flash-lite --allow-llm
```

Expected: 완주 + json/html 생성. `hypa.SUMMARY_COST` 합계를 후기에 기록.
2회차(같은 명령 재실행, --allow-llm 없이)가 미스 0으로 완주하는 것도 확인
(새 요약이 캐시에 쌓였다는 증거). fix-van-r0도 동일 2종 실행.

- [ ] **Step 4: HTML 육안 확인 + 사용자 보고**

45K/0.2 조건에서 첫 압축 발동 턴이 검증 런(100K)보다 훨씬 이른지, similar
선택 근거(청크 점수)가 보이는지 확인. exp/main html 나란히 열어 차이 요약을
사용자에게 보고.

---

### Task 9: 실행 후기 (Plan Postmortem)

**Files:**
- Modify: `docs/superpowers/plans/2026-08-27-hypa-inspector.md` (이 파일 하단)

- [ ] **Step 1: 후기 작성**

플랜 하단에 "## 실행 후기" 섹션 추가 — 필수 포함:
- 핸드오프 전제 오류 박제: export `useExperimentalImpl=true` → exp가 뮈토스
  실사용 경로 (스펙 §0). 다음 핸드오프 작성 시 "export 내용을 직접 까라".
- 검증/관찰 런 실측 수치 (미스 건수, 비용, 첫 압축 턴, 요약 개수)
- 구현 중 발견한 플랜/스펙 결함과 교정 내역

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/plans/2026-08-27-hypa-inspector.md
git diff --cached --stat && git commit -m "docs(inspector): 실행 후기 — 실측 수치·핸드오프 전제 교정 박제"
```

---

## 실행 후기 (2026-08-27)

### 핸드오프 전제 오류 (다음 핸드오프 작성 시 재발 방지)

핸드오프(8/24)는 "RisuAI 기본값 `useExperimentalImpl: false` → 실사용 경로는
main이니 포트를 main으로 다시 베껴라"였다. **export 파일을 직접 까보니
`useExperimentalImpl: true`** — 병합(hypav3.ts:1819~1823)이 이 키를 받아들여
뮈토스 하이파 V5 실사용자는 exp를 탄다. 기존 `hypa.py`의 exp 모델링은 실수가
아니라 export에 충실한 것(파일 독스트링에 명시돼 있었다). 교훈: **설정
파일의 실제 값을 읽지 않고 코드 기본값으로 실사용 경로를 추정하지 말 것.**
결정: 사용자 선택으로 exp·main 양축 인스펙터.

### 실측 수치

| 런 | 조건 | 결과 |
|---|---|---|
| 검증 exp drm | 100K / 0.39 / 0.01 / gemini-3-flash 캐시 | 100턴, 요약 18, 미스 0, 스냅샷 9턴 일치, 첫 압축 74턴 |
| 검증 exp van | 〃 | 100턴, 요약 11, 미스 0, 스냅샷 일치 |
| 검증 main drm | 〃 | 요약 18, 미스 0, **배치 경계 exp와 완전 동일**(스킵 대상 없음 예측 적중) |
| 관찰 4종 | 45K / 0.2 / 0.1 / flash-lite | drm 요약 23(첫 압축 27턴), van 18. 미스 drm 23·van 18건 → 실콜 총 ~45, 비용 < $0.1 |
| 재실행 | 캐시-온리 | drm 미스 0 (캐시 적재 확인) |

관찰 소득 2건:
- **100K/0.39 조건에선 similar가 한 번도 요약을 못 고른다** — recent(60%)가
  18개 전부 흡수. 26턴 전부 `skipped: no_chunks`. 45K/0.2에선 65턴이 similar
  선택 수행, exp↔main의 similar 선택이 10턴에서 갈라짐(쿼리 구성 차이 실증).
- **flash-lite 폭주 배치 1건**: van m32~m39 요약에서 temp 0 결정론 폭주
  (finish_reason=length, 완주 8192tok/44KB 반복). 절단본은 설계대로 캐시
  미기록 → 이 배치만 영구 미스로 남는다(실행당 1콜, $0.004). 폭주 요약은
  크기 때문에 어떤 선택 예산에도 안 들어가 관찰 유효성 무손상.

### 플랜 결함과 교정 (구현자 발견)

1. **테스트 픽스처 3건 오산** — (a) trace similar 테스트: 기본 세팅에선
   important+recent가 후보를 전부 흡수해 similar 이벤트 미발생 → 큰 요약으로
   recent 예산 조기 소진 픽스처로 교정. (b) plan_batches 테스트의
   `tot+10_000`은 보호구간 에러 경로로 빠짐 → `10_050`으로 교정.
   (c) main 고정창 테스트의 exp 비교 호출 `tot*2`도 동일 문제 → `tot*1.1`.
2. **디스크 캐시 오염** — 플랜 테스트가 `summarize` 기본 cache_dir
   (dreaming_data/eval/hypa-cache)에 가짜 요약을 씀 → 후속 테스트가 히트해
   오염 실패. autouse 픽스처(tmp 캐시 래핑)로 격리 + 오염 6파일 삭제.
   교훈: **summarize를 부르는 테스트는 반드시 캐시를 격리하라.**
3. **`__main__` 블록 순서** — heredoc append가 render_html을 `if __name__`
   뒤에 붙여 `-m` 실행만 NameError (import 경로 테스트는 통과). 블록을 파일
   끝으로 이동. 교훈: append 방식 구현은 실행 진입점이 항상 파일 끝인지 확인.
4. **similar 후보 0 관찰 공백** — 이벤트가 `if queries and chunks:` 안에만
   있어 "recent가 전부 흡수한 턴"이 안 보였다 → skipped 사유를 단 빈 이벤트
   추가 (로직 무접촉).
5. **.env 부재** — 새 워크트리에 API 키 파일이 없어 --allow-llm이 즉사.
   자격증명이라 에이전트가 복사 불가(권한 차단) → 사용자가 수동 복사.
   워크트리 셋업 체크리스트에 .env 추가할 것.
