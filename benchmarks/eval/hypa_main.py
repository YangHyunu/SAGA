"""HypaV3 기본 구현(hypaMemoryV3Main, hypav3.ts:955~1618) 포트.

exp 포트(hypa.py)와의 차이는 스펙 §3 표(docs/superpowers/specs/
2026-08-27-hypa-inspector-design.md) — 배치 고정 창, 즉시 요약,
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
            for s in ranked:
                t = hypa._summary_tokens(s)
                if t + used_similar > reserved_similar:
                    break              # 1428~1438
                sel_sim.append(s)
                used_similar += t
            selected.extend(sel_sim)
            hypa._tr(trace, ev="similar", queries=queries,
                     chunk_scores=[[[ch["id"], round(sc, 6)] for ch, sc in lst]
                                   for lst in scored_lists],
                     cc_rank=[ch["id"] for ch in ranked_chunks],
                     rrf_rank=[hypa._index_of(summaries, s) for s in ranked],
                     picked=[hypa._index_of(summaries, s) for s in sel_sim],
                     reserved=reserved_similar, used=used_similar)
        else:
            # 후보/쿼리 없음 — recent가 전부 흡수한 턴도 관찰돼야 한다
            hypa._tr(trace, ev="similar", queries=queries, chunk_scores=[],
                     cc_rank=[], rrf_rank=[], picked=[],
                     reserved=reserved_similar, used=0,
                     skipped=("no_chunks" if not chunks else "no_queries"))

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
    # available 예산 안에서 그리디라 초과 도달 불가 (hypa.py hypa_step 주석).
    return memory, history[start_idx:], start_idx, data, None
