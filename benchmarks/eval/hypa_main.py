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
