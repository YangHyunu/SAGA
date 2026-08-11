"""RisuAI 키워드 로어 활성화 에뮬 (lorebook.svelte.ts 준거).

키 매칭: lowercase + 공백 제거 substring (174-222행). 스캔: 최근 scan_depth개
메시지 (84행, 기본 5). 정렬: constant·활성 keyed 합쳐 priority(=
insertion_order) 내림차순, 동점은 북(book) 원위치 오름차순, 그 결과를 다시
전체 reverse (608-662행) — charx2card._split_lore가 constant 블록에 이미
적용한 규칙(T)과 동일하다.

주의(2026-08-11 리뷰 교정): `card["lore"]`는 T가 이미 한 번 적용된 최종
순서다. order만 복원해 여기서 T를 재적용하면 동점 그룹에 한해 T∘T ≠ T가
되어 원시 북 순서로 되돌아간다 — stable sort + 전체 reverse는 그 자체로
멱등이 아니다(동점 항목의 상대 순서가 reverse 한 번마다 뒤집힌다). 그래서
charx2card가 넘기는 북 원위치(`lore_indices`/keyed 각 원소의 "index")로
매번 원시 데이터부터 T를 다시 계산한다 — 몇 번을 합쳐 재정렬해도 결과가
안정적이다.

depth 0 활성 엔트리는 로어 블록이 아니라 postEverything
(index.svelte.ts:582-590).

이 카드(Alternate Hunters)가 안 쓰는 기능은 미구현: selective/regex/
case_sensitive 플래그/재귀 스캔/토큰 예산 (2026-08-11 실측 0건).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s).lower()


def activate(card: Dict, recent_texts: List[str]) -> Tuple[List[str], str]:
    keyed = card.get("keyed_lore") or []
    lore = list(card.get("lore", []))
    if not keyed:
        return lore, ""
    depth = int(card.get("lore_settings", {}).get("scan_depth", 5))
    scan = _norm("\x00".join(recent_texts[-depth:]))
    hits = [e for e in keyed
            if any(_norm(k) and _norm(k) in scan for k in e["keys"])]
    post = [e["content"] for e in hits if e.get("depth") == 0]
    block_hits = [e for e in hits if e.get("depth") != 0]
    orders = card.get("lore_orders") or [0] * len(lore)
    consts = list(zip(card["lore_indices"], orders, lore))
    hits_t = [(e["index"], e["order"], e["content"]) for e in block_hits]
    merged = consts + hits_t
    merged.sort(key=lambda t: t[0])       # 북 원순서 복원 (RisuAI fullLore 순회 순서)
    merged.sort(key=lambda t: -t[1])      # priority desc — 파이썬 안정 정렬이 동점을 북 순서로 유지
    merged.reverse()                      # RisuAI 최종 reverse (lorebook.svelte.ts:662)
    return [c for _, _, c in merged], "\n\n".join(post)
