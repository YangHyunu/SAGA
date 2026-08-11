"""RisuAI 키워드 로어 활성화 에뮬 (lorebook.svelte.ts 준거).

키 매칭: lowercase + 공백 제거 substring (174-222행). 스캔: 최근 scan_depth개
메시지 (84행, 기본 5). 정렬: constant·활성 keyed 합쳐 sort(-order) 후
reverse (608-662행) — charx2card._split_lore가 constant에 이미 적용한 규칙과
동일해, lore_orders로 원래 order를 복원해 재병합한다. depth 0 활성 엔트리는
로어 블록이 아니라 postEverything (index.svelte.ts:582-590).

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
    merged = ([(o, b) for o, b in zip(orders, lore)]
              + [(e["order"], e["content"]) for e in block_hits])
    merged.sort(key=lambda t: -t[0])
    merged.reverse()
    return [b for _, b in merged], "\n\n".join(post)
