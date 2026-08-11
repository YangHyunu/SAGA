"""RisuAI 키워드 로어 활성화 에뮬 (lorebook.svelte.ts 준거).

키 매칭: lowercase + **스페이스(U+0020)만** 제거 후 substring — 본체는
`replace(/ /g,'')`라 개행·탭은 지우지 않는다(lorebook.svelte.ts:206,208).
스캔: 최근 scan_depth개
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

# 본체는 `replace(/ /g,'')` — U+0020 스페이스만 지운다. \s+였다면 개행·탭도
# 지워 245키 중 공백 포함 11개가 과활성됐다(리뷰 실증: "White Lotus" 키가
# "White\nLotus" 발화에도 붙음). \x00(메시지 구분자)은 공백이 아니라 이
# 정규식이 아무리 넓어져도 안 지워진다 — 크로스 메시지 매칭 차단은 별개.
_WS = re.compile(r" +")


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
    # 한계(기록만, 2026-08-11 리뷰): 활성 keyed depth0은 순서 없이 그냥
    # join해 constant post_everything 뒤에 통짜로 붙는다 — 이 카드는 depth0
    # keyed가 0건이라 무해하지만, 여러 건이 동시 활성되는 카드라면 상대
    # 순서가 어긋날 수 있다.
    post = [e["content"] for e in hits if e.get("depth") == 0]
    block_hits = [e for e in hits if e.get("depth") != 0]
    # lore/lore_orders/lore_indices는 charx2card가 항상 같이 만드는
    # 병렬 3종이다 — 하나라도 없거나 길이가 어긋나면(zip이 조용히
    # 잘라먹기 전에) 여기서 바로 죽는다. .get(...) or 폴백을 안 쓰는 건
    # 의도적: 이 세 값이 안 맞으면 재정렬 결과가 조용히 틀려지므로,
    # 존재·길이 둘 다 명시로 확인한다.
    orders = card["lore_orders"]
    indices = card["lore_indices"]
    if len(orders) != len(lore) or len(indices) != len(lore):
        raise ValueError(
            f"lore({len(lore)})/lore_orders({len(orders)})/"
            f"lore_indices({len(indices)}) 길이 불일치 — zip이 조용히 "
            "잘라먹는다")
    consts = list(zip(indices, orders, lore))
    hits_t = [(e["index"], e["order"], e["content"]) for e in block_hits]
    merged = consts + hits_t
    merged.sort(key=lambda t: t[0])       # 북 원순서 복원 (RisuAI fullLore 순회 순서)
    merged.sort(key=lambda t: -t[1])      # priority desc — 파이썬 안정 정렬이 동점을 북 순서로 유지
    merged.reverse()                      # RisuAI 최종 reverse (lorebook.svelte.ts:662)
    return [c for _, _, c in merged], "\n\n".join(post)
