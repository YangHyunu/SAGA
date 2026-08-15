"""dreaming/worldstate.py — commit 원장 리플레이 (스펙 §4.3).

현재값은 저장하지 않는다. append-only 커밋을 (turn, recorded_at) 순으로
접어서 도출한다 — "소지금이 왜 450인지" 히스토리로 추적 가능해야 한다.
pending_contradiction 커밋은 확정 전까지 제외한다.

리플레이는 **전항 함수다** — 원장은 LLM이 쓴 비신뢰 데이터라 타입이 어긋난
커밋(비수치 슬롯에 add 등)이 실존한다 (실측: set time="…" 뒤 add time).
여기서 raise하면 오염 커밋 1건이 이후 모든 지식 렌더를 영구 사살한다 —
동기 경로 한복판이므로 건너뛰고 경고만 남긴다 (fail-open, 스펙 §2.6).
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Union

from dreaming.records import StateCommit

logger = logging.getLogger(__name__)

Value = Union[float, str]


def _numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def replay(commits: Iterable[StateCommit]) -> Dict[str, Value]:
    state: Dict[str, Value] = {}
    ordered = sorted(commits, key=lambda c: (c.turn, c.recorded_at))
    for c in ordered:
        if c.status == "pending_contradiction":
            continue
        if c.op == "set":
            state[c.slot] = c.value
        elif c.op == "add":
            current = state.get(c.slot, 0)
            if not _numeric(current) or not _numeric(c.value):
                logger.warning("[worldstate] add 불가 커밋 건너뜀: "
                               "slot=%r turn=%s", c.slot, c.turn)
                continue
            state[c.slot] = current + c.value
    return state
