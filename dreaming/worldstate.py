"""dreaming/worldstate.py — commit 원장 리플레이 (스펙 §4.3).

현재값은 저장하지 않는다. append-only 커밋을 (turn, recorded_at) 순으로
접어서 도출한다 — "소지금이 왜 450인지" 히스토리로 추적 가능해야 한다.
pending_contradiction 커밋은 확정 전까지 제외한다.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Union

from dreaming.records import StateCommit

logger = logging.getLogger(__name__)

Value = Union[float, str]


def _numeric(v: object) -> bool:
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
            # 나쁜 커밋 하나가 리플레이 전체를 죽이면 안 된다 (스펙 §2.6 fail-open)
            current = state.get(c.slot, 0)
            if not _numeric(current) or not _numeric(c.value):
                logger.warning(
                    "[worldstate] skip bad add commit %s: slot %r "
                    "current=%r value=%r", c.id, c.slot, current, c.value)
                continue
            state[c.slot] = current + c.value
    return state
