"""dreaming/worldstate.py — commit 원장 리플레이 (스펙 §4.3).

현재값은 저장하지 않는다. append-only 커밋을 (turn, recorded_at) 순으로
접어서 도출한다 — "소지금이 왜 450인지" 히스토리로 추적 가능해야 한다.
pending_contradiction 커밋은 확정 전까지 제외한다.
"""

from __future__ import annotations

from typing import Dict, Iterable, Union

from dreaming.records import StateCommit

Value = Union[float, str]


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
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                raise TypeError(f"cannot add to non-numeric slot {c.slot!r}")
            if not isinstance(c.value, (int, float)) or isinstance(c.value, bool):
                raise TypeError(f"add value must be numeric for slot {c.slot!r}")
            state[c.slot] = current + c.value
    return state
