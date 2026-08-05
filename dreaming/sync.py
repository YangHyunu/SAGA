"""dreaming/sync.py — 동기 경로 오케스트레이터 (스펙 §3.1).

기록·검색·조립·주입만. 턴당 LLM 0콜 — 이해는 전부 Dreamer(Plan 3) 몫.
임베딩 검색은 후속 플랜: v1 지식 렌더링은 state + pinned/confirmed fact +
main actor 결정론 템플릿이다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from dreaming.assembly import clip_knowledge, inject_knowledge
from dreaming.identity import PairLedger, Verdict
from dreaming.marking import mark_cache
from dreaming.resolver import SessionResolver
from dreaming.storage import Storage
from dreaming.store import MemoryStore
from saga.services.pair_ledger import extract_pairs

_MAX_FACTS = 20


def render_knowledge(store: MemoryStore) -> str:
    parts: List[str] = []

    state = store.current_state()
    if state:
        lines = [f"- {slot}: {value}" for slot, value in sorted(state.items())]
        parts.append("[현재 상태]\n" + "\n".join(lines))

    facts = [f for f in store.list_facts()
             if f.pinned or f.status == "confirmed"]
    facts = sorted(facts, key=lambda f: (not f.pinned, f.recorded_at))[:_MAX_FACTS]
    if facts:
        parts.append("[확정 사실]\n" + "\n".join(f"- {f.claim}" for f in facts))

    actors = [a for a in store.list_actors() if a.tier == "main"]
    if actors:
        lines = [f"- {a.names[0]}: {a.profile}" if a.profile else f"- {a.names[0]}"
                 for a in sorted(actors, key=lambda a: a.names[0])]
        parts.append("[주요 인물]\n" + "\n".join(lines))

    return "\n\n".join(parts)


class SyncPath:
    def __init__(self, storage: Storage, session_id: str) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = SessionResolver(storage)
        self._ledger = PairLedger(storage, session_id, resolver=self._resolver)
        self._store = MemoryStore(storage, session_id)

    def process(self, messages: List[Dict]) -> Tuple[List[Dict], Verdict]:
        pairs, last_user_hash = extract_pairs(messages)
        verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
        knowledge = clip_knowledge(render_knowledge(self._store))
        out = inject_knowledge(messages, knowledge)
        out = mark_cache(out)
        return out, verdict

    def record_response(self, verdict: Verdict, messages: List[Dict],
                        assistant_text: str) -> None:
        pairs, last_user_hash = extract_pairs(messages)
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        self._ledger.record_turn(
            verdict, last_user_hash, user_text, assistant_text,
            turn_number=verdict.position,
        )
