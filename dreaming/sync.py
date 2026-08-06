"""dreaming/sync.py — 동기 경로 오케스트레이터 (스펙 §3.1).

기록·검색·조립·주입만. 턴당 LLM 0콜 — 이해는 전부 Dreamer(Plan 3) 몫.
임베딩 검색은 후속 플랜: v1 지식 렌더링은 state + pinned/confirmed fact +
main actor 결정론 템플릿이다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from dreaming.assembly import clip_knowledge, inject_knowledge
from dreaming.chunks import apply_compression
from dreaming.identity import PairLedger, Verdict
from dreaming.lore_shift import shift_keyed
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


def demote_after(storage: Storage, session: str, from_turn: int) -> None:
    """리롤/분기 시점 이후에서 배운 지식을 잠정화한다 (스펙 §3.1).

    Fact → provisional (user_edited 보호), 해당 구간 applied commit →
    pending_contradiction, 꿈 커서 되감기 → 다음 꿈이 대체 응답을 재추출.
    """
    stale_hashes = {row["user_hash"] for _, row in storage.scan(f"{session}/raw")
                    if row["turn_number"] >= from_turn}
    store = MemoryStore(storage, session)
    for f in store.list_facts():
        if f.user_edited or f.status == "provisional":
            continue
        if any(e.pair_hash in stale_hashes for e in f.evidence):
            store.save_fact(f.model_copy(update={"status": "provisional"}))
    for c in store.list_commits():
        if c.turn >= from_turn and c.status == "applied":
            store.update_commit_status(c.id, "pending_contradiction")
    # 분기점이 압축 구간 안이면 플랜 폐기 + 걸친 에피소드 삭제 —
    # 다음 꿈이 재조립한다 (TTL 창구라 캐시 비용 0, 스펙 §6.3)
    plan = storage.get(f"{session}/compression", "plan")
    if plan is not None and plan["covers_until_turn"] > from_turn:
        storage.delete(f"{session}/compression", "plan")
    for e in store.list_episodes():
        if e.end_turn is not None and e.end_turn >= from_turn:
            storage.delete(f"{session}/episodes", e.id)
    cursor = storage.get(f"{session}/dreamer", "cursor")
    if cursor is not None and cursor["next_turn"] > from_turn:
        storage.put(f"{session}/dreamer", "cursor", {"next_turn": from_turn})


class SyncPath:
    def __init__(self, storage: Storage, session_id: str,
                 keyed_lore: Optional[List[str]] = None) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = SessionResolver(storage)
        self._ledger = PairLedger(storage, session_id, resolver=self._resolver)
        self._store = MemoryStore(storage, session_id)
        self._keyed_lore = keyed_lore or []

    def process(self, messages: List[Dict]) -> Tuple[List[Dict], Verdict]:
        pairs, last_user_hash = extract_pairs(messages)
        verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
        if (verdict.kind in ("reroll", "diverged")
                and verdict.reroll_turn_number is not None):
            demote_after(self._storage, self._session, verdict.reroll_turn_number)
        out, _ = shift_keyed(messages, self._keyed_lore)   # 1안 (스펙 §5)
        knowledge = clip_knowledge(render_knowledge(self._store))
        bp2 = None
        plan = self._storage.get(f"{self._session}/compression", "plan")
        if plan is not None:
            out, bp2 = apply_compression(out, plan)
        out = inject_knowledge(out, knowledge)
        out = mark_cache(out, bp2_index=bp2)
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
