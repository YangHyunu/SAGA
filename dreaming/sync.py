"""dreaming/sync.py — 동기 경로 오케스트레이터 (스펙 §3.1).

기록·검색·조립·주입만. 턴당 LLM 0콜 — 이해는 전부 Dreamer(Plan 3) 몫.
임베딩 검색은 후속 플랜: v1 지식 렌더링은 state + pinned/confirmed fact +
main actor 결정론 템플릿이다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from dreaming.assembly import (HOT_ZONE_CHAR_BUDGET, clip_knowledge,
                               inject_knowledge)
from dreaming.chunks import apply_compression
from dreaming.identity import PairLedger, Verdict
from dreaming.lore_shift import shift_keyed
from dreaming.marking import mark_cache
from dreaming.resolver import SessionResolver
from dreaming.storage import Storage
from dreaming import scaffold
from dreaming.store import MemoryStore
from saga.services.pair_ledger import extract_pairs

_BLOCK_SEP = "\n\n"
_FACT_HEADER = "[확정 사실]\n"

# 연속 미정렬이 이 횟수에 닿으면 원장을 버리고 다시 베이스라인을 잡는다.
# 격리 턴은 record_turn을 안 타서 체인이 자라지 않는다 — 한 번 오염된
# 베이스라인은 스스로 못 빠져나온다 (night2-drm-r0: 105/106턴 격리).
_MISALIGN_LIMIT = 3


def render_knowledge(store: MemoryStore,
                     budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    """지식 3블록 렌더 — 상태·인물 먼저 확보하고 사실에 잔여 예산을 준다.

    사실은 **pinned 우선, 그 안에서 최신순**이다. 오름차순으로 앞에서 자르면
    초반 사실에 영구 고정돼 이후 배운 게 하나도 안 들어간다 (실측: confirmed
    179개 중 인덱스 0~19만 주입 — docs/DREAMING_FLAW.md §3).

    예산을 사실이 다 먹게 두면 뒤에 붙는 인물 블록을 clip_knowledge가 통째로
    날린다. 그래서 개수 상한이 아니라 **잔여 예산**으로 자른다.
    """
    state = store.current_state()
    state_block = ""
    if state:
        state_block = "[현재 상태]\n" + "\n".join(
            f"- {slot}: {value}" for slot, value in sorted(state.items()))

    actors = [a for a in store.list_actors() if a.tier == "main"]
    actor_block = ""
    if actors:
        actor_block = "[주요 인물]\n" + "\n".join(
            f"- {a.names[0]}: {a.profile}" if a.profile else f"- {a.names[0]}"
            for a in sorted(actors, key=lambda a: a.names[0]))

    facts = [f for f in store.list_facts()
             if f.pinned or f.status == "confirmed"]
    facts.sort(key=lambda f: (f.pinned, f.recorded_at), reverse=True)
    room = budget - len(_FACT_HEADER)
    for block in (state_block, actor_block):
        if block:
            room -= len(block) + len(_BLOCK_SEP)
    fact_lines: List[str] = []
    for f in facts:
        line = f"- {f.claim}"
        if len(line) + 1 > room:
            break
        fact_lines.append(line)
        room -= len(line) + 1

    parts = [b for b in (state_block,
                         _FACT_HEADER + "\n".join(fact_lines) if fact_lines else "",
                         actor_block) if b]
    return _BLOCK_SEP.join(parts)


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

    def _wire_state(self) -> Dict:
        return self._storage.get(f"{self._session}/wire", "scaffold") or {}

    def _split(self, messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """프리셋 프리필 꼬리를 벗긴다 (dreaming/scaffold.py).

        꼬리를 달고 처리하면 쌍 판정·주입 위치·BP3가 전부 프리필을 가리킨다.
        """
        return scaffold.split(messages, self._wire_state().get("tail_fp"))

    def _rebaseline(self) -> None:
        """오염된 원장을 버린다 — 다음 판정이 이번 요청으로 다시 잡는다.

        어떤 요청 해시와도 안 맞는 행(프리필 등)이 베이스라인이면 정렬이
        영구 실패하고, 격리 턴은 record_turn을 안 타 체인이 자라지도 않는다.
        무효화는 리롤과 같은 primitive(demote_after)를 쓴다 — raw를 읽으므로
        raw 삭제보다 반드시 먼저 부른다.
        """
        rows = [r for _, r in self._storage.scan(f"{self._session}/ledger")]
        turns = [r["turn_number"] for r in rows
                 if r.get("turn_number") is not None]
        if turns:
            demote_after(self._storage, self._session, min(turns))
            for key, row in list(self._storage.scan(f"{self._session}/raw")):
                if row.get("turn_number", 0) >= min(turns):
                    self._storage.delete(f"{self._session}/raw", key)
        for key, _ in list(self._storage.scan(f"{self._session}/ledger")):
            self._storage.delete(f"{self._session}/ledger", key)

    def process(self, messages: List[Dict]) -> Tuple[List[Dict], Verdict]:
        state = self._wire_state()
        # 첫 요청은 직전 요청이 없어 프리필 꼬리를 배울 수단이 없다
        # (scaffold.learn은 prev_fp 없이 항상 None)
        first_request = not state.get("prev_fp")
        tail_fp = state.get("tail_fp") or scaffold.learn(messages,
                                                         state.get("prev_fp"))
        new_state = {"prev_fp": scaffold.fingerprint(messages),
                     "tail_fp": tail_fp, "misaligned": 0}
        messages, tail = scaffold.split(messages, tail_fp)

        ledger_was_empty = first_request and not self._ledger.chain()
        pairs, last_user_hash = extract_pairs(messages)
        verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
        if verdict.quarantine:
            n = (state.get("misaligned") or 0) + 1
            if n >= _MISALIGN_LIMIT:
                self._rebaseline()                 # 자기치유 — 재판정
                verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
            else:
                new_state["misaligned"] = n
        self._storage.put(f"{self._session}/wire", "scaffold", new_state)
        if pairs and ledger_was_empty:
            # 꼬리를 못 배운 첫 요청의 pair가 진짜 히스토리인지 프리셋
            # 프리필인지 가릴 정보가 없다 — 베이스라인 기록을 한 턴 미룬다.
            # (원장이 이미 있으면 정렬이 pair의 실재성을 증명하므로 제외)
            verdict = verdict.model_copy(update={"baseline_deferred": True})
        if (verdict.kind in ("reroll", "diverged")
                and verdict.reroll_turn_number is not None):
            demote_after(self._storage, self._session, verdict.reroll_turn_number)
        if verdict.quarantine:
            # 판정 불확실 — 주입·압축·마킹 없이 무가공 passthrough,
            # 기록은 격리 버퍼로 (스펙 §3.1)
            return messages + tail, verdict
        out, _ = shift_keyed(messages, self._keyed_lore)   # 1안 (스펙 §5)
        knowledge = clip_knowledge(render_knowledge(self._store))
        bp2 = None
        plan = self._storage.get(f"{self._session}/compression", "plan")
        if (plan is not None and verdict.aligned
                and verdict.offset is not None):
            out, bp2 = apply_compression(out, plan,
                                         window_start_turn=verdict.offset)
        out = inject_knowledge(out, knowledge)
        out = mark_cache(out, bp2_index=bp2)
        return out + tail, verdict

    def record_response(self, verdict: Verdict, messages: List[Dict],
                        assistant_text: str) -> None:
        if verdict.baseline_deferred:
            return                  # 다음 턴이 꼬리를 벗기고 베이스라인을 잡는다
        messages, _ = self._split(messages)
        pairs, last_user_hash = extract_pairs(messages)
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_text = m.get("content", "")
                break
        if verdict.quarantine:
            if last_user_hash:
                ns = f"{self._session}/quarantine"
                n = len(list(self._storage.scan(ns)))
                self._storage.put(ns, f"{n:06d}", {
                    "user_text": user_text, "assistant_text": assistant_text,
                    "user_hash": last_user_hash, "kind": verdict.kind,
                })
            return
        self._ledger.record_turn(
            verdict, last_user_hash, user_text, assistant_text,
            turn_number=verdict.position,
        )
