"""dreaming/sync.py — 동기 경로 오케스트레이터 (스펙 §3.1).

기록·검색·조립·주입만. 턴당 LLM 0콜 — 이해는 전부 Dreamer(Plan 3) 몫.
임베딩 검색은 후속 플랜: v1 지식 렌더링은 state + pinned/confirmed fact +
main actor 결정론 템플릿이다.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from dreaming.assembly import (HOT_ZONE_CHAR_BUDGET, clip_knowledge,
                               inject_knowledge)
from dreaming.chunks import apply_compression
from dreaming.identity import PairLedger, Verdict
from dreaming.lore_shift import shift_keyed
from dreaming.marking import mark_cache
from dreaming.records import Episode
from dreaming.resolver import SessionResolver
from dreaming.retrieval import (expand_query, last_user_query, rank_episodes,
                                rank_facts, scene_query)
from dreaming.storage import Storage
from dreaming import scaffold, tracing
from dreaming.store import MemoryStore
from saga.services.pair_ledger import extract_pairs

logger = logging.getLogger(__name__)

_BLOCK_SEP = "\n\n"
_FACT_HEADER = "[확정 사실]\n"

# 핫존 에피소드 주입 (§3.1 확장): 프리픽스 청크에 있는 정보도 활용이 안 되는
# 실측(fix-drm-r0 T39·T79 — 값이 청크에 실렸는데 나레이터가 무시, 핫존 전달분만
# 활용 3/3) 때문에, 장면 관련 에피소드를 핫존으로 끌어온다.
_EP_TOP_K = 2        # 예산 정책: 2×600=1200자 = 총예산의 20%, 사실이 주력 유지
_EP_CHAR_CAP = 600   # 실측 P100 커버(에피 렌더 최소215/중앙392/최대587) — 이상치 방어


def _render_episode(e: Episode, cap: int = _EP_CHAR_CAP) -> str:
    """에피소드 핫존 렌더 — 요약·실마리는 우선, 발췌는 통째로 들어갈 때만.

    반쪽 문장이 주입되면 모델이 이어 쓰려는 사고가 나므로 문장 중간 절단
    금지 — 초과분은 뒤에서부터 통째로 떨군다. open_threads는 짧고(한 줄)
    "그때 그 얘기" 회상의 단골 목표물이라 발췌보다 먼저 넣는다 (실측:
    fix-drm-r0 T39 — 값이 threads에만 있어 빼면 전달 불가).
    """
    text = f"[지난 장면 · {e.title}]\n{e.summary.strip()}"
    if len(text) > cap:
        return text[:cap]        # 병리 방어 — 실측 분포에선 도달하지 않는다
    if e.open_threads:
        line = "\n남은 실마리: " + " / ".join(e.open_threads)
        if len(text) + len(line) <= cap:
            text += line
    for ex in e.key_excerpts:
        line = f'\n원문: "{ex}"'
        if len(text) + len(line) > cap:
            break
        text += line
    return text

# 연속 미정렬이 이 횟수에 닿으면 원장을 버리고 다시 베이스라인을 잡는다.
# 격리 턴은 record_turn을 안 타서 체인이 자라지 않는다 — 한 번 오염된
# 베이스라인은 스스로 못 빠져나온다 (night2-drm-r0: 105/106턴 격리).
_MISALIGN_LIMIT = 3


def render_knowledge_meta(store: MemoryStore, query: str = "",
                          budget: int = HOT_ZONE_CHAR_BUDGET,
                          ep_query: Optional[str] = None
                          ) -> Tuple[str, Dict]:
    """지식 4블록 렌더 + 관측 메타 — 상태·인물·에피소드 먼저 확보하고
    사실에 잔여 예산을 준다.

    사실은 **pinned 우선, 그 안에서 장면 관련도순**이다(어휘 랭킹, 빈
    쿼리면 최신순). 오름차순으로 앞에서 자르면 초반 사실에 영구 고정돼
    이후 배운 게 하나도 안 들어간다 (실측: confirmed 179개 중 인덱스
    0~19만 주입 — docs/DREAMING_FLAW.md §3).

    에피소드는 쿼리와 겹칠 때만(top-2, 개당 캡) — rank_episodes 주석 참조.
    ep_query가 따로 오면 에피소드 랭킹은 그걸 쓴다(기본은 query와 동일).
    사실 쿼리엔 직전 응답이 필요(대명사 선행사)하지만 에피소드엔 독이다
    (현재 장면 어휘가 과거 회상을 밀어냄 — last_user_query 주석의 실측).

    예산을 사실이 다 먹게 두면 뒤에 붙는 인물 블록을 clip_knowledge가 통째로
    날린다. 그래서 개수 상한이 아니라 **잔여 예산**으로 자른다.

    메타는 검색 관측 로그(rlog)용: 주입된 사실 id·에피소드 id/점수·자수.
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

    ep_blocks: List[str] = []
    ep_meta: List[Dict] = []
    eq = query if ep_query is None else ep_query
    for ep, score in rank_episodes(store.list_episodes(), eq)[:_EP_TOP_K]:
        ep_blocks.append(_render_episode(ep))
        ep_meta.append({"id": ep.id, "title": ep.title,
                        "score": round(score, 2)})

    facts = [f for f in store.list_facts()
             if f.pinned or f.status == "confirmed"]
    facts = rank_facts(facts, query)
    room = budget - len(_FACT_HEADER)
    for block in (state_block, actor_block, *ep_blocks):
        if block:
            room -= len(block) + len(_BLOCK_SEP)
    fact_lines: List[str] = []
    fact_ids: List[str] = []
    for f in facts:
        line = f"- {f.claim}"
        if len(line) + 1 > room:
            break
        fact_lines.append(line)
        fact_ids.append(f.id)
        room -= len(line) + 1

    parts = [b for b in (state_block,
                         _FACT_HEADER + "\n".join(fact_lines) if fact_lines else "",
                         *ep_blocks,
                         actor_block) if b]
    text = _BLOCK_SEP.join(parts)
    return text, {"facts": fact_ids, "episodes": ep_meta, "chars": len(text)}


def render_knowledge(store: MemoryStore, query: str = "",
                     budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    return render_knowledge_meta(store, query, budget)[0]


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
                 keyed_lore: Optional[List[str]] = None,
                 mark_cache_enabled: bool = True) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = SessionResolver(storage)
        self._ledger = PairLedger(storage, session_id, resolver=self._resolver)
        self._store = MemoryStore(storage, session_id)
        self._keyed_lore = keyed_lore or []
        self._rlog_n = sum(1 for _ in storage.scan(f"{session_id}/rlog"))
        # cache_control(BP 마킹)은 Anthropic 규약이다. 이해 못 하는 업스트림
        # (DeepSeek 본가 등)엔 미지 필드 + parts 변환(upstream.to_wire)을
        # 보내게 되므로 프록시가 업스트림 보고 끈다 (스펙 §6.3 프로바이더 한정).
        self._mark_cache = mark_cache_enabled

    def _wire_state(self) -> Dict:
        return self._storage.get(f"{self._session}/wire", "scaffold") or {}

    def _rlog(self, rec: Dict) -> None:
        """검색 관측 로그 — 실사용 미스를 추출/전달 퍼널로 분해할 데이터.

        턴당 한 건, <session>/rlog/<연번>. 벤치의 값 생존 감사를 실사용에서
        재현하는 장치이자 격리 침묵 사망의 가시화. 로깅 실패가 채팅을 막으면
        본말전도라 fail-open.
        """
        try:
            self._storage.put(f"{self._session}/rlog",
                              f"{self._rlog_n:06d}", rec)
            self._rlog_n += 1
        except Exception:
            logger.exception("[sync] rlog failed: %s", self._session)

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
            # 기록은 격리 버퍼로 (스펙 §3.1). 격리도 rlog에 남긴다 —
            # fail-open의 그림자는 침묵 사망이라 가시화가 필수다.
            self._rlog({"quarantine": True, "kind": verdict.kind})
            tracing.log_run("dreaming.turn",
                            inputs={"session": self._session},
                            outputs={"quarantine": True, "kind": verdict.kind},
                            metadata={"session": self._session})
            return messages + tail, verdict
        out, _ = shift_keyed(messages, self._keyed_lore)   # 1안 (스펙 §5)
        # 첫 요청은 tail_fp를 못 배워 프리셋 프리필이 messages에 섞여 있다
        # (위 first_request 계산과 동일 조건) — 쿼리가 영문 보일러플레이트로
        # 오염되므로 빈 쿼리(=최신순 폴백)로 렌더한다.
        if first_request:
            query = ep_q = ""
        else:
            actors = self._store.list_actors()
            query = expand_query(scene_query(messages), actors)
            ep_q = expand_query(last_user_query(messages), actors)
        knowledge, kmeta = render_knowledge_meta(self._store, query=query,
                                                 ep_query=ep_q)
        knowledge = clip_knowledge(knowledge)
        self._rlog({"quarantine": False, "kind": verdict.kind,
                    "turn": verdict.position, "query": query[-300:],
                    **kmeta})
        tracing.log_run(
            "dreaming.turn",
            inputs={"query": query, "ep_query": ep_q},
            outputs={"knowledge": knowledge,
                     "episodes": kmeta["episodes"],
                     "n_facts": len(kmeta["facts"])},
            metadata={"session": self._session, "turn": verdict.position,
                      "kind": verdict.kind, "chars": kmeta["chars"]})
        bp2 = None
        plan = self._storage.get(f"{self._session}/compression", "plan")
        if (plan is not None and verdict.aligned
                and verdict.offset is not None):
            out, bp2 = apply_compression(out, plan,
                                         window_start_turn=verdict.offset)
        out = inject_knowledge(out, knowledge)
        if self._mark_cache:
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
