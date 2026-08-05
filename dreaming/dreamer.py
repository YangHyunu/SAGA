"""dreaming/dreamer.py — 유휴 이해 사이클 B-0~B-3 (스펙 §3.2).

B-1(에피소드 경계)과 B-2(추출)는 단일 구조화 출력 1콜로 병합한다 —
스펙의 "사이클 = LLM 1콜 + 임베딩"에서 임베딩은 후속 플랜.
B-4 재압축(청크 조립)은 Plan 4 — 이 모듈의 꿈은 지식 계층만 쓰므로
프리픽스가 불변이고, 꿈 도중 요청이 와도 캐시 충돌이 없다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from dreaming.facts import dreamer_can_modify, supersede
from dreaming.llm import LLMClient
from dreaming.records import (Actor, Episode, Evidence, Fact, StateCommit,
                              TypedNumber)
from dreaming.storage import Storage
from dreaming.store import MemoryStore

logger = logging.getLogger(__name__)

_MAX_PROMPT_FACTS = 50


# ------------------------------------------------------------------ #
# 추출 스키마 — 단일 구조화 출력 (B-1 + B-2)
# ------------------------------------------------------------------ #

class ExtractedNumber(BaseModel):
    name: str
    value: float
    unit: str = ""


class ExtractedFact(BaseModel):
    claim: str
    entities: List[str] = []
    numbers: List[ExtractedNumber] = []
    evidence_turn: int
    action: Literal["ADD", "UPDATE", "DELETE", "NOOP"] = "ADD"   # mem0 4분류
    target_fact_id: Optional[str] = None
    learned_by: List[str] = []


class ExtractedCommit(BaseModel):
    slot: str
    op: Literal["set", "add"]
    value: Union[float, str]
    turn: int


class ExtractedActor(BaseModel):
    names: List[str]
    profile: str = ""
    tier: Literal["main", "support", "extra"] = "support"


class ExtractedEpisode(BaseModel):
    start_turn: int
    end_turn: int
    title: str
    summary: str
    open_threads: List[str] = []


class DreamExtraction(BaseModel):
    episodes: List[ExtractedEpisode] = []
    facts: List[ExtractedFact] = []
    commits: List[ExtractedCommit] = []
    actors: List[ExtractedActor] = []


_FENCE_RE = re.compile(r"^```[A-Za-z]*\s*\n?|\n?```\s*$")


def parse_extraction(text: str) -> DreamExtraction:
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"dream output is not JSON: {e}") from e
    return DreamExtraction.model_validate(data)


# ------------------------------------------------------------------ #
# 프롬프트 — 결정론적 조립
# ------------------------------------------------------------------ #

_SYSTEM = """너는 RP 세션 로그에서 구조화 지식을 추출하는 분석기다.
반드시 JSON 객체 하나만 출력한다. 마크다운 펜스·설명·사과 금지.

스키마:
{
  "episodes": [{"start_turn": int, "end_turn": int, "title": str, "summary": str, "open_threads": [str]}],
  "facts": [{"claim": str, "entities": [str],
             "numbers": [{"name": str, "value": number, "unit": str}],
             "evidence_turn": int, "action": "ADD|UPDATE|DELETE|NOOP",
             "target_fact_id": str|null, "learned_by": [str]}],
  "commits": [{"slot": str, "op": "set|add", "value": number|str, "turn": int}],
  "actors": [{"names": [str], "profile": str, "tier": "main|support|extra"}]
}

규칙:
- fact 1개 = 원자 명제 1개. 복합 문장 금지.
- numbers는 원문에 명시된 숫자만. 추측·계산 금지.
- 기존 사실을 갱신하면 action=UPDATE + target_fact_id(기존 지식의 id).
  기존 사실이 무효가 됐으면 DELETE. 변화 없으면 NOOP. 새 사실은 ADD.
- commits는 수치·상태 변화만 (소지금, 위치, 시각 등). 서술은 fact로.
- episodes는 장면 경계로 나눈다. open_threads엔 미회수 복선만.
- learned_by는 그 사실을 알게 된 인물 이름 목록."""


def build_dream_prompt(raw_turns: List[Dict], facts: List[Fact],
                       state: Dict, actors: List[Actor]) -> Tuple[str, str]:
    known: List[str] = []
    if state:
        known.append("현재 상태:\n" + "\n".join(
            f"- {slot}: {value}" for slot, value in sorted(state.items())))
    recent = sorted(facts, key=lambda f: f.recorded_at)[-_MAX_PROMPT_FACTS:]
    if recent:
        known.append("기존 사실:\n" + "\n".join(
            f"- [{f.id}] {f.claim} ({f.status})" for f in recent))
    if actors:
        known.append("인물:\n" + "\n".join(
            f"- {'/'.join(a.names)} ({a.tier})"
            for a in sorted(actors, key=lambda a: a.names[0])))

    turns = "\n\n".join(
        f"[턴 {r['turn_number']}]\nuser: {r['user_text']}\nassistant: {r['assistant_text']}"
        for r in raw_turns)

    user = ("[기존 지식]\n" + ("\n\n".join(known) if known else "없음")
            + "\n\n[미처리 원문]\n" + turns)
    return _SYSTEM, user


# ------------------------------------------------------------------ #
# B-3: 검증·적용
# ------------------------------------------------------------------ #

def verify_numbers(numbers: List[ExtractedNumber], text: str) -> bool:
    """숫자 정규식 재검증 (스펙 §3.2 B-3): 원문에 문자 그대로 있어야 한다."""
    plain = text.replace(",", "")
    for n in numbers:
        v = n.value
        s = str(int(v)) if float(v).is_integer() else str(v)
        if s not in plain:
            return False
    return True


def _turn_text(raw: Dict) -> str:
    return raw["user_text"] + "\n" + raw["assistant_text"]


def _build_fact(ef: ExtractedFact, raw_by_turn: Dict[int, Dict]) -> Fact:
    raw = raw_by_turn.get(ef.evidence_turn)
    verified = raw is not None and verify_numbers(ef.numbers, _turn_text(raw))
    return Fact(
        claim=ef.claim,
        entities=ef.entities,
        numbers=[TypedNumber(name=n.name, value=n.value, unit=n.unit or None)
                 for n in ef.numbers],
        evidence=[Evidence(pair_hash=raw["user_hash"])] if raw else [],
        learned_by=ef.learned_by,
        status="confirmed" if verified else "provisional",
    )


def apply_extraction(store: MemoryStore, ext: DreamExtraction,
                     raw_by_turn: Dict[int, Dict]) -> Dict[str, int]:
    report = {"facts": 0, "blocked": 0, "commits": 0, "actors": 0, "episodes": 0}

    for ef in ext.facts:
        if ef.action == "NOOP":
            continue
        target = store.get_fact(ef.target_fact_id) if ef.target_fact_id else None
        if ef.action == "DELETE":
            if target is None:
                continue
            if not dreamer_can_modify(target):
                report["blocked"] += 1
                continue
            store.save_fact(target.model_copy(update={"status": "superseded"}))
            continue
        new = _build_fact(ef, raw_by_turn)
        if ef.action == "UPDATE" and target is not None:
            if dreamer_can_modify(target):
                old2, new = supersede(target, new)
                store.save_fact(old2)
            else:
                # 유저 편집이 ground truth (스펙 §2.7) — 모순으로 관찰만
                new = new.model_copy(update={"status": "pending_contradiction"})
                report["blocked"] += 1
        store.save_fact(new)
        report["facts"] += 1

    for ec in ext.commits:
        raw = raw_by_turn.get(ec.turn)
        status = "applied"
        if isinstance(ec.value, (int, float)) and not isinstance(ec.value, bool):
            text = _turn_text(raw) if raw else ""
            probe = ExtractedNumber(name=ec.slot, value=float(ec.value))
            if not verify_numbers([probe], text):
                status = "pending_contradiction"
        store.append_commit(StateCommit(
            slot=ec.slot, op=ec.op, value=ec.value, turn=ec.turn,
            evidence=Evidence(pair_hash=raw["user_hash"]) if raw else None,
            status=status,
        ))
        report["commits"] += 1

    existing = store.list_actors()
    for ea in ext.actors:
        match = next((a for a in existing if set(a.names) & set(ea.names)), None)
        if match is not None:
            names = list(match.names) + [n for n in ea.names if n not in match.names]
            store.save_actor(match.model_copy(update={
                "names": names,
                "profile": ea.profile or match.profile,
                "tier": ea.tier,
            }))
        else:
            store.save_actor(Actor(names=ea.names, profile=ea.profile, tier=ea.tier))
        report["actors"] += 1

    for ep in ext.episodes:
        start = raw_by_turn.get(ep.start_turn)
        end = raw_by_turn.get(ep.end_turn)
        if start is None or end is None:
            continue
        store.save_episode(Episode(
            range_start=start["user_hash"], range_end=end["user_hash"],
            title=ep.title, summary=ep.summary, open_threads=ep.open_threads,
        ))
        report["episodes"] += 1

    return report


# ------------------------------------------------------------------ #
# 사이클 오케스트레이터 (B-0 스냅샷 → 1콜 → B-3 적용 → 커서 전진)
# ------------------------------------------------------------------ #

class Dreamer:
    def __init__(self, storage: Storage, llm: LLMClient) -> None:
        self._storage = storage
        self._llm = llm
        self._locks: Dict[str, asyncio.Lock] = {}

    def _cursor(self, session: str) -> int:
        doc = self._storage.get(f"{session}/dreamer", "cursor")
        return doc["next_turn"] if doc else 0

    def snapshot(self, session: str) -> List[Dict]:
        cur = self._cursor(session)
        rows = [row for _, row in self._storage.scan(f"{session}/raw")
                if row["turn_number"] >= cur]
        return sorted(rows, key=lambda r: r["turn_number"])

    def has_backlog(self, session: str) -> bool:
        return bool(self.snapshot(session))

    async def dream(self, session: str) -> Optional[Dict]:
        lock = self._locks.setdefault(session, asyncio.Lock())
        if lock.locked():
            return None
        async with lock:
            try:
                return await self._cycle(session)
            except Exception:
                # 사이클 폐기, 커서 불변 → 다음 유휴에 재시도 (스펙 §2.6, §3.2)
                logger.exception("[dreamer] cycle discarded: %s", session)
                return None

    async def _cycle(self, session: str) -> Optional[Dict]:
        raw_turns = self.snapshot(session)                       # B-0
        if not raw_turns:
            return None
        store = MemoryStore(self._storage, session)
        system, user = build_dream_prompt(
            raw_turns, store.list_facts(), store.current_state(),
            store.list_actors())
        ext = parse_extraction(await self._llm.complete(system, user))  # B-1+B-2
        raw_by_turn = {r["turn_number"]: r for r in raw_turns}
        report = apply_extraction(store, ext, raw_by_turn)       # B-3
        self._storage.put(f"{session}/dreamer", "cursor",
                          {"next_turn": raw_turns[-1]["turn_number"] + 1})
        return report
