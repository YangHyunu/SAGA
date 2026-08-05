"""dreaming/dreamer.py — 유휴 이해 사이클 B-0~B-3 (스펙 §3.2).

B-1(에피소드 경계)과 B-2(추출)는 단일 구조화 출력 1콜로 병합한다 —
스펙의 "사이클 = LLM 1콜 + 임베딩"에서 임베딩은 후속 플랜.
B-4 재압축(청크 조립)은 Plan 4 — 이 모듈의 꿈은 지식 계층만 쓰므로
프리픽스가 불변이고, 꿈 도중 요청이 와도 캐시 충돌이 없다.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from dreaming.records import Actor, Fact

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
