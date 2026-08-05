"""dreaming/records.py — 레코드 4종 (스펙 §4).

Fact / Episode / StateCommit / Actor. 전부 Pydantic v2 모델 —
Storage에는 model_dump(mode="json") dict로 들어가고 model_validate로 복원된다.
덮어쓰기 금지: Fact 갱신은 버전 체인(supersedes)으로만 한다 (스펙 §4.1, WISE).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


class TypedNumber(BaseModel):
    """구체 숫자 (가격/금액/수량) — 결정론 오라클 채점의 재료 (스펙 §4.1, §9)."""

    name: str
    value: float
    unit: Optional[str] = None


class Evidence(BaseModel):
    """원문 포인터: pair ledger의 pair_hash + 문자 오프셋 (스펙 §4.1)."""

    pair_hash: str
    offset: Optional[int] = None


FactStatus = Literal["provisional", "confirmed", "pending_contradiction", "superseded"]


class Fact(BaseModel):
    """원자 명제 1개 = 레코드 1개 (스펙 §4.1)."""

    id: str = Field(default_factory=_new_id)
    claim: str
    entities: List[str] = []
    numbers: List[TypedNumber] = []
    evidence: List[Evidence] = []
    valid_time: Optional[str] = None
    recorded_at: str = Field(default_factory=utc_now_iso)
    learned_by: List[str] = []
    status: FactStatus = "provisional"
    supersedes: Optional[str] = None
    user_edited: bool = False
    pinned: bool = False


class Episode(BaseModel):
    """서사 단위 — 꿈(B-1)이 경계를 판정한다 (스펙 §4.2). 청크 조립의 재료."""

    id: str = Field(default_factory=_new_id)
    range_start: str  # 시작 pair_hash
    range_end: str    # 끝 pair_hash
    title: str
    summary: str
    causes: List[str] = []        # 선행 에피소드 id
    open_threads: List[str] = []  # 미회수 복선 (스펙 §4.2, CFPG)
    embedding: Optional[List[float]] = None
    recorded_at: str = Field(default_factory=utc_now_iso)


ActorTier = Literal["main", "support", "extra"]


class Actor(BaseModel):
    """등장인물 — knows[]로 POV 격리 (스펙 §4.4). extra는 주입 제외."""

    id: str = Field(default_factory=_new_id)
    names: List[str] = Field(min_length=1)  # 한/영 별칭 통합
    profile: str = ""
    knows: List[str] = []                   # visibility-gated Fact id
    tier: ActorTier = "support"
    last_seen: Optional[str] = None


CommitOp = Literal["set", "add"]
CommitStatus = Literal["applied", "pending_contradiction", "manual"]


class StateCommit(BaseModel):
    """WorldState 변경의 유일한 경로 (스펙 §4.3). append-only 원장."""

    id: str = Field(default_factory=_new_id)
    slot: str
    op: CommitOp
    value: object  # float(set/add) 또는 str(set) — replay가 타입 검증
    turn: int
    evidence: Optional[Evidence] = None
    actor: Optional[str] = None
    recorded_at: str = Field(default_factory=utc_now_iso)
    status: CommitStatus = "applied"
