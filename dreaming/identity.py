"""dreaming/identity.py — KV pair ledger + 판정 5종 (스펙 §3.1).

순수 로직(hash/extract/classify)은 saga.services.pair_ledger에서 승계한다.
이 모듈은 Phase 2(플러그인)에 포팅되지 않는다 — 플러그인은 chat.id가
ground truth라 판정 자체가 불필요하다 (스펙 §8).

원장: {session}/ledger, key=f"{index:06d}", 인덱스당 문서 1개(전이는 덮어쓰기).
saga와 달리 superseded 이력은 보존하지 않는다 — 원문은 {session}/raw에 남고,
지식 이력은 Fact 버전 체인이 담당한다 (의도적 단순화).
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

from dreaming.resolver import SessionResolver
from dreaming.storage import Storage
from saga.services.pair_ledger import classify, hash_text

ACTIVE_STATUSES = ("provisional", "confirmed")

VerdictKind = Literal["new_session", "next_turn", "continuation", "reroll", "diverged"]


class Verdict(BaseModel):
    kind: VerdictKind
    position: int
    reroll_turn_number: Optional[int] = None
    aligned: bool = False


def _map_kind(raw: Dict, chain_len: int, request_pairs: List[Dict],
              last_user_hash: Optional[str]) -> VerdictKind:
    """saga 3종(new/append/reroll) → 스펙 5종."""
    if raw["kind"] == "reroll":
        return "reroll" if raw["position"] == chain_len - 1 else "diverged"
    if chain_len == 0 and not request_pairs:
        return "new_session"
    if last_user_hash is None:
        return "continuation"
    return "next_turn"


class PairLedger:
    def __init__(self, storage: Storage, session_id: str,
                 resolver: "SessionResolver | None" = None) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = resolver

    def _ns(self) -> str:
        return f"{self._session}/ledger"

    @staticmethod
    def _key(index: int) -> str:
        return f"{index:06d}"

    def chain(self, active_only: bool = True) -> List[Dict]:
        rows = [row for _, row in self._storage.scan(self._ns())]
        if active_only:
            rows = [r for r in rows if r["status"] in ACTIVE_STATUSES]
        return rows

    def analyze_and_apply(self, pairs: List[Dict],
                          last_user_hash: Optional[str]) -> Verdict:
        chain = self.chain()
        raw = classify(chain, pairs, last_user_hash)
        kind = _map_kind(raw, len(chain), pairs, last_user_hash)

        # index → 저장 키 매핑: chain은 active만이라 위치가 곧 index가 아님.
        # classify가 주는 인덱스는 chain 리스트 기준 → 실제 row의 index 필드 사용.
        for ci in raw["superseded_indices"]:
            self._transition(chain[ci], "superseded")
        for ci in raw["quarantined_indices"]:
            self._transition(chain[ci], "quarantined")
        for ci, client_asst_hash in raw["confirm"]:
            row = dict(chain[ci])
            row["status"] = "confirmed"
            if client_asst_hash:
                # display script가 본문을 바꿨을 수 있음 — 클라이언트 버전이 정본
                row["assistant_hash"] = client_asst_hash
            self._storage.put(self._ns(), self._key(row["index"]), row)

        return Verdict(
            kind=kind,
            position=raw["position"],
            reroll_turn_number=raw["reroll_turn_number"],
            aligned=raw["aligned"],
        )

    def _transition(self, row: Dict, status: str) -> None:
        updated = dict(row)
        updated["status"] = status
        self._storage.put(self._ns(), self._key(row["index"]), updated)

    def record_turn(self, verdict: Verdict, last_user_hash: Optional[str],
                    user_text: str, assistant_text: str,
                    turn_number: int) -> None:
        if not last_user_hash:
            return
        asst_hash = hash_text(assistant_text)
        self._storage.put(self._ns(), self._key(verdict.position), {
            "index": verdict.position,
            "user_hash": last_user_hash,
            "assistant_hash": asst_hash,
            "status": "provisional",
            "turn_number": turn_number,
        })
        # Dreamer(B-2) 추출 입력용 원문 보존 (스펙 §3.2)
        self._storage.put(f"{self._session}/raw", f"{turn_number:06d}", {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "user_hash": last_user_hash,
            "assistant_hash": asst_hash,
            "turn_number": turn_number,
        })
        # 세션 해석 역색인 갱신 (resolver가 없으면 no-op)
        if self._resolver is not None:
            self._resolver.index_pair(self._session, last_user_hash, asst_hash)
