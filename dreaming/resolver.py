"""dreaming/resolver.py — 해시 역색인 세션 해석 (saga resolve_session 이식).

와이어에 세션 식별자가 없으므로(§0.1) 요청의 최근 pair 해시로 세션을 찾는다.
색인: 전역 네임스페이스 "pair-index", key=해시, 값={"sessions":[...]}.
확정 규칙(saga 승계): assistant 매치 ≥1 또는 총 매치 ≥2.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from dreaming.storage import Storage

_RESOLVE_WINDOW = 6
_NS = "pair-index"


class SessionResolver:
    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    def index_pair(self, session_id: str, user_hash: str,
                   assistant_hash: Optional[str]) -> None:
        for h in (user_hash, assistant_hash):
            if not h:
                continue
            doc = self._storage.get(_NS, h) or {"sessions": []}
            if session_id not in doc["sessions"]:
                doc["sessions"].append(session_id)
                self._storage.put(_NS, h, doc)

    def resolve(self, pairs: List[Dict]) -> Optional[str]:
        if not pairs:
            return None
        recent = pairs[-_RESOLVE_WINDOW:]
        scores: Dict[str, Dict[str, int]] = {}
        for p in recent:
            for h, kind in ((p.get("user_hash"), "user"),
                            (p.get("assistant_hash"), "asst")):
                if not h:
                    continue
                doc = self._storage.get(_NS, h)
                if not doc:
                    continue
                for sid in doc["sessions"]:
                    s = scores.setdefault(sid, {"user": 0, "asst": 0})
                    s[kind] += 1
        best, best_key = None, (-1, -1)
        for sid, s in sorted(scores.items()):
            total = s["user"] + s["asst"]
            if s["asst"] >= 1 or total >= 2:
                key = (s["asst"], total)
                if key > best_key:
                    best, best_key = sid, key
        return best
