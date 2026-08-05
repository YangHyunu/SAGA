"""dreaming/store.py — 세션 스코프 저장 파사드.

이후 컴포넌트(동기 경로, Dreamer, 대시보드)는 Storage를 직접 만지지 않고
이 파사드만 쓴다. 네임스페이스: <session_id>/{facts,episodes,commits,actors}.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from dreaming.records import Actor, Episode, Fact, StateCommit
from dreaming.storage import Storage
from dreaming.worldstate import replay


class MemoryStore:
    def __init__(self, storage: Storage, session_id: str) -> None:
        self._storage = storage
        self._session = session_id

    def _ns(self, kind: str) -> str:
        return f"{self._session}/{kind}"

    # -- Fact ---------------------------------------------------------
    def save_fact(self, f: Fact) -> None:
        self._storage.put(self._ns("facts"), f.id, f.model_dump(mode="json"))

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        data = self._storage.get(self._ns("facts"), fact_id)
        return Fact.model_validate(data) if data is not None else None

    def list_facts(self, include_superseded: bool = False) -> List[Fact]:
        facts = [Fact.model_validate(v) for _, v in self._storage.scan(self._ns("facts"))]
        if not include_superseded:
            facts = [f for f in facts if f.status != "superseded"]
        return facts

    # -- Episode ------------------------------------------------------
    def save_episode(self, e: Episode) -> None:
        self._storage.put(self._ns("episodes"), e.id, e.model_dump(mode="json"))

    def list_episodes(self) -> List[Episode]:
        return [Episode.model_validate(v) for _, v in self._storage.scan(self._ns("episodes"))]

    # -- WorldState ---------------------------------------------------
    def append_commit(self, c: StateCommit) -> None:
        self._storage.put(self._ns("commits"), c.id, c.model_dump(mode="json"))

    def list_commits(self) -> List[StateCommit]:
        return [StateCommit.model_validate(v) for _, v in self._storage.scan(self._ns("commits"))]

    def current_state(self) -> Dict[str, Union[float, str]]:
        return replay(self.list_commits())

    # -- Actor --------------------------------------------------------
    def save_actor(self, a: Actor) -> None:
        self._storage.put(self._ns("actors"), a.id, a.model_dump(mode="json"))

    def list_actors(self) -> List[Actor]:
        return [Actor.model_validate(v) for _, v in self._storage.scan(self._ns("actors"))]
