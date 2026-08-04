# Dreaming Core (Plan 1/6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dreaming의 코어 기반 — KV 문서 저장소, 레코드 4종(Fact/Episode/WorldState commit/Actor), WorldState 리플레이, Fact 버전 체인 — 를 스펙 §2·§4·§8에 맞게 구축한다.

**Architecture:** 저장은 `Storage` 프로토콜(get/put/delete/scan) 뒤의 KV 문서 샤드 — Phase 1은 JSON 파일 디렉터리, Phase 2에서 RisuAI pluginStorage로 백엔드만 교체된다 (스펙 §8: SQL·외부 DB 금지). 레코드는 Pydantic v2 모델로 JSON 직렬화가 공짜다. WorldState는 append-only commit 원장을 리플레이해서 현재값을 도출하고, Fact는 덮어쓰기 대신 supersede 버전 체인을 쓴다 (스펙 §4).

**Tech Stack:** Python 3, Pydantic v2 (기존 의존성, `requirements.txt`의 `pydantic>=2.0`), pytest. 신규 의존성 추가 금지.

**Spec:** [docs/dreaming/SPEC.md](../../dreaming/SPEC.md) — 특히 §2 원칙 8(저장 중립), §4 스키마 v2, §8 저장 모델.

## Global Constraints

- 이 플랜의 모든 신규 코드는 `dreaming/` 패키지에, 테스트는 `tests/test_dreaming_*.py`에 둔다 (tests/는 flat 구조).
- `saga/`와 `external/`은 **수정 금지** (external/은 읽기 전용 심링크 — AGENTS.md).
- 저장은 KV 문서 모델만. SQL, SQLite, 외부 DB(PostgreSQL/Honcho/Mem0류) 금지 (스펙 §8).
- 신규 pip 의존성 금지. Pydantic v2와 표준 라이브러리만.
- 인터프리터는 `python3` (이 머신에 `python` 없음). 테스트 실행: `python3 -m pytest <파일> -q`.
- 테스트 스타일은 기존 `tests/test_pair_ledger.py`를 따른다: 모듈 docstring + `# ---- #` 섹션 주석 + 함수 단위 테스트.
- 커밋 메시지는 기존 컨벤션(`feat:`/`docs:` + 한국어 요약) 유지, 브랜치는 `dreaming/spec`.
- 시간 문자열은 전부 ISO 8601 UTC (`datetime.now(timezone.utc).isoformat()`). 테스트에서는 결정론을 위해 명시적으로 주입한다.

---

### Task 1: Storage 프로토콜 + JSON 디렉터리 백엔드

**Files:**
- Create: `dreaming/__init__.py` (빈 파일)
- Create: `dreaming/storage.py`
- Test: `tests/test_dreaming_storage.py`

**Interfaces:**
- Consumes: 없음 (최초 태스크)
- Produces: `Storage` 프로토콜 — `get(namespace: str, key: str) -> dict | None`, `put(namespace: str, key: str, value: dict) -> None`, `delete(namespace: str, key: str) -> None`, `scan(namespace: str) -> Iterator[tuple[str, dict]]`. 구현체 `JsonDirStorage(root: Path)`. Task 6의 `MemoryStore`가 이 프로토콜만 본다. namespace는 `/`로 구분된 세그먼트 허용(예: `"sess1/facts"`), key와 각 세그먼트는 `[A-Za-z0-9._-]+`만 허용.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_storage.py
"""JsonDirStorage: KV 문서 저장 백엔드 (스펙 §8 — pluginStorage와 동일 모델)."""
import os

import pytest

from dreaming.storage import JsonDirStorage


# ------------------------------------------------------------------ #
# put / get / delete
# ------------------------------------------------------------------ #

def test_put_get_roundtrip(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("sess1/facts", "f1", {"claim": "포션 가격은 50골드", "pinned": False})
    assert s.get("sess1/facts", "f1") == {"claim": "포션 가격은 50골드", "pinned": False}


def test_get_missing_returns_none(tmp_path):
    s = JsonDirStorage(tmp_path)
    assert s.get("sess1/facts", "nope") is None


def test_put_overwrites(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    s.put("ns", "k", {"v": 2})
    assert s.get("ns", "k") == {"v": 2}


def test_delete(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    s.delete("ns", "k")
    assert s.get("ns", "k") is None
    # 없는 키 delete는 no-op (fail-open, 스펙 §2.6)
    s.delete("ns", "k")


# ------------------------------------------------------------------ #
# scan
# ------------------------------------------------------------------ #

def test_scan_yields_sorted_key_value_pairs(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "b", {"v": 2})
    s.put("ns", "a", {"v": 1})
    assert list(s.scan("ns")) == [("a", {"v": 1}), ("b", {"v": 2})]


def test_scan_missing_namespace_is_empty(tmp_path):
    s = JsonDirStorage(tmp_path)
    assert list(s.scan("ghost")) == []


def test_namespaces_are_isolated(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("sess1/facts", "k", {"v": 1})
    s.put("sess2/facts", "k", {"v": 2})
    assert s.get("sess1/facts", "k") == {"v": 1}
    assert s.get("sess2/facts", "k") == {"v": 2}


# ------------------------------------------------------------------ #
# 안전성
# ------------------------------------------------------------------ #

def test_put_leaves_no_tmp_files(tmp_path):
    # write-temp+rename crash 안전 (스펙 §8) — 성공 경로에 임시 파일 잔존 금지
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"v": 1})
    leftovers = [p for p in tmp_path.rglob("*") if p.is_file() and not p.name.endswith(".json")]
    assert leftovers == []


def test_rejects_path_traversal_key(tmp_path):
    s = JsonDirStorage(tmp_path)
    with pytest.raises(ValueError):
        s.put("ns", "../evil", {"v": 1})
    with pytest.raises(ValueError):
        s.get("ns/..", "k")


def test_korean_content_survives_roundtrip(tmp_path):
    s = JsonDirStorage(tmp_path)
    s.put("ns", "k", {"이름": "리사", "메모": "한/영 별칭"})
    assert s.get("ns", "k") == {"이름": "리사", "메모": "한/영 별칭"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_storage.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/__init__.py
```
(빈 파일)

```python
# dreaming/storage.py
"""dreaming/storage.py — KV 문서 저장 (스펙 §8).

Storage 프로토콜은 RisuAI pluginStorage(key -> JSON blob)와 1:1 대응한다.
Phase 1 백엔드는 JSON 파일 디렉터리, Phase 2는 pluginStorage로 교체된다.
SQL·외부 DB 금지.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Iterator, Optional, Protocol, Tuple

_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class Storage(Protocol):
    """KV 문서 저장 인터페이스. 값은 JSON 직렬화 가능한 dict."""

    def get(self, namespace: str, key: str) -> Optional[Dict]: ...

    def put(self, namespace: str, key: str, value: Dict) -> None: ...

    def delete(self, namespace: str, key: str) -> None: ...

    def scan(self, namespace: str) -> Iterator[Tuple[str, Dict]]: ...


def _check_segment(segment: str) -> str:
    if segment in (".", "..") or not _SEGMENT_RE.match(segment):
        raise ValueError(f"invalid storage path segment: {segment!r}")
    return segment


class JsonDirStorage:
    """디렉터리 기반 KV: <root>/<namespace...>/<key>.json, 원자적 쓰기."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _ns_dir(self, namespace: str) -> Path:
        parts = [_check_segment(p) for p in namespace.split("/")]
        return self.root.joinpath(*parts)

    def _path(self, namespace: str, key: str) -> Path:
        _check_segment(key)
        return self._ns_dir(namespace) / f"{key}.json"

    def get(self, namespace: str, key: str) -> Optional[Dict]:
        path = self._path(namespace, key)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, namespace: str, key: str, value: Dict) -> None:
        path = self._path(namespace, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # write-temp + atomic rename: 크래시 시에도 반쪽 파일이 남지 않는다
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def delete(self, namespace: str, key: str) -> None:
        path = self._path(namespace, key)
        if path.is_file():
            path.unlink()

    def scan(self, namespace: str) -> Iterator[Tuple[str, Dict]]:
        ns_dir = self._ns_dir(namespace)
        if not ns_dir.is_dir():
            return
        for path in sorted(ns_dir.glob("*.json")):
            yield path.stem, json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_storage.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/__init__.py dreaming/storage.py tests/test_dreaming_storage.py
git commit -m "feat(dreaming): Storage 프로토콜 + JsonDirStorage — KV 문서 샤드 백엔드 (스펙 §8)"
```

---

### Task 2: Fact 레코드 (+TypedNumber, Evidence)

**Files:**
- Create: `dreaming/records.py`
- Test: `tests/test_dreaming_records.py`

**Interfaces:**
- Consumes: 없음 (storage와 독립 — 직렬화는 Pydantic `model_dump(mode="json")` / `model_validate`)
- Produces: `utc_now_iso() -> str`, `TypedNumber(name, value, unit=None)`, `Evidence(pair_hash, offset=None)`, `Fact` — 필드: `id`(uuid4 hex 자동), `claim`, `entities: list[str]`, `numbers: list[TypedNumber]`, `evidence: list[Evidence]`, `valid_time: str|None`, `recorded_at: str`(자동), `learned_by: list[str]`, `status: Literal["provisional","confirmed","pending_contradiction","superseded"]`(기본 `"provisional"`), `supersedes: str|None`, `user_edited: bool=False`, `pinned: bool=False`. Task 5·6이 이 타입을 그대로 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_records.py
"""레코드 4종 직렬화 (스펙 §4). 이 파일은 Task 2~3에 걸쳐 자란다."""
import pytest
from pydantic import ValidationError

from dreaming.records import Evidence, Fact, TypedNumber


# ------------------------------------------------------------------ #
# Fact (스펙 §4.1)
# ------------------------------------------------------------------ #

def test_fact_minimal_defaults():
    f = Fact(claim="리사는 포션을 50골드에 판다")
    assert f.status == "provisional"          # 꿈이 확정하기 전까지 잠정 (스펙 §4.1)
    assert f.user_edited is False
    assert f.pinned is False
    assert f.supersedes is None
    assert len(f.id) == 32                    # uuid4 hex
    assert "T" in f.recorded_at               # ISO 8601


def test_fact_typed_numbers_and_evidence():
    f = Fact(
        claim="리사는 포션을 50골드에 판다",
        entities=["리사"],
        numbers=[TypedNumber(name="포션 가격", value=50, unit="골드")],
        evidence=[Evidence(pair_hash="abc123", offset=140)],
        learned_by=["user"],
        recorded_at="2026-08-04T00:00:00+00:00",
    )
    assert f.numbers[0].value == 50
    assert f.evidence[0].pair_hash == "abc123"


def test_fact_json_roundtrip():
    f = Fact(
        claim="리사는 포션을 50골드에 판다",
        numbers=[TypedNumber(name="포션 가격", value=50, unit="골드")],
        recorded_at="2026-08-04T00:00:00+00:00",
    )
    data = f.model_dump(mode="json")          # Storage에 넣는 dict
    assert isinstance(data, dict)
    restored = Fact.model_validate(data)
    assert restored == f


def test_fact_rejects_unknown_status():
    with pytest.raises(ValidationError):
        Fact(claim="x", status="deleted")      # 삭제는 상태가 아님 — supersede만 (스펙 §4.1)


def test_fact_ids_are_unique():
    assert Fact(claim="a").id != Fact(claim="a").id


def test_fact_default_lists_are_not_shared():
    a, b = Fact(claim="a"), Fact(claim="b")
    a.entities.append("리사")
    assert b.entities == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_records.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.records'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/records.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_records.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/records.py tests/test_dreaming_records.py
git commit -m "feat(dreaming): Fact 레코드 — 원자 명제 + typed numbers + evidence 포인터 (스펙 §4.1)"
```

---

### Task 3: Episode + Actor 레코드

**Files:**
- Modify: `dreaming/records.py` (Task 2 코드 아래에 추가)
- Test: `tests/test_dreaming_records.py` (아래 섹션 추가)

**Interfaces:**
- Consumes: Task 2의 모듈 구조 (`dreaming/records.py`)
- Produces: `Episode` — 필드: `id`(자동), `range_start: str`, `range_end: str`(pair_hash), `title`, `summary`, `causes: list[str]`, `open_threads: list[str]`, `embedding: list[float]|None`, `recorded_at`(자동). `Actor` — 필드: `id`(자동), `names: list[str]`(최소 1개), `profile: str=""`, `knows: list[str]`(Fact id 목록), `tier: Literal["main","support","extra"]="support"`, `last_seen: str|None`. Task 6이 이 타입을 그대로 쓴다.

- [ ] **Step 1: Write the failing test**

`tests/test_dreaming_records.py` 맨 아래에 추가:

```python
# ------------------------------------------------------------------ #
# Episode (스펙 §4.2)
# ------------------------------------------------------------------ #

def test_episode_roundtrip():
    from dreaming.records import Episode

    e = Episode(
        range_start="hash_a",
        range_end="hash_z",
        title="시장에서의 흥정",
        summary="리사와 가격을 흥정해 50골드에 합의했다.",
        causes=[],
        open_threads=["리사가 언급한 '밀수품'의 정체"],   # 미회수 복선 (CFPG)
    )
    assert e.embedding is None
    restored = type(e).model_validate(e.model_dump(mode="json"))
    assert restored == e


# ------------------------------------------------------------------ #
# Actor (스펙 §4.4)
# ------------------------------------------------------------------ #

def test_actor_defaults_and_roundtrip():
    from dreaming.records import Actor

    a = Actor(names=["리사", "Lisa"], profile="시장 상인")
    assert a.tier == "support"
    assert a.knows == []
    restored = type(a).model_validate(a.model_dump(mode="json"))
    assert restored == a


def test_actor_requires_at_least_one_name():
    from dreaming.records import Actor
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Actor(names=[])


def test_actor_rejects_unknown_tier():
    from dreaming.records import Actor
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Actor(names=["리사"], tier="villain")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_records.py -q`
Expected: FAIL — `ImportError: cannot import name 'Episode'`

- [ ] **Step 3: Write minimal implementation**

`dreaming/records.py` 맨 아래에 추가:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_records.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/records.py tests/test_dreaming_records.py
git commit -m "feat(dreaming): Episode·Actor 레코드 — open_threads 복선 추적 + POV 격리 (스펙 §4.2, §4.4)"
```

---

### Task 4: WorldState commit + 리플레이

**Files:**
- Modify: `dreaming/records.py` (`StateCommit` 추가)
- Create: `dreaming/worldstate.py`
- Test: `tests/test_dreaming_worldstate.py`

**Interfaces:**
- Consumes: Task 2의 `Evidence`, `utc_now_iso`, `_new_id`
- Produces: `StateCommit` — 필드: `id`(자동), `slot: str`, `op: Literal["set","add"]`, `value: float|str`, `turn: int`, `evidence: Evidence|None`, `actor: str|None`, `recorded_at`(자동), `status: Literal["applied","pending_contradiction","manual"]="applied"`. `replay(commits: Iterable[StateCommit]) -> dict[str, float|str]` — `(turn, recorded_at)` 순 적용, `pending_contradiction` 제외. Task 6의 `current_state()`가 이걸 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_worldstate.py
"""WorldState: typed commit 원장 + 리플레이 (스펙 §4.3 — 서술은 상태를 못 바꾼다)."""
import pytest
from pydantic import ValidationError

from dreaming.records import StateCommit
from dreaming.worldstate import replay


def _c(slot, op, value, turn, status="applied", recorded_at="2026-08-04T00:00:00+00:00"):
    return StateCommit(
        slot=slot, op=op, value=value, turn=turn, status=status, recorded_at=recorded_at
    )


# ------------------------------------------------------------------ #
# replay
# ------------------------------------------------------------------ #

def test_replay_set_then_add():
    # 소지금 500 → -50 → +300 = 750: 왜 750인지 커밋 히스토리로 추적 가능 (스펙 §7.1)
    state = replay([
        _c("소지금", "set", 500, turn=1),
        _c("소지금", "add", -50, turn=2),
        _c("소지금", "add", 300, turn=3),
    ])
    assert state == {"소지금": 750}


def test_replay_orders_by_turn_not_input_order(): 
    state = replay([
        _c("소지금", "add", -50, turn=2),
        _c("소지금", "set", 500, turn=1),
    ])
    assert state == {"소지금": 450}


def test_replay_excludes_pending_contradiction():
    # 모순 커밋은 확정 전까지 상태에 반영 금지 (스펙 §3.2 B-3, Stubbornness 2502.04390)
    state = replay([
        _c("소지금", "set", 500, turn=1),
        _c("소지금", "set", 9999, turn=2, status="pending_contradiction"),
    ])
    assert state == {"소지금": 500}


def test_replay_add_on_missing_slot_starts_from_zero():
    assert replay([_c("빚", "add", 100, turn=1)]) == {"빚": 100}


def test_replay_string_slots_set_only():
    state = replay([_c("현재_장소", "set", "시장", turn=1),
                    _c("현재_장소", "set", "여관", turn=2)])
    assert state == {"현재_장소": "여관"}


def test_replay_add_to_string_raises():
    with pytest.raises(TypeError):
        replay([_c("현재_장소", "set", "시장", turn=1),
                _c("현재_장소", "add", 1, turn=2)])


def test_replay_manual_commits_apply():
    # 유저 직접 수정 = manual 커밋으로 기록, 상태에 반영 (스펙 §7.2)
    state = replay([_c("소지금", "set", 500, turn=1),
                    _c("소지금", "set", 100, turn=2, status="manual")])
    assert state == {"소지금": 100}


# ------------------------------------------------------------------ #
# StateCommit 검증
# ------------------------------------------------------------------ #

def test_commit_rejects_unknown_op():
    with pytest.raises(ValidationError):
        _c("소지금", "multiply", 2, turn=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_worldstate.py -q`
Expected: FAIL — `ImportError: cannot import name 'StateCommit'`

- [ ] **Step 3: Write minimal implementation**

`dreaming/records.py` 맨 아래에 추가:

```python
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
```

새 파일:

```python
# dreaming/worldstate.py
"""dreaming/worldstate.py — commit 원장 리플레이 (스펙 §4.3).

현재값은 저장하지 않는다. append-only 커밋을 (turn, recorded_at) 순으로
접어서 도출한다 — "소지금이 왜 450인지" 히스토리로 추적 가능해야 한다.
pending_contradiction 커밋은 확정 전까지 제외한다.
"""

from __future__ import annotations

from typing import Dict, Iterable, Union

from dreaming.records import StateCommit

Value = Union[float, str]


def replay(commits: Iterable[StateCommit]) -> Dict[str, Value]:
    state: Dict[str, Value] = {}
    ordered = sorted(commits, key=lambda c: (c.turn, c.recorded_at))
    for c in ordered:
        if c.status == "pending_contradiction":
            continue
        if c.op == "set":
            state[c.slot] = c.value
        elif c.op == "add":
            current = state.get(c.slot, 0)
            if not isinstance(current, (int, float)) or isinstance(current, bool):
                raise TypeError(f"cannot add to non-numeric slot {c.slot!r}")
            if not isinstance(c.value, (int, float)) or isinstance(c.value, bool):
                raise TypeError(f"add value must be numeric for slot {c.slot!r}")
            state[c.slot] = current + c.value
    return state
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_worldstate.py -q`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/records.py dreaming/worldstate.py tests/test_dreaming_worldstate.py
git commit -m "feat(dreaming): WorldState commit 원장 + 리플레이 — pending_contradiction 격리 (스펙 §4.3)"
```

---

### Task 5: Fact 버전 체인 (supersede / 유저 편집 보호)

**Files:**
- Create: `dreaming/facts.py`
- Test: `tests/test_dreaming_facts.py`

**Interfaces:**
- Consumes: Task 2의 `Fact`, `utc_now_iso`
- Produces: `supersede(old: Fact, new: Fact) -> tuple[Fact, Fact]` (old는 `status="superseded"` 사본, new는 `supersedes=old.id` 사본 — 입력 불변), `dreamer_can_modify(fact: Fact) -> bool` (`user_edited`면 False — 스펙 §2.7), `apply_user_edit(fact: Fact, **changes) -> tuple[Fact, Fact]` (새 버전은 새 id + `user_edited=True` + `supersedes` 링크). Task 6과 이후 Dreamer 플랜(B-3 스윕)이 이 함수만 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_facts.py
"""Fact 버전 체인: 덮어쓰기 금지, invalidate-and-append (스펙 §4.1, WISE 2405.14768)."""
from dreaming.facts import apply_user_edit, dreamer_can_modify, supersede
from dreaming.records import Fact


# ------------------------------------------------------------------ #
# supersede
# ------------------------------------------------------------------ #

def test_supersede_links_chain_and_keeps_inputs_immutable():
    old = Fact(claim="포션은 50골드다")
    new = Fact(claim="포션은 30골드다 (할인)")
    old2, new2 = supersede(old, new)
    assert old2.status == "superseded"
    assert new2.supersedes == old.id
    # 원본 불변 — 저장은 호출자(MemoryStore) 몫
    assert old.status == "provisional"
    assert new.supersedes is None


# ------------------------------------------------------------------ #
# 유저 편집 보호 (스펙 §2.7, §7.2)
# ------------------------------------------------------------------ #

def test_dreamer_cannot_modify_user_edited_fact():
    f = Fact(claim="유저가 고친 사실", user_edited=True)
    assert dreamer_can_modify(f) is False


def test_dreamer_can_modify_ordinary_fact():
    assert dreamer_can_modify(Fact(claim="평범한 사실")) is True


def test_apply_user_edit_creates_protected_new_version():
    f = Fact(claim="포션은 50골드다")
    old2, new2 = apply_user_edit(f, claim="포션은 45골드다")
    assert old2.status == "superseded"
    assert new2.claim == "포션은 45골드다"
    assert new2.user_edited is True          # 이후 Dreamer가 못 덮음
    assert new2.supersedes == f.id
    assert new2.id != f.id                   # 새 버전 = 새 레코드
    assert new2.recorded_at != "" 


def test_apply_user_edit_preserves_untouched_fields():
    f = Fact(claim="포션은 50골드다", entities=["리사"], pinned=True)
    _, new2 = apply_user_edit(f, claim="포션은 45골드다")
    assert new2.entities == ["리사"]
    assert new2.pinned is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_facts.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.facts'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/facts.py
"""dreaming/facts.py — Fact 버전 체인 (스펙 §4.1).

갱신은 항상 invalidate-and-append: 기존 레코드는 superseded로 표시만 하고
새 레코드가 supersedes로 링크한다. 유저 편집(user_edited)은 ground truth —
Dreamer는 수정할 수 없다 (스펙 §2.7).
"""

from __future__ import annotations

import uuid
from typing import Tuple

from dreaming.records import Fact, utc_now_iso


def supersede(old: Fact, new: Fact) -> Tuple[Fact, Fact]:
    """old를 무효화하고 new를 체인에 링크한 사본 쌍을 돌려준다."""
    old2 = old.model_copy(update={"status": "superseded"})
    new2 = new.model_copy(update={"supersedes": old.id})
    return old2, new2


def dreamer_can_modify(fact: Fact) -> bool:
    """유저가 편집한 사실은 Dreamer가 덮을 수 없다 (스펙 §2.7)."""
    return not fact.user_edited


def apply_user_edit(fact: Fact, **changes: object) -> Tuple[Fact, Fact]:
    """유저 편집 = 수동 supersede. 새 버전은 user_edited로 보호된다."""
    new = fact.model_copy(
        update={
            **changes,
            "id": uuid.uuid4().hex,
            "supersedes": fact.id,
            "user_edited": True,
            "recorded_at": utc_now_iso(),
        }
    )
    old = fact.model_copy(update={"status": "superseded"})
    return old, new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_facts.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/facts.py tests/test_dreaming_facts.py
git commit -m "feat(dreaming): Fact 버전 체인 — supersede + 유저 편집 보호 (스펙 §4.1, §2.7)"
```

---

### Task 6: MemoryStore 파사드 (세션 스코프 저장 조합)

**Files:**
- Create: `dreaming/store.py`
- Test: `tests/test_dreaming_store.py`

**Interfaces:**
- Consumes: Task 1 `Storage`/`JsonDirStorage`, Task 2~4의 `Fact`/`Episode`/`Actor`/`StateCommit`, Task 4 `replay`
- Produces: `MemoryStore(storage: Storage, session_id: str)` — 메서드:
  `save_fact(f: Fact) -> None`, `get_fact(fact_id: str) -> Fact | None`,
  `list_facts(include_superseded: bool = False) -> list[Fact]`,
  `save_episode(e: Episode) -> None`, `list_episodes() -> list[Episode]`,
  `append_commit(c: StateCommit) -> None`, `list_commits() -> list[StateCommit]`,
  `current_state() -> dict[str, float | str]`,
  `save_actor(a: Actor) -> None`, `list_actors() -> list[Actor]`.
  네임스페이스는 `{session_id}/facts` 등 4개. 이후 플랜(동기 경로·Dreamer·대시보드)은 전부 이 파사드만 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_store.py
"""MemoryStore: 세션 스코프 파사드 — 이후 모든 플랜의 저장 진입점."""
from dreaming.facts import apply_user_edit
from dreaming.records import Actor, Episode, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore


def _store(tmp_path, session="sess1"):
    return MemoryStore(JsonDirStorage(tmp_path), session_id=session)


# ------------------------------------------------------------------ #
# Fact
# ------------------------------------------------------------------ #

def test_fact_save_get_roundtrip(tmp_path):
    ms = _store(tmp_path)
    f = Fact(claim="포션은 50골드다")
    ms.save_fact(f)
    assert ms.get_fact(f.id) == f
    assert ms.get_fact("missing") is None


def test_list_facts_hides_superseded_by_default(tmp_path):
    ms = _store(tmp_path)
    f = Fact(claim="포션은 50골드다")
    old2, new2 = apply_user_edit(f, claim="포션은 45골드다")
    ms.save_fact(old2)
    ms.save_fact(new2)
    visible = ms.list_facts()
    assert [x.claim for x in visible] == ["포션은 45골드다"]
    assert len(ms.list_facts(include_superseded=True)) == 2


def test_sessions_are_isolated(tmp_path):
    a = _store(tmp_path, "sess_a")
    b = _store(tmp_path, "sess_b")
    a.save_fact(Fact(claim="A만의 사실"))
    assert b.list_facts() == []


# ------------------------------------------------------------------ #
# Episode / Actor
# ------------------------------------------------------------------ #

def test_episode_roundtrip(tmp_path):
    ms = _store(tmp_path)
    e = Episode(range_start="h1", range_end="h2", title="흥정", summary="합의했다")
    ms.save_episode(e)
    assert ms.list_episodes() == [e]


def test_actor_roundtrip(tmp_path):
    ms = _store(tmp_path)
    a = Actor(names=["리사"])
    ms.save_actor(a)
    assert ms.list_actors() == [a]


# ------------------------------------------------------------------ #
# WorldState
# ------------------------------------------------------------------ #

def test_commits_replay_to_current_state(tmp_path):
    ms = _store(tmp_path)
    ms.append_commit(StateCommit(slot="소지금", op="set", value=500, turn=1))
    ms.append_commit(StateCommit(slot="소지금", op="add", value=-50, turn=2))
    assert ms.current_state() == {"소지금": 450}
    assert len(ms.list_commits()) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_store.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/store.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_store.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Run the whole dreaming test suite + existing tests for regressions**

Run: `python3 -m pytest tests/ -q`
Expected: 전부 PASS (기존 344개 + 신규 ~40개). 기존 saga 테스트가 깨지면 이 플랜 범위 밖의 수정이 새어 들어간 것 — 되돌릴 것.

- [ ] **Step 6: Commit**

```bash
git add dreaming/store.py tests/test_dreaming_store.py
git commit -m "feat(dreaming): MemoryStore 파사드 — 세션 스코프 KV 저장 진입점"
```

---

## Self-Review 결과

- **Spec coverage (이 플랜 범위 = 코어)**: §8 KV 저장 모델 → Task 1. §4.1 Fact 전 필드 → Task 2. §4.2 Episode(open_threads 포함)·§4.4 Actor(knows/tier) → Task 3. §4.3 typed commit + 리플레이 + pending_contradiction 격리 → Task 4. §2.7·§7.2 유저 편집 보호 → Task 5. §2.8 저장 중립(파사드 뒤로 격리) → Task 6. **의도적 제외 (후속 플랜)**: slot-level 검증 상세·숫자 정규식 재검증(B-3, Dreamer 플랜), embedding 생성(검색 플랜), 청크 조립(청크 플랜), pair ledger 연동(동기 경로 플랜 — saga 자산 재사용이므로 코어에 안 넣음).
- **Placeholder scan**: 통과 — 모든 스텝에 실제 코드/명령/기대 출력 있음.
- **Type consistency**: `Storage.scan`이 `(key, dict)` 튜플 반환 — Task 6 파사드가 `for _, v in scan()` 으로 소비, 일치. `StateCommit.value: object`는 replay에서 런타임 타입 검증 — Task 4 테스트가 커버. `apply_user_edit` 반환 순서 `(old, new)` — Task 5·6 테스트 모두 동일 순서 사용.
