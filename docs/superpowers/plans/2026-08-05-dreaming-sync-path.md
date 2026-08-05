# Dreaming Sync Path (Plan 2/6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 동기 경로(스펙 §3.1)의 로직 계층 — KV 기반 pair ledger(판정 5종), 세션 해석, 지식 주입 조립, 3-BP 캐시 마킹, SyncPath 오케스트레이터 — 를 턴당 LLM 0콜로 구축한다.

**Architecture:** 정체성 판정의 순수 함수(`hash_text`/`extract_pairs`/`classify`)는 `saga/services/pair_ledger.py`에서 **import로 승계**하고(검증된 자산, 수정 금지), SQLite 서비스 계층만 Plan 1의 `Storage` KV 위에 재작성한다. saga의 3종 판정(new/append/reroll)을 스펙 §3.1의 5종(new_session/next_turn/continuation/reroll/diverged)으로 매핑한다. 조립은 결정론: 지식은 마지막 user prepend(캐시 밖), BP1(마지막 system)·BP3(마지막 assistant)만 이 플랜에서 마킹 — **BP2는 청크가 생기는 Plan 4에서** 추가한다. HTTP 서버 연결은 이 플랜 범위 밖(Plan 3+).

**Tech Stack:** Python 3, Pydantic v2, pytest. Plan 1의 `dreaming.storage.Storage`/`JsonDirStorage`, `dreaming.store.MemoryStore` 사용. 신규 의존성 금지.

**Spec:** [docs/dreaming/SPEC.md](../../dreaming/SPEC.md) §3.1(동기 경로), §5(레이아웃), §2.6(fail-open). 정체성 근거는 §0.1 (chat.id 미전송·리롤 무플래그·매크로 재작성).

## Global Constraints

- 신규 코드는 `dreaming/`, 테스트는 `tests/test_dreaming_*.py` (flat).
- `saga/`와 `external/`은 수정 금지. 단, **`saga.services.pair_ledger`의 순수 함수 3개(`hash_text`, `extract_pairs`, `classify`)는 import 허용** — 이 모듈은 Phase 2에 포팅되지 않으므로(플러그인은 chat.id ground truth, 스펙 §8) saga 의존이 코어 중립성을 해치지 않는다.
- 저장은 Plan 1 `Storage` KV만. SQL·외부 DB 금지 (스펙 §8).
- 신규 pip 의존성 금지. 인터프리터 `python3`, 테스트 `python3 -m pytest <파일> -q`.
- 테스트 스타일: 모듈 docstring + `# ---- #` 섹션 주석 (기존 컨벤션).
- 커밋 컨벤션 유지, 브랜치 `dreaming/spec`.
- fail-open (스펙 §2.6): 동기 경로 어떤 함수도 예외로 채팅을 막으면 안 된다 — 판정 실패 시 안전한 기본값을 돌려준다.
- 메시지 wire 포맷은 OpenAI-호환 dict (`{"role": ..., "content": ...}`) — saga와 동일. 캐시 마킹은 dict에 `"cache_control": {"type": "ephemeral"}` 키 추가 (saga `ChatMessage.cache_control`과 동일 표현).

---

### Task 1: KV PairLedger — 체인 저장 + 판정 5종 매핑

**Files:**
- Create: `dreaming/identity.py`
- Test: `tests/test_dreaming_identity.py`

**Interfaces:**
- Consumes: `saga.services.pair_ledger.hash_text/extract_pairs/classify` (순수 함수, 시그니처: `classify(chain: list[dict], request_pairs: list[dict], last_user_hash: str | None) -> dict` — 반환 dict 키: `kind('new'|'append'|'reroll')`, `position`, `reroll_turn_number`, `superseded_indices`, `quarantined_indices`, `confirm`, `aligned`, `offset`). Plan 1 `Storage`.
- Produces: `VerdictKind = Literal["new_session","next_turn","continuation","reroll","diverged"]`, `Verdict(BaseModel)` — 필드 `kind: VerdictKind`, `position: int`, `reroll_turn_number: int|None`, `aligned: bool`. `PairLedger(storage: Storage, session_id: str)` — 메서드 `chain() -> list[dict]`, `analyze_and_apply(pairs: list[dict], last_user_hash: str|None) -> Verdict`, `record_turn(verdict: Verdict, last_user_hash: str|None, user_text: str, assistant_text: str, turn_number: int) -> None`. 원장 네임스페이스 `{session}/ledger` (key = `f"{index:06d}"`), 원문 네임스페이스 `{session}/raw` (key = `f"{turn_number:06d}"`, 값 `{user_text, assistant_text, user_hash, assistant_hash, turn_number}` — Dreamer(Plan 3)의 추출 입력). Task 3·5가 `Verdict`를 소비한다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_identity.py
"""KV pair ledger: 판정 5종 (스펙 §3.1). 순수 로직은 saga에서 승계."""
from dreaming.identity import PairLedger, Verdict
from dreaming.storage import JsonDirStorage

from saga.services.pair_ledger import extract_pairs, hash_text


def _ledger(tmp_path, session="sess1"):
    return PairLedger(JsonDirStorage(tmp_path), session_id=session)


def _msgs(*texts):
    """user/assistant 교대 메시지 생성. 홀수 개면 마지막이 trailing user."""
    roles = ["user", "assistant"]
    return [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]


def _advance(ledger, user_text, assistant_text, history):
    """한 턴 진행: 요청 분석 → 응답 기록. history는 (u, a) 텍스트 리스트."""
    flat = [t for pair in history for t in pair] + [user_text]
    pairs, last_user = extract_pairs(_msgs(*flat))
    verdict = ledger.analyze_and_apply(pairs, last_user)
    ledger.record_turn(verdict, last_user, user_text, assistant_text,
                       turn_number=verdict.position)
    return verdict


# ------------------------------------------------------------------ #
# 판정 5종
# ------------------------------------------------------------------ #

def test_first_turn_is_new_session(tmp_path):
    ledger = _ledger(tmp_path)
    pairs, last_user = extract_pairs(_msgs("안녕"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "new_session"
    assert v.position == 0


def test_second_turn_is_next_turn(tmp_path):
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "next_turn"
    assert v.position == 1


def test_resend_without_trailing_user_is_continuation(tmp_path):
    # autoContinue류: 새 user 입력 없이 히스토리만 재전송
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와."))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert last_user is None
    assert v.kind == "continuation"


def test_tail_resend_is_reroll(tmp_path):
    # 마지막 assistant pop 후 같은 user 재전송 (RisuAI 리롤 — §0.1)
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    _advance(ledger, "포션 얼마야?", "50골드다.", history=[("안녕", "어서 와.")])
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "reroll"
    assert v.position == 1
    assert v.reroll_turn_number == 1


def test_mid_history_edit_is_diverged(tmp_path):
    # 중간 턴 편집: 그 지점 재전송 → 이후 턴들은 quarantine (스펙 §3.1)
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    _advance(ledger, "포션 얼마야?", "50골드다.",
             history=[("안녕", "어서 와.")])
    _advance(ledger, "3개 줘", "150골드다.",
             history=[("안녕", "어서 와."), ("포션 얼마야?", "50골드다.")])
    # 1번 턴의 user를 편집해 그 지점에서 재전송
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    v = ledger.analyze_and_apply(pairs, last_user)
    assert v.kind == "diverged"
    assert v.position == 1
    # 이후 턴(2번)은 quarantined
    statuses = {row["index"]: row["status"] for row in ledger.chain(active_only=False)}
    assert statuses[2] == "quarantined"


# ------------------------------------------------------------------ #
# 원장 상태 전이 + 원문 보존
# ------------------------------------------------------------------ #

def test_recorded_turn_is_provisional_then_confirmed(tmp_path):
    ledger = _ledger(tmp_path)
    _advance(ledger, "안녕", "어서 와.", history=[])
    assert ledger.chain()[0]["status"] == "provisional"
    # 다음 요청에서 같은 pair가 다시 보이면 confirmed
    pairs, last_user = extract_pairs(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    ledger.analyze_and_apply(pairs, last_user)
    assert ledger.chain()[0]["status"] == "confirmed"


def test_raw_pair_stored_for_dreamer(tmp_path):
    # Dreamer(Plan 3)의 추출 입력 — 원문이 KV에 남아야 한다
    storage = JsonDirStorage(tmp_path)
    ledger = PairLedger(storage, session_id="sess1")
    _advance(ledger, "안녕", "어서 와.", history=[])
    raw = storage.get("sess1/raw", "000000")
    assert raw["user_text"] == "안녕"
    assert raw["assistant_text"] == "어서 와."
    assert raw["user_hash"] == hash_text("안녕")


def test_fail_open_on_garbage_input(tmp_path):
    # 어떤 입력에서도 예외로 채팅을 막지 않는다 (스펙 §2.6)
    ledger = _ledger(tmp_path)
    v = ledger.analyze_and_apply([], None)
    assert isinstance(v, Verdict)
    assert v.kind == "new_session"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_identity.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.identity'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/identity.py
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
    def __init__(self, storage: Storage, session_id: str) -> None:
        self._storage = storage
        self._session = session_id

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_identity.py -q`
Expected: PASS (8 passed)

주의: `test_mid_history_edit_is_diverged`에서 saga `classify`의 chain 인덱스는 active-only 리스트 기준이다. quarantine 후 재조회 시 `chain(active_only=False)`로 실제 index 필드를 확인하는 이유. 실패하면 `_transition`의 index 필드 사용을 먼저 의심할 것 — 테스트를 약화하지 말 것.

- [ ] **Step 5: Commit**

```bash
git add dreaming/identity.py tests/test_dreaming_identity.py
git commit -m "feat(dreaming): KV PairLedger — 판정 5종 매핑 + 원문 보존 (스펙 §3.1)"
```

---

### Task 2: SessionResolver — 해시 역색인으로 세션 찾기

**Files:**
- Create: `dreaming/resolver.py`
- Modify: `dreaming/identity.py` (record_turn 끝에 색인 갱신 1줄 — 아래 명시)
- Test: `tests/test_dreaming_resolver.py`

**Interfaces:**
- Consumes: Task 1 `PairLedger`, Plan 1 `Storage`
- Produces: `SessionResolver(storage: Storage)` — 메서드 `index_pair(session_id: str, user_hash: str, assistant_hash: str|None) -> None`, `resolve(pairs: list[dict]) -> str | None`. 색인 네임스페이스는 전역 `pair-index`, key = 해시값, 값 = `{"sessions": [session_id, ...]}`. saga 규칙 승계: assistant 해시 매치 ≥1 또는 총 매치 ≥2 (단일 user 해시는 "계속" 같은 충돌 위험). 최근 6쌍만 사용. Task 5 `SyncPath`가 세션 미지정 시 이걸 쓴다.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_resolver.py
"""세션 해석: 해시 역색인 (saga resolve_session의 KV 이식)."""
from dreaming.resolver import SessionResolver
from dreaming.storage import JsonDirStorage

from saga.services.pair_ledger import hash_text


def _pairs(*uas):
    return [{"index": i, "user_hash": hash_text(u), "assistant_hash": hash_text(a)}
            for i, (u, a) in enumerate(uas)]


# ------------------------------------------------------------------ #
# resolve
# ------------------------------------------------------------------ #

def test_resolves_by_assistant_hash(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("안녕"), hash_text("어서 와."))
    assert r.resolve(_pairs(("안녕", "어서 와."))) == "sess1"


def test_single_user_hash_match_is_not_enough(tmp_path):
    # "계속" 같은 흔한 입력의 user 해시 하나로는 세션 확정 금지 (saga 규칙)
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("계속"), None)
    assert r.resolve([{"index": 0, "user_hash": hash_text("계속"),
                       "assistant_hash": hash_text("다른 응답")}]) is None


def test_two_user_matches_resolve(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("안녕"), None)
    r.index_pair("sess1", hash_text("포션 얼마야?"), None)
    got = r.resolve([
        {"index": 0, "user_hash": hash_text("안녕"), "assistant_hash": hash_text("x")},
        {"index": 1, "user_hash": hash_text("포션 얼마야?"), "assistant_hash": hash_text("y")},
    ])
    assert got == "sess1"


def test_no_match_returns_none(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    assert r.resolve(_pairs(("안녕", "어서 와."))) is None
    assert r.resolve([]) is None


def test_best_scoring_session_wins(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess_a", hash_text("안녕"), hash_text("어서 와."))
    r.index_pair("sess_b", hash_text("안녕"), hash_text("어서 와."))
    r.index_pair("sess_b", hash_text("포션 얼마야?"), hash_text("50골드다."))
    got = r.resolve(_pairs(("안녕", "어서 와."), ("포션 얼마야?", "50골드다.")))
    assert got == "sess_b"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_resolver.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.resolver'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/resolver.py
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
```

`dreaming/identity.py`의 `record_turn` 끝(원문 보존 put 다음)에 색인 갱신 추가:

```python
        # 세션 해석 역색인 갱신 (resolver가 없으면 no-op)
        if self._resolver is not None:
            self._resolver.index_pair(self._session, last_user_hash, asst_hash)
```

그리고 `__init__` 시그니처를 확장:

```python
    def __init__(self, storage: Storage, session_id: str,
                 resolver: "SessionResolver | None" = None) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = resolver
```

파일 상단 import에 추가 (순환 없음 — resolver는 identity를 모른다):

```python
from dreaming.resolver import SessionResolver
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_resolver.py tests/test_dreaming_identity.py -q`
Expected: PASS (13 passed — resolver 5 + 기존 identity 8 회귀 포함)

- [ ] **Step 5: Commit**

```bash
git add dreaming/resolver.py dreaming/identity.py tests/test_dreaming_resolver.py
git commit -m "feat(dreaming): SessionResolver — 해시 역색인 세션 해석 (saga 규칙 승계)"
```

---

### Task 3: 지식 주입 조립 (마지막 user prepend)

**Files:**
- Create: `dreaming/assembly.py`
- Test: `tests/test_dreaming_assembly.py`

**Interfaces:**
- Consumes: 없음 (순수 함수 — 메시지 dict 리스트만)
- Produces: `HOT_ZONE_CHAR_BUDGET = 6000` (≈2K tokens, 스펙 §3.1 — 한/영 혼합 보수 추정, 후속 플랜에서 토크나이저 기반으로 교체 예정), `clip_knowledge(text: str, budget: int = HOT_ZONE_CHAR_BUDGET) -> str`, `inject_knowledge(messages: list[dict], knowledge: str) -> list[dict]` — 마지막 user 메시지 content 앞에 `<dreaming_context>` 블록 prepend, 원본 리스트 불변(사본 반환). knowledge가 빈 문자열이면 무변경. Task 5가 소비.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_assembly.py
"""지식 주입: 마지막 user prepend — 캐시 밖 (스펙 §3.1, §5)."""
from dreaming.assembly import HOT_ZONE_CHAR_BUDGET, clip_knowledge, inject_knowledge


def _msgs():
    return [
        {"role": "system", "content": "너는 상인 리사다."},
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "어서 와."},
        {"role": "user", "content": "포션 얼마야?"},
    ]


# ------------------------------------------------------------------ #
# inject_knowledge
# ------------------------------------------------------------------ #

def test_injects_into_last_user_only(chosen=None):
    msgs = _msgs()
    out = inject_knowledge(msgs, "소지금: 450골드")
    assert out[3]["content"].startswith("<dreaming_context>\n소지금: 450골드\n</dreaming_context>\n\n")
    assert out[3]["content"].endswith("포션 얼마야?")
    # 다른 메시지 무변경 — 프리픽스(캐시 계층) 불가침 (스펙 §5)
    assert out[0] == msgs[0] and out[1] == msgs[1] and out[2] == msgs[2]


def test_original_list_is_not_mutated():
    msgs = _msgs()
    inject_knowledge(msgs, "소지금: 450골드")
    assert msgs[3]["content"] == "포션 얼마야?"


def test_empty_knowledge_is_noop():
    msgs = _msgs()
    assert inject_knowledge(msgs, "") == msgs


def test_no_user_message_is_noop():
    msgs = [{"role": "system", "content": "x"}]
    assert inject_knowledge(msgs, "지식") == msgs


# ------------------------------------------------------------------ #
# clip_knowledge
# ------------------------------------------------------------------ #

def test_clip_respects_budget():
    text = "가" * (HOT_ZONE_CHAR_BUDGET + 100)
    assert len(clip_knowledge(text)) == HOT_ZONE_CHAR_BUDGET


def test_clip_short_text_unchanged():
    assert clip_knowledge("짧다") == "짧다"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_assembly.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.assembly'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/assembly.py
"""dreaming/assembly.py — 지식 주입 조립 (스펙 §3.1, §5).

지식 계층은 아무리 바뀌어도 캐시를 깨지 않아야 한다 → 주입 위치는
오직 마지막 user 메시지 prepend (프리픽스 밖). system 주입 금지 —
hypaV3의 선두 system 삽입(hypav3.ts:1593)이 반면교사다.
"""

from __future__ import annotations

import copy
from typing import Dict, List

# ≈2K tokens (스펙 §3.1 hot zone). 한/영 혼합 보수 추정 — 후속 플랜에서
# 토크나이저 기반으로 교체한다.
HOT_ZONE_CHAR_BUDGET = 6000


def clip_knowledge(text: str, budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    return text[:budget]


def inject_knowledge(messages: List[Dict], knowledge: str) -> List[Dict]:
    if not knowledge:
        return messages
    last_user = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user = i
            break
    if last_user is None:
        return messages
    out = [copy.deepcopy(m) for m in messages]
    out[last_user]["content"] = (
        f"<dreaming_context>\n{knowledge}\n</dreaming_context>\n\n"
        f"{out[last_user]['content']}"
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_assembly.py -q`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/assembly.py tests/test_dreaming_assembly.py
git commit -m "feat(dreaming): 지식 주입 조립 — 마지막 user prepend, 캐시 밖 (스펙 §5)"
```

---

### Task 4: 3-BP 캐시 마킹 (이 플랜은 BP1·BP3)

**Files:**
- Create: `dreaming/marking.py`
- Test: `tests/test_dreaming_marking.py`

**Interfaces:**
- Consumes: 없음 (순수 함수)
- Produces: `mark_cache(messages: list[dict], ttl: str = "5m") -> list[dict]` — 마지막 system 메시지(BP1)와 마지막 assistant 메시지(BP3)에 `"cache_control": {"type": "ephemeral", "ttl": ttl}` 추가, 사본 반환. 이미 있는 cache_control은 제거 후 재마킹(우리가 유일한 마킹 주체 — §0.1 cachePoint 제거 근거). **BP2(첫 청크 assistant)는 Plan 4(청크)에서 이 함수에 추가된다.** Task 5가 소비.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_marking.py
"""3-BP 캐시 마킹 (스펙 §3.1). RisuAI가 cachePoint를 지우므로 마킹은 우리 몫(§0.1)."""
from dreaming.marking import mark_cache


def _msgs():
    return [
        {"role": "system", "content": "너는 상인 리사다."},
        {"role": "system", "content": "세계관: 판타지 시장."},
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "어서 와."},
        {"role": "user", "content": "포션 얼마야?"},
    ]


# ------------------------------------------------------------------ #
# BP1 / BP3
# ------------------------------------------------------------------ #

def test_marks_last_system_and_last_assistant():
    out = mark_cache(_msgs())
    assert out[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}   # BP1
    assert out[3]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}   # BP3
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[2]
    assert "cache_control" not in out[4]


def test_ttl_configurable():
    out = mark_cache(_msgs(), ttl="1h")
    assert out[1]["cache_control"]["ttl"] == "1h"


def test_strips_preexisting_marks():
    msgs = _msgs()
    msgs[2]["cache_control"] = {"type": "ephemeral"}   # 낯선 마킹 — 제거돼야 함
    out = mark_cache(msgs)
    assert "cache_control" not in out[2]


def test_original_not_mutated():
    msgs = _msgs()
    mark_cache(msgs)
    assert "cache_control" not in msgs[1]


def test_no_system_no_assistant_is_safe():
    out = mark_cache([{"role": "user", "content": "안녕"}])
    assert "cache_control" not in out[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_marking.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.marking'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/marking.py
"""dreaming/marking.py — 3-BP 캐시 마킹 (스펙 §3.1, §5).

BP1 = 마지막 system (Anthropic 변환은 선두 연속 system만 병합 — anthropic.ts:209),
BP3 = 마지막 assistant (원문 꼬리 끝).
BP2 = 첫 청크 assistant — 청크가 생기는 Plan 4에서 추가된다.
RisuAI는 cachePoint를 전송 직전 제거하므로(requests.ts:141) 마킹 주체는
프록시/프로바이더인 우리다. 기존 마킹은 전부 제거 후 재마킹한다.
"""

from __future__ import annotations

import copy
from typing import Dict, List


def mark_cache(messages: List[Dict], ttl: str = "5m") -> List[Dict]:
    out = [copy.deepcopy(m) for m in messages]
    for m in out:
        m.pop("cache_control", None)

    last_system = None
    last_assistant = None
    for i, m in enumerate(out):
        if m.get("role") == "system":
            last_system = i
        elif m.get("role") == "assistant":
            last_assistant = i

    mark = {"type": "ephemeral", "ttl": ttl}
    if last_system is not None:
        out[last_system]["cache_control"] = dict(mark)   # BP1
    if last_assistant is not None:
        out[last_assistant]["cache_control"] = dict(mark)  # BP3
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_marking.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/marking.py tests/test_dreaming_marking.py
git commit -m "feat(dreaming): 3-BP 캐시 마킹 — BP1 system + BP3 assistant (스펙 §3.1)"
```

---

### Task 5: 지식 렌더링 + SyncPath 오케스트레이터

**Files:**
- Create: `dreaming/sync.py`
- Test: `tests/test_dreaming_sync.py`

**Interfaces:**
- Consumes: Task 1 `PairLedger`/`Verdict`, Task 2 `SessionResolver`, Task 3 `inject_knowledge`/`clip_knowledge`, Task 4 `mark_cache`, Plan 1 `MemoryStore`(`list_facts`/`current_state`/`list_actors`), `saga.services.pair_ledger.extract_pairs`
- Produces: `render_knowledge(store: MemoryStore) -> str` — 결정론 템플릿: WorldState 현재값(slot 정렬) + pinned/confirmed Fact(pinned 우선, recorded_at 내림차순, 최대 20개) + main tier Actor. 빈 저장소면 `""`. `SyncPath(storage: Storage, session_id: str)` — 메서드 `process(messages: list[dict]) -> tuple[list[dict], Verdict]` (판정→주입→마킹, LLM 0콜), `record_response(verdict: Verdict, messages: list[dict], assistant_text: str) -> None`. 임베딩 검색은 후속 플랜 — v1 렌더링은 pinned+state+main actor만.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dreaming_sync.py
"""SyncPath: 동기 경로 오케스트레이터 — 턴당 LLM 0콜 (스펙 §3.1)."""
from dreaming.records import Actor, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, render_knowledge


def _msgs(*texts):
    roles = ["user", "assistant"]
    out = [{"role": "system", "content": "너는 상인 리사다."}]
    out += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
    return out


# ------------------------------------------------------------------ #
# render_knowledge
# ------------------------------------------------------------------ #

def test_render_includes_state_pinned_facts_main_actors(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    ms.save_fact(Fact(claim="리사는 밀수품을 취급한다", pinned=True, status="confirmed"))
    ms.save_actor(Actor(names=["리사"], profile="시장 상인", tier="main"))
    text = render_knowledge(ms)
    assert "소지금: 450" in text
    assert "리사는 밀수품을 취급한다" in text
    assert "리사" in text


def test_render_excludes_provisional_unpinned_and_extras(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.save_fact(Fact(claim="잠정 사실", status="provisional"))
    ms.save_actor(Actor(names=["행인1"], tier="extra"))
    text = render_knowledge(ms)
    assert "잠정 사실" not in text
    assert "행인1" not in text


def test_render_empty_store_is_empty(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    assert render_knowledge(ms) == ""


def test_render_is_deterministic(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    ms.save_fact(Fact(claim="사실", status="confirmed", pinned=True))
    assert render_knowledge(ms) == render_knowledge(ms)


# ------------------------------------------------------------------ #
# SyncPath
# ------------------------------------------------------------------ #

def test_process_injects_and_marks(tmp_path):
    storage = JsonDirStorage(tmp_path)
    ms = MemoryStore(storage, "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    sp = SyncPath(storage, "sess1")
    out, verdict = sp.process(_msgs("안녕"))
    assert verdict.kind == "new_session"
    assert "<dreaming_context>" in out[-1]["content"]     # 지식 주입 (캐시 밖)
    assert out[0].get("cache_control") is not None        # BP1
    assert "소지금: 450" in out[-1]["content"]


def test_full_turn_cycle_then_reroll(tmp_path):
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "sess1")
    msgs1 = _msgs("안녕")
    out1, v1 = sp.process(msgs1)
    sp.record_response(v1, msgs1, "어서 와.")

    msgs2 = _msgs("안녕", "어서 와.", "포션 얼마야?")
    out2, v2 = sp.process(msgs2)
    assert v2.kind == "next_turn"
    sp.record_response(v2, msgs2, "50골드다.")

    # 리롤: 같은 요청 재전송
    out3, v3 = sp.process(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    assert v3.kind == "reroll"
    assert v3.reroll_turn_number == 1


def test_process_never_raises_on_weird_input(tmp_path):
    # fail-open (스펙 §2.6): 판정 불가여도 메시지는 통과시킨다
    sp = SyncPath(JsonDirStorage(tmp_path), "sess1")
    out, verdict = sp.process([{"role": "system", "content": "x"}])
    assert out[-1]["content"] == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_dreaming_sync.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dreaming.sync'`

- [ ] **Step 3: Write minimal implementation**

```python
# dreaming/sync.py
"""dreaming/sync.py — 동기 경로 오케스트레이터 (스펙 §3.1).

기록·검색·조립·주입만. 턴당 LLM 0콜 — 이해는 전부 Dreamer(Plan 3) 몫.
임베딩 검색은 후속 플랜: v1 지식 렌더링은 state + pinned/confirmed fact +
main actor 결정론 템플릿이다.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from dreaming.assembly import clip_knowledge, inject_knowledge
from dreaming.identity import PairLedger, Verdict
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


class SyncPath:
    def __init__(self, storage: Storage, session_id: str) -> None:
        self._storage = storage
        self._session = session_id
        self._resolver = SessionResolver(storage)
        self._ledger = PairLedger(storage, session_id, resolver=self._resolver)
        self._store = MemoryStore(storage, session_id)

    def process(self, messages: List[Dict]) -> Tuple[List[Dict], Verdict]:
        pairs, last_user_hash = extract_pairs(messages)
        verdict = self._ledger.analyze_and_apply(pairs, last_user_hash)
        knowledge = clip_knowledge(render_knowledge(self._store))
        out = inject_knowledge(messages, knowledge)
        out = mark_cache(out)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_dreaming_sync.py -q`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add dreaming/sync.py tests/test_dreaming_sync.py
git commit -m "feat(dreaming): SyncPath 오케스트레이터 + 지식 렌더링 — 턴당 LLM 0콜 (스펙 §3.1)"
```

---

### Task 6: 전체 회귀 + push

**Files:**
- 없음 (검증만)

**Interfaces:**
- Consumes: Task 1~5 전부
- Produces: 회귀 없는 그린 스위트

- [ ] **Step 1: Run the full test suite**

Run: `python3 -m pytest tests/ -q`
Expected: 전부 PASS (Plan 1까지 383개 + 신규 ~31개 ≈ 414개). saga 기존 테스트가 깨지면 이 플랜의 수정이 새어 들어간 것 — `git diff origin/main -- saga/`가 비어 있는지 확인하고 되돌릴 것.

- [ ] **Step 2: Push**

```bash
git push origin dreaming/spec
```

---

## Self-Review 결과

- **Spec coverage (이 플랜 범위 = 동기 경로 로직)**: §3.1 판정 5종 → Task 1 (`_map_kind` + 테스트 5개 각 판정별). §3.1 세션 해석(와이어 무식별자) → Task 2. §3.1 지식 주입 + §5 레이아웃(마지막 user prepend, 프리픽스 불가침) → Task 3. §3.1 3-BP 중 BP1·BP3 + TTL → Task 4 (BP2는 Plan 4 몫 — 청크 없이는 마킹 대상이 없음). §3.1 오케스트레이션 + §2.2 턴당 0콜 + §2.6 fail-open → Task 5. **의도적 제외**: HTTP 프록시 서버 배선(Plan 3에서 Dreamer 트리거와 함께), 임베딩 검색(후속), 청크 조립/BP2(Plan 4), 로어북 델타 이동(§5 — 프록시 배선 플랜에서).
- **Placeholder scan**: 통과 — 전 스텝 실코드/실명령/기대값.
- **Type consistency**: `Verdict.position` = saga `classify`의 `position` (int) — Task 5 `record_response`가 `turn_number=verdict.position`으로 사용, 일치. `PairLedger.__init__(storage, session_id, resolver=None)` — Task 2에서 확장, Task 5가 키워드로 호출, 일치. `chain(active_only)` — Task 1 테스트가 `active_only=False` 사용, 시그니처에 있음. `extract_pairs` 반환 `(pairs, last_user_hash)` — saga 원본 시그니처, Task 1·5 동일 사용.
- **알려진 설계 트레이드오프 (구현자 주의)**: KV 원장은 인덱스당 문서 1개라 superseded 이력을 보존하지 않는다(saga는 행 누적) — 리롤 후 swipe-back 복원은 v1 비지원, 의도적 단순화.
