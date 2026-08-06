# Dreaming Plan 4.5 — 트림 정렬: 원장 밀집 뷰 + 격리 버퍼 + 압축 윈도우 앵커

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RisuAI 클라이언트측 트림(슬라이딩 윈도우) 정상상태에서 실캡처로 실증된 두 결함 — PairLedger 턴 번호 붕괴, 위치 기반 압축 오정렬 — 을 고치고, 스펙 §3.1의 격리 버퍼를 이행한다.

**Architecture:** ① classify(saga, 동결)는 "chain 리스트 위치 == pair index"인 밀집 배열을 전제하는데 identity.py가 active row 리스트를 넘겨 트림 정상상태에서 정렬이 무너진다 — identity.py가 **밀집 뷰**(갭은 매칭 불가 자리표시자)를 만들어 넘기고, 트림된 윈도우로 세션이 시작되면 **베이스라인 패드(1024)**를 더해 뒤늦게 윈도우가 앞으로 자라도(maxContext 상향) 음수 오프셋이 안 나게 한다. ② 정렬 실패 + 낯선 pair는 본원장 대신 `{session}/quarantine`에 기록하고 무가공 passthrough (스펙 §3.1 "판정 불확실 → fail-open, 기록은 격리 버퍼에"). ③ 압축 치환은 `verdict.offset`(윈도우 시작 턴)을 받아 "윈도우 안에 남은 압축 대상 pair만" 드롭한다 — 윈도우가 이미 압축 구간을 지나 트림됐으면 드롭 0 + 청크 prepend = **트림으로 사라진 컨텍스트를 청크가 복원**한다 (이 기능의 본래 가치).

**Tech Stack:** Python 3 표준 라이브러리 + 기존 의존성만. 새 의존성 금지.

## Global Constraints

- **기반: PR #5 머지 후의 최신 `origin/main`** (SyncPath는 shift_keyed 직조본, marking은 BP1 선두 한정 + bp2_index). 실행 첫 단계에서 워크트리를 main과 동기화하고 `dreaming/lore_shift.py` 존재를 확인할 것 — 없으면 중단하고 질문.
- 실행은 전부 `python3`. 테스트: `python3 -m pytest`. 파이프는 exit code를 삼킨다(zsh: `${pipestatus[1]}`) — 실행과 exit 확인 분리.
- **saga diff 0 유지.** `classify`/`hash_text`/`extract_pairs`는 수정 금지 — 밀집 뷰·패드·격리는 전부 `dreaming/identity.py`·`dreaming/sync.py`에서 해결한다.
- 저장은 KV 문서 샤드 단일 (스펙 §8). fail-open: 어떤 경우에도 채팅을 막지 않는다 (스펙 §2.6).
- pytest-asyncio 금지 (asyncio.run 패턴). API 키 값 출력·커밋 금지.
- **corpus 데이터(`dreaming_data/corpus*`) 커밋 절대 금지** — 카드 저작물 포함 (ba42ff 로컬 보관). 재생 **도구**만 커밋한다.
- worldstate.replay 비숫자 add 크래시 방어는 별도 세션(eloquent-dirac) 담당 — 구현 금지.
- 비스코프: 격리 버퍼의 사후 복권(수동 병합 UI), 세션 자동 분리, Tier3, 임베딩 검색.

## 실증 근거 (이 플랜이 고치는 것)

corpus3(트림 정상상태)·corpus4(리롤/편집) — ba42ff의 진짜 RisuAI 캡처. 재생 결과:

| # | 결함 | 재생 관측 | 근인 |
|---|---|---|---|
| ① | 원장 턴 붕괴 | 턴 18 기록 직후 다음 턴이 **턴 1**로 기록, "*says nothing*"이 턴 7·8·9 3중 기록, 진짜 리롤이 diverged 오판 | `_align_offset`(pair_ledger.py:77)은 chain **리스트 위치 == index** 전제. identity.py `chain()`은 active 리스트를 넘김 — 트림 세션은 index 18 하나로 시작해 위치 0 == index 18 → `offset = 0 - 17 < 0` → 정렬 실패 → fallback이 매번 어긋난 번호 배정 |
| ② | 압축 오정렬 | covers_until=4 플랜이 트림된 윈도우의 **선두 4 pair**(전혀 다른 턴)를 드롭 | `apply_compression`이 순수 위치 기반 — 윈도우 시작 턴을 모름 |
| ③ | §3.1 미이행 | aligned=False인데 본원장에 그대로 기록 | 격리 버퍼 자체가 미구현 |

재생 스크립트: 세션 scratchpad `replay_corpus.py` (Task 4에서 커밋 가능한 도구로 정식화).

## File Structure

- Modify: `dreaming/identity.py` — `_dense_chain()` 밀집 뷰, 베이스라인 패드, `Verdict.offset`/`Verdict.quarantine`
- Modify: `dreaming/sync.py` — 격리 passthrough·격리 기록, 압축 게이트(`verdict.offset` 전달)
- Modify: `dreaming/chunks.py` — `apply_compression(..., window_start_turn=0)`
- Create: `benchmarks/capture/replay_ledger.py` — corpus 재생·정합성 검사 도구 (데이터는 미커밋)
- Test: `tests/test_dreaming_identity.py`, `tests/test_dreaming_sync.py`, `tests/test_dreaming_chunks.py`, `tests/test_dreaming_proxy.py`

---

### Task 0: 기반 동기화

- [ ] **Step 1: main 동기화 + 전제 확인**

```bash
git fetch origin && git merge --ff-only origin/main || git merge origin/main
ls dreaming/lore_shift.py
python3 -m pytest tests/ -q
```

Expected: `lore_shift.py` 존재, 전체 스위트 PASS (PR #5 머지본 ~500개). 실패 시 중단하고 질문.

---

### Task 1: 밀집 체인 뷰 + 트림 베이스라인 패드

**Files:**
- Modify: `dreaming/identity.py`
- Test: `tests/test_dreaming_identity.py`

**Interfaces:**
- Consumes: `saga.services.pair_ledger.classify` (동결 — chain 리스트 위치 == index 전제).
- Produces: `PairLedger._dense_chain() -> List[Dict]` — 저장 index를 리스트 위치로 복원, 갭은 `{"user_hash": None, "assistant_hash": None, "status": "gap", "turn_number": None, "index": i}`. `_BASELINE_PAD = 1024` (모듈 상수) — **빈 원장 + 요청에 pair가 있으면**(= 프록시가 이미 트림된 대화 중간에 합류) position에 패드를 더해 세션 턴 번호를 1024부터 시작. 신규 세션(첫 메시지, pair 없음)은 패드 없음 → 기존 플로우 턴 번호 불변. `Verdict`에 `offset: Optional[int] = None` 추가 (classify의 offset — **윈도우 첫 pair의 세션 턴 번호**, Task 3의 압축 앵커).

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_identity.py`에 추가 (기존 import에 `_BASELINE_PAD` 추가; `JsonDirStorage`는 기존 테스트가 이미 씀):

```python
def _window(start, count, current):
    """트림된 윈도우 시뮬레이션: u{start}..u{start+count-1} pair + 현재 user."""
    pairs = [{"index": i, "user_hash": f"u{start + i}",
              "assistant_hash": f"a{start + i}"} for i in range(count)]
    return pairs, f"u{current}"


def test_trimmed_session_baseline_is_padded(tmp_path):
    # corpus3 실증: 트림된 대화 중간 합류 — 이후 윈도우가 앞으로 자라도
    # (maxContext 상향) 음수 오프셋이 안 나게 베이스라인을 띄운다
    from dreaming.identity import _BASELINE_PAD, PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    pairs, cur = _window(7, 18, 25)
    v = ledger.analyze_and_apply(pairs, cur)
    assert v.position == _BASELINE_PAD + 18


def test_trim_steady_state_alignment(tmp_path):
    # corpus3 실증 붕괴: 턴 18 기록 후 다음 턴이 턴 1로 기록되던 버그
    from dreaming.identity import _BASELINE_PAD, PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    pairs, cur = _window(7, 18, 25)
    v1 = ledger.analyze_and_apply(pairs, cur)
    ledger.record_turn(v1, cur, "턴25 유저", "턴25 응답",
                       turn_number=v1.position)

    from saga.services.pair_ledger import hash_text
    pairs2 = [{"index": i, "user_hash": f"u{8 + i}",
               "assistant_hash": f"a{8 + i}"} for i in range(17)]
    pairs2.append({"index": 17, "user_hash": "u25",
                   "assistant_hash": hash_text("턴25 응답")})
    v2 = ledger.analyze_and_apply(pairs2, "u26")
    assert v2.aligned
    assert v2.position == _BASELINE_PAD + 19      # ← 버그 시절엔 1
    assert v2.offset == _BASELINE_PAD + 1         # 윈도우 첫 pair(u8)의 턴


def test_trimmed_reroll_rewinds_correct_turn(tmp_path):
    # 리롤 = 같은 윈도우 재전송 (마지막 assistant pop) — 트림 상태에서도
    # 방금 기록한 턴을 정확히 되감아야 한다 (corpus4의 "*says nothing*"
    # 3중 기록 재발 방지)
    from dreaming.identity import _BASELINE_PAD, PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    pairs, cur = _window(7, 18, 25)
    v1 = ledger.analyze_and_apply(pairs, cur)
    ledger.record_turn(v1, cur, "턴25 유저", "턴25 응답",
                       turn_number=v1.position)
    v2 = ledger.analyze_and_apply(pairs, cur)      # 동일 재전송
    assert v2.kind == "reroll"
    assert v2.reroll_turn_number == _BASELINE_PAD + 18


def test_fresh_session_positions_unchanged(tmp_path):
    # 신규 세션(pair 없는 첫 메시지)은 패드 없음 — 기존 번호 체계 그대로
    from dreaming.identity import PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    v = ledger.analyze_and_apply([], "u0")
    assert v.position == 0
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_identity.py -q`
Expected: FAIL — `_BASELINE_PAD` 없음(ImportError), 정렬·리롤 테스트는 어긋난 position.

- [ ] **Step 3: 구현** — `dreaming/identity.py`.

모듈 상수 추가 (`ACTIVE_STATUSES` 아래):

```python
# 트림된 대화 중간에 합류하면 윈도우 앞이 몇 턴인지 알 수 없다.
# 베이스라인을 띄워 두면 나중에 윈도우가 앞으로 자라도(maxContext 상향,
# corpus3 실측) 음수 오프셋 없이 앞 턴 번호를 배정할 수 있다.
_BASELINE_PAD = 1024
```

`Verdict`에 필드 추가:

```python
class Verdict(BaseModel):
    kind: VerdictKind
    position: int
    reroll_turn_number: Optional[int] = None
    aligned: bool = False
    offset: Optional[int] = None      # 윈도우 첫 pair의 세션 턴 번호
```

`PairLedger`에 밀집 뷰 추가 + `analyze_and_apply` 교체:

```python
    def _dense_chain(self) -> List[Dict]:
        """저장 index를 리스트 위치로 복원한 밀집 뷰 — classify의 전제.

        트림 정상상태에서 원장은 index 1042 하나로 시작할 수 있다. active
        리스트를 그대로 넘기면 위치 0 == index 1042가 되어 _align_offset이
        음수 오프셋으로 실패한다 (corpus3 재생으로 실증). 갭은 어떤
        user_hash와도 매칭되지 않는 자리표시자로 채운다.
        """
        rows = {r["index"]: r for r in self.chain()}
        if not rows:
            return []
        gap = {"user_hash": None, "assistant_hash": None,
               "status": "gap", "turn_number": None}
        return [rows.get(i, {**gap, "index": i})
                for i in range(max(rows) + 1)]

    def analyze_and_apply(self, pairs: List[Dict],
                          last_user_hash: Optional[str]) -> Verdict:
        dense = self._dense_chain()
        raw = classify(dense, pairs, last_user_hash)
        kind = _map_kind(raw, len(dense), pairs, last_user_hash)
        if not dense and pairs:
            # 트림된 대화 중간 합류 — 베이스라인 패드 (모듈 주석 참조)
            raw["position"] += _BASELINE_PAD
            if raw["offset"] is not None:
                raw["offset"] += _BASELINE_PAD

        for ci in raw["superseded_indices"]:
            if dense[ci]["status"] != "gap":
                self._transition(dense[ci], "superseded")
        for ci in raw["quarantined_indices"]:
            if dense[ci]["status"] != "gap":
                self._transition(dense[ci], "quarantined")
        for ci, client_asst_hash in raw["confirm"]:
            row = dict(dense[ci])
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
            offset=raw["offset"],
        )
```

(기존 "index → 저장 키 매핑" 주석 블록은 삭제 — 밀집 뷰로 위치 == index가 복원됐다.)

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_identity.py tests/test_dreaming_sync.py tests/test_dreaming_demote.py tests/test_dreaming_proxy.py -q; echo "exit=${pipestatus[1]}"`
Expected: PASS — 기존 신규-세션 플로우는 패드 미적용이라 번호 불변.

- [ ] **Step 5: 커밋**

```bash
git add dreaming/identity.py tests/test_dreaming_identity.py
git commit -m "fix(dreaming): 원장 밀집 뷰 + 트림 베이스라인 패드 — 트림 정상상태 턴 붕괴 수정"
```

---

### Task 2: 격리 버퍼 — 스펙 §3.1 이행

**Files:**
- Modify: `dreaming/identity.py` (`Verdict.quarantine`, analyze_and_apply 한 줄)
- Modify: `dreaming/sync.py` (`SyncPath.process` 선두 게이트, `record_response` 분기)
- Test: `tests/test_dreaming_sync.py`

**Interfaces:**
- Consumes: Task 1의 `Verdict`.
- Produces: `Verdict.quarantine: bool = False` — **원장 비어있지 않음 + 요청에 pair 있음 + 정렬 실패 + reroll 아님**일 때 True (reroll은 fallback이 trailing user로 출처를 확정한 것이라 제외 — 트림 직후 리롤이 격리당하면 안 된다). True면: `process`는 주입·압축·마킹 없이 **원본 그대로 반환**, `record_response`는 본원장(raw/ledger/resolver) 대신 `{session}/quarantine`(key = 연번 6자리)에 기록.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_sync.py`에 추가 (기존 헬퍼 재사용, 파일에 이미 있는 import 기준으로 `json` 등 필요분 추가):

```python
def test_stranger_history_is_quarantined(tmp_path):
    # 원장 있는 세션에 전혀 무관한 히스토리 — 본원장 오염 금지 (스펙 §3.1)
    import json
    from dreaming.storage import JsonDirStorage
    from dreaming.sync import SyncPath
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "s")
    m1 = [{"role": "system", "content": "너는 리사다."},
          {"role": "user", "content": "안녕"}]
    _, v1 = sp.process(m1)
    sp.record_response(v1, m1, "어서 와.")
    assert storage.get("s/raw", "000000") is not None

    stranger = [{"role": "system", "content": "너는 리사다."},
                {"role": "user", "content": "전혀 다른 이야기"},
                {"role": "assistant", "content": "낯선 응답"},
                {"role": "user", "content": "다음 질문"}]
    out, v = sp.process(stranger)
    assert v.quarantine
    assert out == stranger                       # 무가공 passthrough
    assert "cache_control" not in json.dumps(out, ensure_ascii=False)
    sp.record_response(v, stranger, "응답")
    assert storage.get("s/quarantine", "000000") is not None
    raws = [k for k, _ in storage.scan("s/raw")]
    assert raws == ["000000"]                    # 본원장 무오염


def test_trimmed_reroll_is_not_quarantined(tmp_path):
    # 트림 직후 리롤: 정렬은 실패해도 trailing user가 출처를 확정 → 격리 금지
    from dreaming.identity import PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    pairs = [{"index": i, "user_hash": f"u{7 + i}",
              "assistant_hash": f"a{7 + i}"} for i in range(18)]
    v1 = ledger.analyze_and_apply(pairs, "u25")
    ledger.record_turn(v1, "u25", "유저", "응답", turn_number=v1.position)
    v2 = ledger.analyze_and_apply(pairs, "u25")  # 동일 재전송 = 리롤
    assert v2.kind == "reroll"
    assert not v2.quarantine
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_sync.py -q`
Expected: FAIL — `quarantine` 속성 없음(AttributeError) 또는 stranger가 본원장에 기록됨.

- [ ] **Step 3: 구현**

`dreaming/identity.py` — `Verdict`에 필드 추가:

```python
    quarantine: bool = False
```

`analyze_and_apply`의 `return Verdict(...)`에 인자 추가:

```python
            quarantine=(bool(dense) and bool(pairs)
                        and not raw["aligned"] and raw["kind"] != "reroll"),
```

`dreaming/sync.py` — `SyncPath.process`에서 demote 블록 **직후**, shift_keyed 앞에 추가:

```python
        if verdict.quarantine:
            # 판정 불확실 — 주입·압축·마킹 없이 무가공 passthrough,
            # 기록은 격리 버퍼로 (스펙 §3.1)
            return messages, verdict
```

`record_response`의 `self._ledger.record_turn(...)` 앞에 추가 (user_text 추출 아래):

```python
        if verdict.quarantine:
            if last_user_hash:
                ns = f"{self._session}/quarantine"
                n = len(list(self._storage.scan(ns)))
                self._storage.put(ns, f"{n:06d}", {
                    "user_text": user_text, "assistant_text": assistant_text,
                    "user_hash": last_user_hash, "kind": verdict.kind,
                })
            return
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest tests/test_dreaming_sync.py tests/test_dreaming_identity.py tests/test_dreaming_proxy.py -q; echo "exit=${pipestatus[1]}"`
Expected: PASS.

- [ ] **Step 5: 커밋**

```bash
git add dreaming/identity.py dreaming/sync.py tests/test_dreaming_sync.py
git commit -m "feat(dreaming): 격리 버퍼 — 정렬 실패 요청은 무가공 passthrough + 별도 기록 (스펙 §3.1)"
```

---

### Task 3: 압축 윈도우 앵커 — 트림 구간 청크 복원

**Files:**
- Modify: `dreaming/chunks.py` (`apply_compression`)
- Modify: `dreaming/sync.py` (`SyncPath.process` 압축 게이트)
- Test: `tests/test_dreaming_chunks.py`, `tests/test_dreaming_sync.py`

**Interfaces:**
- Consumes: Task 1의 `Verdict.offset` (윈도우 첫 pair의 세션 턴 번호).
- Produces: `apply_compression(messages: List[Dict], plan: Dict, window_start_turn: int = 0) -> Tuple[List[Dict], Optional[int]]` — 드롭 수 = `max(0, covers_until_turn - window_start_turn)`. 드롭 0이어도 청크는 prepend된다 (트림으로 사라진 구간 복원). 기본값 0 = 기존 전체-히스토리 동작 그대로. SyncPath는 `plan and verdict.aligned and verdict.offset is not None`일 때만 적용, `window_start_turn=verdict.offset` 전달.

- [ ] **Step 1: 실패 테스트 작성** — `tests/test_dreaming_chunks.py`에 추가:

```python
def test_window_past_covers_restores_chunks_without_drop():
    # 트림이 이미 압축 구간을 지나감 (window_start 5 ≥ covers 2) —
    # 드롭 0 + 청크 prepend = 사라진 컨텍스트 복원 (이 기능의 본래 가치)
    msgs = _msgs(3)
    out, bp2 = apply_compression(msgs, _PLAN, window_start_turn=5)
    texts = [m["content"] for m in out]
    assert "[지난 이야기 · 초반]" in texts
    assert all(f"질문{i}" in "".join(texts) for i in range(3))   # 전량 보존
    assert bp2 == 2                                # system+인사 다음
    assert len(out) == len(msgs) + 1


def test_window_inside_covers_drops_remainder_only():
    # 윈도우 시작 1, covers 2 → 윈도우에 남은 압축 대상은 1 pair뿐
    msgs = _msgs(4)
    out, bp2 = apply_compression(msgs, _PLAN, window_start_turn=1)
    texts = [m["content"] for m in out]
    assert "질문0" not in "".join(texts)           # 첫 pair(턴1)만 드롭
    assert "질문1" in "".join(texts)               # 턴2부터 보존
    assert bp2 == 2


def test_default_window_start_keeps_existing_behavior():
    msgs = _msgs(4)
    assert apply_compression(msgs, _PLAN) == \
        apply_compression(msgs, _PLAN, window_start_turn=0)
```

`tests/test_dreaming_sync.py`에 게이트 테스트 추가:

```python
def test_compression_uses_window_offset(tmp_path):
    # 트림 합류 세션(패드 베이스라인)에서 플랜이 구간을 못 덮으면
    # 드롭 없이 청크만 prepend — 위치 오치환(재생으로 실증된 결함 ②) 금지
    import json
    from dreaming.identity import _BASELINE_PAD
    from dreaming.storage import JsonDirStorage
    from dreaming.sync import SyncPath
    storage = JsonDirStorage(tmp_path)
    storage.put("s/compression", "plan", {
        "covers_until_turn": _BASELINE_PAD,        # 패드 이전 구간만 커버
        "messages": [{"role": "assistant", "content": "[지난 이야기 · 복원]"}]})
    sp = SyncPath(storage, "s")
    msgs = [{"role": "system", "content": "너는 리사다."}]
    for i in range(3):
        msgs += [{"role": "user", "content": f"질문{i}"},
                 {"role": "assistant", "content": f"답{i}"}]
    msgs.append({"role": "user", "content": "현재 질문"})
    out, v = sp.process(msgs)
    joined = json.dumps(out, ensure_ascii=False)
    assert v.aligned and v.offset == _BASELINE_PAD
    assert "[지난 이야기 · 복원]" in joined        # 청크 복원
    assert "질문0" in joined                       # 윈도우 pair는 무드롭
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest tests/test_dreaming_chunks.py tests/test_dreaming_sync.py -q`
Expected: FAIL — `window_start_turn` 파라미터 없음(TypeError), 게이트 미배선.

- [ ] **Step 3: 구현**

`dreaming/chunks.py` — `apply_compression` 교체:

```python
def apply_compression(messages: List[Dict], plan: Dict,
                      window_start_turn: int = 0
                      ) -> Tuple[List[Dict], Optional[int]]:
    """히스토리 선두의 압축 대상 pair를 청크로 치환 (스펙 §5 레이아웃).

    window_start_turn = 요청 첫 pair의 세션 턴 번호 (트림 시 0이 아님).
    드롭 수 = 압축 구간 중 윈도우에 아직 남아 있는 pair 수 — 트림이 이미
    구간을 지나갔으면 0이고, 그때 청크 prepend는 사라진 컨텍스트의 복원이다.
    선두 system 블록과 인사(첫 user 이전 assistant)는 보존한다.
    히스토리가 드롭 수보다 짧으면 원본 그대로 — fail-open.
    반환: (메시지, 첫 청크 인덱스 | None).
    """
    to_drop = plan["covers_until_turn"] - window_start_turn
    if to_drop < 0:
        to_drop = 0
    i = 0
    while i < len(messages) and messages[i].get("role") != "user":
        i += 1                             # 첫 user 앞(system·인사)은 보존
    pairs, j = 0, i
    while j < len(messages) and pairs < to_drop:
        if messages[j].get("role") == "user":
            if (j + 1 < len(messages)
                    and messages[j + 1].get("role") == "assistant"):
                j += 2
                pairs += 1
            else:
                break                      # 미완 pair(현재 턴) — 압축 불가
        else:
            j += 1
    if pairs < to_drop:
        return messages, None
    out = messages[:i] + copy.deepcopy(plan["messages"]) + messages[j:]
    return out, i
```

`dreaming/sync.py` — `SyncPath.process`의 압축 블록 교체:

```python
        bp2 = None
        plan = self._storage.get(f"{self._session}/compression", "plan")
        if (plan is not None and verdict.aligned
                and verdict.offset is not None):
            out, bp2 = apply_compression(out, plan,
                                         window_start_turn=verdict.offset)
```

- [ ] **Step 4: 통과 확인 (전체 회귀 포함)**

Run: `python3 -m pytest tests/ -q; echo "exit=${pipestatus[1]}"`
Expected: 전체 PASS — 기존 chunks·proxy e2e는 신규 세션 플로우(offset 0)라 동작 불변.

- [ ] **Step 5: 커밋**

```bash
git add dreaming/chunks.py dreaming/sync.py tests/test_dreaming_chunks.py tests/test_dreaming_sync.py
git commit -m "feat(dreaming): 압축 윈도우 앵커 — 트림 구간은 드롭 없이 청크로 복원"
```

---

### Task 4: corpus 재생 도구 + 교차 기능 회귀 + push

**Files:**
- Create: `benchmarks/capture/replay_ledger.py`
- Test: `tests/test_dreaming_proxy.py` (교차 기능 1개)
- 수동 검증: ba42ff corpus3→4 재생 (데이터 미커밋)

**Interfaces:**
- Consumes: 전부.
- Produces: `python3 -m benchmarks.capture.replay_ledger <corpus_dir> [<corpus_dir> ...]` — req-*.json을 순서대로 SyncPath에 재생, 원장 정합성(중복 턴 기록·시간순 역전) 검사, 위반 시 exit 1. 응답 텍스트는 다음 요청의 assistant에서 합성.

- [ ] **Step 1: 교차 기능 테스트 작성** — `tests/test_dreaming_proxy.py`에 추가 (PR #5의 lore_shift와 Plan 4 압축 동시 가동 — 지금까지 수동 스모크로만 확인):

```python
def test_lore_shift_and_compression_together(tmp_path):
    storage = JsonDirStorage(tmp_path)
    storage.put("sess1/compression", "plan", {
        "covers_until_turn": 1,
        "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]})
    keyed = ("성문 경비는 해가 지면 두 배로 늘어나며, 통행증 없는 외지인은 "
             "동문 초소에서 하룻밤 억류된 뒤 아침에 심문을 받는다.")
    from dreaming.sync import SyncPath
    sp = SyncPath(storage, "sess1", keyed_lore=[keyed])
    msgs = [{"role": "system", "content": f"너는 리사다.\n\n{keyed}"},
            {"role": "user", "content": "질문0"},
            {"role": "assistant", "content": "답0"},
            {"role": "user", "content": "질문1"},
            {"role": "assistant", "content": "답1"},
            {"role": "user", "content": "질문2"}]
    out, v = sp.process(msgs)
    texts = [str(m.get("content")) for m in out]
    assert keyed not in texts[0]                          # 프리픽스에서 제거
    chunk_i = next(i for i, t in enumerate(texts) if "[지난 이야기" in t)
    marks = [i for i, m in enumerate(out) if "cache_control" in m]
    last_user = max(i for i, m in enumerate(out) if m["role"] == "user")
    assert "질문0" not in "".join(texts) and "질문1" in "".join(texts)
    assert marks == [0, chunk_i, len(out) - 2]            # BP1·BP2·BP3
    assert max(marks) < last_user                         # mutable 존은 캐시 밖
    assert any("<active_lorebook>" in t for t in texts)
```

- [ ] **Step 2: 도구 작성** — `benchmarks/capture/replay_ledger.py` 신규:

```python
"""corpus 캡처를 PairLedger에 재생해 원장 시간순 정합성을 검사한다.

usage: python3 -m benchmarks.capture.replay_ledger <corpus_dir> [<corpus_dir> ...]

corpus_dir 안의 req-*.json(진짜 RisuAI 와이어 캡처)을 이름순으로 SyncPath에
먹이고, 끝난 뒤 원장을 검사한다. 응답 텍스트는 캡처에 없으므로 다음 요청의
"마지막 user 직전 assistant"에서 합성한다 (마지막 요청은 기록 생략).

검사 항목 (corpus3→4 재생으로 실증된 붕괴의 재발 방지):
  1) 같은 유저 발화가 복수 턴 번호로 기록 (리롤 오판)
  2) 기록 턴 순서가 마지막 요청의 대화 순서와 역전
위반 시 exit 1. 코퍼스 데이터는 카드 저작물 포함 — 커밋 금지, 도구만 커밋.
"""
import json
import pathlib
import sys
import tempfile

from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath


def _load(dirs):
    reqs = []
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("req-*.json")):
            reqs.append((f"{pathlib.Path(d).name}/{p.stem}",
                         json.loads(p.read_text())["messages"]))
    return reqs


def _last_user_idx(msgs):
    return max((i for i, m in enumerate(msgs) if m["role"] == "user"),
               default=None)


def _reply_for(reqs, n):
    if n + 1 >= len(reqs):
        return None
    nxt = reqs[n + 1][1]
    li = _last_user_idx(nxt)
    for i in range(li - 1, -1, -1):
        if nxt[i]["role"] == "assistant":
            return nxt[i]["content"]
    return None


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    reqs = _load(argv)
    storage = JsonDirStorage(pathlib.Path(tempfile.mkdtemp(prefix="replay-")))
    sp = SyncPath(storage, "cap")
    quarantined = 0
    for n, (name, msgs) in enumerate(reqs):
        _, v = sp.process(msgs)
        reply = _reply_for(reqs, n)
        if reply is not None:
            sp.record_response(v, msgs, reply)
        quarantined += bool(v.quarantine)
        print(f"{name}: {v.kind:12s} pos={v.position} offset={v.offset} "
              f"aligned={v.aligned} quarantine={v.quarantine}")

    rows = sorted(storage.scan("cap/raw"), key=lambda kv: kv[0])
    by_text = {}
    for _, r in rows:
        by_text.setdefault(r["user_text"], []).append(r["turn_number"])
    dups = {t[:24]: ns for t, ns in by_text.items() if len(ns) > 1}

    gt = [m["content"] for m in reqs[-1][1] if m["role"] == "user"]
    pos = {t: i for i, t in enumerate(gt)}
    seq = [r["user_text"] for _, r in
           sorted(rows, key=lambda kv: kv[1]["turn_number"])]
    known = [t for t in seq if t in pos]
    inversions = [(a[:16], b[:16]) for a, b in zip(known, known[1:])
                  if pos[a] > pos[b]]

    print(f"\nrecorded={len(rows)} quarantined={quarantined}")
    print("중복 턴 기록:", dups if dups else "없음")
    print("시간순 역전:", inversions if inversions else "없음")
    return 1 if (dups or inversions) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 3: 테스트·도구 확인**

Run: `python3 -m pytest tests/ -q; echo "exit=${pipestatus[1]}"`
Expected: 전체 PASS.

Run (수동 — ba42ff 코퍼스가 로컬에 있을 때):

```bash
python3 -m benchmarks.capture.replay_ledger \
  /Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-ba42ff/dreaming_data/corpus3 \
  /Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-ba42ff/dreaming_data/corpus4; echo "exit=$?"
```

Expected: exit=0 — 중복 턴 기록 없음, 시간순 역전 없음 (수정 전에는 "*says nothing*" 3중 기록 + 역전 2건). 결과를 최종 보고에 포함할 것.

- [ ] **Step 4: saga diff 0 확인 + 커밋 + push + PR**

```bash
git diff --stat origin/main -- saga/
git add benchmarks/capture/replay_ledger.py tests/test_dreaming_proxy.py
git commit -m "test(dreaming): lore_shift×압축 교차 회귀 + corpus 원장 재생 검사 도구"
git push -u origin dreaming/spec
```

PR 생성 (`gh pr create --base main`) — 제목: `fix(dreaming): 트림 정렬 — 원장 밀집 뷰·격리 버퍼(§3.1)·압축 윈도우 앵커`. 본문에 결함 ①~③ ↔ Task 매핑과 corpus 재생 전/후 결과.

---

## Self-Review 결과 (작성 시 수행)

1. **커버리지**: 결함 ①(T1 밀집 뷰+패드), ②(T3 윈도우 앵커), ③(T2 격리 버퍼) — 재생으로 실증된 3건 전부 태스크 존재. 리롤 예외(트림 직후 리롤이 격리당하면 안 됨)는 T1 테스트 + T2 quarantine 조건(`kind != "reroll"`)으로 양쪽에서 고정.
2. **자리표시자 없음**: 전 태스크 실코드 포함.
3. **타입 일관성**: `Verdict.offset`(T1 생산) → T3 소비, `Verdict.quarantine`(T2 생산) → sync 게이트, `apply_compression(messages, plan, window_start_turn=0)` T3 정의 == T3 sync 호출부. `_BASELINE_PAD` T1 정의, T1·T2·T3 테스트에서 동일 import 경로(`dreaming.identity`). 기존 테스트 보호: 신규 세션은 패드 미적용(포지션 불변), `window_start_turn` 기본값 0(기존 호출부 무변경), 기존 e2e는 offset 0 경로.

## 알려진 한계 (의도된 범위)

- 윈도우와 압축 구간 사이 갭(트림됐지만 아직 꿈이 에피소드로 못 만든 턴)은 프롬프트에서 빠진다 — Risu 단독 대비 손해는 아니고(어차피 트림됨), 꿈이 covers를 따라잡으면 청크로 채워진다.
- 격리 버퍼는 기록만 한다 — 열람·복권 UI는 대시보드 플랜(로드맵 Plan 5) 몫.
- 리롤 직후 1턴은 aligned=False라 압축 미적용(풀 히스토리) — 다음 턴에 재정렬.
