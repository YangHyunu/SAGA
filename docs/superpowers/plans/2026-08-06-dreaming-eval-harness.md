# Plan 6: 평가 하네스 (스펙 §9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 같은 고정 대본을 Dreaming/순정/트림/turn-retrieval 4개 변형에 재생하고, 심어둔 사실(지뢰)을 트림 윈도우 밖에서 되묻는 프로브를 결정론 오라클로 자동 채점해 $·캐시율·지연과 함께 비교 표를 낸다.

**Architecture:** `benchmarks/eval/` 신규 패키지. 대본(script)·채점(oracle)·변형별 조립(variants)은 순수 함수로 분리해 유닛 테스트하고, 네트워크는 run.py 드라이버에만 둔다. 유저 턴은 첫 실행(dreaming)에서 시뮬레이터로 생성 후 JSON으로 동결 — 이후 변형은 동일 텍스트를 재생해 변형 간 입력을 공정하게 고정한다. 채점은 LLM 0콜(한글 수사 포함 문자열 매칭), 프로브 응답 원문을 리포트에 병기해 수동 감사(§9 "10~20% 수동 감사")를 지원한다.

**Tech Stack:** Python 3, httpx, pytest. 재사용: `benchmarks/cardsim/lorebook.py`(load_card/activate/build_messages), `dreaming/numerals.korean_spellings`.

## Global Constraints

- 스펙 §9: 대조군 의무 — "Risu 순정 / HypaV3 / 단순 turn-retrieval 베이스라인 / Dreaming". **HypaV3는 RisuAI 클라이언트 필요 → 이 플랜 비스코프** (플러그인 단계에서). LLM judge(서사)도 비스코프 — 이 플랜은 결정론 오라클만.
- 스펙 §9: 병기 지표 — "$ / 캐시율 / 턴 준비시간".
- 결과·대본 JSON은 전부 `dreaming_data/eval/` 아래 (gitignored — 카드 파생 텍스트 커밋 금지).
- API 키는 `.env`의 `DREAMING_UPSTREAM_KEY`를 읽기만 — 값 출력·커밋 금지.
- CLAUDE.md: 최소 코드, 단일 사용 추상화 금지, 기존 스타일(한국어 주석) 유지.
- 트림 윈도우 `W=8` pair — TAIL_KEEP(6)보다 크고, 모든 프로브 시점에 지뢰 턴이 윈도우 밖에 있도록 대본이 설계됨.

---

## File Structure

- `benchmarks/eval/__init__.py` — 패키지 docstring 한 줄
- `benchmarks/eval/script.py` — BEATS(30) + Probe 정의(기대답 그룹) + 대본 동결/로드
- `benchmarks/eval/oracle.py` — `num_terms`, `score_reply`, `score_recall` (결정론 채점)
- `benchmarks/eval/variants.py` — `trim_window`, `retrieve_turns`, `prepare_request` (변형별 메시지 조립)
- `benchmarks/eval/run.py` — CLI 드라이버 (네트워크, 시뮬레이터, 결과 JSON 기록)
- `benchmarks/eval/report.py` — 결과 JSON들 → 비교 표 + 프로브 응답 원문
- `tests/test_eval_harness.py` — 위 순수 함수 전부

---

### Task 1: 대본 + 프로브 정의 (`script.py`)

**Files:**
- Create: `benchmarks/eval/__init__.py`
- Create: `benchmarks/eval/script.py`
- Test: `tests/test_eval_harness.py`

**Interfaces:**
- Produces: `BEATS: List[str]` (길이 30), `PAUSES: Dict[int, int]` (`{9: 12, 19: 12}`), `Probe(turn: int, label: str, expect: List[List[str]], recall: bool)` dataclass, `PROBES: List[Probe]`, `freeze_script(path, turns: List[Dict]) -> None`, `load_script(path) -> List[Dict]` (`[{"turn": int, "user_text": str}]`).
- 지뢰 배치: 이름·나이(0), 소지금 300(3), 선물 세 개(5), 자정 약속(7), 50 지불(11), 왼손잡이(13), 보름달 약속(15). 프로브: 21(이름·나이), 23(잔액), 25(첫 선물), 27(약속 시각), 28(약속 행사), 29(종합 회상). W=8이면 모든 프로브 시점에 해당 지뢰가 윈도우 밖.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""평가 하네스 — 대본/오라클/변형 조립 순수 함수 (스펙 §9)."""
import json

from benchmarks.eval.script import BEATS, PROBES, freeze_script, load_script


def test_script_shape_and_probe_positions():
    assert len(BEATS) == 30
    turns = [p.turn for p in PROBES]
    assert turns == [21, 23, 25, 27, 28, 29]
    assert all(0 <= t < len(BEATS) for t in turns)
    # 지뢰(0~15)와 프로브(21~29) 사이 간격이 트림 윈도우(8 pair)보다 큼
    assert min(turns) - 15 >= 6


def test_probe_expectations_are_nonempty():
    for p in PROBES:
        assert p.expect and all(group for group in p.expect)
    recall = [p for p in PROBES if p.recall]
    assert len(recall) == 1 and recall[0].turn == 29
    assert len(recall[0].expect) == 5          # 회상 항목 5개


def test_freeze_and_load_roundtrip(tmp_path):
    turns = [{"turn": 0, "user_text": "안녕, 나는 한결이야."}]
    p = tmp_path / "script.json"
    freeze_script(p, turns)
    assert load_script(p) == turns
    assert json.loads(p.read_text())           # 평문 JSON
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError: No module named 'benchmarks.eval'`

- [ ] **Step 3: 구현**

`benchmarks/eval/__init__.py`:

```python
"""평가 하네스 — 고정 대본 재생 + 대조군 + 결정론 오라클 (스펙 §9)."""
```

`benchmarks/eval/script.py`:

```python
"""평가 대본 — 지뢰 심기 + 트림 윈도우 밖 프로브 (스펙 §9 드라이버).

지뢰는 0~15턴에 심고 프로브는 21턴 이후에 되묻는다. 트림 윈도우 W=8 pair
기준으로 모든 프로브 시점에 해당 지뢰 원문이 윈도우 밖 — 대본 자체가
장기기억 시험이 되도록 설계됐다. 리롤은 없다 (기억 비교에 집중).

유저 턴 동결: 첫 실행이 시뮬레이터 발화를 freeze_script로 저장하고, 이후
변형은 load_script로 같은 텍스트를 재생한다 — 변형 간 입력 공정성.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List

from dreaming.numerals import korean_spellings

BEATS: List[str] = [
    "정중히 자기소개를 한다. 이름은 '한결', 나이는 '스물일곱'이라고 명확히 밝힌다.",
    "상대에 대해 물어본다 — 어떻게 불러야 할지, 어떤 사람인지.",
    "지금 있는 장소/상황에 대해 자연스럽게 묻는다.",
    "자신의 소지금이 정확히 300(이 세계관의 화폐 단위)뿐이라고 대화 중에 언급한다. 숫자 300을 명시.",
    "상대의 취향이나 좋아하는 것을 묻는다.",
    "작은 선물로 먹을 것(세계관에 어울리는 것) '세 개'를 건넨다. 개수 '세 개'를 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "오늘은 물러가겠다며, '내일 자정'에 다시 오겠다고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "짧게 작별 인사를 한다.",
    # -- pause: dream #1 --
    "약속대로 다시 찾아왔다고 인사한다.",
    "값으로 50(화폐)을 치르고 마실 것이나 먹을 것을 산다. 숫자 50을 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "자신이 사실 '왼손잡이'라는 것을 고백한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "'다음 보름달'에 함께 축제나 나들이를 가자고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "지난 며칠을 회상하며 짧게 감상을 말한다.",
    "짧게 작별 인사를 한다.",
    # -- pause: dream #2 --
    "시간이 지나 다시 찾아왔다고 인사한다.",
    "자신의 이름과 나이를 기억하고 있는지 상대에게 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "장부를 잃어버렸다며, 처음 소지금과 그간 쓴 돈을 감안하면 지금 얼마가 남았을지 아는지 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "처음 만난 날 자신이 건넨 선물이 무엇이었고 몇 개였는지 기억하는지 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "예전에 자신이 '몇 시'에 다시 오겠다고 약속했었는지 묻는다.",
    "언제 어디에 함께 가자고 약속했었는지 묻는다.",
    "지금까지 자신(한결)에 대해 알게 된 것을 전부 말해달라고 한다.",
]

PAUSES: Dict[int, int] = {9: 12, 19: 12}       # beat index → idle 초 (꿈 트리거)


@dataclass
class Probe:
    turn: int                        # beat index (0-based)
    label: str
    expect: List[List[str]]          # 그룹 간 AND, 그룹 내 OR
    recall: bool = False             # True면 그룹 적중 수 m/n으로 채점


def _num(value: int, *extra: str) -> List[str]:
    return [str(value)] + korean_spellings(value) + list(extra)


PROBES: List[Probe] = [
    Probe(21, "이름·나이", [["한결"], _num(27)]),
    Probe(23, "잔액 250", [_num(250)]),
    Probe(25, "선물 세 개", [["세 개", "세개", "3개", "셋"]]),
    Probe(27, "약속 시각", [["자정", "밤 12", "12시"]]),
    Probe(28, "약속 행사", [["보름달", "보름"]]),
    Probe(29, "종합 회상", [["한결"], _num(27), ["왼손잡이", "왼손"],
                          _num(250) + _num(300), ["보름달", "보름"]],
          recall=True),
]


def freeze_script(path, turns: List[Dict]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(turns, ensure_ascii=False, indent=1))


def load_script(path) -> List[Dict]:
    return json.loads(pathlib.Path(path).read_text())
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/__init__.py benchmarks/eval/script.py tests/test_eval_harness.py
git commit -m "feat(eval): 평가 대본 — 지뢰 7개 + 트림 밖 프로브 6개 + 대본 동결 (스펙 §9)"
```

---

### Task 2: 결정론 오라클 (`oracle.py`)

**Files:**
- Create: `benchmarks/eval/oracle.py`
- Test: `tests/test_eval_harness.py` (추가)

**Interfaces:**
- Consumes: `Probe` (Task 1).
- Produces: `score_reply(reply: str, probe: Probe) -> Dict` — `{"label": str, "hit": "full"|"partial"|"miss", "matched": int, "total": int}`. 일반 프로브: 전 그룹 매치=full, 1개 이상=partial, 0=miss. recall 프로브: matched=적중 그룹 수, full=전부, partial=1 이상.
- 매칭: reply와 후보 양쪽에서 공백 제거 후 substring (cardsim `_norm_for_match`와 같은 정신 — "세 개"가 "세개"로 붙어 나와도 잡는다).

- [ ] **Step 1: 실패하는 테스트 작성 (test_eval_harness.py에 추가)**

```python
from benchmarks.eval.oracle import score_reply
from benchmarks.eval.script import Probe


def test_score_full_partial_miss():
    p = Probe(0, "이름·나이", [["한결"], ["27", "스물일곱", "이십칠"]])
    assert score_reply("한결님, 스물일곱이시죠.", p)["hit"] == "full"
    r = score_reply("한결님이라는 건 기억해요.", p)
    assert r["hit"] == "partial" and r["matched"] == 1 and r["total"] == 2
    assert score_reply("글쎄요, 기억나지 않네요.", p)["hit"] == "miss"


def test_score_ignores_whitespace():
    p = Probe(0, "선물", [["세 개"]])
    assert score_reply("사과를 세\n개 주셨죠.", p)["hit"] == "full"


def test_score_recall_counts_groups():
    p = Probe(0, "회상", [["한결"], ["왼손잡이"], ["보름달"]], recall=True)
    r = score_reply("한결님은 왼손잡이시고...", p)
    assert r["hit"] == "partial" and r["matched"] == 2 and r["total"] == 3


def test_korean_numeral_expectation_matches():
    p = Probe(0, "잔액", [["250", "이백오십"]])
    assert score_reply("이백오십 남으셨을 거예요.", p)["hit"] == "full"
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError` (oracle)

- [ ] **Step 3: 구현**

`benchmarks/eval/oracle.py`:

```python
"""결정론 오라클 — 프로브 응답 채점, LLM 0콜 (스펙 §9 "숫자/상태").

공백 제거 후 substring 매칭 — 한글 수사("이백오십")는 대본 단계에서
korean_spellings로 기대답 그룹에 이미 포함돼 있다.
"""

from __future__ import annotations

import re
from typing import Dict

from benchmarks.eval.script import Probe

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text)


def score_reply(reply: str, probe: Probe) -> Dict:
    hay = _norm(reply)
    matched = sum(1 for group in probe.expect
                  if any(_norm(c) in hay for c in group))
    total = len(probe.expect)
    if matched == total:
        hit = "full"
    elif matched > 0:
        hit = "partial"
    else:
        hit = "miss"
    return {"label": probe.label, "hit": hit,
            "matched": matched, "total": total}
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/oracle.py tests/test_eval_harness.py
git commit -m "feat(eval): 결정론 오라클 — 공백 무시 그룹 매칭, full/partial/miss (스펙 §9)"
```

---

### Task 3: 변형별 조립 (`variants.py`)

**Files:**
- Create: `benchmarks/eval/variants.py`
- Test: `tests/test_eval_harness.py` (추가)

**Interfaces:**
- Consumes: `benchmarks.cardsim.lorebook.build_messages/activate/Card` (기존).
- Produces:
  - `TRIM_WINDOW = 8`
  - `trim_window(history: List[Dict], w: int = TRIM_WINDOW) -> List[Dict]` — 마지막 w개 user-pair만 남김 (마지막 미완 user 포함). RisuAI 클라이언트 트림 근사.
  - `retrieve_turns(history: List[Dict], query: str, k: int = 3) -> List[str]` — 트림으로 잘려나갈 과거 pair를 문자 bigram 겹침으로 top-k 선별, `"유저: …\n캐릭터: …"` 문자열 목록 (결정론, 동률은 오래된 순).
  - `prepare_request(variant: str, card, history: List[Dict]) -> List[Dict]` — variant ∈ `{"dreaming", "vanilla", "trim", "retrieval"}`. dreaming/trim/retrieval은 `trim_window` 적용, vanilla는 전체. retrieval은 발췌 블록을 마지막 user 콘텐츠 앞에 prepend (2511.17208 단순 turn-retrieval 베이스라인).
- 참고: 순정 RisuAI의 automaticCachePoint(마지막 user 3개 마킹)는 재현하지 않는다 — vanilla는 "마킹 없음" 하한 대조군.

- [ ] **Step 1: 실패하는 테스트 작성 (추가)**

```python
from benchmarks.eval.variants import prepare_request, retrieve_turns, trim_window


def _hist(pairs):
    h = []
    for i in range(pairs):
        h.append({"role": "user", "content": f"질문{i} 사과 이야기"})
        h.append({"role": "assistant", "content": f"답{i}"})
    h.append({"role": "user", "content": "마지막 질문"})
    return h


def test_trim_window_keeps_last_pairs_and_trailing_user():
    out = trim_window(_hist(12), w=8)
    assert out[0]["content"] == "질문4 사과 이야기"    # 앞 4 pair 잘림
    assert out[-1]["content"] == "마지막 질문"
    assert trim_window(_hist(3), w=8) == _hist(3)      # 짧으면 그대로


def test_retrieve_turns_is_deterministic_topk():
    h = _hist(12)
    h[0]["content"] = "질문0 보름달 축제 약속"
    got = retrieve_turns(h, "보름달 약속 기억해?", k=2)
    assert got == retrieve_turns(h, "보름달 약속 기억해?", k=2)
    assert any("보름달" in g for g in got)
    assert len(got) <= 2
    # 윈도우 안 pair는 검색 대상 아님 (원문이 이미 있음)
    assert not any("질문11" in g for g in got)


def test_prepare_request_variants_differ():
    from benchmarks.cardsim.lorebook import Card
    card = Card(name="리사", description="너는 리사다.", post_history="",
                greeting="어서 와요.")
    h = _hist(12)
    full = prepare_request("vanilla", card, h)
    trimmed = prepare_request("trim", card, h)
    retr = prepare_request("retrieval", card, h)
    assert len(full) > len(trimmed)
    assert "질문0" in json.dumps(full, ensure_ascii=False)
    assert "질문0" not in json.dumps(trimmed, ensure_ascii=False)
    assert "[과거 대화 발췌]" in retr[-1]["content"]
    assert prepare_request("dreaming", card, h) == trimmed  # 전송분 동일, 차이는 프록시
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError` (variants)

- [ ] **Step 3: 구현**

`benchmarks/eval/variants.py`:

```python
"""대조군 변형별 요청 조립 (스펙 §9 대조군).

- vanilla   : 전체 히스토리, 마킹 없음 — Risu 순정 하한 근사
- trim      : 마지막 W pair만 — 클라이언트 트림 치매 베이스라인
- retrieval : trim + 잘린 구간에서 top-k 어휘 검색 발췌 prepend
              (2511.17208 단순 turn-retrieval — Dreaming이 이겨야 하는 선)
- dreaming  : 전송분은 trim과 동일 — 주입/압축/마킹은 프록시가 한다

HypaV3 대조군은 RisuAI 클라이언트가 필요해 플러그인 단계로 미룬다.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from benchmarks.cardsim.lorebook import Card, activate, build_messages

TRIM_WINDOW = 8


def trim_window(history: List[Dict], w: int = TRIM_WINDOW) -> List[Dict]:
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    if history and history[-1]["role"] == "user":
        starts = starts[:-1]                   # 진행 중인 현재 턴은 카운트 제외
    if len(starts) <= w:
        return history
    return history[starts[len(starts) - w]:]


def _bigrams(text: str) -> set:
    t = "".join(text.split())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def retrieve_turns(history: List[Dict], query: str, k: int = 3) -> List[str]:
    """트림으로 잘릴 과거 pair에서 질의와 겹치는 top-k 발췌 (결정론)."""
    kept = trim_window(history)
    cut = history[:len(history) - len(kept)]
    q = _bigrams(query)
    scored = []
    for i in range(0, len(cut) - 1, 2):
        if cut[i]["role"] != "user" or cut[i + 1]["role"] != "assistant":
            continue
        text = f"유저: {cut[i]['content']}\n캐릭터: {cut[i + 1]['content']}"
        scored.append((len(q & _bigrams(text)), -i, text))
    scored.sort(reverse=True)
    return [t for s, _, t in scored[:k] if s > 0]


def prepare_request(variant: str, card: Card, history: List[Dict]) -> List[Dict]:
    window = history if variant == "vanilla" else trim_window(history)
    actives = activate(card, window)
    msgs = build_messages(card, actives, window)
    if variant == "retrieval":
        query = history[-1]["content"]
        excerpts = retrieve_turns(history, query)
        if excerpts:
            block = "[과거 대화 발췌]\n" + "\n---\n".join(excerpts)
            msgs = copy.deepcopy(msgs)
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = block + "\n\n" + m["content"]
                    break
    return msgs
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/variants.py tests/test_eval_harness.py
git commit -m "feat(eval): 대조군 변형 조립 — 트림 윈도우 + turn-retrieval 베이스라인 (스펙 §9)"
```

---

### Task 4: 드라이버 (`run.py`)

**Files:**
- Create: `benchmarks/eval/run.py`
- Test: `tests/test_eval_harness.py` (추가 — 결과 조립 순수 함수만)

**Interfaces:**
- Consumes: Task 1~3 전부, `benchmarks.cardsim.bench`의 패턴(`_upstream_key`, usage 파싱).
- Produces: CLI `python3 -m benchmarks.eval.run <card.charx> <variant> --session S [--script PATH] [--reset]`. 결과 JSON `dreaming_data/eval/result-{session}.json` — `{"variant", "session", "model", "turns": [{"turn", "user", "reply", "prompt", "cached", "write", "cost", "sec"}], "probes": [score_reply 결과 + {"turn", "reply"}], "totals": {"cost", "avg_hit_t2", "avg_sec", "oracle_full", "oracle_partial", "recall"}}`.
- `build_result(variant, session, model, turns) -> Dict` 순수 함수로 분리 — 채점·집계를 네트워크 없이 테스트.
- dreaming 변형: 프록시(`http://127.0.0.1:8787`) + `x-dreaming-session-id` 헤더 + PAUSES idle + 종료 후 cursor 대기. 그 외 변형: OpenRouter 직결, pause 없음.
- 대본: `--script` 있으면 재생(시뮬레이터 0콜), 없으면 Flash 시뮬레이터로 생성 후 `dreaming_data/eval/script-{session}.json` 동결.

- [ ] **Step 1: 실패하는 테스트 작성 (추가)**

```python
from benchmarks.eval.run import build_result


def test_build_result_scores_probes_and_totals():
    turns = []
    for i in range(30):
        reply = "기억해요, 한결님. 스물일곱이시죠." if i == 21 else f"응답{i}"
        turns.append({"turn": i, "user": f"발화{i}", "reply": reply,
                      "prompt": 100, "cached": 90 if i else 0, "write": 0,
                      "cost": 0.001, "sec": 1.0})
    r = build_result("trim", "s1", "m", turns)
    assert r["totals"]["oracle_full"] == 1            # 21번 프로브만 적중
    assert r["totals"]["cost"] == 0.03
    assert abs(r["totals"]["avg_hit_t2"] - 90.0) < 1e-6
    p21 = next(p for p in r["probes"] if p["turn"] == 21)
    assert p21["hit"] == "full" and "한결" in p21["reply"]
    recall = next(p for p in r["probes"] if p["turn"] == 29)
    assert recall["matched"] == 0 and recall["total"] == 5
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError` (run)

- [ ] **Step 3: 구현**

`benchmarks/eval/run.py`:

```python
"""평가 드라이버 — 한 변형을 대본으로 완주하고 결과 JSON을 남긴다 (스펙 §9).

usage:
    python3 -m benchmarks.eval.run <card.charx> dreaming --session ev1
    python3 -m benchmarks.eval.run <card.charx> trim --session ev1-trim \
        --script dreaming_data/eval/script-ev1.json

첫 실행(--script 없음)이 시뮬레이터로 유저 턴을 생성해 동결하고, 대조군은
--script로 같은 텍스트를 재생한다. dreaming만 프록시를 경유하며 idle pause로
꿈을 트리거한다. 결과·대본은 dreaming_data/eval/ (gitignored — 커밋 금지).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time
from typing import Dict, List

import httpx

from benchmarks.cardsim.lorebook import load_card
from benchmarks.eval.oracle import score_reply
from benchmarks.eval.script import (BEATS, PAUSES, PROBES, freeze_script,
                                    load_script)
from benchmarks.eval.variants import prepare_request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = "http://127.0.0.1:8787"
UPSTREAM = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4.5"
SIM_MODEL = "google/gemini-2.5-flash"
USER_NAME = "한결"
VARIANTS = ("dreaming", "vanilla", "trim", "retrieval")


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def build_result(variant: str, session: str, model: str,
                 turns: List[Dict]) -> Dict:
    probes = []
    for p in PROBES:
        reply = turns[p.turn]["reply"] if p.turn < len(turns) else ""
        probes.append({**score_reply(reply, p), "turn": p.turn,
                       "reply": reply})
    hits = [t["cached"] / t["prompt"] for t in turns[1:] if t["prompt"]]
    totals = {
        "cost": round(sum(t["cost"] for t in turns), 4),
        "avg_hit_t2": round(sum(hits) / len(hits) * 100, 1) if hits else 0.0,
        "avg_sec": round(sum(t["sec"] for t in turns) / len(turns), 1)
        if turns else 0.0,
        "oracle_full": sum(1 for p in probes
                           if p["hit"] == "full" and not _is_recall(p)),
        "oracle_partial": sum(1 for p in probes
                              if p["hit"] == "partial" and not _is_recall(p)),
        "recall": next((f'{p["matched"]}/{p["total"]}' for p in probes
                        if _is_recall(p)), "-"),
    }
    return {"variant": variant, "session": session, "model": model,
            "turns": turns, "probes": probes, "totals": totals}


def _is_recall(scored: Dict) -> bool:
    return scored["turn"] == next(p.turn for p in PROBES if p.recall)


def _gen_user(client: httpx.Client, setting: str, last_reply: str,
              beat: str) -> str:
    sys_p = ("너는 RP에서 유저(1인칭 남성, 이름 한결) 역할을 연기하는 시뮬레이터다. "
             "작품 설정과 직전 장면에 자연스럽게 이어지는 유저의 다음 발화 하나만 출력한다. "
             "지문과 대사 혼합 가능, 3문장 이내. 메타 발언·설명 금지.\n\n[작품 설정 요약]\n"
             + setting)
    usr_p = (f"[직전 캐릭터 응답]\n{last_reply[-800:]}\n\n[이번 턴 지시]\n{beat}\n\n"
             "유저 발화:")
    try:
        r = client.post("/chat/completions", json={
            "model": SIM_MODEL, "max_tokens": 250,
            "messages": [{"role": "system", "content": sys_p},
                         {"role": "user", "content": usr_p}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip() or beat
    except Exception as e:                     # noqa: BLE001 — 벤치는 계속 돈다
        print(f"  (sim fail: {e} — 비트 원문 사용)", flush=True)
        return beat


def _call(variant: str, session: str, key: str, msgs: List[Dict]) -> Dict:
    t0 = time.time()
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session},
                       json={"model": MODEL, "max_tokens": 300,
                             "messages": msgs})
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "max_tokens": 300,
                             "messages": msgs, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    return {"reply": d["choices"][0]["message"]["content"],
            "prompt": u.get("prompt_tokens", 0),
            "cached": det.get("cached_tokens", 0),
            "write": det.get("cache_write_tokens", 0),
            "cost": u.get("cost", 0.0),
            "sec": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("--session", required=True)
    ap.add_argument("--script", help="동결 대본 JSON — 없으면 시뮬레이터 생성")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    sess_dir = DATA / args.session
    if sess_dir.exists():
        if not args.reset:
            raise SystemExit(f"{sess_dir} 이미 있음 — --reset 또는 다른 세션 ID")
        shutil.rmtree(sess_dir)

    card = load_card(args.card, USER_NAME)
    key = _key()
    scripted = {t["turn"]: t["user_text"]
                for t in load_script(args.script)} if args.script else {}
    sim = (None if scripted else
           httpx.Client(base_url=UPSTREAM, timeout=60,
                        headers={"Authorization": f"Bearer {key}"}))

    history: List[Dict] = []
    last_reply = card.greeting or "(첫 장면)"
    turns: List[Dict] = []
    frozen: List[Dict] = []
    for i, beat in enumerate(BEATS):
        utext = scripted.get(i) or _gen_user(
            sim, card.description[:2500], last_reply, beat)
        frozen.append({"turn": i, "user_text": utext})
        history.append({"role": "user", "content": utext})
        msgs = prepare_request(args.variant, card, history)
        st = _call(args.variant, args.session, key, msgs)
        history.append({"role": "assistant", "content": st["reply"]})
        last_reply = st["reply"]
        hit = st["cached"] / st["prompt"] * 100 if st["prompt"] else 0
        print(f"T{i + 1:02d} prompt={st['prompt']} cached={st['cached']}"
              f" ({hit:.0f}%) ${st['cost']:.4f} {st['sec']}s", flush=True)
        turns.append({"turn": i, "user": utext, **st})
        if args.variant == "dreaming" and i in PAUSES:
            print(f"-- idle {PAUSES[i]}s (dream) --", flush=True)
            time.sleep(PAUSES[i])

    if args.variant == "dreaming":
        cursor = sess_dir / "dreamer" / "cursor.json"
        deadline = time.time() + 120
        while time.time() < deadline:
            if cursor.is_file() and \
                    json.loads(cursor.read_text())["next_turn"] >= len(BEATS) - 1:
                break
            time.sleep(3)

    if not args.script:
        freeze_script(EVAL_DIR / f"script-{args.session}.json", frozen)
        print(f"대본 동결 → eval/script-{args.session}.json", flush=True)

    result = build_result(args.variant, args.session, MODEL, turns)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"result-{args.session}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    t = result["totals"]
    print(f"[{args.variant}] full={t['oracle_full']}/5 "
          f"partial={t['oracle_partial']} recall={t['recall']} "
          f"${t['cost']} hit={t['avg_hit_t2']}% {t['avg_sec']}s/turn",
          flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: 11 passed. 추가: `python3 -m benchmarks.eval.run --help` 정상 출력.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/eval/run.py tests/test_eval_harness.py
git commit -m "feat(eval): 변형 드라이버 — 대본 생성/재생 + usage 계측 + 결과 JSON (스펙 §9)"
```

---

### Task 5: 비교 리포트 (`report.py`)

**Files:**
- Create: `benchmarks/eval/report.py`
- Test: `tests/test_eval_harness.py` (추가)

**Interfaces:**
- Consumes: Task 4 결과 JSON 스키마.
- Produces: `render_report(results: List[Dict]) -> str` (마크다운 표 + 프로브 응답 원문 절), CLI `python3 -m benchmarks.eval.report dreaming_data/eval/result-*.json`.
- 표 (The Seed 표 포맷 — §9 병기 지표): variant / oracle(full·partial/5) / recall / $ / cache% / sec.

- [ ] **Step 1: 실패하는 테스트 작성 (추가)**

```python
from benchmarks.eval.report import render_report


def test_render_report_table_and_audit_section():
    res = [{"variant": "dreaming", "session": "a", "model": "m",
            "turns": [],
            "probes": [{"label": "이름·나이", "hit": "full", "matched": 2,
                        "total": 2, "turn": 21, "reply": "한결님이시죠"}],
            "totals": {"cost": 0.21, "avg_hit_t2": 93.0, "avg_sec": 2.1,
                       "oracle_full": 1, "oracle_partial": 0,
                       "recall": "4/5"}}]
    md = render_report(res)
    assert "| dreaming |" in md and "4/5" in md and "93.0" in md
    assert "한결님이시죠" in md                     # 수동 감사용 원문 병기
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError` (report)

- [ ] **Step 3: 구현**

`benchmarks/eval/report.py`:

```python
"""변형별 결과 JSON → 비교 표 (스펙 §9 병기 지표 + 수동 감사용 원문).

usage: python3 -m benchmarks.eval.report dreaming_data/eval/result-*.json
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

_MARK = {"full": "○", "partial": "△", "miss": "×"}


def render_report(results: List[Dict]) -> str:
    lines = ["| variant | oracle(full/5) | partial | recall | $ | cache% | sec/turn |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: x["variant"]):
        t = r["totals"]
        lines.append(
            f"| {r['variant']} | {t['oracle_full']}/5 | {t['oracle_partial']} "
            f"| {t['recall']} | {t['cost']} | {t['avg_hit_t2']} "
            f"| {t['avg_sec']} |")
    lines.append("")
    for r in sorted(results, key=lambda x: x["variant"]):
        lines.append(f"## {r['variant']} — 프로브 응답 원문 (수동 감사용)")
        for p in r["probes"]:
            lines.append(f"- T{p['turn'] + 1:02d} {p['label']} "
                         f"{_MARK[p['hit']]} ({p['matched']}/{p['total']}): "
                         f"{' '.join(p['reply'].split())[:300]}")
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [json.loads(pathlib.Path(p).read_text()) for p in argv]
    print(render_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: 통과 확인**

Run: `python3 -m pytest "tests/test_eval_harness.py" -x -q; echo EXIT=$?`
Expected: 12 passed

- [ ] **Step 5: 전체 회귀 + Commit**

Run: `python3 -m pytest -q 2>&1 | tail -3` — 기존 504 + 신규 12 전부 green 확인 (exit는 `${pipestatus[1]}`).

```bash
git add benchmarks/eval/report.py tests/test_eval_harness.py
git commit -m "feat(eval): 변형 비교 리포트 — 오라클·비용·캐시율 표 + 감사용 원문 (스펙 §9)"
```

---

### Task 6: 라이브 4-변형 실행 + 리포트 (검증, 코드 변경 없음)

**Files:** 없음 (실행만). 산출물은 전부 `dreaming_data/eval/` (커밋 금지).

전제: 프록시 기동(`.env` 키, 포트 8787), REALM 카드 경로는 기존 벤치와 동일한 `dreaming_data/` 아래 charx. 예상 비용 ~$1 (haiku 30턴 × 4 + Flash 시뮬레이터 30콜).

- [ ] **Step 1: dreaming 변형 (대본 생성 + 동결)**

```bash
python3 -m benchmarks.eval.run <card.charx> dreaming --session ev1 --reset
```

확인: 대본 동결 메시지, cache hit이 후반 turn에서 85%+ 유지, oracle 요약 라인.

- [ ] **Step 2: 대조군 3종 (동결 대본 재생)**

```bash
python3 -m benchmarks.eval.run <card.charx> vanilla --session ev1-van --script dreaming_data/eval/script-ev1.json --reset
python3 -m benchmarks.eval.run <card.charx> trim --session ev1-trim --script dreaming_data/eval/script-ev1.json --reset
python3 -m benchmarks.eval.run <card.charx> retrieval --session ev1-ret --script dreaming_data/eval/script-ev1.json --reset
```

- [ ] **Step 3: 리포트 생성 + 판독**

```bash
python3 -m benchmarks.eval.report dreaming_data/eval/result-ev1*.json
```

기대 서열 (스펙 §9 정당화 조건): oracle에서 dreaming ≥ retrieval > trim, vanilla는 oracle 높되 $·prompt 토큰 최다. dreaming이 retrieval에 지면 4타입 스키마 정당화 실패 — 결과 그대로 보고 (좋게 포장 금지). 프로브 응답 원문 전부 육안 감사.

- [ ] **Step 4: 결과 요약을 커밋 메시지 없이 대화로 보고** — 수치 표 + 감사 소견. 코퍼스/결과 JSON은 커밋하지 않는다.

---

## Self-Review

- **스펙 커버리지**: §9 드라이버(대본+지뢰) → Task 1; 결정론 오라클 → Task 2; 대조군(순정/trim/retrieval/Dreaming) → Task 3~4; 병기 지표 → Task 4~5; 수동 감사 → Task 5 원문 병기 + Task 6 Step 3. 비스코프 명시: HypaV3 대조군(클라이언트 필요), LLM judge(서사), 거리 기반 재질문의 "10막 80비트" 전체 규모(30비트 축소판).
- **플레이스홀더**: 없음 — 전 코드 블록 완결.
- **타입 일관성**: `Probe(turn, label, expect, recall)` Task 1 정의 = Task 2/4 사용 일치. `prepare_request(variant, card, history)` Task 3 = Task 4 호출 일치. 결과 JSON 스키마 Task 4 `build_result` = Task 5 `render_report` 소비 일치. `score_reply` 반환 dict에 Task 4가 `turn`/`reply` 병합 — Task 5 `_MARK[p['hit']]` 접근 일치.
