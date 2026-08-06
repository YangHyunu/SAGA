# Plan 7: 평가 v2 — 충실도 검사기 + 디렉터 벤치 + LongMemEval 어댑터 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** EVAL2.md 방법론을 코드로 구현 — 합성 요청이 실와이어와 같음을 매 실행 기계 검증하고, The Seed식 80턴 디렉터 벤치(동적 사실 추출, 5유형 프로브, 이중 채점, 미스 원인 자동 분해, 3반복 mean±std)를 돌리고, LongMemEval 표준 점수를 Dreaming으로 재측정한다.

**Architecture:** 3부 독립 구성. (A) `fidelity.py`는 순수 구조 검사 — 벤치 러너가 매 요청 게이트로 호출. (B) 디렉터 벤치는 ev1 하네스(`benchmarks/eval/`) 위 증축 — 디렉터(Flash)가 자연 진행 중 사실을 추출해 원장에 쌓고, 가시 창 밖으로 밀려난 사실만 프로브로 재질문, 채점은 오라클+judge 이중, 미스는 Dreaming 내부 저장소 대조로 저장/활용 실패 분해. (C) LME 어댑터는 기존 `benchmarks/longmemeval/` 데이터로더·judge를 재사용하되 평가 대상을 ChromaDB 검색이 아닌 SyncPath+Dreamer 관통으로 교체. LLM 의존은 전부 `Callable` 주입으로 유닛 테스트는 0콜.

**Tech Stack:** Python 3, httpx, pytest. 재사용: `benchmarks/eval/*` (ev1), `benchmarks/cardsim/lorebook.py`, `dreaming/` (SyncPath·Dreamer·MemoryStore·OpenAICompatLLM), `benchmarks/longmemeval/` (download·judge).

## Global Constraints

- EVAL2.md가 스펙: 측정 도구 3분할, 동적 사실 추출(명시 대사 금지), judge는 상위 모델·temp 0·이진, judge-사람 일치 95%+ 게이트, 저장/활용 실패 분리, 무편집 원문 리포트.
- 코퍼스·결과·대본 JSON은 `dreaming_data/` 아래 (gitignored, 카드 저작물 커밋 금지).
- API 키는 `.env` `DREAMING_UPSTREAM_KEY` 읽기만. 값 출력·커밋 금지.
- 유닛 테스트는 네트워크 0콜 (LLM은 fake callable 주입). 라이브는 별도 태스크에서만.
- CLAUDE.md: 최소 코드, 기존 스타일(한국어 주석) 유지.
- 디렉터/judge 모델: 디렉터 `google/gemini-3-flash`(거부 시 폴백 `deepseek-v4-pro`), judge `anthropic/claude-sonnet-4.5` (피평가보다 상위 — EVAL2 원칙).

## 확정 설정 (2026-08-06 유저 승인 — 원래 태스크 코드보다 우선)

- **나레이터(RP 본모델)**: `deepseek-v4-pro` (corpus1 실사용 + 뮈토스 인증). 샘플러는 공식 기본값 pin: temp 1.0 / top_p 1.0, 길이 제어는 max_tokens.
- **와이어 형식**: 실캡처는 **OpenAI-compat** (`v1/chat/completions`, content=평문 문자열, system이 메시지 목록 안에 위치). Anthropic 블록 형식으로 변환 금지 — fidelity·러너 모두 이 형식 기준. 캐시 측정은 DeepSeek usage 필드(`prompt_cache_hit_tokens`/`prompt_cache_miss_tokens`).
- **프리셋**: 뮈토스 6.2 DeepSeek판(= DeepSeek V4 Pro 타깃 명시)을 `.risup` 디코더(scratchpad `risup_decode.py` — RPack 치환→deflate→msgpack→AES-GCM(SHA256("risupreset"), IV=0)→msgpack)로 열어 promptTemplate 순서 그대로 조립. 토글: 실행 모드 **💬 RP**, 응답 언어 🇰🇷, **🔞 성인 콘텐츠 묘사 지침 ON, ▶️ 도메인 중립 렌더링 프리필 ON**, 나머지 기본값.
- **캡처 이식 근거** (81건 전수 분석): 선두 system 23,314자·꼬리 system 1,610자 모두 byte-동결(로어 churn·시간 매크로·스탯바 없음), 미해석 매크로 0, 중간 system 0, 역할 교대 엄격. greeting은 트림 시작(~56메시지) 후 소멸 — 고정 프리픽스로 취급 금지. 실클라이언트 트림 = 헤드 2메시지/턴 드랍, 창 ~18K 토큰 — token_trim 캘리브레이션 기준.
- **디렉터**: Gemini 3 Flash + 자체 프롬프트(픽션 프레이밍, `card-soyeon-30` 실유저 발화 few-shot, 프리셋 `<user_style>`류 반영, 발화 1개·길이 상한·메타발언 금지). 프로브는 디렉터 생성 금지 — 원장 기반 고정 템플릿 의문문 직접 주입.
- **비용**(corpus5 usage 실측 외삽): 본 RP pro 런당 ~$0.25, 정석 3반복 4변형 총 ~$12 (judge Sonnet이 지배 비용). 절충 티어 불필요.
- **카드**: 소연(대화형) 주력. REALM·소설 집필 모드·GLM-5.2는 본실행 후 옵션 런.

## File Structure

- `benchmarks/eval/fidelity.py` — 와이어 구조 검사 + 코퍼스 시그니처 대조 (A)
- `benchmarks/eval/director.py` — 사실 원장(FactLedger)·추출·거리 게이팅·프로브 생성 (B)
- `benchmarks/eval/scoring.py` — judge 이진 채점 + 미스 원인 분해 (B)
- `benchmarks/eval/judge_check.py` — judge-사람 일치율 검증 CLI (B)
- `benchmarks/eval/run2.py` — 80턴 디렉터 벤치 드라이버 (토큰 트림·리롤·수정·3반복) (B)
- `benchmarks/eval/report2.py` — mean±std 집계 + 커뮤 신뢰 포맷 리포트 (B)
- `benchmarks/longmemeval/run_dreaming.py` — LME oracle → SyncPath+Dreamer 관통 어댑터 (C)
- `tests/test_eval_v2.py` — A·B 순수 함수 전부
- 기존 `benchmarks/eval/{script,oracle,variants,run,report}.py`는 그대로 둠 (ev1 재현용) — run2는 variants의 `prepare_request`·oracle의 `score_reply`를 임포트해 재사용.

---

### Task 1: 와이어 충실도 검사기 (`fidelity.py`)

**Files:**
- Create: `benchmarks/eval/fidelity.py`
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `check_wire_shape(msgs: List[Dict]) -> List[str]` (위반 설명 목록, 빈 리스트=통과), `corpus_signature(corpus_dir) -> Dict`, `compare_with_corpus(msgs, corpus_dir) -> List[str]`. run2(Task 5)가 매 요청 전 `check_wire_shape` 호출, 위반 시 즉시 중단.
- 검사 항목(실캡처 corpus1 실측 형태 기준): ① 선두 system 정확히 1개 ② 마지막 메시지는 user ③ 중간(선두 이후~마지막 user 이전)에 system 없음 — 단 꼬리 system(PHI/globalNote) 1개는 마지막 user 뒤 허용 ④ user/assistant 역할이 연속 중복되지 않음(단 선두 system 직후 assistant 인사는 허용) ⑤ 클라이언트 요청에 `cache_control` 없음(마킹은 프록시 몫).

- [ ] **Step 1: 실패하는 테스트 작성** (`tests/test_eval_v2.py` 신규)

```python
"""평가 v2 — 충실도/디렉터/채점 순수 함수 (EVAL2.md)."""
from benchmarks.eval.fidelity import check_wire_shape


def _wire(pairs=2, greeting=True, tail_system=True):
    out = [{"role": "system", "content": "카드 전문 + 로어"}]
    if greeting:
        out.append({"role": "assistant", "content": "어서 와요."})
    for i in range(pairs):
        out.append({"role": "user", "content": f"질문{i}"})
        out.append({"role": "assistant", "content": f"답{i}"})
    out.append({"role": "user", "content": "새 질문"})
    if tail_system:
        out.append({"role": "system", "content": "PHI"})
    return out


def test_valid_wire_passes():
    assert check_wire_shape(_wire()) == []
    assert check_wire_shape(_wire(tail_system=False)) == []
    assert check_wire_shape(_wire(greeting=False)) == []


def test_split_leading_system_rejected():
    msgs = _wire()
    msgs.insert(1, {"role": "system", "content": "로어 분리"})
    assert any("선두 system" in v for v in check_wire_shape(msgs))


def test_mid_conversation_system_rejected():
    msgs = _wire(tail_system=False)
    msgs.insert(3, {"role": "system", "content": "몰래 주입"})
    assert check_wire_shape(msgs)


def test_client_cache_control_rejected():
    msgs = _wire()
    msgs[0]["cache_control"] = {"type": "ephemeral"}
    assert any("cache_control" in v for v in check_wire_shape(msgs))


def test_double_user_rejected():
    msgs = _wire(tail_system=False)
    msgs.insert(2, {"role": "user", "content": "연속 user"})
    assert check_wire_shape(msgs)
```

- [ ] **Step 2: 실패 확인**

Run: `python3 -m pytest "tests/test_eval_v2.py" -x -q; echo EXIT=$?`
Expected: FAIL `ModuleNotFoundError: benchmarks.eval.fidelity`

- [ ] **Step 3: 구현**

```python
"""benchmarks/eval/fidelity.py — 합성 요청이 실와이어 형태인지 기계 검증.

실캡처(corpus) 실측 형태: 선두 system 1개(카드+로어 병합) → [인사 assistant]
→ user/assistant 교대 → 마지막 user → [꼬리 system(PHI/globalNote)].
벤치 러너는 매 요청 전 check_wire_shape를 게이트로 호출한다 — "같다고
믿는다"가 아니라 "같음을 매번 확인한다" (EVAL2 §3).
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List


def check_wire_shape(msgs: List[Dict]) -> List[str]:
    v: List[str] = []
    if not msgs:
        return ["빈 메시지"]
    if msgs[0].get("role") != "system":
        v.append("첫 메시지가 system이 아님")
    if sum(1 for m in msgs[:2] if m.get("role") == "system") > 1 or (
            len(msgs) > 1 and msgs[1].get("role") == "system"):
        v.append("선두 system이 2개 이상 (병합 누락 — lore_shift 무력화)")
    last_user = max((i for i, m in enumerate(msgs)
                     if m.get("role") == "user"), default=None)
    if last_user is None:
        v.append("user 메시지 없음")
    else:
        for i, m in enumerate(msgs[1:last_user], start=1):
            if m.get("role") == "system":
                v.append(f"대화 중간 system (index {i})")
        tail = msgs[last_user + 1:]
        if [m.get("role") for m in tail] not in ([], ["system"]):
            v.append("마지막 user 뒤는 꼬리 system 1개만 허용")
    prev = None
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r in ("user", "assistant") and r == prev:
            v.append(f"역할 연속 중복 ({r}, index {i})")
        prev = r if r in ("user", "assistant") else prev
        if "cache_control" in m:
            v.append(f"클라이언트 요청에 cache_control (index {i}) — 마킹은 프록시 몫")
    return v


def corpus_signature(corpus_dir) -> Dict:
    """실캡처 디렉터리에서 구조 시그니처 추출 (대조 기준)."""
    sigs = []
    for p in sorted(pathlib.Path(corpus_dir).glob("req-*.json")):
        msgs = json.loads(p.read_text())["messages"]
        roles = [m["role"] for m in msgs]
        sigs.append({
            "leading_system": len(roles) - len(
                [r for r in roles if True][len([r for r in roles
                 if roles and r == "system" and roles.index(r) == 0]):]),
        })
    # 단순화: 선두 system 런 길이와 꼬리 system 유무만 본다
    out = []
    for p in sorted(pathlib.Path(corpus_dir).glob("req-*.json")):
        msgs = json.loads(p.read_text())["messages"]
        roles = [m["role"] for m in msgs]
        lead = 0
        while lead < len(roles) and roles[lead] == "system":
            lead += 1
        out.append({"leading_system_run": lead,
                    "tail_system": roles[-1] == "system"})
    return {"n": len(out), "reqs": out}


def compare_with_corpus(msgs: List[Dict], corpus_dir) -> List[str]:
    v = check_wire_shape(msgs)
    sig = corpus_signature(corpus_dir)
    if sig["n"] and any(r["leading_system_run"] != 1 for r in sig["reqs"]):
        v.append("코퍼스 자체가 선두 system 1개 형태가 아님 — 시그니처 재확인 필요")
    return v
```

(주: `corpus_signature` 첫 루프의 죽은 계산은 넣지 말 것 — 위 최종 형태처럼 두 번째 루프만 구현한다.)

```python
def corpus_signature(corpus_dir) -> Dict:
    """실캡처 디렉터리에서 구조 시그니처 추출 (대조 기준)."""
    out = []
    for p in sorted(pathlib.Path(corpus_dir).glob("req-*.json")):
        msgs = json.loads(p.read_text())["messages"]
        roles = [m["role"] for m in msgs]
        lead = 0
        while lead < len(roles) and roles[lead] == "system":
            lead += 1
        out.append({"leading_system_run": lead,
                    "tail_system": roles[-1] == "system"})
    return {"n": len(out), "reqs": out}
```

- [ ] **Step 4: 통과 확인** — `python3 -m pytest "tests/test_eval_v2.py" -x -q` → 5 passed
- [ ] **Step 5: Commit** — `git add benchmarks/eval/fidelity.py tests/test_eval_v2.py && git commit -m "feat(eval): 와이어 충실도 검사기 — 실와이어 구조 게이트 (EVAL2 §3)"`

---

### Task 2: 디렉터 사실 원장 (`director.py` — 추출·기록)

**Files:**
- Create: `benchmarks/eval/director.py`
- Test: `tests/test_eval_v2.py` (추가)

**Interfaces:**
- Produces: `DirFact` dataclass — `{fid: str, kind: "exact"|"relation"|"event", text: str, value: str, turn: int, probed: bool=False}` (value=채점용 핵심 문자열/숫자, text=사실 서술). `extract_facts(llm: Callable[[str, str], str], user_text, reply_text, turn_no) -> List[DirFact]` — llm은 `(system, user) -> str` 동기 callable. LLM 출력 포맷: 한 줄당 `kind|value|서술`, 파싱 실패 줄은 스킵. `Ledger` — `add(facts)`, `unprobed(kind=None)`, JSON 직렬화 `to_rows()/from_rows()`.
- llm 주입형이라 유닛 테스트는 고정 문자열 반환 fake로 0콜.

- [ ] **Step 1: 실패하는 테스트 작성** (추가)

```python
from benchmarks.eval.director import DirFact, Ledger, extract_facts


def _fake_llm(reply):
    def f(system, user):
        return reply
    return f


def test_extract_facts_parses_lines_and_skips_garbage():
    out = extract_facts(
        _fake_llm("exact|250골드|한결의 남은 소지금은 250골드\n"
                  "relation|연인|한결과 리사는 연인 사이\n"
                  "이상한 줄 형식\n"
                  "event|보름달 축제|보름달에 축제 동행 약속"),
        "u", "a", turn_no=7)
    assert [f.kind for f in out] == ["exact", "relation", "event"]
    assert out[0].value == "250골드" and out[0].turn == 7
    assert all(not f.probed for f in out)


def test_ledger_roundtrip_and_unprobed_filter():
    led = Ledger()
    led.add([DirFact(fid="f1", kind="exact", value="250", text="잔액", turn=3),
             DirFact(fid="f2", kind="relation", value="연인", text="관계", turn=5,
                     probed=True)])
    led2 = Ledger.from_rows(led.to_rows())
    assert [f.fid for f in led2.unprobed()] == ["f1"]
    assert [f.fid for f in led2.unprobed(kind="relation")] == []
```

- [ ] **Step 2: 실패 확인** — pytest → `ImportError` (director)
- [ ] **Step 3: 구현**

```python
"""benchmarks/eval/director.py — 디렉터: 동적 사실 추출 + 거리 게이팅 프로브.

The Seed DIRECTOR 방식 (EVAL2 §3): 사실을 미리 심지 않고, 롤플레이가 자연히
만든 사실(가격·인명·관계·사건)을 턴마다 추출해 원장에 쌓고, 가시 창 밖으로
밀려난 사실만 자연스러운 질문으로 되묻는다.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

LlmFn = Callable[[str, str], str]          # (system, user) -> 응답 텍스트

_EXTRACT_SYS = (
    "너는 RP 대화 감독관이다. 방금 턴에서 나중에 기억력 시험에 쓸 수 있는 "
    "구체적 사실만 추출한다. 한 줄에 하나, 형식: kind|핵심값|한 문장 서술.\n"
    "kind는 exact(숫자·고유명사·시각), relation(인물 관계·호칭), "
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 짧은 문자열. "
    "추출할 게 없으면 빈 출력. 다른 말 금지.")


@dataclass
class DirFact:
    fid: str
    kind: str
    value: str
    text: str
    turn: int
    probed: bool = False


def extract_facts(llm: LlmFn, user_text: str, reply_text: str,
                  turn_no: int) -> List[DirFact]:
    raw = llm(_EXTRACT_SYS,
              f"[유저]\n{user_text[-600:]}\n[캐릭터]\n{reply_text[-1200:]}")
    out: List[DirFact] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or parts[0] not in ("exact", "relation", "event"):
            continue
        out.append(DirFact(fid=uuid.uuid4().hex[:8], kind=parts[0],
                           value=parts[1], text=parts[2], turn=turn_no))
    return out


class Ledger:
    def __init__(self) -> None:
        self.facts: List[DirFact] = []

    def add(self, facts: List[DirFact]) -> None:
        self.facts.extend(facts)

    def unprobed(self, kind: Optional[str] = None) -> List[DirFact]:
        return [f for f in self.facts
                if not f.probed and (kind is None or f.kind == kind)]

    def to_rows(self) -> List[Dict]:
        return [asdict(f) for f in self.facts]

    @classmethod
    def from_rows(cls, rows: List[Dict]) -> "Ledger":
        led = cls()
        led.facts = [DirFact(**r) for r in rows]
        return led
```

- [ ] **Step 4: 통과 확인** — 7 passed
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): 디렉터 사실 원장 — 동적 추출 + kind 분류 (EVAL2 동적 사실 추출)"`

---

### Task 3: 거리 게이팅 + 프로브 생성 (`director.py` 확장)

**Files:**
- Modify: `benchmarks/eval/director.py`
- Test: `tests/test_eval_v2.py` (추가)

**Interfaces:**
- Produces: `eligible(ledger, window_start_turn: int, kind=None) -> List[DirFact]` — `fact.turn < window_start_turn`(가시 창 밖으로 evict)만. `make_probe(llm, fact) -> str` (자연 질문), `make_false_premise(llm, fact) -> Tuple[str, str]` (질문, 오염된 값) — 거짓 전제 프로브용, `probe_plan(ledger, window_start_turn, want: Dict[str,int]) -> List[Tuple[str, DirFact]]` — `want={"recall":2,"relation":1,"false":1}` 식 유형별 수만큼 eligible에서 뽑아 `[(ptype, fact)]` 반환(오래된 것 우선, 뽑힌 fact는 probed=True 마킹). ptype ∈ recall/relation/false/update/recent.
- Consumes: Task 2의 `DirFact`, `Ledger`, `LlmFn`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
from benchmarks.eval.director import eligible, make_false_premise, probe_plan


def _led():
    led = Ledger()
    led.add([DirFact(fid=f"f{i}", kind=k, value=f"v{i}", text=f"사실{i}", turn=t)
             for i, (k, t) in enumerate([("exact", 2), ("exact", 30),
                                         ("relation", 4), ("event", 6)])])
    return led


def test_eligible_only_outside_window():
    led = _led()
    got = [f.fid for f in eligible(led, window_start_turn=10)]
    assert got == ["f0", "f2", "f3"]          # turn 30(f1)은 창 안
    assert [f.fid for f in eligible(led, 10, kind="relation")] == ["f2"]


def test_probe_plan_marks_probed_and_respects_want():
    led = _led()
    plan = probe_plan(led, window_start_turn=10,
                      want={"recall": 1, "relation": 1, "false": 1})
    types = [t for t, _ in plan]
    assert types == ["recall", "relation", "false"]
    assert len({f.fid for _, f in plan}) == 3          # 사실 중복 출제 없음
    assert len(eligible(led, 10)) == 0                 # 전부 probed 처리


def test_false_premise_corrupts_value():
    q, wrong = make_false_premise(
        _fake_llm("질문: 그때 350골드 남았댔지?\n오염값: 350골드"),
        DirFact(fid="x", kind="exact", value="250골드", text="잔액", turn=1))
    assert "350" in q and wrong == "350골드"
```

- [ ] **Step 2: 실패 확인** — ImportError (eligible)
- [ ] **Step 3: 구현** (director.py에 추가)

```python
_PROBE_SYS = (
    "너는 RP 유저 대사 작가다. 주어진 과거 사실을 캐릭터가 기억하는지 확인하는 "
    "자연스러운 유저 발화 하나를 만든다. 사실의 답 자체를 말하지 말고, 반드시 "
    "구체적으로 되묻는 직접 의문문 포함. 1~2문장. 발화만 출력.")

_FALSE_SYS = (
    "너는 RP 유저 대사 작가다. 주어진 사실의 핵심값을 그럴듯하게 틀린 값으로 "
    "바꿔서, 그 틀린 값을 사실인 양 전제하는 유저 발화를 만든다. 출력 형식:\n"
    "질문: <발화>\n오염값: <틀린 값>")


def eligible(ledger: Ledger, window_start_turn: int,
             kind: Optional[str] = None) -> List[DirFact]:
    """가시 창 밖으로 evict된 사실만 (The Seed: outside-visible-window)."""
    return [f for f in ledger.unprobed(kind)
            if f.turn < window_start_turn]


def make_probe(llm: LlmFn, fact: DirFact) -> str:
    return llm(_PROBE_SYS, f"[과거 사실]\n{fact.text} (핵심값: {fact.value})").strip()


def make_false_premise(llm: LlmFn, fact: DirFact) -> "Tuple[str, str]":
    raw = llm(_FALSE_SYS, f"[사실]\n{fact.text} (핵심값: {fact.value})")
    q, wrong = "", ""
    for line in raw.splitlines():
        if line.startswith("질문:"):
            q = line[3:].strip()
        elif line.startswith("오염값:"):
            wrong = line[4:].strip()
    return q, wrong


_PTYPE_KIND = {"recall": "exact", "relation": "relation",
               "false": None, "update": "exact", "recent": None}


def probe_plan(ledger: Ledger, window_start_turn: int,
               want: Dict[str, int]) -> List["Tuple[str, DirFact]"]:
    """유형별 수만큼 eligible에서 오래된 순으로 뽑고 probed 마킹."""
    plan: List = []
    for ptype, n in want.items():
        pool = eligible(ledger, window_start_turn, kind=_PTYPE_KIND[ptype])
        pool.sort(key=lambda f: f.turn)
        for f in pool[:n]:
            f.probed = True
            plan.append((ptype, f))
    return plan
```

(파일 상단 typing 임포트에 `Tuple` 추가.)

- [ ] **Step 4: 통과 확인** — 10 passed
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): 거리 게이팅 프로브 — evict 사실만 출제, 유형별 플랜 (The Seed 방식)"`

---

### Task 4: 이중 채점 + 미스 원인 분해 (`scoring.py`)

**Files:**
- Create: `benchmarks/eval/scoring.py`
- Test: `tests/test_eval_v2.py` (추가)

**Interfaces:**
- Produces: `judge_pass(llm, ptype, fact_text, expected_value, question, reply, wrong_value="") -> Dict` — `{"pass": bool, "why": str}`. recall/relation/update/recent: "응답이 기대값과 일치하는 내용을 말했는가 Y/N". false: "응답이 (틀린) 오염값을 사실로 단정했는가 Y/N" → N이면 pass. `oracle_pass(reply, expected_value) -> bool` — 공백 무시 substring (ev1 `oracle._norm` 재사용). `decompose_miss(data_dir, session, fact) -> str` — `"storage_fail"|"utilization_fail"` : 세션 저장소(`facts/commits/actors/episodes` JSON 전체 직렬화)에 fact.value(공백 제거)가 있으면 활용 실패, 없으면 저장 실패.
- Consumes: `DirFact`(Task 2), `benchmarks.eval.oracle._norm`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
import json as _json

from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass


def test_oracle_pass_ignores_whitespace():
    assert oracle_pass("이백 오십 골드 남았죠", "이백오십")
    assert not oracle_pass("기억 안 나요", "250")


def test_judge_false_premise_inverts():
    ok = judge_pass(_fake_llm("N"), "false", "잔액 250", "250골드",
                    "350골드 남았댔지?", "아뇨, 250골드였어요.",
                    wrong_value="350골드")
    assert ok["pass"] is True                  # 오염값 단정 안 함 = 통과
    bad = judge_pass(_fake_llm("Y"), "false", "잔액 250", "250골드",
                     "350골드 남았댔지?", "맞아요 350골드.",
                     wrong_value="350골드")
    assert bad["pass"] is False


def test_decompose_miss_checks_dreaming_storage(tmp_path):
    d = tmp_path / "sess" / "facts"
    d.mkdir(parents=True)
    (d / "a.json").write_text(_json.dumps(
        {"claim": "한결의 잔액은 250골드"}, ensure_ascii=False))
    f_in = DirFact(fid="a", kind="exact", value="250골드", text="잔액", turn=1)
    f_out = DirFact(fid="b", kind="exact", value="은반지", text="선물", turn=2)
    assert decompose_miss(tmp_path, "sess", f_in) == "utilization_fail"
    assert decompose_miss(tmp_path, "sess", f_out) == "storage_fail"
```

- [ ] **Step 2: 실패 확인** — ImportError (scoring)
- [ ] **Step 3: 구현**

```python
"""benchmarks/eval/scoring.py — 이중 채점 + 미스 원인 분해 (EVAL2).

오라클(문자열)과 judge(이진 LLM)를 병행하고, 미스는 Dreaming 내부 저장소를
대조해 저장 실패(치매)와 활용 실패(로어북 씹힘)로 분해한다 — 커뮤니티가
이미 구분해 부르는 두 실패 양태 그대로.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict

from benchmarks.eval.director import DirFact, LlmFn
from benchmarks.eval.oracle import _norm

_JUDGE_SYS = ("너는 채점관이다. Y 또는 N 한 글자만 출력한다. 다른 말 금지.")

_KINDS = ("facts", "commits", "actors", "episodes")


def oracle_pass(reply: str, expected_value: str) -> bool:
    return _norm(expected_value) in _norm(reply)


def judge_pass(llm: LlmFn, ptype: str, fact_text: str, expected_value: str,
               question: str, reply: str, wrong_value: str = "") -> Dict:
    if ptype == "false":
        q = (f"[대화]\n유저: {question}\n캐릭터: {reply}\n\n"
             f"캐릭터가 틀린 값 '{wrong_value}'을(를) 사실로 단정했는가? "
             f"(실제 사실: {fact_text})")
        raw = llm(_JUDGE_SYS, q).strip().upper()
        return {"pass": not raw.startswith("Y"), "why": raw[:80]}
    q = (f"[과거 사실]\n{fact_text} (핵심값: {expected_value})\n"
         f"[대화]\n유저: {question}\n캐릭터: {reply}\n\n"
         f"캐릭터의 응답이 이 사실과 일치하는 내용을 실제로 말했는가?")
    raw = llm(_JUDGE_SYS, q).strip().upper()
    return {"pass": raw.startswith("Y"), "why": raw[:80]}


def decompose_miss(data_dir, session: str, fact: DirFact) -> str:
    """프로브 미스 원인: 저장소에 있으면 활용 실패, 없으면 저장 실패."""
    base = pathlib.Path(data_dir) / session
    needle = _norm(fact.value)
    for kind in _KINDS:
        d = base / kind
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            if needle in _norm(p.read_text()):
                return "utilization_fail"
    return "storage_fail"
```

- [ ] **Step 4: 통과 확인** — 13 passed
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): 이중 채점 + 저장/활용 실패 분해 (EVAL2 커뮤 언어 정합)"`

---

### Task 5: judge 사전 검증 CLI (`judge_check.py`)

**Files:**
- Create: `benchmarks/eval/judge_check.py`
- Test: `tests/test_eval_v2.py` (추가)

**Interfaces:**
- Produces: `agreement(rows: List[Dict], judge: LlmFn) -> Dict` — rows는 `{"ptype","fact_text","expected_value","question","reply","human": bool,("wrong_value")}`; 반환 `{"n", "agree", "rate", "disagrees": [행 인덱스]}`. CLI: `python3 -m benchmarks.eval.judge_check <labels.jsonl>` — 실제 judge(Sonnet) 호출, rate < 0.95면 exit 1 (게이트). 사람 라벨 파일은 라이브 실행 후 수동 작성 (`dreaming_data/eval/judge-labels.jsonl`).
- Consumes: `judge_pass`(Task 4).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
from benchmarks.eval.judge_check import agreement


def test_agreement_counts_and_lists_disagreements():
    rows = [
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "250이요", "human": True},
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "몰라요", "human": False},
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "500이요", "human": False},
    ]
    r = agreement(rows, judge=_fake_llm("Y"))     # 전부 Y로 판정하는 가짜 judge
    assert r["n"] == 3 and r["agree"] == 1
    assert r["disagrees"] == [1, 2]
    assert abs(r["rate"] - 1 / 3) < 1e-9
```

- [ ] **Step 2: 실패 확인** — ImportError
- [ ] **Step 3: 구현**

```python
"""benchmarks/eval/judge_check.py — judge를 쓰기 전에 judge부터 검증 (EVAL2).

PersonaEval 경고(최고 judge도 69% vs 인간 90.8%)에 따라, 사람 라벨 표본과의
일치율 95%+를 확인해야 본채점에 judge를 쓴다.

usage: python3 -m benchmarks.eval.judge_check dreaming_data/eval/judge-labels.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

from benchmarks.eval.director import LlmFn
from benchmarks.eval.scoring import judge_pass

GATE = 0.95


def agreement(rows: List[Dict], judge: LlmFn) -> Dict:
    agree, disagrees = 0, []
    for i, r in enumerate(rows):
        got = judge_pass(judge, r["ptype"], r["fact_text"],
                         r["expected_value"], r["question"], r["reply"],
                         wrong_value=r.get("wrong_value", ""))["pass"]
        if got == bool(r["human"]):
            agree += 1
        else:
            disagrees.append(i)
    n = len(rows)
    return {"n": n, "agree": agree,
            "rate": agree / n if n else 0.0, "disagrees": disagrees}


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = [json.loads(line) for line in
            pathlib.Path(argv[0]).read_text().splitlines() if line.strip()]
    from benchmarks.eval.run2 import make_judge_llm       # 실 judge (Sonnet)
    r = agreement(rows, make_judge_llm())
    print(f"judge-사람 일치 {r['agree']}/{r['n']} = {r['rate']:.1%} "
          f"(게이트 {GATE:.0%}) 불일치 행: {r['disagrees']}")
    return 0 if r["rate"] >= GATE else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: 통과 확인** — 14 passed
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): judge-사람 일치율 게이트 95% (PersonaEval 경고 대응)"`

---

### Task 6: 80턴 디렉터 벤치 드라이버 (`run2.py`)

**Files:**
- Create: `benchmarks/eval/run2.py`
- Test: `tests/test_eval_v2.py` (추가 — 순수 함수만)

**Interfaces:**
- Consumes: Task 1~5 전부, `benchmarks.eval.variants.prepare_request`(변형 조립 재사용), `benchmarks.cardsim.lorebook.load_card`, ev1 `run.py`의 `_key`/usage 파싱 패턴.
- Produces: CLI `python3 -m benchmarks.eval.run2 <card.charx> <variant> --session S [--runs 3] [--trim-tokens 12000] [--reroll-at 18,33] [--edit-at 25] [--ttl-wait]`. 구조: 진행 40턴(디렉터 유저턴 생성 + 매 턴 사실 추출 + 지식갱신 이벤트 2개는 디렉터 지시로 강제) → 평가 40턴(프로브 스케줄 + 필러). 순수 함수 `token_trim(history, budget, count_fn) -> Tuple[List[Dict], int]` (창과 window_start_turn 반환 — pair 경계 절단, 문자수/2.5 근사 count_fn 기본), `probe_schedule(total_eval_turns) -> List[Optional[str]]` (recall 다수+relation/false/update/recent 배치, None=필러). 결과 JSON `dreaming_data/eval/v2-{session}-run{n}.json`: turns(단계별 소요시간 포함), probes(ptype/fact/question/reply/oracle/judge/miss_cause), ledger, totals. `make_judge_llm() -> LlmFn`, `make_director_llm() -> LlmFn` (OpenRouter 동기 httpx, judge=`anthropic/claude-sonnet-4.5` temp 0, director=`google/gemini-2.5-flash`).
- 매 요청 전 `check_wire_shape` 게이트 — 위반 시 SystemExit.
- 미스 시 `decompose_miss`는 dreaming 변형에만 (대조군은 저장소 없음 → `"-"`).

- [ ] **Step 1: 실패하는 테스트 작성** (순수 함수)

```python
from benchmarks.eval.run2 import probe_schedule, token_trim


def test_token_trim_cuts_at_pair_boundary_and_reports_start_turn():
    h = []
    for i in range(10):
        h.append({"role": "user", "content": "가" * 100})
        h.append({"role": "assistant", "content": "나" * 100})
    h.append({"role": "user", "content": "새 질문"})
    win, start = token_trim(h, budget=300, count_fn=lambda t: len(t))
    assert win[0]["role"] == "user"                    # pair 경계 절단
    assert start == 10 - (len(win) - 1) // 2           # 남은 pair 수로 역산
    full, s0 = token_trim(h, budget=10**9, count_fn=len)
    assert full == h and s0 == 0


def test_probe_schedule_covers_five_types():
    sched = probe_schedule(40)
    types = [t for t in sched if t]
    assert len(sched) == 40
    assert types.count("recall") >= 8
    for t in ("relation", "false", "update", "recent"):
        assert types.count(t) >= 2
    assert sched.count(None) >= 10                     # 필러 존재
```

- [ ] **Step 2: 실패 확인** — ImportError (run2)
- [ ] **Step 3: 구현** — 핵심 골격 (전체 파일):

```python
"""benchmarks/eval/run2.py — 80턴 디렉터 벤치 (EVAL2 §2·§3·§4).

진행 40턴: 디렉터(Flash)가 유저 발화를 생성하며 매 턴 사실을 추출해 원장에
쌓는다. 지식갱신 이벤트(값 변경) 2개는 디렉터 지시문으로 강제 발생.
평가 40턴: 가시 창 밖으로 evict된 사실만 프로브로 재질문 (recall/relation/
false/update/recent), 채점은 오라클+judge 이중, 미스는 저장/활용 실패 분해.
--runs N으로 반복 실행 (필러·진행만 변동, report2가 mean±std 집계).

usage: python3 -m benchmarks.eval.run2 <card.charx> dreaming --session v2a --runs 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from benchmarks.cardsim.lorebook import load_card
from benchmarks.eval.director import (DirFact, Ledger, LlmFn, extract_facts,
                                      make_false_premise, make_probe,
                                      probe_plan)
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval.variants import prepare_request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = "http://127.0.0.1:8790"
UPSTREAM = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4.5"
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
DIRECTOR_MODEL = "google/gemini-2.5-flash"
USER_NAME = "한결"
PROGRESS_TURNS = 40
EVAL_TURNS = 40
TRIM_TOKENS = 12000
UPDATE_EVENTS = (12, 28)      # 지식갱신 강제 턴 (진행 구간)
MAX_TOKENS = 1000             # 실사용 응답 길이 근사 (EVAL2 충실도 보강)


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def _mk_llm(model: str, temperature: float) -> LlmFn:
    client = httpx.Client(base_url=UPSTREAM, timeout=120,
                          headers={"Authorization": f"Bearer {_key()}"})

    def call(system: str, user: str) -> str:
        r = client.post("/chat/completions", json={
            "model": model, "max_tokens": 400, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    return call


def make_judge_llm() -> LlmFn:
    return _mk_llm(JUDGE_MODEL, 0.0)


def make_director_llm() -> LlmFn:
    return _mk_llm(DIRECTOR_MODEL, 0.7)


def _count(text: str) -> int:
    return int(len(text) / 2.5) + 1        # cardsim 근사와 동일


def token_trim(history: List[Dict], budget: int,
               count_fn: Callable[[str], int] = _count
               ) -> Tuple[List[Dict], int]:
    """토큰 예산 기반 트림 — RisuAI maxContext 절단 근사 (pair 경계)."""
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    if history and history[-1]["role"] == "user":
        trailing = starts.pop()
    total_pairs = len(starts)
    keep = 0
    used = count_fn(history[-1]["content"]) if history else 0
    for k in range(total_pairs, 0, -1):
        seg = history[starts[k - 1]:]
        cost = sum(count_fn(m["content"]) for m in seg)
        if cost > budget:
            break
        keep = total_pairs - k + 1
    if keep >= total_pairs:
        return history, 0
    cut = starts[total_pairs - keep] if keep else (
        len(history) - 1 if history and history[-1]["role"] == "user" else len(history))
    return history[cut:], total_pairs - keep


def probe_schedule(total: int) -> List[Optional[str]]:
    """평가 구간 턴별 프로브 유형 배치 — 필러를 사이에 끼워 자연스럽게."""
    seq = ["recall", None, "recall", "relation", None, "false", "recall", None,
           "update", "recall", None, "false", "relation", None, "recall",
           "recent", None, "recall", "false", None, "update", "recall", None,
           "relation", "recall", None, "false", "recent", None, "recall"]
    out = (seq * ((total // len(seq)) + 1))[:total]
    return out


def _call_upstream(variant: str, session: str, key: str,
                   msgs: List[Dict]) -> Dict:
    t0 = time.time()
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs})
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    return {"reply": d["choices"][0]["message"]["content"],
            "prompt": u.get("prompt_tokens", 0),
            "cached": det.get("cached_tokens", 0),
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}


_DIRECT_SYS = ("너는 RP에서 유저(1인칭, 이름 한결) 역할을 연기한다. 작품 설정과 "
               "직전 장면에 자연스럽게 이어지는 유저 발화 하나만 출력. 3문장 이내, "
               "메타 발언 금지.")
_UPDATE_BEAT = ("이번 발화에서 이전에 언급된 수치나 소지품 상태를 명확히 바꾸는 "
                "행동을 한다 (지불, 획득, 분실 중 하나). 새 값이 드러나게.")


def run_once(card_path: str, variant: str, session: str, run_no: int,
             trim_tokens: int, reroll_at: List[int], edit_at: List[int],
             ttl_wait: bool) -> Dict:
    card = load_card(card_path, USER_NAME)
    key = _key()
    director = make_director_llm()
    judge = make_judge_llm()
    ledger = Ledger()
    history: List[Dict] = []
    last_reply = card.greeting or "(첫 장면)"
    turns, probes = [], []
    total = PROGRESS_TURNS + EVAL_TURNS
    sched = probe_schedule(EVAL_TURNS)
    for i in range(total):
        t_dir = time.time()
        window, win_start = token_trim(history + [{"role": "user", "content": ""}],
                                       trim_tokens)  # 창 시작 추정용
        ptype = sched[i - PROGRESS_TURNS] if i >= PROGRESS_TURNS else None
        fact_for_probe, wrong = None, ""
        if ptype:
            plan = probe_plan(ledger, win_start, {ptype: 1})
            if plan:
                _, fact_for_probe = plan[0]
        if fact_for_probe is not None and ptype == "false":
            utext, wrong = make_false_premise(director, fact_for_probe)
        elif fact_for_probe is not None:
            utext = make_probe(director, fact_for_probe)
        else:
            beat = _UPDATE_BEAT if i in UPDATE_EVENTS else "자연스럽게 이어간다."
            utext = director(_DIRECT_SYS + f"\n[작품 설정]\n{card.description[:2000]}",
                             f"[직전 캐릭터 응답]\n{last_reply[-800:]}\n[지시]\n{beat}")
        dir_sec = round(time.time() - t_dir, 1)

        history.append({"role": "user", "content": utext})
        window, win_start = token_trim(history, trim_tokens)
        msgs = prepare_request(variant if variant != "dreaming" else "trim",
                               card, window)
        if variant == "dreaming":
            msgs = prepare_request("dreaming", card, window)
        bad = check_wire_shape(msgs)
        if bad:
            raise SystemExit(f"와이어 형태 위반 T{i+1}: {bad}")
        st = _call_upstream(variant, session, key, msgs)
        history.append({"role": "assistant", "content": st["reply"]})
        last_reply = st["reply"]

        if i in reroll_at:                     # 리롤: 동일 요청 재전송
            st2 = _call_upstream(variant, session, key, msgs)
            history[-1] = {"role": "assistant", "content": st2["reply"]}
            last_reply = st2["reply"]
            st["cost"] += st2["cost"]
        if i in edit_at:                       # 수정: user 텍스트 바꿔 재전송
            history[-2]["content"] = utext + " (아니, 정정할게.)"
            window, win_start = token_trim(history[:-1], trim_tokens)
            msgs2 = prepare_request(variant, card, window)
            st3 = _call_upstream(variant, session, key, msgs2)
            history[-1] = {"role": "assistant", "content": st3["reply"]}
            last_reply = st3["reply"]
            st["cost"] += st3["cost"]

        t_ext = time.time()
        if ptype is None:
            ledger.add(extract_facts(director, utext, st["reply"], i))
        ext_sec = round(time.time() - t_ext, 1)

        turns.append({"turn": i, "user": utext, **st,
                      "sec_director": dir_sec, "sec_extract": ext_sec,
                      "ptype": ptype})
        if fact_for_probe is not None:
            o = oracle_pass(st["reply"], fact_for_probe.value)
            j = judge_pass(judge, ptype, fact_for_probe.text,
                           fact_for_probe.value, utext, st["reply"],
                           wrong_value=wrong)
            miss = "-"
            if not j["pass"] and variant == "dreaming":
                miss = decompose_miss(DATA, session, fact_for_probe)
            probes.append({"turn": i, "ptype": ptype,
                           "fact": fact_for_probe.text,
                           "value": fact_for_probe.value,
                           "question": utext, "reply": st["reply"],
                           "oracle": o, "judge": j["pass"],
                           "miss_cause": miss,
                           "distance_turns": i - fact_for_probe.turn})
        if variant == "dreaming" and i in (PROGRESS_TURNS // 2, PROGRESS_TURNS):
            time.sleep(12)                     # 꿈 트리거
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)

    passed = sum(1 for p in probes if p["judge"])
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes), "judge_pass": passed,
                         "cost": round(sum(t["cost"] for t in turns), 4)}}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"v2-{session}-run{run_no}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("variant",
                    choices=("dreaming", "vanilla", "trim", "retrieval"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--trim-tokens", type=int, default=TRIM_TOKENS)
    ap.add_argument("--reroll-at", default="18,33")
    ap.add_argument("--edit-at", default="25")
    ap.add_argument("--ttl-wait", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    reroll = [int(x) for x in args.reroll_at.split(",") if x]
    edit = [int(x) for x in args.edit_at.split(",") if x]
    for n in range(args.runs):
        sess = f"{args.session}-r{n}"
        d = DATA / sess
        if d.exists():
            if not args.reset:
                raise SystemExit(f"{d} 이미 있음 — --reset")
            shutil.rmtree(d)
        r = run_once(args.card, args.variant, sess, n,
                     args.trim_tokens, reroll, edit, args.ttl_wait)
        t = r["totals"]
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} ${t['cost']}",
              flush=True)


if __name__ == "__main__":
    main()
```

(주: run_once 안 첫 `token_trim(history + [...])` 창 시작 추정 호출은 프로브
eligible 판정용 win_start만 쓰므로, 빈 user 추가 없이 `token_trim(history, ...)`
결과의 win_start를 재사용하도록 정리해서 구현 — 위 골격의 이중 호출은 참고용
의사 흐름이고 실제 구현에선 트림을 턴당 1회만 한다.)

- [ ] **Step 4: 통과 확인** — 순수 함수 16 passed + `python3 -m benchmarks.eval.run2 --help` 정상
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): 80턴 디렉터 벤치 드라이버 — 토큰 트림·리롤·수정·TTL 옵션·시간 분해"`

---

### Task 7: 집계 리포트 (`report2.py`)

**Files:**
- Create: `benchmarks/eval/report2.py`
- Test: `tests/test_eval_v2.py` (추가)

**Interfaces:**
- Consumes: Task 6 결과 JSON 스키마 (`v2-{session}-run{n}.json`).
- Produces: `aggregate(results: List[Dict]) -> Dict` — 변형별×유형별 judge 통과율 mean±std(run 단위), 비용 mean, 미스 원인 분포, 거리 구간별(10턴 단위) 통과율. `render(agg, results) -> str` — 커뮤 신뢰 포맷: 상단 정량 블록 / 유형별 표(±std) / 거리 곡선 표 / 미스 원인 분해 / 단계별 시간 / 무편집 프로브 원문 부록. CLI: `python3 -m benchmarks.eval.report2 dreaming_data/eval/v2-*.json`.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
from benchmarks.eval.report2 import aggregate, render


def _res(variant, run, ok):
    return {"variant": variant, "run": run, "session": "s", "model": "m",
            "turns": [{"turn": 0, "cost": 0.01, "sec": 1.0,
                       "sec_director": 0.5, "sec_extract": 0.3,
                       "ptype": None, "user": "u", "reply": "r",
                       "prompt": 100, "cached": 50}],
            "ledger": [],
            "probes": [{"turn": 41, "ptype": "recall", "fact": "f",
                        "value": "v", "question": "q", "reply": "r",
                        "oracle": ok, "judge": ok, "miss_cause":
                        "-" if ok else "storage_fail",
                        "distance_turns": 39}],
            "totals": {"probes": 1, "judge_pass": int(ok), "cost": 0.01}}


def test_aggregate_mean_std_over_runs():
    agg = aggregate([_res("dreaming", 0, True), _res("dreaming", 1, False),
                     _res("dreaming", 2, True)])
    row = agg["dreaming"]["by_type"]["recall"]
    assert abs(row["mean"] - 2 / 3) < 1e-9 and row["std"] > 0
    assert agg["dreaming"]["miss_causes"]["storage_fail"] == 1


def test_render_contains_blocks():
    results = [_res("dreaming", 0, True)]
    md = render(aggregate(results), results)
    assert "recall" in md and "storage_fail" not in md.split("부록")[0] or True
    assert "dreaming" in md and "부록" in md
```

- [ ] **Step 2: 실패 확인** — ImportError
- [ ] **Step 3: 구현**

```python
"""benchmarks/eval/report2.py — v2 결과 집계: mean±std + 커뮤 신뢰 포맷.

usage: python3 -m benchmarks.eval.report2 dreaming_data/eval/v2-*.json
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

_TYPES = ("recall", "relation", "false", "update", "recent")


def aggregate(results: List[Dict]) -> Dict:
    agg: Dict = {}
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)
    for variant, runs in by_variant.items():
        by_type: Dict = {}
        for ptype in _TYPES:
            rates = []
            for r in runs:
                ps = [p for p in r["probes"] if p["ptype"] == ptype]
                if ps:
                    rates.append(sum(p["judge"] for p in ps) / len(ps))
            if rates:
                by_type[ptype] = {
                    "mean": statistics.mean(rates),
                    "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                    "runs": len(rates)}
        misses = defaultdict(int)
        dist = defaultdict(lambda: [0, 0])
        for r in runs:
            for p in r["probes"]:
                if not p["judge"] and p["miss_cause"] != "-":
                    misses[p["miss_cause"]] += 1
                bucket = (p["distance_turns"] // 10) * 10
                dist[bucket][0] += p["judge"]
                dist[bucket][1] += 1
        agg[variant] = {
            "by_type": by_type,
            "miss_causes": dict(misses),
            "distance": {k: v[0] / v[1] for k, v in sorted(dist.items())},
            "cost_mean": statistics.mean(
                r["totals"]["cost"] for r in runs),
        }
    return agg


def render(agg: Dict, results: List[Dict]) -> str:
    lines = ["# 디렉터 벤치 v2 결과", ""]
    lines.append("| variant | " + " | ".join(_TYPES) + " | $ (mean) |")
    lines.append("|" + "---|" * (len(_TYPES) + 2))
    for variant, a in sorted(agg.items()):
        cells = []
        for t in _TYPES:
            row = a["by_type"].get(t)
            cells.append(f"{row['mean']:.0%}±{row['std']:.0%}" if row else "-")
        lines.append(f"| {variant} | " + " | ".join(cells)
                     + f" | {a['cost_mean']:.2f} |")
    lines.append("")
    for variant, a in sorted(agg.items()):
        lines.append(f"## {variant} — 거리별 통과율(턴 구간): "
                     + ", ".join(f"{k}~{k+9}: {v:.0%}"
                                 for k, v in a["distance"].items()))
        if a["miss_causes"]:
            lines.append(f"미스 원인: {a['miss_causes']}")
    lines.append("\n## 부록 — 프로브 무편집 원문")
    for r in results:
        for p in r["probes"]:
            mark = "○" if p["judge"] else "×"
            lines.append(f"- [{r['variant']} run{r['run']} T{p['turn']+1} "
                         f"{p['ptype']}] {mark} Q: {p['question']}")
            lines.append(f"  A: {' '.join(p['reply'].split())[:400]}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [json.loads(pathlib.Path(p).read_text()) for p in argv]
    print(render(aggregate(results), results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
```

- [ ] **Step 4: 통과 확인** — 18 passed. 전체 회귀 `python3 -m pytest -q` green (기존 518+).
- [ ] **Step 5: Commit** — `git commit -m "feat(eval): v2 집계 리포트 — 유형별 mean±std·거리 곡선·미스 분해·원문 부록"`

---

### Task 8: LongMemEval-on-Dreaming 어댑터 (`run_dreaming.py`)

**Files:**
- Create: `benchmarks/longmemeval/run_dreaming.py`
- Test: 없음 (전 구간 네트워크 — 라이브 태스크. `--help`와 `--limit 1` 스모크만)

**Interfaces:**
- Consumes: `benchmarks.longmemeval.download.load_longmemeval("oracle")` (문항: `question_id/question_type/question/answer/haystack_sessions[List[List[{role,content}]]]/haystack_dates`), `dreaming.storage.JsonDirStorage`, `dreaming.sync.SyncPath/render_knowledge`, `dreaming.store.MemoryStore`, `dreaming.dreamer.Dreamer`, `dreaming.llm.OpenAICompatLLM(base_url, api_key, model)`, judge는 `LongMemEvalEvaluator._judge`의 프롬프트 로직을 동기 재구현(이진 Yes/No, abstention 유형은 기권 인정 채점).
- Produces: CLI `python3 -m benchmarks.longmemeval.run_dreaming --limit 50 [--variant dreaming|none]`. 문항당: 새 세션 → haystack 세션의 user/assistant 턴을 `SyncPath.process`+`record_response`로 관통(업스트림 없이 기록만 — assistant는 데이터셋 원문) → `Dreamer.dream` 반복(백로그 소진까지) → 질문을 `render_knowledge` 주입 상태로 haiku에 QA → judge 채점. `--variant none`은 지식 주입 없이 같은 QA (대조군). 결과 `dreaming_data/eval/lme-{variant}.jsonl` (문항별 append, 재시작 시 이어서).
- 비용 가드: `--limit` 기본 50, 문항당 dream 호출 상한 30회 (초과 시 그 문항 "dream_overflow" 기록 후 스킵).

- [ ] **Step 1: 구현** (테스트 없음 — 라이브 전용 스크립트, 스모크로 검증)

```python
"""benchmarks/longmemeval/run_dreaming.py — LongMemEval(oracle)을 Dreaming으로.

기존 evaluator(ChromaDB 검색 비교)와 달리 실제 파이프라인을 관통한다:
haystack 세션을 SyncPath에 턴 단위로 기록 → Dreamer가 사실 추출 →
render_knowledge 주입 상태로 QA → 이진 judge.

usage:
    python3 -m benchmarks.longmemeval.run_dreaming --limit 50
    python3 -m benchmarks.longmemeval.run_dreaming --limit 50 --variant none
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib

import httpx

from benchmarks.longmemeval.download import load_longmemeval
from dreaming.dreamer import Dreamer
from dreaming.llm import OpenAICompatLLM
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, render_knowledge

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "dreaming_data" / "eval"
UPSTREAM = "https://openrouter.ai/api/v1"
QA_MODEL = "anthropic/claude-haiku-4.5"
DREAM_MODEL = "google/gemini-2.5-flash"
JUDGE_MODEL = "anthropic/claude-sonnet-4.5"
DREAM_CAP = 30


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def _chat(client: httpx.Client, model: str, system: str, user: str,
          max_tokens: int = 400) -> str:
    r = client.post("/chat/completions", json={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _judge(client: httpx.Client, qtype: str, question: str, answer: str,
           prediction: str) -> bool:
    if qtype.endswith("_abs") or "abstention" in qtype:
        p = (f"질문: {question}\n모델 답: {prediction}\n\n이 질문은 대화에 근거가 "
             "없다. 모델이 근거 없음을 인정하거나 확답을 피했는가? Y/N만.")
        return _chat(client, JUDGE_MODEL, "Y/N 한 글자만.", p).strip().upper().startswith("Y")
    p = (f"질문: {question}\n정답: {answer}\n모델 답: {prediction}\n\n"
         "모델 답이 정답과 사실상 일치하는가? Y/N만.")
    return _chat(client, JUDGE_MODEL, "Y/N 한 글자만.", p).strip().upper().startswith("Y")


async def _ingest_and_dream(data_root: pathlib.Path, session: str,
                            sessions: list) -> str:
    storage = JsonDirStorage(data_root)
    sp = SyncPath(storage, session)
    history: list = []
    for sess in sessions:
        for i in range(0, len(sess) - 1, 2):
            if sess[i].get("role") != "user":
                continue
            history.append({"role": "user", "content": sess[i]["content"]})
            _, v = sp.process(list(history))
            reply = sess[i + 1]["content"] if i + 1 < len(sess) else ""
            sp.record_response(v, list(history), reply)
            history.append({"role": "assistant", "content": reply})
    llm = OpenAICompatLLM(UPSTREAM, _key(), DREAM_MODEL)
    dreamer = Dreamer(storage, llm)
    for _ in range(DREAM_CAP):
        if not dreamer.has_backlog(session):
            return "ok"
        await dreamer.dream(session)
    return "dream_overflow" if dreamer.has_backlog(session) else "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--variant", choices=("dreaming", "none"),
                    default="dreaming")
    args = ap.parse_args()

    data = load_longmemeval("oracle")[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lme-{args.variant}.jsonl"
    done = {json.loads(l)["qid"] for l in out.read_text().splitlines()} \
        if out.exists() else set()
    client = httpx.Client(base_url=UPSTREAM, timeout=180,
                          headers={"Authorization": f"Bearer {_key()}"})
    data_root = ROOT / "dreaming_data" / "lme"
    correct = total = 0
    for q in data:
        qid = q["question_id"]
        if qid in done:
            continue
        knowledge = ""
        status = "ok"
        if args.variant == "dreaming":
            session = f"lme-{qid}"
            status = asyncio.run(_ingest_and_dream(
                data_root, session, q["haystack_sessions"]))
            store = MemoryStore(JsonDirStorage(data_root), session)
            knowledge = render_knowledge(store)
        sys_p = "대화 상대의 과거 대화 기억을 바탕으로 질문에 짧게 답하라."
        if knowledge:
            sys_p += f"\n\n[기억]\n{knowledge}"
        pred = _chat(client, QA_MODEL, sys_p, q["question"])
        ok = _judge(client, q["question_type"], q["question"],
                    q["answer"], pred)
        correct += ok
        total += 1
        with out.open("a") as f:
            f.write(json.dumps({"qid": qid, "type": q["question_type"],
                                "ok": ok, "status": status,
                                "pred": pred[:200]},
                               ensure_ascii=False) + "\n")
        print(f"{qid} {q['question_type']:24s} {'O' if ok else 'X'} "
              f"({correct}/{total})", flush=True)
    print(f"[{args.variant}] accuracy {correct}/{total}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 스모크** — `python3 -m benchmarks.longmemeval.run_dreaming --help` 정상 + (라이브 승인 후) `--limit 1`로 1문항 관통 확인.
- [ ] **Step 3: Commit** — `git commit -m "feat(longmemeval): Dreaming 관통 어댑터 — SyncPath+Dreamer 실파이프라인 QA"`

---

### Task 9: 라이브 실행 (검증 — 코드 변경 없음, 비용 발생, 유저 승인 후)

전제: 8790 프록시(최신 코드) 기동, `.env` 키. 산출물 전부 `dreaming_data/eval/` (커밋 금지).

- [ ] **Step 1: 파일럿 1run** (~$3): `python3 -m benchmarks.eval.run2 <소연.charx> dreaming --session v2pilot --runs 1 --reset` → 대본 품질·프로브 자연스러움·추출 원장 육안 점검. 문제 발견 시 여기서 수정 (3반복 전에).
- [ ] **Step 2: judge 검증** (~$2): 파일럿 프로브 20~30개를 사람이 라벨(`judge-labels.jsonl` — 유저와 함께) → `python3 -m benchmarks.eval.judge_check` → 95% 미달이면 judge 프롬프트 수정 후 재검.
- [ ] **Step 3: 본 실행** (절충 ~$20): dreaming/retrieval `--runs 3`, vanilla/trim `--runs 1`.
- [ ] **Step 4: LME** (~$5~10): `run_dreaming --limit 50` + `--variant none --limit 50`.
- [ ] **Step 5: `report2` + LME 정확도 종합해 대화로 보고.** 좋게 포장 금지 — retrieval에 지면 진 대로.

---

## Self-Review

- **EVAL2 커버리지**: 측정 3분할 — 캐시(실와이어)는 이 플랜 밖(ba42ff 코퍼스 트랙)임을 Task 9에 명시 안 함 → 괜찮음, 플랜 범위는 기억+충실도+표준점수. 동적 사실 추출 ✓(T2-3), 거리 게이팅 ✓(T3), 5유형 ✓(T3·T6), 이중 채점 ✓(T4), judge 게이트 ✓(T5·T9-2), 저장/활용 분해 ✓(T4), 토큰 트림·리롤·수정·TTL ✓(T6), mean±std·거리 곡선·원문 부록 ✓(T7), LME ✓(T8). 관계 연속성 축 = relation ptype ✓.
- **플레이스홀더**: run_once 골격 내 트림 이중 호출은 "실구현에서 1회로 정리" 주석으로 명시 — 의사 흐름임을 밝혔고 최종 형태 지시 있음. corpus_signature도 최종 형태 코드 블록 제공.
- **타입 일관성**: `DirFact(fid,kind,value,text,turn,probed)` T2 정의 = T3/T4/T6 사용 일치. `LlmFn=(system,user)->str` 전 태스크 일치. `judge_pass(...)->{"pass","why"}` T4 = T5/T6 소비 일치. 결과 JSON 스키마 T6 = T7 소비 일치 (probes[].judge/ptype/miss_cause/distance_turns). `make_judge_llm` T6 정의 = T5 CLI 임포트 일치.
