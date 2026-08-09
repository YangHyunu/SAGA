# Eval v2 하네스 구조 리팩터 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** run2.py god object(663줄/8책임)를 역할별 모듈로 분해하고 프롬프트를 단일 모듈에 집결해, 프롬프트 가시성·A/B 실험·오프라인 오케스트레이션 테스트를 가능하게 한다.

**Architecture:** 동작 보존(behavior-preserving) 추출 — 기존 심볼을 새 모듈로 옮기고 run2.py는 재수출(facade)로 남겨 658개 테스트와 스크립트 호환을 유지한다. 의도된 동작 변경은 정확히 3개: ① run2↔hypa 순환 절단(hypa가 config/transport를 직접 봄), ② `HYPA_SUMMARY_MODEL` env 분리, ③ run_once에 `call_fn` 심(seam) 추가. 근거: `.brooks-lint-history.json` 2026-08-10 감사 (score 48, Critical 2건).

**Tech Stack:** Python 3, pytest, ruff (실행은 `python3 -m ruff` — PATH에 없음)

## Global Constraints

- 기준선: `python3 -m pytest -q tests/` **658 passed** 전부 통과 유지 (신규 테스트로 증가만 허용). 테스트의 monkeypatch 대상 경로 수정은 허용 — 단 해당 태스크 스텝에 명시된 것만.
- `python3 -m ruff check benchmarks/eval/ dreaming/` 기존 **6 errors** — 신규 0.
- `dreaming_data/*`, `.env` 커밋 금지. API 키 값 echo/출력 금지.
- 각 태스크 끝에 커밋. 커밋 메시지는 태스크 스텝에 명시된 것 사용.
- 명시된 3개 동작 변경 외에는 로직 수정 금지 — diff는 이동+import 재배선이어야 한다. 이동 시 함수 본문·주석·docstring 바이트 그대로.
- v1 하네스 파일(run.py, variants.py, oracle.py, script.py, report.py)은 Task 7의 배너·의존 방향 작업 외 수정 금지.

## 파일 구조 (최종 상태)

| 파일 | 책임 (한 문장) |
|---|---|
| `benchmarks/eval/config.py` | **신설** — env 읽기와 벤치 상수 전부 (모델·경로·턴·NPC·TOGGLES) |
| `benchmarks/eval/prompts.py` | **신설** — LLM에 보내는 모든 프롬프트 상수 + A/B 오버라이드 로더 |
| `benchmarks/eval/transport.py` | **신설** — 업스트림/프록시 HTTP 호출, 재시도, LLM 팩토리, 비용 추출 |
| `benchmarks/eval/windowing.py` | **신설** — 토큰 카운트, token_trim, wire_history, hypa_in_window |
| `benchmarks/eval/quality.py` | **신설** — reply_flaw, reroll_until_clean, abort_reroll_count |
| `benchmarks/eval/run2.py` | 얇아짐 — run_once(오케스트레이션) + main(CLI) + 위 모듈 재수출 |
| `benchmarks/eval/hypa.py` | 불변 (import 경로만 run2→config/transport로 교체) |
| `benchmarks/eval/director.py` | 불변 (프롬프트 상수만 prompts.py로 이동, 별칭 유지) |
| `benchmarks/eval/scoring.py` | `_JUDGE_SYS`→prompts 이동 + v1 oracle private 의존 방향 뒤집기 |
| `benchmarks/eval/oracle.py` (v1) | `_norm`/`_STATBAR`/`expect_alternatives`를 scoring에서 역수입 |

---

### Task 1: config.py — 설정·상수 추출

**Files:**
- Create: `benchmarks/eval/config.py`
- Modify: `benchmarks/eval/run2.py` (상수 정의 제거 → config에서 import)
- Test: 기존 스위트가 안전망 (신규 테스트 없음 — 값 이동만)

**Interfaces:**
- Produces: `config.ROOT, DATA, EVAL_DIR, PROXY, UPSTREAM, MODEL, JUDGE_MODEL, DIRECTOR_MODEL, HYPA_EXPORT, TURNS, PROBE_EVERY, MAX_CONTEXT, UPDATE_EVENTS, NPC_NAME, NPC_EVENT_TURN, NPC_EVENT_RETRY, MAX_TOKENS, MAX_RUN_REROLLS, TOGGLES` (run2.py 41–92줄의 모듈 상수 전부 — `_ENC` 제외, 그건 Task 4의 windowing 소관)
- Consumes: 없음 (최하층 — 어떤 eval 모듈도 import하지 않는다. os/pathlib만)

- [ ] **Step 1: config.py 생성** — run2.py 상단의 상수 블록(ROOT~TOGGLES, `_ENC`·`_FULL_HISTORY` 제외)을 바이트 그대로 이동. 파일 docstring:

```python
"""eval v2 설정 — env 오버라이드 + 벤치 상수. 최하층: eval 내부 import 금지."""
```

- [ ] **Step 2: run2.py 재배선** — 이동한 상수 정의를 지우고 맨 위 import 블록에 추가:

```python
from benchmarks.eval.config import (DATA, DIRECTOR_MODEL, EVAL_DIR, HYPA_EXPORT,
                                    JUDGE_MODEL, MAX_CONTEXT, MAX_RUN_REROLLS,
                                    MAX_TOKENS, MODEL, NPC_EVENT_RETRY,
                                    NPC_EVENT_TURN, NPC_NAME, PROBE_EVERY, PROXY,
                                    ROOT, TOGGLES, TURNS, UPDATE_EVENTS, UPSTREAM)
```

주의: `from X import Y`로 가져오면 run2.NAME이 그대로 존재해 기존 테스트의 `run2.MAX_CONTEXT` 참조·monkeypatch가 계속 동작한다 (rebind는 run2 네임스페이스에만 적용 — 현행과 동일 의미).

- [ ] **Step 3: 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `658 passed`. Run: `python3 -m ruff check benchmarks/eval/ dreaming/ 2>&1 | tail -1` Expected: `Found 6 errors.`
- [ ] **Step 4: 커밋**

```bash
git add benchmarks/eval/config.py benchmarks/eval/run2.py
git commit -m "refactor(eval): 설정·상수를 config.py로 추출 — 최하층 분리"
```

---

### Task 2: prompts.py — 프롬프트 집결 + A/B 오버라이드

**Files:**
- Create: `benchmarks/eval/prompts.py`
- Modify: `benchmarks/eval/run2.py`, `benchmarks/eval/director.py`, `benchmarks/eval/scoring.py` (상수 정의 제거 → 별칭 import)
- Test: `tests/test_eval_v2.py` (신규 2개 추가)

**Interfaces:**
- Produces: `prompts.DIRECT_SYS, UPDATE_BEAT, BEATS, NPC_BEAT, EXTRACT_SYS, PROBE_SYS, FALSE_SYS, JUDGE_SYS` + `prompts.override_from(path: str) -> None` + `prompts.active() -> Dict[str, str]`
- Consumes: `config.NPC_NAME` (NPC_BEAT f-string용)

- [ ] **Step 1: 실패 테스트 작성** — tests/test_eval_v2.py 끝에:

```python
def test_prompts_override_replaces_named_prompt(tmp_path):
    # A/B: JSON 파일로 이름 붙은 프롬프트를 통째로 교체한다
    from benchmarks.eval import prompts
    p = tmp_path / "ab.json"
    p.write_text('{"JUDGE_SYS": "대체 채점 프롬프트"}', encoding="utf-8")
    before = prompts.JUDGE_SYS
    try:
        prompts.override_from(str(p))
        assert prompts.JUDGE_SYS == "대체 채점 프롬프트"
        assert prompts.active()["JUDGE_SYS"] == "대체 채점 프롬프트"
    finally:
        prompts.JUDGE_SYS = before          # 모듈 전역 원복


def test_prompts_override_rejects_unknown_key(tmp_path):
    from benchmarks.eval import prompts
    p = tmp_path / "bad.json"
    p.write_text('{"NO_SUCH_PROMPT": "x"}', encoding="utf-8")
    with pytest.raises(KeyError):
        prompts.override_from(str(p))
```

- [ ] **Step 2: FAIL 확인** — Run: `python3 -m pytest -q tests/test_eval_v2.py -k prompts_override 2>&1 | tail -2` Expected: FAIL (`ModuleNotFoundError`)
- [ ] **Step 3: prompts.py 생성** — 8개 상수를 원본에서 바이트 그대로 이동하되 언더스코어 제거명으로 통일 (`_DIRECT_SYS`→`DIRECT_SYS` 등). 출처: run2.py의 `_DIRECT_SYS`/`_UPDATE_BEAT`/`_BEATS`/`_NPC_BEAT`, director.py의 `_EXTRACT_SYS`/`_PROBE_SYS`/`_FALSE_SYS`, scoring.py의 `_JUDGE_SYS`. 각 상수 위에 1줄 주석: 어느 호출이 쓰는지 (예: `# 디렉터 페르소나 — run2.run_once 매 턴 system`). 하단에:

```python
_NAMES = ("DIRECT_SYS", "UPDATE_BEAT", "BEATS", "NPC_BEAT",
          "EXTRACT_SYS", "PROBE_SYS", "FALSE_SYS", "JUDGE_SYS")


def active() -> Dict[str, str]:
    """현재 프롬프트 세트 스냅샷 — 결과 JSON에 기록해 A/B 추적용."""
    return {n: globals()[n] for n in _NAMES}


def override_from(path: str) -> None:
    """A/B 실험: JSON({이름: 프롬프트})로 이름 붙은 프롬프트를 교체한다.

    BEATS처럼 tuple인 항목은 JSON 배열로 넘긴다. 미지의 키는 KeyError —
    조용히 무시하면 오타가 A/B 결과를 침묵 속에 무효화한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    for name, value in overrides.items():
        if name not in _NAMES:
            raise KeyError(f"unknown prompt: {name} (valid: {', '.join(_NAMES)})")
        globals()[name] = tuple(value) if isinstance(value, list) else value
```

- [ ] **Step 4: 원본 3파일 재배선** — 정의 삭제 후 별칭 import (기존 테스트가 `director._PROBE_SYS` 등 언더스코어 이름을 참조하므로 별칭 필수):

```python
# run2.py
from benchmarks.eval import prompts
from benchmarks.eval.prompts import (BEATS as _BEATS, DIRECT_SYS as _DIRECT_SYS,
                                     NPC_BEAT as _NPC_BEAT,
                                     UPDATE_BEAT as _UPDATE_BEAT)
# director.py
from benchmarks.eval.prompts import (EXTRACT_SYS as _EXTRACT_SYS,
                                     FALSE_SYS as _FALSE_SYS,
                                     PROBE_SYS as _PROBE_SYS)
# scoring.py
from benchmarks.eval.prompts import JUDGE_SYS as _JUDGE_SYS
```

주의: `NPC_BEAT`은 `config.NPC_NAME`을 쓰는 f-string — prompts.py가 config를 import하는 것은 허용(config는 최하층).

- [ ] **Step 5: run2 main()에 `--prompts` CLI 플래그 추가** — argparse에 `parser.add_argument("--prompts", default="", help="프롬프트 오버라이드 JSON (A/B)")` + 파싱 직후 `if args.prompts: prompts.override_from(args.prompts)`. run_once 시작부에서 결과 dict에 `"prompt_set": prompts.active()` 기록 (A/B 런 추적).
- [ ] **Step 6: 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `660 passed`. ruff 6 유지.
- [ ] **Step 7: 커밋**

```bash
git add benchmarks/eval/prompts.py benchmarks/eval/run2.py benchmarks/eval/director.py benchmarks/eval/scoring.py tests/test_eval_v2.py
git commit -m "refactor(eval): 프롬프트 8종을 prompts.py로 집결 — A/B 오버라이드 + 런 기록"
```

---

### Task 3: transport.py — HTTP·LLM 팩토리 추출 + 순환 절단

**Files:**
- Create: `benchmarks/eval/transport.py`
- Modify: `benchmarks/eval/run2.py`, `benchmarks/eval/hypa.py`
- Test: `tests/test_eval_hypa.py` (monkeypatch 대상 경로 교체), `tests/test_eval_v2.py` (해당 시)

**Interfaces:**
- Produces: `transport.key() -> str` (구 `run2._key` — 공개명으로 승격), `transport.mk_llm(model, temperature) -> LlmFn`, `transport.make_judge_llm()`, `transport.make_director_llm()`, `transport.call_upstream(variant, session, key, msgs) -> Dict`, `transport.call_upstream_once(...)`, `transport.SUMMARY_MODEL: str` (신설 env `HYPA_SUMMARY_MODEL`, 기본값 `config.DIRECTOR_MODEL`)
- Consumes: `config.UPSTREAM, PROXY, MODEL, MAX_TOKENS, JUDGE_MODEL, DIRECTOR_MODEL`

- [ ] **Step 1: transport.py 생성** — run2.py에서 `_key`, `_mk_llm`, `make_judge_llm`, `make_director_llm`, `_call_upstream`, `_call_upstream_once` 본문 바이트 그대로 이동, 공개명(`key`, `mk_llm`, `call_upstream`, `call_upstream_once`)으로 개명. 추가:

```python
# hypa 요약 모델 — 디렉터 축과 분리 (감사 R2: DIRECTOR_MODEL이 두 축을 동시에 움직임)
SUMMARY_MODEL = os.environ.get("HYPA_SUMMARY_MODEL", config.DIRECTOR_MODEL)
```

- [ ] **Step 2: run2.py 재배선** — 하위호환 별칭 (기존 테스트·스크립트가 `run2._key`, `run2._call_upstream`을 참조):

```python
from benchmarks.eval import transport
from benchmarks.eval.transport import (call_upstream as _call_upstream,
                                       call_upstream_once as _call_upstream_once,
                                       key as _key, make_director_llm,
                                       make_judge_llm, mk_llm as _mk_llm)
```

- [ ] **Step 3: hypa.py 순환 절단** — `from benchmarks.eval import run2` 2곳(lazy import)을 교체:
  - `_director_model()` 본문: `from benchmarks.eval import transport; return transport.SUMMARY_MODEL` (docstring도 "요약 모델 — HYPA_SUMMARY_MODEL env, 기본 DIRECTOR_MODEL 상속"으로 갱신)
  - `_summarize_call_once()`: `from benchmarks.eval import config, transport` 후 `config.UPSTREAM`, `transport.key()` 사용
  - lazy import 유지 이유 주석: 없음 — 이제 최하층만 보므로 **top-level import로 승격** (`import httpx` 옆에 `from benchmarks.eval import config, transport` 추가, 함수 내 import 삭제). run2→hypa 단방향 확정.
- [ ] **Step 4: 테스트 monkeypatch 대상 교체** — tests/test_eval_hypa.py에서 `monkeypatch.setattr(run2, "_key", ...)` → `monkeypatch.setattr(transport, "key", ...)` (4곳: retries/truncation/4xx/pins_params 테스트), `monkeypatch.setattr(run2, "DIRECTOR_MODEL", ...)` (cache-key 테스트) → `monkeypatch.setattr(transport, "SUMMARY_MODEL", ...)`. 파일 상단 import에 `from benchmarks.eval import transport` 추가. hypa.summary_cache_key가 `_director_model()`을 경유하는지 확인 — 경유하면 그대로 동작.
- [ ] **Step 5: 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `660 passed`. 추가 확인: `grep -n "import run2" benchmarks/eval/hypa.py` → 0건 (순환 소멸).
- [ ] **Step 6: 커밋**

```bash
git add benchmarks/eval/transport.py benchmarks/eval/run2.py benchmarks/eval/hypa.py tests/test_eval_hypa.py
git commit -m "refactor(eval): transport.py 추출 — run2↔hypa 순환 절단 + HYPA_SUMMARY_MODEL 분리"
```

---

### Task 4: windowing.py + quality.py 추출

**Files:**
- Create: `benchmarks/eval/windowing.py`, `benchmarks/eval/quality.py`
- Modify: `benchmarks/eval/run2.py`
- Test: 기존 스위트가 안전망 (이동만 — 로직 불변)

**Interfaces:**
- Produces: `windowing._ENC, count(text) -> int` (구 `_count`), `token_trim(history, budget, count_fn) -> (window, win_start)`, `FULL_HISTORY` (구 `_FULL_HISTORY`), `wire_history(variant, history, window)`, `hypa_in_window(fact_turn, kept_start_msg, has_greeting)` / `quality.REFUSAL_MARKS, LOOP_LOOKBACK, LOOP_RATIO, reply_flaw(reply, prior_replies)`, `reroll_until_clean(call, prior_replies, max_rerolls)`, `abort_reroll_count(flaw_history)`
- Consumes: windowing → 없음 (tiktoken만). quality → 없음 (difflib/re만)

- [ ] **Step 1: 두 파일 생성** — run2.py에서 심볼 본문 바이트 그대로 이동 (위 Produces 목록이 전체). quality의 `_HANGUL` 정규식도 함께.
- [ ] **Step 2: run2.py 재배선** — 별칭 import:

```python
from benchmarks.eval.windowing import (FULL_HISTORY as _FULL_HISTORY,
                                       count as _count, hypa_in_window,
                                       token_trim, wire_history)
from benchmarks.eval.quality import (abort_reroll_count, reply_flaw,
                                     reroll_until_clean)
```

- [ ] **Step 3: 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `660 passed`. ruff 6 유지.
- [ ] **Step 4: 커밋**

```bash
git add benchmarks/eval/windowing.py benchmarks/eval/quality.py benchmarks/eval/run2.py
git commit -m "refactor(eval): windowing·quality 추출 — run2는 오케스트레이션+CLI만 남김"
```

---

### Task 5: run_once 심(seam) + 오프라인 오케스트레이션 테스트

**Files:**
- Modify: `benchmarks/eval/run2.py` (run_once 시그니처)
- Test: `tests/test_eval_v2.py` (신규 1개)

**Interfaces:**
- Produces: `run_once(preset_path, card_path, variant, session, *, turns=None, max_context=None, call_fn=None)` — `call_fn` 기본값 None이면 `transport.call_upstream` 사용. 기존 위치 인자 호출과 하위호환 (기존 키워드 인자들은 현행 시그니처 그대로 유지 — 이 태스크는 `call_fn` 하나만 추가).
- Consumes: Task 3의 `transport.call_upstream`

- [ ] **Step 1: 실패 테스트 작성** — 가짜 나레이터로 5턴 vanilla를 오프라인 완주. 디렉터/저지 LLM은 이미 주입 가능(`_mk_llm` 팩토리)이므로 monkeypatch로 상수 응답 고정:

```python
def test_run_once_offline_with_fake_narrator(tmp_path, monkeypatch):
    # run_once 오케스트레이션이 라이브 HTTP 없이 완주 — seam 검증.
    # 나레이터: call_fn 주입. 디렉터/저지: 팩토리 monkeypatch.
    from benchmarks.eval import run2, transport

    def fake_call(variant, session, key, msgs):
        return {"reply": "…소연은 조용히 고개를 끄덕였다.", "prompt": 100,
                "cached": 0, "cost": 0.0, "latency": 0.1}

    monkeypatch.setattr(transport, "key", lambda: "offline")
    monkeypatch.setattr(run2, "make_director_llm",
                        lambda: (lambda sys, user: "장터를 함께 걷자고 말한다"))
    monkeypatch.setattr(run2, "make_judge_llm",
                        lambda: (lambda sys, user: "PASS"))
    monkeypatch.setattr(run2, "EVAL_DIR", tmp_path)
    out = run2.run_once(PRESET, CARD, "vanilla", "seam-test",
                        turns=5, call_fn=fake_call)
    assert out["turns"] and len(out["turns"]) == 5
    assert not out.get("aborted")
```

주의: `PRESET`/`CARD`는 test_eval_v2.py 상단에 이미 있는 실경로 상수 사용. 실제 run_once의 반환 dict 키·`turns`/`EVAL_DIR` 사용 방식은 구현 시점에 run_once 본문을 읽고 정합 (테스트 골격은 유지하되 키 이름은 실코드 기준으로 조정 — 조정 내역을 커밋 메시지 본문에 기록).

- [ ] **Step 2: FAIL 확인** — Run: `python3 -m pytest -q tests/test_eval_v2.py -k offline 2>&1 | tail -2` Expected: FAIL (`TypeError: unexpected keyword argument 'call_fn'`)
- [ ] **Step 3: 최소 구현** — run_once 시그니처에 `call_fn=None` 추가, 본문 첫머리 `call = call_fn or transport.call_upstream`, 내부의 `_call_upstream(...)` 호출을 `call(...)`로 교체 (재롤 람다 포함 전 지점). 다른 로직 변경 금지.
- [ ] **Step 4: PASS 확인 + 전체 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `661 passed`.
- [ ] **Step 5: 커밋**

```bash
git add benchmarks/eval/run2.py tests/test_eval_v2.py
git commit -m "feat(eval): run_once에 call_fn 심 — 오프라인 오케스트레이션 테스트 확보"
```

---

### Task 6: run_once 단계 분해 (가독성 — 동작 불변)

**Files:**
- Modify: `benchmarks/eval/run2.py`
- Test: Task 5의 오프라인 테스트 + 기존 스위트가 안전망

**Interfaces:**
- Produces: run_once 내부에서만 쓰는 모듈-레벨 헬퍼 3개 (이름·시그니처는 본문 구조 기준으로 구현자가 확정하되, 다음 경계를 지킨다): ① 턴 1회 실행+리롤 블록 → `_play_turn(...)`, ② 프로브 계획+기록 블록 → `_record_probe(...)`, ③ totals 집계 블록 → `_collect_totals(...)`. run_once는 이 셋을 부르는 ~60줄 루프로.
- Consumes: Task 5까지의 전부

- [ ] **Step 1: 현행 run_once 정독** — 214줄을 3블록 경계로 표시 (주석 아님, 작업 메모). 각 블록의 입력/출력 로컬 변수를 나열해 헬퍼 시그니처 확정.
- [ ] **Step 2: 한 블록씩 추출** — 블록 하나 추출할 때마다 Run: `python3 -m pytest -q tests/test_eval_v2.py 2>&1 | tail -2` Expected: 전부 통과. 세 블록 반복. 본문 코드 바이트 이동 원칙 (제어 흐름 재작성 금지 — `continue`/`break`가 걸린 블록은 반환값 튜플로 승격).
- [ ] **Step 3: 전체 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `661 passed`. ruff 6 유지. `wc -l benchmarks/eval/run2.py` — 400줄 미만 확인.
- [ ] **Step 4: 커밋**

```bash
git add benchmarks/eval/run2.py
git commit -m "refactor(eval): run_once 214줄을 단계 헬퍼 3개로 분해 — 동작 불변"
```

---

### Task 7: v1 배너 + scoring↔oracle 의존 방향 뒤집기

**Files:**
- Modify: `benchmarks/eval/scoring.py`, `benchmarks/eval/oracle.py`, `benchmarks/eval/run.py`, `benchmarks/eval/variants.py`, `benchmarks/eval/script.py`, `benchmarks/eval/report.py`
- Test: 기존 스위트 (test_eval_harness.py가 v1 커버)

**Interfaces:**
- Produces: `scoring._norm`, `scoring._STATBAR`, `scoring.expect_alternatives` (oracle에서 본문 이동 — v2가 소유)
- Consumes: oracle이 `from benchmarks.eval.scoring import _STATBAR, _norm, expect_alternatives`로 역수입

- [ ] **Step 1: 의존 방향 뒤집기** — oracle.py의 `_norm`/`_STATBAR`/`expect_alternatives` 본문을 scoring.py로 이동(바이트 그대로), oracle.py에는 `from benchmarks.eval.scoring import _STATBAR, _norm, expect_alternatives` 한 줄. scoring.py의 기존 `from benchmarks.eval.oracle import ...` 삭제. 순환 확인: scoring은 이제 oracle을 import하지 않아야 한다 (`grep -n "from benchmarks.eval.oracle" benchmarks/eval/scoring.py` → 0건).
- [ ] **Step 2: v1 배너** — v1 파일 5개(run.py, variants.py, oracle.py, script.py, report.py) docstring 첫 줄 위에 주석 1줄 추가: `# [v1 하네스] 현행은 run2 계열 — 이 파일은 구 테스트·PR 호환용으로 보존.`
- [ ] **Step 3: 검증** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `661 passed`. ruff 6 유지.
- [ ] **Step 4: 커밋**

```bash
git add benchmarks/eval/scoring.py benchmarks/eval/oracle.py benchmarks/eval/run.py benchmarks/eval/variants.py benchmarks/eval/script.py benchmarks/eval/report.py
git commit -m "refactor(eval): v2→v1 private 의존 역전 (_norm 등 scoring 소유) + v1 배너"
```

---

### Task 8: 통합 검증 — 동작 동일성

**Files:**
- Modify: 없음 (검증만 — 발견된 결함은 별도 수정 커밋)

- [ ] **Step 1: 전체 게이트** — Run: `python3 -m pytest -q tests/ 2>&1 | tail -2` Expected: `661 passed`. Run: `python3 -m ruff check benchmarks/eval/ dreaming/ 2>&1 | tail -1` Expected: `Found 6 errors.`
- [ ] **Step 2: import 그래프 확인** — 순환 부재 검증:

```bash
python3 -c "
import ast, pathlib
for f in sorted(pathlib.Path('benchmarks/eval').glob('*.py')):
    tree = ast.parse(f.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, 'module', '') or ''
            if 'run2' in mod and f.name in ('hypa.py',):
                print('CYCLE:', f.name, '->', mod)
print('cycle check done')
"
```

Expected: `cycle check done` (CYCLE 줄 없음)

- [ ] **Step 3: 라이브 미니 스모크 (동작 동일성, ~$0.05)** — vanilla 4턴 1회:

```bash
DREAMING_UPSTREAM_KEY_SET=1 python3 -m benchmarks.eval.run2 --variant vanilla --session refactor-smoke --turns 4
```

(키는 스크립트/셸에서 `grep '^DREAMING_UPSTREAM_KEY=' .env | cut -d= -f2-`로 읽어 env로 전달 — 값 출력 금지. 정확한 CLI 인자명은 main() 기준으로 조정.) Expected: exit 0, `dreaming_data/eval/v2-refactor-smoke-*.json` 생성, aborted 없음.

- [ ] **Step 4: 결과 JSON에 prompt_set 기록 확인** — Run: `python3 -c "import json;d=json.load(open(sorted(__import__('glob').glob('dreaming_data/eval/v2-refactor-smoke*'))[-1]));print('prompt_set' in d)"` Expected: `True`
- [ ] **Step 5: 커밋 없음** — 검증 태스크. 실패 시 원인 수정 후 `fix(eval):` 커밋.

---

## Self-Review 결과

- 감사 finding 커버리지: R1 god object → T1–T6 / R5 순환 → T3 / R2 프롬프트 산재 → T2 / R5 v1 private → T7 / R2 모델 축 결합 → T3 / seam 부재 → T5 / v1 배너 → T7. 잔여: hypa.py 내부 분할(감사 Suggestion — 의도적 미착수, 스펙 대조 용이성 트레이드오프).
- 테스트 수 추적: 658 → T2 +2 → T5 +1 = **661** (T8 게이트 기준).
- 유의: Task 5 오프라인 테스트는 run_once 반환 키를 실코드 기준으로 조정할 여지를 명시했다 — 골격 고정, 키 이름만 가변.
