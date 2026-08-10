# Lucid 개편 P1 — 유출 차단 · 런 유효성 게이트 · TTL 창구 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 벤치가 뱉는 숫자를 (a) 오염되지 않게 만들고 (b) 무효일 때 자동으로 무효라고 말하게 만들고 (c) SPEC 주장 2건(C11 TTL 재압축, C12 캐치업)을 새로 시험 가능하게 만든다.

**Architecture:** 우선순위는 [LUCID-CONTRACT.md §7](../../LUCID-CONTRACT.md)에서 도출했다 — 정렬 기준은 "SPEC 주장을 새로 여는가 > 현재 측정값의 오염을 제거하는가 > 도구 위생". `docs/HANDOFF-lucid.md` §3의 원 순서(3층 분리 → 점진 공개 → 목표=상태)는 이 검사를 거치지 않았고, 셋 다 SPEC 주장을 하나도 열지 않아 각각 4순위·폐기·범위밖으로 재배치됐다.

**Tech Stack:** Python 3, pytest, ruff (실행은 `python3 -m ruff` — PATH에 없음)

## 계약에서 확정된 결정 (재론 금지)

| # | 결정 | 이 플랜에서 |
|---|---|---|
| D1 | `EVAL2.md:99-105`(동적 사실 추출)가 정본, `:56`(고정 대사) 표는 낡음 | Task 3에서 `:56` 정정 |
| D2 | 점진 공개 **폐기** | 미구현 |
| D3 | `--ttl-wait` 본런 기본 ON | Task 3 |
| D4 | 프로브 전달률 80% 미만 = 런 실패 | Task 2 (G4) |
| D5 | 값 마스킹 + 하드 게이트 | Task 1 |
| D6 | `lucid_model` + 프롬프트 해시 필수 박제 | Task 0 + Task 2 (G8) |

## 감사 후 확정된 결정 (2026-08-10, 유저 승인)

플랜 v1을 서브에이전트 3팀으로 코드 대조 감사한 결과 BLOCKER 7건이 나왔다. 아래가 그 처리다.

| # | 발견 | 결정 |
|---|---|---|
| A1 | `expect_alternatives`는 `scoring.py:42` 소유이고 `scoring.py:28`이 이미 `director`를 import → `lucid`에서 scoring을 import하면 순환 | **하위 레이어 `matching.py` 신설**로 분리 (Task 1) |
| A2 | Task 4 층 표의 "복창 금지"·"날조 금지"는 `DIRECT_SYS`에 grep 0건 | **삭제** — Task 4는 순수 이동만. 규칙 추가는 별건 |
| A3 | `[작품 설정]`·`{beat}`는 `DIRECT_SYS` 밖 (`run2.py:396`·`398`) | **2층화**(RULES+PERSONA)로 축소. SCENARIO 층 미신설 |
| A4 | `layer_hashes()`를 Task 2가 쓰는데 Task 4가 만듦 (순서 버그) | **`layer_hashes()`를 Task 2로 이동**. 층과 무관한 `active()` 해시라 선행 가능 |
| A5 | `evaluate() -> List[Tuple]`인데 저장은 `{failed, warnings}` 2버킷 | 반환형을 **`Dict[str, List[Tuple[str,str]]]`**로 확정 |
| A6 | `MIN_PROBE_AGE=15`가 리롤 직후(나이 1) 프로브를 구조적으로 금지 | **Task 5 보류** — 범위 밖으로 이동 |
| A7 | `dream_cycles`를 셀 방법이 디스크에 없음 (`dreamer.py:341`이 cursor 단일 문서를 덮어씀) | G1을 **`dream_ran and episodes_written >= 1`**로 재정의 + 계약 §6 정정 |

**감사가 과장한 것 1건**(기록용): "단일 `compose_lucid_sys` 호출은 구조적으로 불가"는 틀렸다. `dir_sys`는 `run2.py:346`에서 1회 조립되고 396행이 붙이는 `[작품 설정]`은 루프 불변 상수다. 변하는 `{beat}`는 **user 메시지**(398행)라 시스템 프롬프트에 애초에 없다. 2층화를 고른 건 이 제약 때문이 아니라 변경량 최소화 때문이다.

## Track A 교정 (2026-08-10, annyeong-3b2696 세션 — 착수 전 반영 필수)

야간 본런 사후 조치로 dreaming/spec 팁이 이동했고, 본런 실측에서 이 플랜의 전제 몇 개가 낡았다.

| # | 교정 | 상세 |
|---|---|---|
| C1 | **베이스 커밋 리베이스: `2f1a56e` → `58b1387`** | `58b1387`이 transport.py(provider throughput 정렬 + 추론 예산 캡)와 night_run.sh(REASONING 기본값 4000, 주석 2줄 추가)를 변경. Task 0이 transport.py를, Task 0·3이 night_run.sh를 만지므로 리베이스 없이 착수하면 충돌/드리프트 |
| C2 | **행번호 드리프트: `night_run.sh:88` → ~:90** | C1의 주석 2줄 추가로 밀림. 착수 시 실조회 (플랜 후기 규칙 그대로) |
| C3 | **기준선 테스트 수 재측정** | 플랜 "659 passed, 4 skipped" vs Track A 실측 663 passed — 환경차(skip 조건) 추정. `58b1387`에서 다시 재고 Task별 기대 수를 그 값 기준으로 보정 |
| C4 | **PERSONA 층 후속 1순위 예약: 존댓말 → 카드 준수** | 유저 확정 피드백: 렌 카드는 반말(casual)인데 `DIRECT_SYS`가 존댓말 강제 — 디렉터가 카드 persona를 아예 안 읽는 문제. Task 4는 설계상 바이트 보존 이동이므로 **이 플랜에서 고치지 말고**, Task 4 완료 직후 `lucid_persona.md` 단일 파일 수정으로 처리 (층 분리의 실익이 바로 이것) |
| C5 | **이 플랜이 안 고치는 것 명시 (기대 관리)** | "렌으로서 활동 안 함"·"어거지 plot"·"어거지 테스트(자기 이름 오염 '한결', 전투 중 자기소개 요구)"는 P1 범위 밖. P1은 숫자 신뢰성(유출·게이트·프로비넌스) 판. D5 마스킹은 정답 유출만 차단하고 그럴듯함은 못 보장 |

**C1~C3 처리 결과 (2026-08-10, 이 세션 실측)**

- **C1 완료 — 실제로는 `d85d32f`로 맞췄다.** 착수 시점에 `dreaming/spec` 팁이 `58b1387`에서 2커밋 더 나가 있었다 (`c557b7e`, `d85d32f` — 둘 다 `docs/DREAMING_FLAW.md`만 건드리는 문서 커밋). 코드는 `58b1387`과 동일하므로 C1의 의도(코드 드리프트 제거)를 만족하면서 팁에 정렬했다. 로컬 커밋 0건이라 `git reset --hard`로 충분 (재생할 커밋 없음).
- **C2 완료 — `scripts/night_run.sh:90`.** 실조회 확인 (`c = t["cost"] + t.get("cost_director", 0) + ...`). 플랜 v1의 `:82`, C2의 "~:90" 추정 모두 이 값으로 확정.
- **C3 해소 — 환경차 아니었다.** `d85d32f`에서 `python3 -m pytest -q` = **659 passed, 4 skipped** (2f1a56e와 동일). Track A의 "663"은 659 passed + 4 skipped를 합산한 수치다. 기준선은 659로 유지하고 Task별 기대 수 보정 불필요.
- **C6(신규) — `DREAMING_FLAW.md`가 Task 2 G1을 무력화한다.** 본런 실측이 **에피소드 76개 저장 / 압축 청크 0개**였다 (`DREAMING_FLAW.md:22-23`). 즉 A7로 재정의한 G1(`dream_ran and episodes_written >= 1`)은 **핵심 압축 층이 완전히 죽은 런을 유효로 통과시켰을 것이다.** G1에 `compression_planned`(= `{session}/compression/plan.json` 존재)를 추가한다 (Task 2). `dreamer.py:345-346`이 `plan is not None`일 때만 그 파일을 쓰므로 청크 0건은 파일 부재로 곧장 드러난다. 압축 버그 수정은 Track A 소관이고, 고쳐지기 전까지 dreaming 런이 G1에서 계속 실패하는 것은 **게이트가 제 일을 하는 것**이다 — "이 런은 압축을 시험하지 못했다"가 사실이므로.

## Global Constraints

- 베이스 커밋 **`d85d32f`** (dreaming/spec 팁, C1 완료). 기준선 `python3 -m pytest -q` = **659 passed, 4 skipped** (이 베이스에서 실측 확인, C3 해소).
- `python3 -m ruff check benchmarks/eval/ dreaming/` 신규 에러 0.
- **모든 검증은 오프라인** (`call_fn` 주입 / monkeypatch). 라이브 런은 이 플랜 범위 밖 — 프록시 8790·OpenRouter 쿼터는 Track A 소관.
- 결과 JSON 변경은 **추가 또는 리네임+폴백**만. 소비자(`report2.py`, `viewer.py`, `scripts/night_run.sh`)를 같은 커밋에서 함께 고친다.
- `dreaming_data/*`, `.env`, `research/*` 커밋 금지. API 키 값 echo 금지.
- `external/` 이하는 읽기 전용 — 수정 금지.
- 카드 지식 선취 금지 — `NPC_NAME`(당채련) 유도만 예외 (`config.py:29` 주석).
- **패치 표면 규약** (CLAUDE.md §5): 오버라이드가 도달해야 하는 호출부는 `module.NAME` 점 접근. import 시점 별칭은 스냅샷이라 무효.
- 각 태스크 끝에 커밋. 커밋 전제는 pytest 통과.

## 파일 구조 (최종 상태)

| 파일 | 변화 |
|---|---|
| `benchmarks/eval/lucid.py` | `director.py`에서 리네임. + `mask_value`, `probe_leaks_value` |
| `benchmarks/eval/matching.py` | **신설** — 값 동치 매칭 최하층 (`_norm`, `expect_alternatives`, `value_hit`) |
| `benchmarks/eval/gates.py` | **신설** — 런 유효성 게이트 판정 (계약 §6) |
| `benchmarks/eval/prompts/lucid_{rules,persona}.md` | **신설** (Task 4) |
| `benchmarks/eval/prompts.py` | 층 로더 + `compose_lucid_sys()` + `layer_hashes()` |
| `benchmarks/eval/config.py` | `DIRECTOR_MODEL`→`LUCID_MODEL`, env `DREAMING_EVAL_LUCID` |
| `benchmarks/eval/transport.py` | `make_director_llm`→`make_lucid_llm` |
| `benchmarks/eval/hypa.py` | `_director_model()`→`_lucid_model()` (**실제 식별자** — 주석 아님) |
| `benchmarks/eval/scoring.py` | `matching`에서 재수출 (`oracle.py:15` 호환 유지) |
| `benchmarks/eval/run2.py` | 유출 게이트·전달률·게이트 판정 배선. 결과 키 `cost_lucid`/`lucid_calls`/`sec_lucid`/`lucid_model` |
| `benchmarks/eval/report2.py`, `viewer.py`, `scripts/night_run.sh` | 신키 우선 + 구키 폴백, 게이트 표기 |
| `docs/dreaming/EVAL2.md` | `:56` 표 정정 (D1) |
| `docs/LUCID-CONTRACT.md` | §6 G1 재정의 (A7) |

---

### Task 0: 전면 리네임 (director → lucid) + 모델 박제

동작 변경은 2건뿐 (`lucid_model` 추가, 구 env 폴백). 나머지는 순수 이동 — 리네임 diff와 개편 diff를 분리해야 리뷰가 읽힌다.

**Files:**
- Rename: `benchmarks/eval/director.py` → `benchmarks/eval/lucid.py` (`git mv`)
- Modify: `config.py`, `transport.py`, `run2.py`, `scoring.py`, `judge_check.py`, `hypa.py`, `prompts.py`(주석), `report2.py`, `viewer.py`, `scripts/night_run.sh`, `tests/test_eval_v2.py`, `tests/test_eval_hypa.py`

**Steps:**
- [ ] `git mv benchmarks/eval/director.py benchmarks/eval/lucid.py`. docstring "디렉터"→"Lucid(유저 시뮬레이터)". 별칭 재노출 `_EXTRACT_SYS`/`_PROBE_SYS`/`_FALSE_SYS`는 **유지** (기존 테스트가 이 이름으로 참조 — `lucid.py:19-21`).
- [ ] `config.py:15` — `DIRECTOR_MODEL`→`LUCID_MODEL`, env `DREAMING_EVAL_LUCID`. **구 env 폴백 포함**: `DREAMING_EVAL_DIRECTOR`가 있고 신 env가 없으면 그 값 사용 + `sys.stderr` 1회 경고. 근거: 셸에 구 export가 남아 있으면 조용히 기본 모델로 떨어져 런 전체가 다른 모델로 돈다.
- [ ] `transport.py` — `make_director_llm`→`make_lucid_llm`, `SUMMARY_MODEL` 기본값 `config.LUCID_MODEL`, `from benchmarks.eval.lucid import LlmFn`.
- [ ] `run2.py` — import 경로, 지역변수 `director`→`lucid`, `_record_probe` 파라미터명, 결과 키 3종 (`cost_director`→`cost_lucid`, `director_calls`→`lucid_calls`, `turns[*].sec_director`→`sec_lucid`), `main()` 출력 문자열.
- [ ] **`_collect_totals`에 `"lucid_model": config.LUCID_MODEL` 추가** (D6 절반). `MODEL`(나레이터)은 이미 최상위에 있으므로 `totals` 안에 넣어 축을 구분한다.
- [ ] `scoring.py:28`, `judge_check.py:31` — import 경로만.
- [ ] **`hypa.py`는 주석이 아니라 코드다** — `_director_model()` 정의(214행)와 호출부 2곳(243·314행)을 `_lucid_model()`로 바꾸고, `tests/test_eval_hypa.py:226`의 `hypa._director_model()` 호출도 함께 바꾼다. 플랜 v1이 "주석만"이라 적은 건 오기였다.
- [ ] `prompts.py` 주석 정정 — 3행 `director.py(추출·프로브·false)`, 55·66·83행의 `director.extract_facts`/`make_probe`/`make_false_premise` 모듈 경로 표기.
- [ ] **소비자 폴백**: `viewer.py:114-117` → `t.get("cost_lucid", t.get("cost_director"))` / `t.get("lucid_calls", t.get("director_calls", 0))`. **`scripts/night_run.sh:90`** (실조회 확정 — v1 ":82", C2 추정 "~:90"). 근거: 어제 본런 JSON(`v2-night2-*`)이 계속 렌더돼야 비교가 된다.
- [ ] **`transport.py`의 혼재 import 주의** — 17행이 `from ...config import DIRECTOR_MODEL`(import 시점 별칭)인데 22행은 `config.DIRECTOR_MODEL`(점 접근)이다. 리네임은 **현 동작을 보존**한다 (별칭→별칭, 점 접근→점 접근). 이 혼재를 고치는 건 별건. 다만 **env 폴백 테스트는 `config.LUCID_MODEL`을 단언 대상으로 삼는다** — `config`를 리로드해도 `transport`가 잡아둔 별칭 스냅샷은 안 바뀐다 (CLAUDE.md §5 그대로).
- [ ] `tests/test_eval_v2.py` — `from benchmarks.eval.director import` 8곳, `monkeypatch.setattr(run2, "make_director_llm", ...)` 2곳, 픽스처 `"sec_director"` 1곳. **추가로** 테스트 함수명 4곳(785·804·973·982행)과 주석 4곳(1114·1123·1153·1176행)이 남는다 — 함수명·주석도 같이 바꿔야 Verify grep이 통과한다.
- [ ] 신규 테스트 2건: ① `totals["lucid_model"]`이 `config.LUCID_MODEL`과 일치 ② 구 env만 설정 시 폴백 동작 (`monkeypatch.setenv` + 모듈 리로드).

**Verify:**
- [ ] `python3 -m pytest -q` → **661 passed, 4 skipped**.
- [ ] `python3 -m ruff check benchmarks/eval/ dreaming/` 신규 0.
- [ ] `grep -rn "director" benchmarks/ scripts/ tests/` → 남은 건 한국어 산문뿐. **Python 식별자가 하나라도 남으면 실패다.**
- [ ] 구 JSON 렌더 회귀: **이 워크트리에 `dreaming_data/`가 없다** (gitignore — Track A 워크트리 소유). 원 워크트리 `/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696/dreaming_data/eval/`의 구 본런 JSON 하나를 **읽기 전용**으로 지목해 `viewer` 실행 → 비용 칸에 값이 뜬다. 거기 아무것도 쓰지 말 것. 접근 불가면 구 키만 든 최소 JSON을 스크래치패드에 만들어 대신 검증하고 리포트에 명시.

**Commit:** `refactor(eval): 디렉터 → Lucid 전면 리네임 + lucid_model 박제, 소비자는 구키 폴백`

---

### Task 1: 정답 유출 차단 (계약 D5 / I1·I2)

**현행 결함**: `lucid._probe_user()`가 `(핵심값: {fact.value})`를 프로브 생성기에 그대로 넘긴다. 발화에 정답이 섞였는지 검사하는 코드는 **0건**이다 (`_probe_mentions_fact_object`는 반대 방향 검사 — 대상 명사가 **빠졌는지** 보는 것이고, 하드 차단 없이 `drift_suspected` 로깅만 한다).

**Files:** Create `benchmarks/eval/matching.py`. Modify `benchmarks/eval/scoring.py`, `lucid.py`, `run2.py`, `tests/test_eval_v2.py`

#### 1a. 순환 import 해소 — `matching.py` 신설 (A1)

`lucid`가 값 동치 매칭을 쓰려면 `scoring`을 봐야 하는데 `scoring.py:28`이 이미 `lucid`를 본다. 최하층을 하나 판다.

- [ ] `benchmarks/eval/matching.py` 신설. eval 내부 import **금지** (`config.py`와 같은 최하층 규약). 의존은 `re` + `dreaming.numerals.korean_spellings`뿐.
- [ ] `scoring.py`에서 **이동**: `_WS`, `_DIGITS`, `_norm()`, `expect_alternatives()`, 그리고 `_hit()`을 **`value_hit()`으로 공개 이름 변경**해 이동. `_STATBAR`는 스탯바 벗기기라 scoring에 남긴다.
- [ ] `scoring.py`는 재수출로 호환 유지: `from benchmarks.eval.matching import _norm, expect_alternatives, value_hit as _hit  # noqa: F401`. **근거**: `oracle.py:15`가 `from benchmarks.eval.scoring import _STATBAR, _norm, expect_alternatives`를 하고 있어 그대로 살려야 한다.
- [ ] `value_hit(hay, expected_value)`의 `hay`는 **이미 `_norm` 처리된 문자열**을 전제한다 (현행 `_hit`의 계약 그대로). 새 호출부는 이걸 지킨다 — docstring에 명시.

#### 1b. 마스킹 + 하드 게이트

- [ ] `lucid.mask_value(text: str, value: str) -> str` — `text` 안의 `value` 출현을 `◻︎`로 치환. **가드**: `len(value) < 2`면 마스킹하지 않고 원문 반환 (짧은 값이 무관한 글자를 지워 `fact.text`를 훼손하는 것 방지).
- [ ] `_probe_user()` — `make_probe` 경로에서 `fact.text`를 마스킹하고 `(핵심값: ...)` 꼬리를 **삭제**. `make_false_premise`는 오염값 생성에 원값이 필요하므로 **예외** (계약 I2) — 시그니처에 `mask: bool = True`를 두고 false 경로만 `False`.
- [ ] `lucid.probe_leaks_value(utext: str, fact: DirFact) -> bool` — `matching.value_hit(matching._norm(utext), fact.value)`. 표기 변형("250"/"이백오십")까지 잡힌다.
  - **비대칭 주의**: `mask_value`는 리터럴 치환이고 게이트는 변형까지 본다. 변형 표기를 가진 사실은 재생성으로 몰릴 수 있다 — 발생 빈도를 `probe_leak_retries`로 관측하고, 높으면 후속에서 마스킹을 변형까지 확장한다. 이 절충을 코드 주석에 남긴다.

#### 1c. run2 배선

- [ ] **선행 리팩터**: 현행 필러 생성은 `run2.py:384-398`의 `else:` 블록에 인라인돼 있고 NPC 유인 스케줄(`npc_due`)과 얽혀 있다. 재사용 가능한 `_filler_turn(...) -> str`로 **추출**한다. 이걸 안 하면 "필러로 강등"을 구현할 수 없다.
- [ ] 프로브 분기(**현행 370-398행** — v1의 "364-384"는 오기로, 필러 로직을 잘라먹은 범위였다): 생성 → `probe_leaks_value`면 **1회 재생성** → 재위반이면 그 턴을 `_filler_turn(...)`으로 **강등**하고 `fact.probed = False`로 되돌린다 (사실을 태우지 않는다). false 프로브는 **원값** 기준으로 같은 검사 (발화에는 오염값만 있어야 한다).
- [ ] `totals`에 2키 추가: `probe_leak_retries`, `probe_leak_dropped`. 0이 정상.

**Steps(테스트):** 신규 7건
- [ ] ① `mask_value`가 값을 지우고 나머지 문장은 보존 ② 1글자 값은 마스킹 안 함 ③ 마스킹된 `_probe_user` 출력에 `fact.value` 부재 ④ `probe_leaks_value`가 한글 수사 변형을 잡음 ⑤ 1회 누출 후 재생성 통과 시 정상 기록 ⑥ 2회 누출 시 필러 강등 + `fact.probed is False` + `probe_leak_dropped == 1` ⑦ **순환 회귀**: `matching.py` 소스에 `benchmarks.eval` import가 없고, `import benchmarks.eval.lucid`를 첫 import로 해도 성공.

**Verify:**
- [ ] `python3 -m pytest -q` → **668 passed, 4 skipped**.
- [ ] 오프라인 20턴 스텁 런 → `probe_leak_dropped == 0`.

**Commit:** `feat(eval): Lucid 정답 유출 차단 — 프로브 값 마스킹 + 누출 하드 게이트, 매칭 최하층 분리`

---

### Task 2: 런 유효성 게이트 (계약 §6)

**목적**: 런이 무효일 때 결과 JSON만 보고 무효임을 알 수 있게 한다. 지금은 꿈이 0회 돌아도, 프로브가 절반만 나가도, 결과가 똑같이 "회수율 X%"로 보인다.

**Files:** Create `benchmarks/eval/gates.py`. Modify `prompts.py`, `run2.py`, `report2.py`, `docs/LUCID-CONTRACT.md`, `tests/test_eval_v2.py`

**Steps:**
- [ ] **`prompts.layer_hashes() -> Dict[str, str]` 신설** (A4 — Task 4에서 앞당김). `active()`의 각 항목에 sha256 앞 12자. 층과 무관하게 현재 `_NAMES` 8개를 덮고, Task 4가 10개로 늘리면 자동 확장된다.
- [ ] `run2` 계측 3키 추가:
  - `totals.probes_scheduled` — `sched`(`run2.py:351`)의 non-None 개수. `probes_delivered`는 기존 `totals.probes`가 그 역할이므로 재사용 (감사 확인: `probes`는 전달 성공분만 append하므로 비율이 항상 1.0이 되지 않는다).
  - `totals.dream_ran` / `totals.episodes_written` / `totals.compression_planned` — dreaming 변형에서 각각 `DATA/{session}/dreamer/cursor.json` 존재, `DATA/{session}/episodes/*.json` 개수, `DATA/{session}/compression/plan.json` 존재 (런 종료 시 1회). 다른 변형은 `None`. 경로 규약은 `scoring.decompose_miss`(`scoring.py:140-148`)의 `base/{kind}/*.json`을 따른다.
    - **A7 근거 주석 필수**: 사이클 수는 셀 수 없다. `dreaming/dreamer.py:341`이 cursor를 단일 문서로 덮어쓰고, 한 사이클이 episode를 여러 개 뱉을 수 있다. 이 키들은 "Dreamer가 돌았는가 / 압축까지 갔는가"만 증명한다.
    - **`compression_planned` 근거 (C6)**: `dreamer.py:345-346`이 `plan is not None`일 때만 `{session}/compression/plan.json`을 쓴다. 압축이 한 번도 성립하지 않으면 **파일 자체가 없다** — 모호함 없는 신호다. 본런 실측이 정확히 이 상태였다 (에피소드 76개 / 청크 0개, `DREAMING_FLAW.md:22-23`).
  - `prompt_hashes` — 최상위 필드. `prompts.layer_hashes()` 결과.
- [ ] `gates.py` — **`evaluate(result: Dict) -> Dict[str, List[Tuple[str, str]]]`**, 키는 `"failed"` / `"warnings"`, 값은 `(게이트 id, 사유)` 목록 (A5로 반환형 확정). 계약 §6 표:
  - G1 `dream_ran and episodes_written >= 1 and compression_planned` (dreaming만) — **A7로 재정의, C6로 압축 조건 추가**. 압축 버그가 고쳐지기 전까지 dreaming 런이 여기서 계속 실패하는 것은 정상이다 — "이 런은 압축 층을 시험하지 못했다"가 사실이므로. 게이트를 느슨하게 만들어 통과시키지 말 것.
  - G2 `distance_turns` 중앙값 ≥ 15; trim/hypa는 추가로 `in_window=False` ≥ 50% (**dreaming/vanilla는 `windowing.FULL_HISTORY`라 항상 창내 — 후자 비대상**)
  - G3 `probe_leak_dropped == 0`
  - G4 `probes / probes_scheduled >= 0.8`
  - G5 `truncated == 0` / G6 `flawed == 0 and aborted == ""` / G7 `judge_unparsed == 0`
  - G8 `lucid_model` 존재 and `prompt_hashes` 비어있지 않음
  - **G9(judge-사람 일치율)는 자동 판정 불가** — 항상 `warnings`에 "미검증" 항목을 넣고, 계약 §6에 따라 judge 기반 지표가 유보 상태임을 표기한다.
  - 구 JSON(신규 키 부재)에서 `KeyError`로 죽지 않는다 — 없는 키는 해당 게이트 실패로 처리.
- [ ] `run_once` 종료 시 `result["gates"] = gates.evaluate(result)` 기록. **런을 죽이지는 않는다** — 부분 결과도 감사 가치가 있다. `main()`이 실패 게이트가 있으면 비영점 종료 (`night_run.sh`가 감지).
- [ ] `report2.render` — 변형별 줄에 실패 게이트 id를 붙인다. 실패가 있으면 그 변형의 회수율 옆에 `[무효]` 표기.
- [ ] **`docs/LUCID-CONTRACT.md` §6 정정** — G1의 `dream_cycles >= 2`를 위 재정의로 갱신하고 A7 근거 한 줄을 단다. 계약이 코드보다 앞서 틀린 상태로 남으면 다음 플랜이 또 물린다.

**Steps(테스트):** 신규 10건 — 게이트 8개 각각 통과/실패 케이스 (G1은 3조건이라 `compression_planned` 부재 케이스를 따로 1건) + `FULL_HISTORY` 변형에서 G2 후반부 비대상 + `layer_hashes()` 8키·내용 변경 시 해시 변화.

**Verify:**
- [ ] `python3 -m pytest -q` → **678 passed, 4 skipped**.
- [ ] 구 본런 JSON 4건에 `gates.evaluate`를 돌린다 (신규 키 부재 → G8/G4/G1이 실패로 뜨는 게 정상, `KeyError` 아님). **구 JSON에서 어떤 게이트가 뜨는지가 곧 "그 런이 뭘 증명 못 했는가"의 목록이다.** 결과를 커밋 메시지 본문이나 실행 후기에 남긴다.

**Commit:** `feat(eval): 런 유효성 게이트 — 꿈 실행·유출·전달률·프로비넌스 자동 판정`

---

### Task 3: TTL 재압축 창구 개방 (계약 D3) + EVAL2 정정 (D1)

**여는 것**: C11(TTL 재압축 한계비용 0), C12(캐치업 드림). 계약 §2에서 유일하게 "코드가 이미 있는데 안 켜서 미시험"인 항목이다.

**실측 근거**: 12초 sleep으로 유휴 Dreamer는 이미 트리거된다 (`night-smoke-r0` 6턴에 episodes 5건). 즉 **C3는 시험되고 C11은 안 된다** — 유휴 임계와 TTL(5m)이 다른 값이기 때문. 이 구분이 이 태스크의 존재 이유다.

**Files:** `scripts/night_run.sh`, `benchmarks/eval/run2.py`(주석만), `docs/dreaming/EVAL2.md`, `tests/test_eval_v2.py`

**Steps:**
- [ ] `night_run.sh` — `run_variant()`에 `--ttl-wait` 추가, `TTL_WAIT="${TTL_WAIT:-1}"` env로 끌 수 있게. 스모크는 이미 `run_variant()`와 분리된 경로라(감사 확인) 자연히 제외된다 — 분리 상태를 주석으로 못박는다.
- [ ] `night_run.sh` 헤더 주석 + `say` 로그에 예상 추가 시간 명시: 100턴 기준 10회 × 305초 = **+51분**.
- [ ] **`run2.py:425`** 주석 보강 (v1의 ":424"는 조건행) — "TTL 5m 만료 재현 (옵션)"을 "C11/C12 개방용. 이걸 끄면 TTL 재압축 창구는 미시험이며 캐시율은 과대측정 (`EVAL2.md:93`)"으로.
- [ ] **`docs/dreaming/EVAL2.md:56` 정정 (D1)** — "지뢰·프로브 턴은 고정 대사(시뮬레이터 0관여), 필러만 시뮬레이터"를 §3(`:99-105`)의 동적 추출 방식으로 갱신하고, `:99` 전환 문단을 참조로 건다. 표가 갱신 누락된 것이지 설계 충돌이 아님을 한 줄로 명시.

**Verify:**
- [ ] `bash -n scripts/night_run.sh` 문법 통과.
- [ ] `python3 -m benchmarks.eval.run2 --help`에 `--ttl-wait` 노출 확인.
- [ ] 신규 테스트 2건 — **`time.sleep` 스파이는 반드시 인자 305로 필터한다.** `run2.py:421-423`에 `ttl_wait`와 무관한 `time.sleep(12)` 드림 트리거가 dreaming 변형에서 별도로 뜬다 (감사 발견). ① `total_turns=3, ttl_wait=True` → `sleep(305)` 0회 (`i % 10 == 9` 미도달) ② `total_turns=10` → `sleep(305)` 1회.
- [ ] `python3 -m pytest -q` → **682 passed, 4 skipped**. (Task 2 수정 라운드에서 종료코드 테스트 2건이 추가돼 기준이 678→680으로 올라갔다.)

**Commit:** `feat(eval): TTL 재압축 창구 개방 — 본런 --ttl-wait 기본 ON + EVAL2 지뢰 방식 표 정정`

---

### Task 4: Lucid 프롬프트 2층 분리 (RULES / PERSONA)

**순위 근거**: 그 자체로는 SPEC 주장을 하나도 열지 않는다 (계약 §7에서 4순위). A/B·버전관리의 전제라 P1에 남긴다.

**범위 축소 근거 (A2·A3)**: `DIRECT_SYS`에 실재하는 문장만 옮긴다. `[작품 설정]`은 `run2.py:396`이 턴 루프 안에서 붙이는 별도 접합이고 `{beat}`는 **user 메시지**(398행)라 시스템 프롬프트에 없다 — SCENARIO 층은 만들지 않는다. v1 층 표에 있던 "복창 금지"·"날조 금지"는 `DIRECT_SYS`에 존재하지 않아 삭제했다.

**적용 범위 (기록)**: `dir_sys`는 **필러 턴에서만** 쓰인다 (`run2.py:396`). 프로브 턴은 `PROBE_SYS`/`FALSE_SYS` 경로라 이 분리의 영향을 받지 않는다.

**Files:** Create `benchmarks/eval/prompts/lucid_rules.md`, `benchmarks/eval/prompts/lucid_persona.md`. Modify `prompts.py`, `run2.py`, `tests/test_eval_v2.py`

**층 분담 (τ² 비중복 원칙) — 문장 단위, 바이트 보존**

| 층 | 문장 |
|---|---|
| PERSONA | "너는 RP에서 유저(1인칭{user}) 역할을 연기한다." / "상대는 연상이자 신비한 존재다 — 정중한 존댓말을 쓴다 (반말 금지)." / "예의는 지키되 굽신거리지 마라 — …" |
| RULES | "작품 설정과 직전 장면에 자연스럽게 이어지는 유저 발화 하나만 출력." / "3문장 이내, 메타 발언 금지." / "상대 캐릭터의 대사나 행동을 네가 대신 쓰지 마라 — …" / 2문단 전체(미공개 정보 선취 금지) / 3문단 전체(조연 무시·조기 퇴장 금지) |

`{user}` 슬롯은 **PERSONA에만** 있다 — 분할이 슬롯을 쪼개지 않는다.

**Steps:**
- [ ] 2개 .md 생성. 베이스 커밋 `2f1a56e`의 `DIRECT_SYS` 본문을 위 표대로 **분해 이동** — 문구는 바이트 보존, 소속만 바꾼다.
- [ ] `prompts.py` — `_LAYER_DIR = pathlib.Path(__file__).parent / "prompts"`, `_read()` 로더. `LUCID_RULES` / `LUCID_PERSONA`를 모듈 전역으로 노출하고 `_NAMES`에 추가 (8→10) → `layer_hashes()`(Task 2)가 자동 확장.
- [ ] `DIRECT_SYS = "{persona}\n\n{rules}"` (메타 템플릿).
- [ ] `prompts.compose_lucid_sys(user: str) -> str` — **단일 조립 지점**. 2단계로 조립한다:
  ```
  tpl = globals()["DIRECT_SYS"].replace("{rules}", globals()["LUCID_RULES"]) \
                               .replace("{persona}", globals()["LUCID_PERSONA"])
  return tpl.format(user=user)
  ```
  - **`.format()`을 1단계에 쓰면 안 된다** — 층 본문의 `{user}`를 만나 `KeyError`가 난다. `str.replace`로 층을 심고, 그 다음에야 `.format(user=...)`을 건다.
  - **`globals()` 접근 필수** (CLAUDE.md §5 / Global Constraints): import 시점 별칭은 스냅샷이라 `override_from`이 안 먹는다. `active()`가 이미 같은 방식이다.
- [ ] `run2.py:346` — `prompts.DIRECT_SYS.format(user=...)`를 `prompts.compose_lucid_sys(user=...)` 호출로 교체. 347-348행의 few-shot 접합과 396행의 `[작품 설정]` 접합은 **그대로 둔다**.
- [ ] 테스트 이관: `_DIRECT_SYS` 문자열 단언 **4건**(v1의 "3건"은 오기 — 787·806·978·986행)의 대상을 `prompts.LUCID_RULES`/`prompts.LUCID_PERSONA`로 변경. **단언 문구 자체는 그대로.**
- [ ] 신규 테스트 5건: ① 조립 결과가 2층을 각각 정확히 1회 포함 ② `LUCID_RULES` 오버라이드가 조립에 반영 ③ `DIRECT_SYS`(메타 템플릿) 오버라이드도 반영 ④ `layer_hashes()`가 10키로 확장, 층 변경 시 해시 변화 ⑤ **등가성** — `compose_lucid_sys(user=X)` 결과의 정규화 텍스트가 베이스 커밋 `DIRECT_SYS.format(user=X)`와 같은 문장 집합 (순서·구분자 차이는 허용, 문장 누락·중복은 실패). 기대값은 테스트 안에 리터럴로 박는다. **이 테스트가 없으면 Task 4도 프롬프트 변경이 돼 이후 델타를 개편에 귀속시킬 수 없다.**

**Verify:**
- [ ] `python3 -m pytest -q` → **687 passed, 4 skipped**.
- [ ] `python3 -m ruff check benchmarks/eval/ dreaming/` 신규 0.
- [ ] 층 중복 육안 점검 — 두 .md에 같은 규칙이 두 번 없는지 (자동 검사 불가, 리뷰 항목).

**Commit:** `feat(eval): Lucid 프롬프트 2층 분리 — 규칙/페르소나 .md + 조립 단일 지점`

---

## 범위 밖 (명시)

| 항목 | 왜 미룸 | 계약 순위 |
|---|---|---|
| **리롤·편집 직후 프로브 (구 Task 5, SPEC C8)** | **A6 — `MIN_PROBE_AGE=15`가 나이 1짜리 사실을 `eligible()`에서 구조적으로 걸러낸다. `turn_range`를 얹어도 교집합이 공집합이라 매 런 자기 스킵.** 게다가 기본 `--reroll-at 18,33`이면 턴 19에 이미 `relation` 슬롯이 있어 충돌 정책도 미정의. 지연 프로브(15턴 뒤)로 갈지 min_age 우회로 갈지는 계약에 먼저 올린다 | 5 |
| 종료 다중화 + 저항 카운터 | 고정 턴수가 개편 전후 비교성에 오히려 유리 | 6 |
| 실패 감지 1급 지표화 (OOC/모순/반복) | 시뮬레이터 자체 품질 축 — P1 판정의 전제 아님 | 7 |
| 목표=상태 (인-픽션 목표 객체) | 프로브 집행 원장은 Task 2에 흡수됨. 인-픽션 목표는 별건이고 SPEC 주장을 안 엶 | 8 |
| 복선/POV/서술-상태 프로브 유형 신설 (C21·C16·C5) | 비용 큼. 절충안 = 대본이 상황을 **발생**만 시키고 채점은 후속 | 9 |
| SCENARIO 층 신설 / Lucid 규칙 추가("복창 금지"·"날조 금지") | A2·A3 — Task 4를 순수 이동으로 묶어야 등가성 테스트가 의미를 갖는다. 규칙 추가는 그 다음 판 | — |
| 비협조 페르소나 | 협조 편향은 구조로만 교정되는데 그 구조(상태 카운터·확률 주입·필수정보 큐)가 전부 별건 | 계약 §1-2 |
| **프로브 그럴듯함 필터** (Track A C5) | 거짓 전제가 **화자 자신이 착각할 수 없는 사실**(자기 이름 → "저, 한결이라고 합니다" 실측)을 오염시키고, 장면 상태 무시(전투 중 자기소개 요구)로 죽은 프로브 발생. D5 마스킹과 별개 축 — 대상 선정에 화자-자신 사실 제외 + 오염값을 작품 어휘에 묶기 + 장면 게이팅. P2로 | — |
| **PERSONA 카드 준수** (Track A C4) | 존댓말 강제 제거 → 렌 카드 말투(반말) 준수. Task 4 완료 직후 `lucid_persona.md` 단일 파일 수정 — 등가성 테스트와 분리하기 위해 이 플랜에선 금지 | — |
| **G9 judge-사람 일치율 수동 감사** | 코드 작업이 아니라 라벨링 작업 (10~20% 표본). **미실시 상태에서는 judge 기반 지표 전부 유보** — Task 2의 `evaluate`가 매번 경고로 알린다 | 계약 §6 |
| 리포트 "점수 = 상한선" 문구 (계약 I6) | 비용 0이지만 리포트 생성 시점 작업 — 본런 결과 분석 때 | 계약 §1 |

## 실행 후기 (구현 완료 후 작성)

**착수 전 감사에서 이미 확정된 v1 플랜 결함** (CLAUDE.md §5 — 구현 전에 잡힌 사례):

- BLOCKER 7건 / WRONG 8건. 원인 분류:
  - **의존 방향을 코드 확인 없이 서술** (A1) — "scoring이 oracle을 역수입"은 정반대였다. 심볼 소유자를 `grep -n "^def "`로 확인하지 않은 결과.
  - **존재하지 않는 문구를 이동 대상으로 지정** (A2·A3) — 층 표를 프롬프트 본문 대조 없이 작성.
  - **태스크 간 심볼 순서 미검** (A4) — 뒤 태스크가 만드는 함수를 앞 태스크가 호출.
  - **반환형과 저장 형태 불일치** (A5) — 같은 태스크 안 두 줄 사이에서 모순.
  - **기존 가드를 못 본 신규 기능** (A6) — `MIN_PROBE_AGE`를 안 읽고 "직후 프로브"를 설계.
  - **관측 불가능한 지표 요구** (A7) — 디스크에 사이클 카운터가 없는데 `dream_cycles >= 2`를 게이트로 지정.
  - **행 번호·개수 오기 다수** — `:82`→`:88`, `364-384`→`370-398`, `:424`→`:425`, 단언 3건→4건, 리네임 표면 3파일 누락.
- **다음 플랜에 적용할 규칙**: 플랜에 적는 모든 `file:line`·심볼 소유자·문자열 존재·개수는 작성 시점에 실제로 조회한 것만 적는다. 조회 안 했으면 "확인 필요"로 표시하고 태스크 첫 스텝에 조회를 넣는다.

**실행 단계에서 드러난 플랜 결함** (Task 0~4 구현 + 태스크별 리뷰 + 전체 리뷰)

| # | 결함 | 어디서 터졌나 | 다음 플랜 규칙 |
|---|---|---|---|
| E1 | **행 번호 부패가 체계적이다** | 5개 태스크 **전부** 플랜의 `file:line`을 최소 1건씩 틀린 것으로 발견. Task 3은 자기가 삽입한 2줄 때문에 **자기가 방금 쓴 상호참조**를 밀어냈다 | 플랜은 행 번호가 아니라 **심볼·문자열**로 지목한다. 행 번호를 쓸 거면 태스크 첫 스텝에 재조회를 넣는다 |
| E2 | **자기가 만든 고아를 안 치운다 (4/5 재발)** | T0 `hypa.py` docstring이 없어진 `DIRECTOR_MODEL`을 계속 호명 · T1 `wrong = ""`이 자기 리팩터로 죽음 · T3 자기 삽입으로 밀린 참조 · T4 `_DIRECT_SYS` 별칭(소비자 4건을 자기가 전부 이관해놓고 별칭만 남김) | 태스크 체크리스트에 **"내 변경이 무엇을 고아로 만들었나" grep**을 명시 스텝으로 넣는다. CLAUDE.md §3이 이미 요구하지만 구현자는 "기존 dead code는 두라"는 반대 조항으로 빠져나간다 |
| E3 | **기준선이 실행 중에 움직인다** | Task 2 수정 라운드가 테스트 2건을 추가 → 678이 680이 됨 → Task 3·4 기대치를 두 번 재유도 | 기대 수를 절대값이 아니라 **"기준선 + 신규 N건"**으로 적는다 |
| E4 | **종료 코드·반환 계약 변경의 소비자를 안 셌다** | Task 2가 `main()`을 게이트 실패 시 비영점 종료로 바꿈 → `night_run.sh` 스모크 분기가 그 코드를 읽음 → 압축 버그로 G1이 상시 실패 → **dreaming 본런이 매일 밤 영구 스킵**될 뻔. 플랜의 Task 2 파일 목록에 `night_run.sh`가 없었다 | 프로세스 종료 코드·예외·반환 타입을 바꾸는 태스크는 **호출부를 플랜에 열거**한다. 파일 목록이 곧 blast radius 선언이다 |
| E5 | **두 태스크가 같은 술어를 반대편에서 건드렸는데 조합을 아무도 안 봤다** | T1은 `len(value) < 2`면 마스킹을 **건너뛰고**, T2의 G3는 그 값도 **검사한다**. 실측 프로브 126건 중 21건(17%)이 1글자 값이고 최빈값은 화자 본인 이름 `렌`(16회) → 자연 발화가 곧 유출 → 2회 드롭 → **런 전체 무효**. 태스크별 리뷰는 각자 통과시켰고 전체 리뷰가 잡았다 | 두 태스크가 같은 조건식을 다른 각도에서 만지면 플랜이 **조합 진리표**를 적는다 |
| E6 | **"바이트 보존"과 실제 산출물이 어긋났다** | Task 4는 문장 집합은 보존했지만 **순서를 바꾸고 문단 구분을 4개 추가**했다. 등가성 테스트가 순서 무관(`Counter`)이라 통과. 플랜은 "바이트 보존"이라 적었다 — 테스트와 문구가 불일치. **필러 턴 시스템 프롬프트는 지난 본런과 다른 토큰 열이다** | "보존"의 단위(바이트/문장/의미)를 플랜이 명시하고, 테스트가 그 단위와 일치하는지 확인한다 |
| E7 | 계약 문서가 태스크 커밋에 묻어 들어갔다 | 플랜은 `docs/LUCID-CONTRACT.md` **§6만 정정**하라 했는데, 파일이 미추적 상태였어서 Task 2 커밋에 132줄 전체가 처음으로 들어갔다 | 플랜이 참조하는 문서의 **추적 상태를 Global Constraints에 적는다** |

**리뷰 체제에 대한 관찰**: 태스크별 리뷰 5회는 전부 통과시켰지만 전체 리뷰가 Important 4건을 더 잡았다(E5 포함). 잡힌 것들의 공통점은 **태스크 경계를 가로지른다**는 것 — 단일 diff 안에서는 결함으로 안 보인다. 태스크별 리뷰를 늘리는 것으로는 이 계열이 안 잡힌다.

<!-- 구현 중 새로 발견되는 결함은 아래에 추가 -->
