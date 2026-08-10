# P0/P1 실소스 충실화 Implementation Plan (v2 — 크리틱 반영)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 자작 흉내(retrieval)를 실제 RisuAI HypaV3 알고리즘의 충실 재현(hypa 변형)으로 교체하고, dreaming 프록시 격리 연쇄를 수술하며, P1 하네스 4건(값-생존·리롤 사유·중복 게이트·프로브 품질)을 반영한다.

**Architecture:** (1) `hypav3.ts`(exp 경로)에서 추출한 알고리즘을 `benchmarks/eval/hypa.py`로 포팅 — 발동·배칭·요약·선택·주입 전부 file:line 근거 스펙 기반, 뮈토스 하이파 V5 export 설정 고정. 주입은 뮈토스 프리셋 memory 카드(index 35) 위치 = 시스템 프롬프트 한가운데 → 캐시 파괴 병리를 그대로 재현하고 **캐시 히트율로 측정한다**. (2) dreaming 프록시는 검증 완료된 3건 수정(베이스라인 가드/자기치유/벤치 풀히스토리). (3) P1은 기존 순수 함수 패턴을 따른다.

**Tech Stack:** Python 3.13, pytest, tiktoken(o200k_base), sentence-transformers(MiniLM 임베딩 — 무거움, 테스트는 페이크 embed 주입이 기본 경로), OpenRouter (요약: Gemini 3 Flash — 뮈토스 subModel과 동일 계열, 라우팅은 Vertex 플러그인 vs OpenRouter로 상이).

## Global Constraints

- API 키 값 출력·커밋 금지. `dreaming_data/*`·`.env` 커밋 금지. NSFW 원문 재현 금지. 결과 스핀 금지.
- 테스트 기준선: `python3 -m pytest -q tests/` = **전체 599 collected (test_eval_v2.py 72개 포함)**. 태스크마다 "599+누적 신규 N, 실패 0"으로 관리. 린트 게이트: `python3 -m ruff check benchmarks/eval/ dreaming/` — 기존 에러 수(6, 전부 E402/기존 항목) 초과 금지. 무인자 `ruff check`(전 저장소 89건)는 게이트 아님.
- 조사 근거 문서 (모든 수치·라인번호의 출처):
  - hypaV3 스펙: `docs/superpowers/plans/2026-08-09-refs/hypav3-algorithm.md` (hypav3.ts:129-1893) — **§2 의사코드와 §7 체크리스트 11개 전부가 규범**
  - RisuAI 트림: `2026-08-09-refs/risuai-trim.md`, 프록시 수술: `2026-08-09-refs/dreaming-proxy-surgery.md` (검증 완료), P1 매핑: `2026-08-09-refs/harness-p1-mapping.md`
  - 크리틱 리뷰 반영 (v2): variants.py 보존, memo 계약, 에러 4튜플, 스모크 26K, 기준선 정정
- RisuAI 소스: `external/risuai/src/ts/process/memory/hypav3.ts`, hypa export: `/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 하이파/hypaV3_export_뮈토스 하이파 V5.json`
- 프리셋: `/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup`, 카드: `dreaming_data/eval/card-soyeon-v2.json`
- **프리셋+카드 고정 비용 실측 11,552 토큰** (빈 히스토리 build_wire 12,022 − greeting 467 − user 3) — 발동 산술의 기준값.

## 확정 설계 결정

| 결정 | 값 | 근거 |
|---|---|---|
| retrieval 변형 운명 | **hypa로 교체하되 `variants.py`는 보존** — run2의 import·분기만 제거. v1 하네스(run.py:29 prepare_request)와 test_eval_harness.py 3건이 계속 쓴다 | 크리틱 실측: 삭제 시 3곳 파손 |
| hypa 실행 경로 | exp 단일 (`useExperimentalImpl: true`) | export 실측 (hypav3.ts:129) |
| maxContext 의미 | CLI `--max-context` (기본 45000) = 프리셋+로어+히스토리+메모리+maxResponse 공유 풀. **원 프리셋 200K 대비 4.4× 축소 — memoryTokens 상시 선점 78,000 → 17,550. 리포트에 이 축소를 명시하고 정량 외삽 주의 문구를 단다** | RisuAI 단일 풀 (index.svelte.ts:614-618); 실유저 55-65K 관찰과의 절충 |
| 요약 모델·파라미터 | DIRECTOR_MODEL(google/gemini-3-flash-preview), **`max_tokens=8192`, `temperature=0.0` 고정** — run2._mk_llm(max_tokens 400)은 요약에 쓰지 않는다. 원본은 db.maxResponse(30000)·temperature 센티널(-1000, 해석 미확정) — 이탈 2건을 리포트에 기록 | hypav3.ts:1709-1720, request.ts:457-458; 크리틱 지적 |
| 요약 메시지 순서 | `[{role:"user", content:원문}, {role:"system", content:19,405자 프롬프트}]` — system이 뒤 | parseChatML null 폴백 실경로 (hypav3.ts:1695-1706) |
| 임베딩 | sentence-transformers `all-MiniLM-L6-v2` (mean pooling + normalize, 코사인). torch 동반 — 설치 무겁고, **테스트 기본 경로는 페이크 embed 주입**. import 실패 시 명확한 에러(조용한 폴백 금지) | 원본 기본값과 동일 모델 (transformers.ts:82), q8→fp32 차이는 근접 순위에서만 |
| 주입 위치 | 뮈토스 memory 카드(promptTemplate[35]) 자리. **preset2wire.assemble에 `memory: str = ""` 파라미터와 type='memory' 카드 처리(_fill + {{slot}})가 이미 구현돼 있다 (preset2wire.py:174, 208-209)** — 갭은 build_wire가 memory를 안 넘기는 1줄뿐 | index.svelte.ts:1429-1443; 크리틱 실측 |
| 토큰 회계 버그 | 원본 그대로 보존 (스킵 챗 토큰도 차감, hypav3.ts:313) | 커뮤 유저가 겪는 그대로 |
| trim 변형 충실화 | 메시지 단위 FIFO + greeting 토큰 포함 + 공유 풀 예산 | index.svelte.ts:1143-1154 |
| 요약 캐시 | `dreaming_data/eval/hypa-cache/{sha256}.txt` + temperature 0.0 — 결정론 2중 | inv-agent2 §6 공통 결정론 조치 |
| chats 변환 계약 | `to_risu_chats(history)` — **memo = 메시지 인덱스 기반 안정 문자열** (아래 Task 4). content 해시 금지 — edit_at 턴(run2.py:415 content 변형)이 memo를 바꾸면 start_idx 매칭 영구 실패 | 크리틱 지적; hypav3 startIdx가 memo에 전적 의존 (hypav3.ts:214-229) |
| loop 리롤 vs 중단 게이트 | "loop" 사유 리롤은 `total_rerolls`(MAX_RUN_REROLLS=10 중단 게이트) 집계에서 **제외** — 게이트는 프로바이더 거부 반복(비용 소각) 탐지용이고 loop는 다른 병리. 오탐으로 런이 중단되면 안 된다 | inv-agent0 open_questions; run2.py:458-461 |
| 캐시 병리 측정 | hypa 런의 `st["cached"]` 히트율을 리포트·스모크 확인 항목에 포함 — 병리 재현이 목적이므로 측정 없이는 무의미 | 크리틱 지적 |

---

### Task 1: dreaming 프록시 — 베이스라인 가드

`refs/dreaming-proxy-surgery.md`가 실프리셋 재현·검증까지 끝낸 수정. 그대로 적용한다.

**Files:**
- Modify: `dreaming/identity.py` (Verdict 필드 1줄), `dreaming/sync.py` (process/record_response)
- Test: `tests/test_dreaming_sync.py`

**Interfaces:**
- Produces: `Verdict.baseline_deferred: bool = False`. record_response가 이 플래그면 즉시 return.

- [ ] **Step 1: 실패 테스트** — `test_first_request_prefill_never_becomes_baseline` 추가 (refs/dreaming-proxy-surgery.md §신규 테스트 초안 코드 그대로 — 뮈토스 프리필 꼬리 6개 `_TAIL` 픽스처 포함. 해당 초안은 완전한 실코드다).
- [ ] **Step 2: FAIL 확인** — `python3 -m pytest -q tests/test_dreaming_sync.py -k prefill_never` → FAIL.
- [ ] **Step 3: 구현** — identity.py Verdict에 `baseline_deferred: bool = False`. sync.py process에 `first_request = not state.get("prev_fp")` (state 읽은 직후), `ledger_was_empty = first_request and not self._ledger.chain()` (판정 전), 판정 후:

```python
        if pairs and ledger_was_empty:
            # 꼬리를 못 배운 첫 요청의 pair가 진짜 히스토리인지 프리셋
            # 프리필인지 가릴 정보가 없다 — 베이스라인 기록을 한 턴 미룬다.
            verdict = verdict.model_copy(update={"baseline_deferred": True})
```
record_response 최상단: `if verdict.baseline_deferred: return`.

**주의**: `first_request and pairs`만으로 걸면 `test_stored_plan_compresses_outbound_but_records_original`이 깨진다 — `ledger_was_empty` 필수.

- [ ] **Step 4: PASS + 전체 회귀** — `python3 -m pytest -q tests/` → 599+신규 전부 PASS.
- [ ] **Step 5: 커밋** — `fix(dreaming): 첫 요청 프리필 오염 차단 — 꼬리 미확정 베이스라인 보류`

---

### Task 2: dreaming 프록시 — 격리 자기치유

**Files:**
- Modify: `dreaming/sync.py`
- Test: `tests/test_dreaming_sync.py`

**Interfaces:**
- Produces: `_MISALIGN_LIMIT = 3`, `SyncPath._rebaseline()`. wire/scaffold 문서에 `misaligned` 카운터.

- [ ] **Step 1: 실패 테스트** — `test_persistent_misalignment_rebaselines` (refs/dreaming-proxy-surgery.md 초안 그대로 — 오염 원장 시드 후 4턴 `seen == [True, True, False, False]`, **내용 기준 단언**: `"prefill-hash" not in hashes` — 재베이스라인 후 index 1027이 우연히 재사용될 수 있어 키 존재로 단언하면 안 된다).
- [ ] **Step 2: FAIL 확인** — `-k rebaselines` → FAIL.
- [ ] **Step 3: 구현** — refs/dreaming-proxy-surgery.md §2 코드: `_MISALIGN_LIMIT = 3`, scaffold put을 판정 후로 이동(`new_state` dict), 격리 시 카운터 증가 → 한계 도달 시 `_rebaseline()` 후 재판정. `_rebaseline`은 **demote_after를 raw 삭제보다 먼저** 호출(raw를 읽어 stale_hashes를 만들기 때문), ledger 전 행 삭제.
- [ ] **Step 4: PASS + 회귀** — `test_stranger_history_is_quarantined`(격리 1회 < 3) 통과 명시 확인.
- [ ] **Step 5: 커밋** — `fix(dreaming): 연속 3턴 미정렬 시 재베이스라인 — 격리 자기영속 차단`

---

### Task 3: 벤치 — dreaming 풀 히스토리 + 야간 스크립트 격리 게이트

**Files:**
- Modify: `benchmarks/eval/run2.py`
- Create: `scripts/night_run.sh` (스크래치패드 `night_run2.sh`를 저장소로 이관 — 휘발 경로에만 있던 게이트 산출물)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `_FULL_HISTORY = ("vanilla", "dreaming")`, `wire_history(variant, history, window) -> List[Dict]`. hypa는 여기 포함되지 않는다 — hypa의 히스토리는 Task 7의 `kept_history`가 결정한다.

- [ ] **Step 1: 실패 테스트** — `test_dreaming_variant_sends_full_history` (refs/dreaming-proxy-surgery.md §3 초안 그대로).
- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — `wire_history` 헬퍼 신설, 본문·edit_at 분기 두 곳 교체, in_window 조건 `variant in _FULL_HISTORY` (dreaming은 프록시가 압축했을 수 있어 **상한값** — 주석 명시). 모듈 docstring "vanilla 변형만 트림 없이" 갱신.
- [ ] **Step 4: 야간 스크립트 이관+게이트** — `scripts/night_run.sh` 생성 (기존 스크래치패드본 기반, 세션 이름·턴 수는 변수화). dreaming 스모크 판정에 추가: 스모크 후 `dreaming_data/<smoke-session>/quarantine/` 파일 수 0 확인, 아니면 본런 스킵. 키 취급 규칙(grep/cut, echo 금지) 유지.
- [ ] **Step 5: PASS + 커밋** — `fix(eval): dreaming 풀 히스토리 전송 + 야간 스크립트 격리 게이트`

---

### Task 4: hypa 변형 — 설정·토크나이저·chats 변환 계약

Task 4~7이 hypaV3 포팅. 규범 스펙 = `refs/hypav3-algorithm.md` §0~§7 — **§7 체크리스트 11개를 태스크 완료 조건으로 삼는다** (#1 maxResponse 차감, #2 스킵 토큰 차감, #3 스캔 범위≠담긴 개수, #4 마지막 3챗 불가침, #5 break/continue, #6 available 차감 시점, #7 잔여 흡수 조건, #8 시간순 재정렬, #9 요약 0개 미삽입, #10 memo 매칭, #11 예약 자기유지).

**Files:**
- Create: `benchmarks/eval/hypa.py`
- Test: `tests/test_eval_hypa.py` (신규 파일)

**Interfaces:**
- Produces:
  - `HypaSettings` (dataclass): max_chats_per_summary(8)/query_chat_count(3)/memory_tokens_ratio(0.39)/extra_summarization_ratio(0.01)/recent_memory_ratio(0.6)/similar_memory_ratio(0.4)/do_not_summarize_user_message(False)/summarization_prompt(19,405자)/summary_chunk_separator(기본 `"\\n\\n"` — export에 없음, hypav3.ts:1803 기본값)
  - `load_hypa_settings(export_path: str) -> HypaSettings` — "키 존재 + typeof 일치" 화이트리스트 병합 (hypav3.ts:1814-1824)
  - `tok_chat(chat: Dict) -> int` = `len(_ENC.encode(chat["content"])) + 3` — 비-gpt 경로 (index.svelte.ts:287-293). name 항은 벤치 chats에 name 필드가 없어 생략 (스펙 이탈 아님을 주석으로)
  - `to_risu_chats(history: List[Dict]) -> List[Dict]` — **memo 계약**: `[{role, content, memo}]`로 변환, `memo = f"m{i}"` (히스토리 내 메시지 인덱스, greeting 포함 0부터). **인덱스 기반이므로 edit_at(content 변형)·리롤(같은 자리 교체)에도 memo가 불변** — start_idx 매칭(hypav3.ts:214-229)이 깨지지 않는다. greeting도 `m0`을 받는다 (원본은 memo undefined지만, undefined 재현 시 §7 #10의 오매칭 버그까지 따라온다 — 벤치에 example 메시지가 없어 그 버그는 발동 불가이므로 안정 인덱스가 동작 동치이면서 안전).
  - `Summary` dict: `{"text": str, "chatMemos": List[str], "isImportant": False}`
  - 세션 상태: `dreaming_data/eval/hypa-state-{session}.json` (summaries 지속)

- [ ] **Step 1: 실패 테스트**:

```python
EXPORT = ("/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 하이파/"
          "hypaV3_export_뮈토스 하이파 V5.json")

def test_hypa_settings_load_from_real_export():
    s = hypa.load_hypa_settings(EXPORT)
    assert s.max_chats_per_summary == 8 and s.query_chat_count == 3
    assert s.memory_tokens_ratio == 0.39
    assert s.recent_memory_ratio == 0.6 and s.similar_memory_ratio == 0.4
    assert len(s.summarization_prompt) > 19000
    assert s.summary_chunk_separator == "\\n\\n"   # export 부재 → 코드 기본값

def test_random_ratio_is_exactly_zero():
    s = hypa.load_hypa_settings(EXPORT)
    assert 1 - s.recent_memory_ratio - s.similar_memory_ratio == 0.0  # IEEE754

def test_to_risu_chats_memo_stable_under_content_edit():
    hist = [{"role": "assistant", "content": "인사"},
            {"role": "user", "content": "원문"},
            {"role": "assistant", "content": "답"}]
    before = [c["memo"] for c in hypa.to_risu_chats(hist)]
    hist[1]["content"] = "원문 (아니, 정정할게.)"     # edit_at 재현
    after = [c["memo"] for c in hypa.to_risu_chats(hist)]
    assert before == after == ["m0", "m1", "m2"]
```

- [ ] **Step 2: FAIL 확인** — `python3 -m pytest -q tests/test_eval_hypa.py` → import 에러.
- [ ] **Step 3: 구현.**
- [ ] **Step 4: PASS 확인.** (커밋은 Task 7에서 hypa 일괄)

---

### Task 5: hypa 변형 — 요약 배칭·summarize

**Files:**
- Modify: `benchmarks/eval/hypa.py`
- Test: `tests/test_eval_hypa.py`

**Interfaces:**
- Produces:
  - `plan_batches(chats, start_idx, current_tokens, max_ctx, S) -> (batches, new_current_tokens, new_start_idx, error: Optional[str])` — 순수 함수. **호출 전에 이미 maxResponse 차감이 끝난 current_tokens를 받는다** (차감은 Task 7 hypa_step 소관).
  - `summarize(send: Callable[[List[Dict]], str], batch: List[Dict], S) -> str` — **send는 OpenAI 메시지 배열을 받아 응답 문자열을 반환하는 콜러블 하나로 통일** (크리틱: LlmFn(system,user) 재사용 금지 — 순서가 뒤집힌다). 프로덕션 send = `_summarize_call`.
  - `_summarize_call(messages) -> str` — OpenRouter 직접 호출: DIRECTOR_MODEL, `max_tokens=8192`, `temperature=0.0`, 비용은 모듈 전역 `SUMMARY_COST`에 누적 (run2가 `cost_hypa`로 수거).
  - `summary_cache_key(batch, S) -> str` (sha256), 캐시 `dreaming_data/eval/hypa-cache/{key}.txt`

- [ ] **Step 1: 실패 테스트**:

```python
def _chat(role, content, memo=None):
    return {"role": role, "content": content, "memo": memo}

def test_batch_planning_respects_query_chat_count_floor():
    # 마지막 queryChatCount(3)개는 절대 배치에 안 들어간다 (hypav3.ts:293-296)
    chats = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(6)]
    batches, _, _, err = hypa.plan_batches(chats, 0, 10**9, 100, S)
    got = [c["memo"] for b in batches for c in b]
    assert not set(got) & {"m3", "m4", "m5"}

def test_batch_planning_charges_skipped_chat_tokens():
    # 스킵 챗 토큰도 차감 — 원본 버그 보존 (hypav3.ts:313, §7 #2)
    skip = _chat("system", "[Start a new chat]" * 50, memo="NewChat")
    real = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(6)]
    _, after_with, _, _ = hypa.plan_batches([skip] + real, 0, 10**6, 100, S)
    _, after_without, _, _ = hypa.plan_batches(list(real), 0, 10**6, 100, S)
    # skip 챗이 배치엔 없어도 그 토큰만큼 current가 더 줄어든다
    assert after_with < after_without

def test_batch_scan_range_exceeds_max_chats_when_skipping():
    # maxChatsPerSummary는 '담긴 개수' 상한 — 스킵이 많으면 스캔은 더 나간다
    # (§7 #3, legacy와 다름)
    skips = [_chat("system", "x", memo="NewChat") for _ in range(5)]
    real = [_chat("user", f"u{i}" * 200, memo=f"m{i}") for i in range(11)]
    batches, _, new_start, _ = hypa.plan_batches(skips + real, 0, 10**9, 100, S)
    assert len(batches[0]) == 8            # 담긴 건 8개
    assert new_start == 13                 # 스캔은 스킵 5 + 유효 8 = 13칸

def test_summarize_message_order_is_user_then_system():
    # parseChatML null 폴백: [user=원문, system=프롬프트] — system이 뒤
    seen = {}
    def send(messages):
        seen["m"] = messages
        return "요약"
    hypa.summarize(send, [_chat("user", "안녕", memo="m0")], S)
    assert seen["m"][0]["role"] == "user"
    assert seen["m"][0]["content"].startswith("user: 안녕")   # role: content 라인
    assert seen["m"][1]["role"] == "system"
    assert "<hypa_memory_extraction>" in seen["m"][1]["content"]

def test_summarize_strips_thoughts_block():
    out = hypa.summarize(lambda m: "<Thoughts>사고</Thoughts>실제 요약",
                         [_chat("user", "x", memo="m0")], S)
    assert out == "실제 요약"
```

- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — refs/hypav3-algorithm.md §2 배치 계획 루프(스킵 토큰 누적 순서 유지, 과요약 방지 분기 포함) + §3 summarize(라인 포맷 `role: content`, Thoughts 제거, 빈 응답 에러). 디스크 캐시로 감싼다. 이탈 기록 주석 2건: max_tokens 8192(원본 30000), temperature 0.0(원본 센티널 -1000 해석 미확정).
- [ ] **Step 4: PASS 확인.**

---

### Task 6: hypa 변형 — 선택 4단계 + 임베딩

**Files:**
- Modify: `benchmarks/eval/hypa.py`
- Test: `tests/test_eval_hypa.py`

**Interfaces:**
- Produces:
  - `select_summaries(summaries, available_tokens, recent_chats, S, embed) -> List[Summary]` — important→recent→similar→(random 스킵), **data.summaries 인덱스 기준 시간순 재정렬 후 반환** (§7 #8)
  - `embed(texts: List[str]) -> "np.ndarray"` — sentence-transformers MiniLM lazy 로드. **테스트는 페이크 embed 주입이 기본** (torch 다운로드를 CI/테스트 경로에서 배제)
  - `simple_cc(scored_lists, weights)`, `child_to_parent_rrf(ranked, key, k=60)` — dict 삽입 순서로 동점 안정성 보장 (JS Map/TimSort 대응)

- [ ] **Step 1: 실패 테스트**:

```python
def _summ(text):
    return {"text": text, "chatMemos": [], "isImportant": False}

def test_selection_break_vs_continue_semantics():
    # recent는 예산 초과 시 break — 더 작은 것 안 찾음 (hypav3.ts:553-569, §7 #5)
    small, big = _summ("나"), _summ("가" * 4000)
    # 역순(최신부터) 스캔: big(최신)에서 초과 → break → small(과거)도 못 담음
    sel = hypa.select_summaries([small, big], available_tokens=50,
                                recent_chats=[], S=S, embed=None)
    assert sel == []

def test_similar_absorbs_unused_recent_budget_when_random_zero():
    # random_ratio==0.0 → recent 잔여가 similar 예산에 합산 (hypav3.ts:596-607, §7 #7)
    # recent가 최신 1개만 담고 남긴 예산을 similar가 흡수해 2개째를 담는다
    old = _summ("옛날 요약.\n\n산에 갔다.")
    new = _summ("최근 요약")
    fake = lambda texts: [[1.0] for _ in texts]        # 전부 동일 벡터
    recent = [{"role": "user", "content": "산에 갔던 얘기", "memo": "m9"}]
    sel = hypa.select_summaries(
        [old, new], available_tokens=hypa.tok_chat(
            {"content": new["text"] + "\n\n"}) * 3,     # recent 몫으론 1개만
        recent_chats=recent, S=S, embed=fake)
    assert old in sel and new in sel                    # 잔여 흡수로 둘 다

def test_selection_returns_chronological_order():
    # 검색 순위가 아니라 원래 순서로 재정렬 (hypav3.ts:871-874, §7 #8)
    a, b, c = _summ("A" * 10), _summ("B" * 10), _summ("C" * 10)
    sel = hypa.select_summaries([a, b, c], available_tokens=10**6,
                                recent_chats=[], S=S, embed=None)
    assert sel == [a, b, c]

def test_rrf_folds_chunk_ranks_into_parent():
    ranked = ["c1@s1", "c2@s2", "c3@s1"]   # s1 청크 2개 상위
    out = hypa.child_to_parent_rrf(ranked, key=lambda ch: ch.split("@")[1])
    assert out[0] == "s1"                   # 1/61 + 1/63 > 1/62

def test_rrf_tie_keeps_insertion_order():
    # 동점이면 첫 등장 순서 유지 — JS Map 삽입순 대응 (inv-agent2 §2 주의)
    out = hypa.child_to_parent_rrf(["x@s1", "y@s2"],
                                   key=lambda ch: ch.split("@")[1], k=60)
    # 두 부모가 (rank1, rank2)로 서로 달라 동점은 아니므로, 진짜 동점 케이스:
    out2 = hypa.child_to_parent_rrf([], key=lambda c: c)
    assert out2 == []
    scores_equal = hypa.simple_cc([[("a", 1.0)], [("b", 1.0)]], [0.5, 0.5])
    assert scores_equal == ["a", "b"]       # 동점 → 삽입 순서
```

- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — refs/hypav3-algorithm.md §2 (a)~(d): important는 available 직접 차감·break (§7 #6 — recent/similar 예산은 important 차감 후 available에 비율 곱), recent 역순·break, similar 청크화(separator 정규식)→최근 3챗 문단 서브쿼리 가중 `(idx+1)/(n(n+1)/2)/len(subs)`→코사인→simple_cc→RRF(k=60)→break, random ratio 0 스킵. requirements.txt에 `sentence-transformers` 추가 + 설치 무게(torch ~2GB) 주석.
- [ ] **Step 4: PASS 확인** — 전부 페이크/None embed로 결정론 실행.

---

### Task 7: hypa 변형 — run2 통합 + memory 주입 + 스모크

**Files:**
- Modify: `benchmarks/eval/hypa.py` (`hypa_step`), `benchmarks/eval/run2.py` (변형 분기, retrieval 분기·import 제거 — **variants.py 파일 자체는 보존**)
- Test: `tests/test_eval_hypa.py`, `tests/test_eval_v2.py`

**Interfaces:**
- Consumes: Task 4 `to_risu_chats`/`tok_chat`, Task 5 `plan_batches`/`summarize`/`_summarize_call`, Task 6 `select_summaries`.
- Produces:
  - `hypa_step(history, preset_tokens, S, data, send, max_ctx, max_response) -> (memory_text: Optional[str], kept_history: List[Dict], kept_start_msg: int, data, error: Optional[str])` — **5튜플**. 내부 순서: `current = max_response + 50 + preset_tokens + Σtok(chats)` → **`current -= max_response`** (§7 #1 — 빼먹으면 조기 발동) → 예약 게이트(`summaries>0 or current>max_ctx`, §7 #11 자기유지 루프 = 재현 대상 병리) → 배칭·요약·선택 → memory_text 반환 (요약 0개면 None — §7 #9 미삽입). `kept_start_msg` = `chats.slice(startIdx)`의 시작 메시지 인덱스 (in_window 판정용). error ≠ None이면 run_once가 턴 기록에 남기고 SystemExit (원본도 요청 실패 — hypav3.ts:263-274, 906-910).
  - run2: variant choices `("vanilla", "trim", "hypa", "dreaming")`. hypa 분기: `use_window = kept_history`; `in_window = fact_msg_index >= kept_start_msg` (token_trim의 win_start 사용 금지 — hypa가 실제 보낸 것과 무관); `build_wire(..., memory=memory_text or "")` **한 줄 배선** (assemble의 기존 memory 파라미터 사용); 비용 `cost_hypa` totals 추가.
- CLI: `--max-context` (기본 45000). trim/hypa 공용. `--trim-tokens`는 하위호환 별칭.

- [ ] **Step 1: 실패 테스트**:

```python
def test_wire_carries_memory_inside_leading_system():
    # 요약이 선두 system(카드 0~36 병합 블록) 안, chat 히스토리 앞에 앉는다
    # — 캐시 파괴 병리의 구조적 재현 (index.svelte.ts:1429-1443)
    msgs = build_wire(preset, card, [{"role": "user", "content": "안녕"}],
                      memory="MEMORY_SENTINEL_XYZ")
    assert "MEMORY_SENTINEL_XYZ" in msgs[0]["content"]      # 선두 system 내부
    without = build_wire(preset, card, [{"role": "user", "content": "안녕"}])
    assert "MEMORY_SENTINEL_XYZ" not in without[0]["content"]

def test_hypa_step_subtracts_max_response_before_trigger():
    # §7 #1: maxResponse 차감 누락이면 조기 발동. 경계값으로 고정한다.
    # current(차감 후) = 50 + preset + hist 가 max_ctx 바로 아래면 미발동.
    hist = [{"role": "user", "content": "가" * 40}]
    hist_tk = sum(hypa.tok_chat(c) for c in hypa.to_risu_chats(hist))
    ctx = 50 + 1000 + hist_tk + 1
    mem, kept, _, data, err = hypa.hypa_step(hist, 1000, S, {"summaries": []},
                                             None, ctx, max_response=4000)
    assert mem is None and err is None and kept == hist     # 발동 안 함

def test_hypa_step_reservation_self_sustains():
    # 요약 1개면 이후 매 턴 memoryTokens 선점 (§7 #11) — 짧은 히스토리에도
    # should_reserve=True 경로로 들어가 memory_text가 계속 나온다
    data = {"summaries": [{"text": "요약", "chatMemos": ["m0"],
                           "isImportant": False}]}
    hist = [{"role": "user", "content": "짧다"}]
    mem, _, _, _, err = hypa.hypa_step(hist, 1000, S, data, None, 10**6, 4000)
    assert err is None and mem is not None and "요약" in mem

def test_hypa_step_errors_when_floor_reached():
    # 남은 챗 ≤ queryChatCount인데 여전히 초과 → 에러 (hypav3.ts:263-274)
    hist = [{"role": "user", "content": "가" * 400} for _ in range(3)]
    mem, _, _, _, err = hypa.hypa_step(hist, 10**6, S, {"summaries": []},
                                       None, 100, 4000)
    assert err is not None and "minimum" in err
```

- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — hypa_step 오케스트레이터 + run2 배선 (retrieval 분기 제거, `from benchmarks.eval.variants import retrieve_turns` 임포트 삭제 — 파일은 보존). totals에 `cost_hypa`. **loop 리롤 제외 규칙은 Task 9에서 오므로 여기선 미구현.**
- [ ] **Step 4: 발동 하한 검증 + 12턴 스모크** — 하한: `available_floor ≈ 50 + preset(11,552) + 최근3챗 + floor(max_ctx×0.39)` → max_ctx 20000이면 ~20,300 > 20,000이라 **에러로 죽는다 (실행 불가)**. 스모크는 `--max-context 26000`:

```bash
python3 -u -m benchmarks.eval.run2 <preset> <card> hypa \
  --session hypa-smoke --turns 12 --max-context 26000 --reset
```
확인: (a) 요약 ≥ 1 생성 (hypa-state json), (b) 와이어 선두 system에 Past Summary 삽입 (턴 기록으로), (c) 크래시 0, (d) **캐시 히트율** — 요약 갱신 턴 전후 `cached` 급락 = 병리 재현 증거.
- [ ] **Step 5: 커밋** — `feat(eval): hypa 변형 — RisuAI HypaV3 exp 경로 충실 포팅 (자작 retrieval 대체)`

---

### Task 8: trim 변형 충실화 (RisuAI OFF 경로)

**Files:**
- Modify: `benchmarks/eval/run2.py` (`token_trim`)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `token_trim(history, budget, count_fn=_count) -> (window, win_start)` — **count_fn 파라미터 유지** (기존 테스트 3건이 주입: tests:427,430,440). 내부를 메시지 단위 FIFO로: greeting 포함 전 메시지 합산, 초과 시 `history[0]`부터 제거 (index.svelte.ts:1143-1154). budget은 run_once에서 `max_context - preset_tokens - MAX_TOKENS - 50` 산출 (공유 풀).
- **win_start 규칙 (명문화)**: 반환값은 "이 턴 번호부터의 사실이 창내" 의미. 창의 첫 메시지가 턴 k의 user면 win_start=k; **턴 k의 assistant면 win_start=k+1** (그 턴의 user 발화는 이미 소실 — 반 잘린 턴은 창밖 취급). 이 규칙이 틀리면 2×2 분해가 한 턴씩 밀린다.

- [ ] **Step 1: 실패 테스트**:

```python
def _hist(pairs, greeting=None):
    out = ([{"role": "assistant", "content": greeting}] if greeting else [])
    for i in range(pairs):
        out += [{"role": "user", "content": f"질문{i}" * 30},
                {"role": "assistant", "content": f"답{i}" * 30}]
    return out

def test_token_trim_cuts_message_unit_fifo():
    # RisuAI는 메시지 단위 — 남은 첫 메시지가 assistant일 수 있다
    h = _hist(4)
    budget = sum(len(m["content"]) for m in h[3:])   # user1 중간에서 끊기게
    window, win_start = token_trim(h, budget, count_fn=len)
    assert window[0]["role"] == "assistant"          # 턴1의 답만 남음
    assert win_start == 2                            # 반 잘린 턴1은 창밖

def test_token_trim_counts_greeting_tokens():
    # greeting(선두 assistant)도 예산 판정에 들어간다
    h = _hist(2, greeting="가" * 500)
    keep_all = token_trim(h, budget=10**6, count_fn=len)[0]
    assert keep_all == h                             # 여유면 전부 유지
    window, _ = token_trim(h, budget=sum(len(m["content"]) for m in h) - 1,
                           count_fn=len)
    assert window[0]["content"] != "가" * 500        # 1토큰 모자라면 greeting부터 제거
```

- [ ] **Step 2: FAIL 확인** (현행 페어 단위 + greeting 무시).
- [ ] **Step 3: 구현 + 기존 token_trim 계약 테스트 갱신** — 페어 단위를 가정한 기존 테스트를 메시지 단위 의미로 수정 (grep으로 `token_trim` 사용 테스트 전수 목록화 후 하나씩; win_start 소비처인 in_window 판정과의 정합 확인).
- [ ] **Step 4: PASS + 커밋** — `fix(eval): trim을 RisuAI 실동작으로 — 메시지 단위 FIFO + greeting 토큰 + 공유 풀 예산`

---

### Task 9: P1-2·3 — 리롤 사유 기록 + 중복 응답 게이트 (한 커밋)

**Files:**
- Modify: `benchmarks/eval/run2.py`
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `reply_flaw(reply, prior_replies=()) -> str` ("loop" 추가, `_LOOP_LOOKBACK=3`, `_LOOP_RATIO=0.97`), `reroll_until_clean(call, prior_replies=(), max_rerolls=2) -> (st, flaw_history)`. 턴 기록에 `flaw_history` 자동 전파(**st 스프레드).
- **중단 게이트 상호작용**: `total_rerolls` 누적에서 "loop" 사유 리롤은 제외 — `total_rerolls += sum(1 for f in flaw_history[:-1] if f != "loop")`. MAX_RUN_REROLLS는 거부 반복(비용 소각) 탐지용; loop 오탐이 런을 죽이면 안 된다.

- [ ] **Step 1: 실패 테스트** — refs/harness-p1-mapping.md §2·§3 초안 5개 그대로 + 게이트 상호작용 1개:

```python
def test_loop_rerolls_do_not_count_toward_abort_gate():
    # 중단 게이트는 거부 반복용 — loop 리롤로 런이 죽으면 안 된다
    from benchmarks.eval.run2 import abort_reroll_count
    assert abort_reroll_count(["loop", "refusal", ""]) == 1
    assert abort_reroll_count(["loop", "loop", "loop"]) == 0
```
(`abort_reroll_count(flaw_history) -> int` 헬퍼 신설 — 마지막 항목 제외(그건 최종 상태지 리롤 아님), "loop" 제외 카운트.)

- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — refs/harness-p1-mapping.md §2 AFTER 코드 (import difflib, Sequence; 호출부 `prior_replies = [m["content"] for m in history[-6:] if m["role"] == "assistant"]`) + `abort_reroll_count`로 total_rerolls 누적 교체.
- [ ] **Step 4: PASS** — 기존 reply_flaw 단항 테스트 2건 디폴트 인자로 통과 확인.
- [ ] **Step 5: 커밋** — `feat(eval): 리롤 flaw_history + 중복 응답(loop) 게이트 — trim T85=T86 실측 병리`

---

### Task 10: P1-1 — value_in_window 값-생존 기록

**Files:**
- Modify: `benchmarks/eval/run2.py`, `benchmarks/eval/report2.py`, `benchmarks/eval/viewer.py`
- Test: `tests/test_eval_v2.py`

- [ ] **Step 1: 실패 테스트** — refs/harness-p1-mapping.md §1 초안 2개 (`test_value_survival_flags_evicted_but_repeated_value`, `test_value_survival_defaults_missing_key_to_not_survived` — 실코드 완비).
- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — probes.append에 `"value_in_window": any(fact.value in m["content"] for m in use_window)` (한글 표기 변형 미탐 = 과소탐지 방향 한계 주석). report2에 `value_survival` 병렬 함수(기존 window_split 불변 — test_report_splits_by_window 보호), aggregate·render 한 줄, viewer 배지.
- [ ] **Step 4: PASS + 커밋** — `feat(eval): 창밖 프로브 값-생존 기록 — narrative rehearsal 오염 분리`

---

### Task 11: P1-4 — 프로브 품질 4종

**Files:**
- Modify: `benchmarks/eval/director.py` (**`import re` 추가 필수** — 현재 import에 re 없음), `benchmarks/eval/run2.py` (`drift_suspected` 기록)
- Test: `tests/test_eval_v2.py`

- [ ] **Step 1: 실패 테스트** — refs/harness-p1-mapping.md §4 초안 4개 (deixis 문구 / T49 실측 드리프트 재현 `_probe_mentions_fact_object` / 시제 앵커 문구 / 자기외모 배제 문구 — 실코드 완비).
- [ ] **Step 2: FAIL 확인.**
- [ ] **Step 3: 구현** — 전부 append 방식 (기존 어서션 문자열 "슬며시"/"시험조"/"존댓말"/"명사형" 보존). 드리프트는 `drift_suspected` 로깅만 — 하드 차단은 무한루프 위험.
- [ ] **Step 4: PASS + 커밋** — `fix(eval): 프로브 품질 — deixis 금지·시제 앵커·값 드리프트 로깅·자기외모 배제`

---

### Task 12: 통합 검증 — 미니 비교런

- [ ] **Step 1: 전체 테스트** — `python3 -m pytest -q tests/` 599+신규 전부 PASS, `python3 -m ruff check benchmarks/eval/ dreaming/` 기존 6건 초과 없음.
- [ ] **Step 2: 4변형 12턴 스모크 병렬** — vanilla/trim/hypa/dreaming 각 12턴 **`--max-context 26000`** (hypa 발동 하한 ~20.5K 위, Task 7 산술). 확인: (a) dreaming quarantine 0건, (b) hypa 요약 생성 + 선두 system 삽입 + **요약 갱신 턴의 캐시 급락 관측** (병리 측정), (c) 프로브 기록에 value_in_window/flaw_history/drift_suspected 존재, (d) 비용 합계 (cost_hypa 포함).
- [ ] **Step 3: 결과 보고** — 스모크 수치 + 이상 유무 보고. **야간 본런(100턴, --max-context 45000) 재발사는 사용자 승인 후.**

## 비용 추정 (야간 본런 재발사 시)

| 항목 | 추정 |
|---|---|
| vanilla / trim | ~$0.8–1.0 각 (기존 실측) |
| dreaming (풀 히스토리化) | ~$0.9–1.2 (vanilla급 프롬프트) |
| hypa | ~$1.0–1.4 — 본문 + 요약 콜. 요약 입력 = 배치당 19.4K자 프롬프트(~7K tk) + 원문(~2K tk), 100턴에 배치 ~10–15개 → Gemini Flash 입력 ~135K tk ≈ **$0.05 수준** (미미). 디스크 캐시로 리런 무료 |
| **4변형 합** | **~$4–5** (예산 내 3회 반복 가능) |

## Self-Review 결과 (v2)

- 크리틱 HIGH 7건 반영: variants.py 보존(실행불가→해소) / maxResponse 차감 명시+경계 테스트 / memo 계약(to_risu_chats, 인덱스 기반) / hypa_step 5튜플 에러 경로 / wire_history↔hypa 분기 모순 해소(hypa는 kept_history + kept_start_msg 기반 in_window) / summarize 시그니처 send 단일화 / 스모크 26000 재산정.
- 크리틱 MED 반영: preset2wire 신설 제거(기존 memory 파라미터 1줄 배선) / token_trim count_fn 유지 + win_start 반 잘린 턴 규칙 명문화 / MAX_RUN_REROLLS 상호작용(abort_reroll_count) / temperature 0.0 고정 / 스캔 범위 테스트 / 기준선 599·ruff 스코프 정정 / night 스크립트 저장소 이관 / 플레이스홀더 테스트 6건 실코드화.
- 크리틱 LOW 반영: import re 명시 / tok_chat name 생략 주석 / subModel 표현 완화 / 캐시 병리 측정 추가 / retrieval 파일 보존.
- 잔여 미결(구현 중 결정): hypa 요약이 갱신 안 된 턴에도 similar 선택이 흔들려 memory_text가 바뀔 수 있음 — 캐시 측정에서 "갱신 턴"과 "선택 요동 턴"을 구분할지 여부는 스모크 관측 후 결정.
