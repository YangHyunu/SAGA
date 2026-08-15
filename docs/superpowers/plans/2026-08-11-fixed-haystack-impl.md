# 고정 건초더미 벤치 구현 계약 — 2026-08-11

설계: [2026-08-10 스펙](../specs/2026-08-10-fixed-haystack-bench-design.md). 이 문서는 **구현 계약**이다 —
모듈 경계·데이터 스키마·불변식을 확정해 병렬 작업자가 서로 안 어긋나게 한다.

> **`benchmarks/eval/` 수정 금지.** 전부 import 재사용. 신규 코드는 `benchmarks/fixed_haystack/`에만.

## 0. 정찰로 확정된 사실 (스펙 수정 사항 포함)

실물 검증 완료 (2026-08-11, 이 워크트리):

| 항목 | 스펙이 말한 것 | **실물** | 영향 |
|---|---|---|---|
| 나레이터 모델 | `config.MODEL` = `deepseek/deepseek-v4-pro` (§6) | **`deepseek/deepseek-v4-flash-0731`** (run JSON `model` 필드, night_run.sh `DREAMING_EVAL_MODEL`) | **스펙 결함.** 얼린 원문을 쓴 모델이 flash다. 프로브 재생성도 flash여야 문체 일관성이 성립 → 모델은 haystack 메타에 박제하고 거기서 읽는다 |
| max_context | `config.MAX_CONTEXT=45000` (§3.1) | **100000** (`fix-run.log` "100턴 @100000") | **스펙 결함.** trim 예산식이 달라진다 |
| max_tokens | env 기본 4000 | **30000** (`DREAMING_EVAL_MAX_TOKENS`) | 트림 예산에서 30K를 선점 |
| reasoning | 미언급 | **`4000`** (`DREAMING_EVAL_REASONING`) | 프로브 콜도 같은 값으로 보내야 응답 길이·비용이 재현된다 |
| 카드 정합성 (§8-9) | 미검증 | **확인** — 재조립 T0 와이어 12,427tok vs 기록 12,725tok (−2.3%, 프로바이더 토크나이저·메시지 오버헤드 차) | `card-soyeon-v2.json` + `DeepSeek V6.2_preset.risup` 조합이 맞다 |
| vanilla 무상한 리스크 (§8-8) | "사전 실측 필요" | **정량화 완료** (아래) | 오버플로 처리 규약 필요 |

원 런 실측 (같은 4변형, 100턴, 프로브 9개) — 이번 벤치의 비교 기준선:

```
dreaming 1/9   hypa 2/9   trim 3/9   vanilla 2/9        (judge_pass, fix-run.log)
```

건초더미 후보 누적 히스토리 토큰(o200k, greeting 제외):

| 프로브 턴 | 19 | 29 | 39 | 49 | 59 | 69 | 79 | 89 | 99 |
|---|---|---|---|---|---|---|---|---|---|
| **A** `fix-drm-r0` | 24.9K | 36.7K | 47.4K | 60.0K | 70.6K | 83.1K | 94.1K | 113.3K | **133.6K** |
| **B** `fix-van-r0` | 23.0K | 31.2K | 42.1K | 53.4K | 62.1K | 72.7K | 81.0K | 88.5K | 95.2K |

고정부(프리셋+카드)가 11,806tok이므로 vanilla 와이어 = 고정부 + 위 값.
> ⚠️ **이 표의 숫자는 o200k다 — 128K 창과 직접 비교하면 안 된다.** 프로바이더 토크나이저 공간에선
> 최대 12% 더 크다(§10.1 C1). 보정 후 결론: **A는 T89·T99 둘 다 창을 넘는다.** 아래 문단은
> 정정 전 원문이며 근거로 쓰지 말 것.

**A의 T99는 145K → DeepSeek 128K 창을 넘는다** (T89도 125K로 아슬아슬). B는 T99에서 107K로 들어간다.
→ vanilla는 오버플로를 **에러가 아니라 관측 결과**로 기록한다(§4.5). 이게 "순정이 이 대본에선 물리적으로 못 돈다"는
발견 그 자체다. 결과를 스핀하지 말 것 — 오버플로는 오버플로로 보고한다.

A와 B의 평균 assistant 길이가 1,929자 vs 1,295자로 다르다 — **두 건초더미의 점수는 절대 평균 내지 않는다**(스펙 §2.3).

## 1. 패키지 구조

```
benchmarks/fixed_haystack/
  __init__.py
  haystack.py    # 동결 스키마 + freeze CLI + 채택 검증
  narrator.py    # 모델 명시 나레이터 콜 (eval/transport 재사용 불가 사유는 §5)
  assemble.py    # 프로브 턴 컨텍스트 조립 — 5변형
  ingest.py      # 유상태 변형(hypa/dreaming) 1회 리플레이 + 스냅샷
  run.py         # CLI: 프로브 루프 + 이중 채점
  report.py      # 집계·렌더
tests/
  test_fixed_haystack_haystack.py
  test_fixed_haystack_assemble.py
  test_fixed_haystack_ingest.py     # 스냅샷 격리 불변식 (LLM 0콜, fake로)
```

데이터 루트: `dreaming_data/fixed_haystack/` (gitignore 승계).

```
dreaming_data/fixed_haystack/
  fix-drm-r0.json                 # 동결 건초더미
  fix-drm-r0/
    dreaming/live/                # 인제스천 중 JsonDirStorage 루트
    dreaming/snap/t19/ … t99/     # 프로브 턴별 저장소 스냅샷 (디렉터리 복사)
    hypa/snap/t19.json … t99.json # hypa data 스냅샷
    ingest-meta.json              # 콜 수·비용·소요 + git commit·꿈 트리거 정책 (스펙 §9-2)
    runs/{variant}-run{n}.json    # 스펙 §9-3 전용 네임스페이스
```

## 2. 동결 스키마 (`haystack.py` — 스키마 소유자)

```python
@dataclass(frozen=True)
class FrozenTurn:
    turn: int          # 0-기준
    user: str
    assistant: str     # 원 런의 reply 원문
    is_probe: bool

@dataclass(frozen=True)
class FrozenProbe:
    turn: int          # 0-기준, 이 턴의 user가 프로브 발화
    ptype: str         # recall|relation|false|update|recent
    fid: str           # ledger 대조용 (없으면 "")
    fact: str          # fact.text
    value: str         # fact.value
    wrong: str         # false 프로브 오염값, 그 외 ""
    question: str      # == turns[turn].user
    fact_turn: int     # ledger에서 찾은 원 기록 턴 (distance_turns 계산용)

@dataclass(frozen=True)
class FixedHaystack:
    haystack_id: str            # "fix-drm-r0"
    source_run: str             # 원 런 JSON 경로 (상대)
    origin_variant: str         # "dreaming" | "vanilla" — 출처 편향 라벨
    narrator_model: str         # 원 런의 model 필드 — 프로브 재생성에 그대로 쓴다
    preset_path: str            # 절대 경로 (Downloads의 risup)
    card_path: str
    run_params: dict            # {"max_context":100000,"max_tokens":30000,"reasoning":"4000"}
    turns: list[FrozenTurn]
    probes: list[FrozenProbe]
    ledger: list[dict]          # 원 런 ledger 그대로 (fid/kind/value/text/turn/probed)
    validation: dict            # §2.2 결과 박제
    provenance: dict            # 스펙 §9-2 생성 이력 — 없으면 load()가 거부 (§2.3)
```

### 2.3 provenance (스펙 §9-2 의무)

`haystack.REQUIRED_PROVENANCE`가 단일 진실이고 `load()`가 그 키 전부를 강제한다 (누락 시
`SystemExit` + 재freeze 안내, 조용한 기본값 채우기 금지). **키 존재가 아니라 값을 본다** —
`PROVENANCE_MAY_BE_UNKNOWN`(git 2종·해시 2종·`source_session`)만 `""`가 정당한 "미상"이고,
그 밖의 자리가 비면 거부한다. 전부 `""`인 provenance는 손편집 스텁이 아니라 **git 없는 머신에서
경로가 깨진 채 얼린 실제 산물**이라, 통과시키면 §9-2가 막으려던 상태 그 자체가 들어온다(§10.5 D1).

| 키 | 내용 |
|---|---|
| `source_session` | 원 런 JSON의 `session` (예 `fix-drm-r0`) |
| `source_run` / `source_run_sha256` | 원 런 파일(ROOT 상대)과 해시 |
| `frozen_at` | ISO8601 UTC |
| `git_commit` / `git_dirty` | freeze 시점 HEAD·워킹트리 오염. git 실패 시 `("", "")` — freeze는 안 죽는다. `git_dirty=""`는 `False`와 다른 뜻(미상 ≠ 깨끗함) |
| `preset_path`/`preset_sha256`, `card_path`/`card_sha256` | 조립 입력 지문. 경로는 ROOT 아래면 **ROOT 상대**(`repo_path`) — 워크트리 절대 경로를 박으면 머지 후 워크트리를 지운 순간 실행 불가다. 읽기 실패는 `""`(미측정). 지문 대조는 freeze 때 한 번이 아니라 **측정 시작마다** `verify_inputs`가 다시 잰다 (§9-1의 "입력은 오직 동결 파일뿐"은 런 JSON에만 참이고 preset·card는 매 런 새로 읽힌다) |
| `narrator_model`, `run_params` | 최상위 필드와 **의도적 중복** — §9-2가 "생성 설정"을 한 블록에 요구한다. 최상위는 호출부 호환용으로 유지 |
| `run_params_source` | `"cli"`(조작자가 플래그로 명시) / `"assumed-default"`(모듈 상수 가정). 원 런 JSON엔 이 값들이 **없어서** 대조할 진실이 없다 — 값만 적으면 가정이 사실로 둔갑한다(§10.5 D4) |
| `source_store_reused` | 항상 `false`. §9-4(병리 세션의 `facts/`·`episodes/` 재사용 금지) 준수를 박제하고, **`load()`가 `is False`를 강제**한다 — 값 검증이 없으면 선언이 아니라 장식이다 |
| `gates` | `"validation"` — 게이트 **결과**는 복제하지 않고 포인터만 둔다 (두 사본은 재freeze 때 갈린다) |

꿈 모델·트리거 정책은 동결 파일이 아니라 **인제스천 산출물**의 속성이라 `ingest-meta.json`이 진다:
`dream_model`·`dream_trigger`(`probe-drain-backlog` = IDLE 타이머 아님, §5.1)·`dream_cap`·`git_commit`/`git_dirty`
+ `source_run_sha256`(어느 동결 원문을 리플레이했는지 — 동결 파일과 스냅샷을 잇는 유일한 식별자).
목록은 `ingest.REQUIRED_META`가 단일 진실이고, **`ingest.require_meta`가 동결 파일 로더와 같은
규약으로 거부한다** — 스킵 경로(재사용)와 `run.py`의 dreaming 변형 양쪽에 걸린다. 미충족 메타는
지금 값으로 메우지 않고 재인제스천을 요구한다(사후에 채우면 "그때 조건"이 아니라 "지금 추측"이다).

JSON 직렬화는 dataclass → dict 그대로. `load(path) -> FixedHaystack`, `freeze(run_json, out_path, **overrides)`.

### 2.1 히스토리 재구성 (모든 변형 공통, 순수 함수)

```python
def messages_upto(hs: FixedHaystack, probe_turn: int, card: dict) -> list[dict]:
    """greeting + turns[0..probe_turn-1]의 (user, assistant) 쌍 + turns[probe_turn].user

    프로브 응답은 절대 포함되지 않는다 — 이 함수가 haystack만 읽는 순수 함수이므로
    스펙 §3.2("프로브 응답은 채점에만 쓰고 history에 반영 금지")가 구조적으로 보장된다.
    """
```
greeting은 `card["greeting"]`이 있으면 `history[0]`에 assistant로 (run2.py:447-448과 동일).

### 2.2 채택 검증 (freeze 시 1회, 결과를 `validation`에 박제)

1. **프로브 누출 소급 검증** (§8-10): 9개 프로브 전부 `lucid.probe_leaks_value(question, DirFact(...))` → 전부 False여야 한다. 하나라도 True면 그 프로브를 `validation["leaky_probes"]`에 남기고 **채택은 계속하되 채점에서 제외 표시**(`FrozenProbe`는 남기고 run.py가 `excluded` 라벨로 스킵). 조용히 버리지 않는다.
2. **거리 분포** (G2 대응): `distance_turns` 중앙값·최소값 기록. `MIN_PROBE_AGE=15` 미만 프로브가 있으면 경고로 남긴다.
3. **와이어 정합성** (§8-9): T0 와이어 재조립 토큰 수 vs 기록 `turns[0].prompt` — 5% 이내면 pass. 넘으면 카드/프리셋 불일치로 **freeze 실패**(exit 1).
4. **vanilla 오버플로 지점**: 프로브 턴별 `고정부 + 누적 히스토리` 토큰을 계산해 `provider_ctx_limit`(기본 128000) 초과 턴 목록을 기록.
5. **ledger 대조**: 각 프로브의 `fact`/`value`로 ledger에서 원 사실을 찾아 `fid`·`fact_turn` 채움. 못 찾으면 원 런의 `probes[].distance_turns` 필드에서 역산(`turn - distance_turns`)하고 `fid=""`.

## 3. 나레이터 (`narrator.py`)

`benchmarks/eval/transport.call_upstream_once`를 **재사용할 수 없다**: 모델을 `MODEL` 모듈 별칭
(import 시점 스냅샷)에서 읽어 런타임 오버라이드가 no-op이고, `benchmarks/eval/` 수정은 금지다
([2026-08-10 후기](2026-08-10-eval-refactor.md)의 "import-시점 별칭은 스냅샷" 교훈과 같은 함정).
→ 모델을 인자로 받는 얇은 호출기를 새로 쓴다. 재시도·응답 파싱 형태는 `transport.py:59-112`와 동일하게.

```python
def call(model: str, msgs: list[dict], key: str, *, max_tokens: int,
         reasoning: str = "", timeout: float = 300) -> dict:
    """반환 {reply, finish, prompt, cached, completion, cost, sec} — transport와 동일 스키마.
    5xx·TransportError는 3회 재시도(15s·30s 백오프), 4xx는 즉시 전파."""
```

- 엔드포인트는 `config.UPSTREAM`(OpenRouter) 고정, `provider={"sort":"throughput"}`, `usage={"include":True}`.
- 키는 `transport.key()` 재사용. **키 값을 로그·리포트에 절대 출력하지 않는다.**
- 품질 리롤은 `quality.reroll_until_clean(lambda: call(...), prior_replies)` 그대로 재사용
  (prior_replies = 얼린 히스토리의 직전 assistant 3개).

## 4. 조립 (`assemble.py`) — 5변형

공통 입력: `(hs, card, preset, probe_turn, fixed_tokens, trim_budget, state)`.
공통 출력:

```python
@dataclass
class Assembly:
    msgs: list[dict]          # 나레이터에 보낼 최종 와이어
    use_window: list[dict]    # 나레이터가 실제로 본 히스토리 (in_window 판정용)
    win_start: int | None     # 창 시작 턴 (trim/retrieval), 그 외 None
    kept_start_msg: int       # hypa 전용, 그 외 0
    memory_text: str          # hypa 요약 블록, 그 외 ""
    overflow: bool            # vanilla 컨텍스트 초과 (§4.5)
    extra: dict               # 변형별 계측 (발췌 수·압축 적용 등)
```

와이어 조립은 전부 `run2.build_wire(preset, card, window, memory=...)` → 이미 reformat까지 포함.
조립 후 `fidelity.check_wire_shape(msgs)` 통과 필수 (위반 시 예외).

예산: `fixed_tokens = sum(windowing.count(m["content"]) for m in build_wire(preset, card, []))`,
`trim_budget = run_params.max_context - fixed_tokens - run_params.max_tokens - 50` (run2.py:471-474 공식).

### 4.1 vanilla
`window = messages_upto(...)` 전체, 트림 없음. `win_start=0`, `use_window=window`.

### 4.2 trim
`window, win_start = windowing.token_trim(messages_upto(...), trim_budget)`.

### 4.3 retrieval (단순 turn-retrieval 베이스라인)
trim과 같은 창을 쓰고, `variants.retrieve_turns(full_msgs, query=probe.question, k=3, window=window)`로
잘린 구간에서 발췌 → `"[과거 대화 발췌]\n" + "\n---\n".join(...)` 블록을 **마지막 user 메시지에 prepend**
(variants.prepare_request:78-86과 동일 위치).
**예산 이식(§8-5)**: 블록을 `assembly.HOT_ZONE_CHAR_BUDGET`(6000자)로 자른다 — dreaming 지식 블록과 동일 상한.
자르는 단위는 발췌(청크) 단위, 문장 중간 절단 금지: 누적 길이가 예산을 넘는 발췌부터 버린다.

### 4.4 hypa
`ingest.py`가 만든 스냅샷 `hypa/snap/t{probe_turn}.json`(= 프로브 턴 **직전까지** 소비한 `data`)을
`copy.deepcopy`한 뒤 프로브 턴 1회분 `hypa.hypa_step(full_msgs, fixed_tokens, S, data_copy,
hypa._summarize_call, max_context, max_tokens)` 실행 → `(memory_text, kept, kept_start_msg, _, err)`.
`err`면 `overflow=False`지만 `extra["hypa_error"]`에 남기고 그 프로브는 채점 제외.
**deepcopy 필수** — 스냅샷 원본을 오염시키면 반복(runs) 간 상태가 갈라진다.

### 4.5 vanilla 오버플로 규약
`fixed_tokens + sum(count(m.content) for m in window) > provider_ctx_limit`(기본 128000, CLI 조정 가능)이면
**나레이터를 호출하지 않고** `overflow=True`로 반환. run.py는 그 프로브를
`{"skipped":"context_overflow","oracle":None,"judge":None}`로 기록하고 **분모에서 제외**한다
(judge 파싱 실패를 분모에서 빼는 `scoring.py:15-18` 관례와 동일). 리포트에 오버플로 프로브 수를 명시한다.

### 4.6 dreaming
스냅샷 `dreaming/snap/t{probe_turn}/`(프로브 턴 직전까지 인제스천된 저장소)를 **임시 디렉터리로 복사**한 뒤
그 복사본에 `JsonDirStorage` → `SyncPath(storage, session, mark_cache_enabled=True)`:

```python
raw = build_wire(preset, card, messages_upto(hs, probe_turn, card))
out, verdict = sp.process(raw)
msgs = upstream.to_wire(out)          # 프로덕션 경로와 동일 (proxy.py:179)
```

**임시 복사 필수** — `sp.process`는 저장소를 쓴다(원장·scaffold). 스냅샷 원본에 직접 돌리면
다음 반복(run)과 다음 프로브가 오염된다.
`use_window`는 `out`에서 role이 user/assistant인 메시지들, `win_start=None`(프록시 압축은 턴 경계와
1:1이 아니다), `extra`에 `verdict.kind`·`quarantine`·압축 플랜 적용 여부·주입된 지식 블록 길이를 기록.

## 5. 인제스천 (`ingest.py`)

**공통 규약**: 진행 구간을 턴 0부터 순차 소비하되, **프로브 턴 i를 소비하기 직전에 스냅샷을 뜬다**.
프로브 턴 자체도 원문(user+assistant)으로 계속 소비한다 — 그래야 T29 스냅샷이 T19까지의 원문을 포함한다.

### 5.1 dreaming
```python
storage = JsonDirStorage(root/"dreaming/live")
sp = SyncPath(storage, session, mark_cache_enabled=True)
hist = [greeting] if card.greeting else []
for t in hs.turns:
    if t.is_probe:  snapshot_dir(live, snap/f"t{t.turn}")     # 소비 전
    hist.append({"role":"user","content":t.user})
    raw = build_wire(preset, card, hist)                       # 프로덕션 = 프록시가 받는 전체 와이어
    out, v = sp.process(raw)
    sp.record_response(v, raw, t.assistant)                    # 원본 raw를 넘긴다 (proxy.py:149-150)
    hist.append({"role":"assistant","content":t.assistant})
    ... 꿈 트리거 (아래)
```
- **`record_response`에는 `process`에 넘긴 원본 `raw`를 넘긴다** — 프록시와 동일(`proxy.py:144-153`).
- **격리(quarantine) 감시**: `v.quarantine`이 True인 턴 수를 세고 `ingest-meta.json`에 기록. 1건이라도 있으면
  경고로 표시한다(night2-drm 사고의 재발 감지선).

**꿈 트리거 정책 (스펙 §8-4 미결 → 여기서 확정)**: 프로브 스냅샷 직전에 backlog를 **완전히 소진**한다
(`while dreamer.has_backlog(session) and n < DREAM_CAP: await dreamer.dream(session)`, `DREAM_CAP=30`).
근거: "유휴 시간이 충분했다"는 조건을 명시적으로 만족시킨 **상한 성능**을 재는 것이고, 프로브 간
꿈 횟수가 변형·반복 간 동일해 통제된다. **한계는 정직하게 라벨한다** — 실사용은 유휴가 짧으면
backlog가 남는다. `ingest-meta.json`에 사이클 수·소요·비용을 기록해 상각 비용을 드러낸다.
인제스천은 haystack당 **1회**만 하고 반복(runs)은 캐시를 재사용한다(§8-4 후자 선택 = 조립 알고리즘의
순수 성능 측정). 이 선택 역시 리포트에 명시한다.

### 5.2 hypa
```python
S = hypa.load_hypa_settings(config.HYPA_EXPORT)
data = {"summaries": []}
for t in hs.turns:
    if t.is_probe: write_json(snap/f"t{t.turn}.json", data)   # 소비 전, deepcopy 저장
    hist.append(user); (mem, kept, ksm, data, err) = hypa.hypa_step(hist, fixed_tokens, S, data,
                                                                    hypa._summarize_call, max_ctx, max_tok)
    if err: 기록하고 중단 (원본도 요청을 실패시킨다 — hypav3.ts:263-274)
    hist.append(assistant)
```
`hypa`의 요약 캐시(`hypa-cache/`, `summary_cache_key`)가 이미 있으므로 재실행이 싸다.
`hypa.SUMMARY_COST`/`SUMMARY_CALLS` 증분을 `ingest-meta.json`에 기록.

### 5.3 무상태 변형
인제스천 없음. `ingest.py`는 `--variant`에 hypa/dreaming만 받는다.

## 6. 실행 (`run.py`)

```bash
python3 -m benchmarks.fixed_haystack.haystack freeze \
    dreaming_data/eval/v2-fix-drm-r0-run0.json --id fix-drm-r0
python3 -m benchmarks.fixed_haystack.ingest --haystack fix-drm-r0 --variants dreaming,hypa
python3 -m benchmarks.fixed_haystack.run --haystack fix-drm-r0 \
    --variants vanilla,trim,retrieval,hypa,dreaming --runs 3
python3 -m benchmarks.fixed_haystack.report dreaming_data/fixed_haystack/fix-drm-r0/runs/*.json
```

프로브 루프 (변형 × 프로브 × 반복):
1. `Assembly` 조립 → `overflow`면 스킵 기록하고 다음
2. `quality.reroll_until_clean(lambda: narrator.call(hs.narrator_model, a.msgs, ...), prior_replies)`
3. 채점: `scoring.oracle_pass(reply, p.value, wrong_value=p.wrong, char_name=card["name"])`
   + `scoring.judge_pass(judge_llm, p.ptype, p.fact, p.value, p.question, reply, wrong_value=p.wrong)`
4. dreaming 미스면 `scoring.decompose_miss(<프로브 임시 저장소의 부모>, session, DirFact(...))`
5. 기록: `distance_turns = p.turn - p.fact_turn`, `in_window`(변형별: vanilla=True,
   trim/retrieval=`p.fact_turn >= win_start`, hypa=`windowing.hypa_in_window(...)`, dreaming=True[상한]),
   `value_in_window = any(p.value in m["content"] for m in use_window)`
6. **프로브 응답은 어디에도 누적하지 않는다** (§3.2 — 코드 주석으로 명시)

결과 JSON: `{"haystack_id","variant","run","narrator_model","probes":[...],"totals":{...},"assembly_meta":{...}}`.
totals에 `probes_scored`(분모), `skipped_overflow`, `judge_unparsed`, `cost`, `cost_judge`를 담는다.

## 7. 리포트 (`report.py`)

변형 × 유형(recall/relation/false/update/recent) 교차표 + 창내/창외 분해 + 오버플로 수 + 비용.
`report2.py:18-48`의 `window_split`/`value_survival` 패턴 참고(스키마가 다르므로 신규 구현).
**두 건초더미 결과를 한 표에 평균 내지 않는다** — haystack_id별로 별도 섹션.

## 8. 불변식 (리뷰 체크리스트)

1. **프로브 응답 비누적** — `messages_upto`는 haystack만 읽는 순수 함수. 어떤 경로로도 프로브 reply가
   히스토리·저장소·hypa data에 들어가면 안 된다.
2. **스냅샷 불변** — 프로브 실행은 항상 스냅샷의 **복사본**(dreaming=temp dir, hypa=deepcopy)에서 돈다.
3. **haystack 불변** — `FixedHaystack`은 frozen dataclass, 실행 중 수정 금지.
4. **eval/ 무수정** — `benchmarks/eval/**` 파일이 diff에 나오면 실패.
5. **모델 출처** — 나레이터 모델은 `hs.narrator_model`에서만 읽는다. `config.MODEL` 참조 금지.
6. **예산 동일** — 5변형이 같은 `max_context`/`max_tokens` 풀을 쓴다. retrieval 발췌 블록과
   dreaming 지식 블록은 같은 6000자 상한.
7. **비밀 금지** — API 키·NSFW 원문을 로그/리포트/커밋에 남기지 않는다. 리포트는 구조적 서술만.

## 9. 테스트 (LLM 0콜)

- `test_fixed_haystack_haystack.py`: freeze 라운드트립, `messages_upto` 경계(프로브 턴에 assistant 미포함,
  greeting 위치), 누출 검증이 leaky 프로브를 잡는지, 와이어 정합성 실패 시 예외.
- `test_fixed_haystack_assemble.py`: vanilla/trim/retrieval 3종을 fake preset/card로 완전 검증 —
  trim이 예산을 지키는지, retrieval 발췌가 6000자 캡을 지키고 마지막 user에 prepend되는지,
  vanilla 오버플로 플래그가 임계에서 켜지는지.
- `test_fixed_haystack_ingest.py`: fake LLM(꿈 0콜)으로 스냅샷이 프로브 턴 **직전** 상태인지,
  프로브 실행이 스냅샷을 오염시키지 않는지(복사본 불변식).

## 10. 실행 후기

구현 → 리뷰 → 교정 1주기 후 기록. **계약 자체의 결함 4건**과 구현 결함을 나눠 적는다.

### 10.1 계약·스펙 결함 (문서가 틀렸다)

**C1. §0/§2.2-4의 오버플로 수치가 단위 혼동 — 헤드라인 주장이 틀렸다. (심각)**

§0 표의 "누적 히스토리 토큰"과 "A의 T99는 145K → DeepSeek 128K 창을 넘는다 (T89도 125K로 아슬아슬)"는
**o200k(`windowing.count`) 값**인데, 비교 대상인 128K는 **프로바이더(DeepSeek) 토크나이저 공간**의 숫자다.
두 숫자가 다른 자다. 원 런 `v2-fix-van-r0-run0.json`의 기록 `prompt`와 대조한 실측:

| 프로브 턴 | 19 | 39 | 59 | 79 | 99 |
|---|---|---|---|---|---|
| o200k | 34,704 | 53,752 | 73,663 | 92,822 | 107,094 |
| 프로바이더 기록 | 38,167 | 59,746 | 82,223 | 103,918 | 120,011 |
| 비율 | 1.100 | 1.112 | 1.116 | 1.120 | **1.121** |

한국어 히스토리 비중이 커질수록 격차가 벌어진다(고정부 1.02 / 한국어 히스토리 1.136).
→ **A의 T89는 o200k 122,268로 가드를 통과하지만 프로바이더 공간에선 137,527로 창을 넘는다.**
T89은 누출 프로브도 아니라 실제로 나레이터를 부르고, `narrator.py`가 4xx를 즉시 전파하므로
§4.5가 "관측 결과로 기록"하라던 상황이 **런 중단**으로 나타났을 것이다.

교정: `haystack.provider_tokens(fixed, history)` 2항 보정(계수는 그 런의 기록 prompt 10점 최소자승,
최대 오차 0.30%)을 도입하고 오버플로 판정을 전부 프로바이더 공간으로 옮겼다. 실물 A의
`overflow_turns`는 `[99]` → **`[89, 99]`**로 정정. **다음 플랜 교훈: 토큰 수치를 표에 적을 때
어느 토크나이저인지 반드시 명시할 것.** 벤치 전체가 o200k 하나로만 살 수 있다는 가정이 깨진 지점이다.

**C2. §2.2의 채택 검증 5개에 값 길이 게이트가 빠졌다. (심각)**

누출 게이트(`probe_leaks_value`)는 소급 재실행하도록 써 놓고, `lucid.eligible`/`probe_plan`이
출제 단계에서 적용하는 `len(value) >= 2` 문턱(lucid.py:84-93, 199-209)은 소급 항목에 없다.
그 결과 **두 건초더미 모두 T29(false)의 기대값이 한글 1글자**로 실렸다 — 그 글자는 얼린 원문에
92/100(drm)·82/100(van)턴에 등장한다. `value_in_window`가 항상 True가 되고 오라클도 문자
기저율에 지배돼, 9개 분모 중 1개(11%)가 노이즈이며 두 건초더미가 **같은 노이즈를 공유**해
A/B 대조로도 상쇄되지 않는다. (원 런 `totals`에 `probe_facts_unprotectable` 키가 없는 것으로 보아
그 런이 게이트 도입 전 코드로 돌았을 가능성이 높다 — 그래서 더더욱 freeze에서 재검증했어야 한다.)

교정: `validation.unprotectable_probes` 추가, run.py가 `probe_unprotectable`로 스킵, 리포트에 컬럼 추가.
**교훈: "출제 단계 게이트를 소급 적용한다"는 규약을 세웠으면 게이트를 열거하지 말고 출처 모듈의
게이트 전부를 대상으로 삼을 것.** 하나만 적으면 나머지는 조용히 빠진다.

**C3. §2.2-3 와이어 정합성 검사가 T0만 본다 — 정작 판정이 내려지는 구간을 보증하지 않는다.**

T0은 히스토리 비중이 최소라 두 토크나이저의 격차가 **최소인 지점**(−2.34%)이다. 5% 허용치를
넉넉히 통과하지만, 같은 비교를 T99에서 하면 −10.8%다. 즉 이 게이트는 오버플로 판정이 실제로
내려지는 영역에 대해 아무것도 말하지 않았다. C1이 T0 검사를 통과한 채로 살아남은 이유가 이것이다.

교정: 기록 `prompt`가 있는 **모든 턴**을 대조하도록 넓혔다(풀 히스토리 출처일 때만 게이트;
dreaming 출처는 프록시 압축 때문에 기록이 정당하게 작아 T0만 문). 실물 `fix-van-r0`는 이제
**100/100턴** 대조에 최악 −0.32%로 통과한다 — C1의 보정이 우연이 아니라는 독립 증거다.
**교훈: 검증 표본은 "제일 싼 지점"이 아니라 "판정이 갈리는 지점"에서 뽑을 것.**

**C4. §2.2-5의 "대조"가 실은 단방향 조회 — (value, text)가 유일하다는 단언이 거짓.**

계약은 "ledger에서 원 사실을 찾아 fid·fact_turn을 채운다"고만 적었고 구현은 dict 컴프리헨션으로
마지막 행을 채택했다. 실측: **`fix-van-r0` 원장 343행 중 2쌍이 충돌한다**(turn 24↔90, 93↔98).
이번 9개 프로브는 우연히 충돌 행을 안 밟았지만, 밟으면 `fact_turn`이 조용히 최대 66턴 어긋나고
`distance_turns` → `_in_window` → `decompose_miss`까지 전파돼 창밖 실패가 창내 실패로 뒤집힌다.
원 런 JSON에 `probes[].distance_turns`라는 **독립적인 진실이 이미 있는데** 폴백으로만 썼다.

교정: 두 출처를 실제로 맞춰 보고(기대 턴과 일치하는 행만 채택), 갈리면 fid를 비운다.
`validation.ledger_ambiguous`로 충돌 쌍 수를 박제. **교훈: "유일하다"는 단언은 플랜에 적기 전에
데이터로 세어 볼 것.** 셀 수 있는 주장을 검증 없이 계약에 박으면 구현이 그 위에 인덱스를 세운다.

### 10.2 구현 결함 (계약은 맞았고 코드가 어긋났다)

| # | 위치 | 증상 |
|---|---|---|
| 1 | `assemble_dreaming` | `workdir` 재호출이 `FileExistsError`(재현함). `copytree`에 `dirs_exist_ok` 없음 + `workspace_temp=False`라 정리도 안 됨 → `--runs 2`가 첫 dreaming 프로브에서 즉사 |
| 2 | `assemble_dreaming` | §4.5 오버플로 가드가 vanilla에만 있음. dreaming도 압축이 안 물면 풀 히스토리를 보내는데 넘치면 4xx 전파 → 런 중단 |
| 3 | `build(session="")` | 기본값이 `JsonDirStorage._check_segment`까지 내려가 `ValueError: invalid storage path segment`(재현함). hypa는 같은 종류의 누락에 친절한 `SystemExit`을 주는데 dreaming만 저장소 내부 예외 |
| 4 | `assemble_dreaming` | `mkdtemp`+`copytree` 뒤 `try/finally` 없음 → `sp.process`가 던지면 100턴 저장소 복사본이 `/var/folders`에 영구 잔류 |
| 5 | `ingest.py` ↔ `assemble.py` | hypa 고정부를 인제스천은 `tok_chat`(11,830), 조립은 `count`(11,806)로 잼. `plan_batches`가 `current > max_ctx` 경계에서 갈리므로 그 경계 턴에서 프로브가 인제스천과 다른 배치를 만들고 **계획에 없던 실제 요약 콜**을 태운다 |
| 6 | `run_variant` | 그 요약 비용을 `totals`가 수거하지 않음 — `report.py`가 "이걸 빼면 유상태 변형이 싸 보인다"고 경고한 바로 그 구멍이 프로브 루프 쪽에 남아 있었다 |
| 7 | `report.aggregate` grid | 반복 간 스킵 불일치를 라벨이 덮음 — 3반복 중 run0만 `hypa_error`면 `1/2`를 버리고 "hypa실패"만 표시 |
| 8 | `report.load_ingest_meta` | `f.parents[1]` 고정 상대 위치라 `run --out` 사용 시 인제스천 비용 섹션이 통째로 침묵 → "비용 0"으로 오독 |
| 9 | `haystack.main` | `freeze(run_params=)`를 CLI가 노출 안 함 — 다른 세팅으로 돈 런을 얼리면 `run_params`가 조용히 오라벨 |

### 10.3 파생 조치

- **동결 산출물 재생성 필요.** `validation` 스키마가 바뀌었으므로(`unprotectable_probes`,
  `probe_provider_tokens`, `ledger_ambiguous`, 보정 대조 필드) 기존 `fix-*.json`은 재freeze해야
  교정이 실제로 발효된다. 두 건 모두 재생성 완료(LLM 0콜, 오프라인).
- **테스트가 계약의 틀린 수치를 박제하고 있었다.** `overflow_turns == [99]`, `probes_scored == 7`
  같은 하드코딩이 C1·C2를 통과시켰다. 이번에 전부 `hs.validation`에서 파생하도록 고쳤다 —
  **교훈: 건초더미가 바뀌면 같이 움직여야 하는 수치는 테스트에 상수로 적지 말 것.**
- 보정 계수(1.020/1.136)는 **이 프리셋·한국어·DeepSeek 조합의 실측**이다. 프리셋/언어/프로바이더가
  바뀌면 재측정 대상이고, freeze가 매번 기록 `prompt`와 대조해 5%를 넘으면 죽는다.

### 10.4 후속: 스펙 §9(데이터 위생) 반영 — 2026-08-11

스펙에 §9 4규칙이 추가되면서 규칙 2(provenance 의무 + 로더 거부)와 규칙 3(`runs/` 네임스페이스)을
구현했다. 규칙 1(세션 디렉터리 직접 읽기 금지)·4(병리 세션 스토어 재사용 금지)는 이미 충족 상태였고
그대로 유지 — 4는 `provenance.source_store_reused=false`로 기계 판독 가능하게 박제했다.

- **두 번째 재freeze.** §10.3과 같은 이유(스키마 변경은 재생성해야 발효)로 `fix-drm-r0`·`fix-van-r0`를
  다시 얼렸다. **로더가 구 스키마를 거부하므로 이번엔 재freeze 누락이 조용히 넘어가지 않는다** —
  §10.3에서 "재생성 필요"를 사람이 기억해야 했던 자리를 게이트가 대신한다.
- **발견: `source_run`이 ROOT 상대가 아닐 수 있다.** `_relpath`가 `resolve()`로 심링크를 따라가는데
  실물 런 JSON(`dreaming_data/eval/v2-fix-*-run0.json`)이 본 저장소를 가리키는 심링크라 절대 경로가
  남는다. 계약 §2의 "원 런 JSON 경로 (상대)"는 워크트리에선 성립하지 않는 서술이었다 — 주석을
  실물에 맞춰 고쳤다. 경로가 어느 쪽이든 `source_run_sha256`이 내용을 못박으므로 대조는 성립한다.
- **`git_dirty`는 미상(`""`)과 깨끗함(`false`)을 구분한다.** git이 없거나 실패했을 때 `false`로 적으면
  provenance가 "그때 워킹트리가 깨끗했다"고 거짓말한다. **교훈: 모르는 값을 기본값으로 메우는 순간
  메타데이터는 기록이 아니라 추측이 된다** — §9-2가 로더 거부를 요구하는 이유와 같은 계열이다.

### 10.5 소급 규칙 라운드 — 리베이스가 완성된 구현을 위반 상태로 만들었다 (2026-08-11)

이 브랜치를 main에 fast-forward 했더니 main이 새로 머지한 `fadafa7 docs(spec): §9 데이터 위생`이
**동결 파일에 대한 의무 4개**를 추가한 상태였다. 즉 §9는 이 구현이 끝난 **뒤에** 계약에 들어왔고,
리베이스 순간 이미 통과하던 구현(818건 그린)이 소급 위반 상태가 됐다. §10.4가 그중 2개를
반영했지만, 리뷰 결과 **반영했다고 적은 두 규칙 모두 형식만 충족**이었다 — 결함 12건 중 7건이
"게이트는 있는데 통과 조건이 형식뿐"이다.

**D1. "규칙을 구현했다"와 "규칙이 위반을 막는다"는 다른 명제다. (심각)**

§10.4는 §9-2(provenance 의무 + 로더 거부)와 §9-3(전용 네임스페이스)을 "구현했다"고 적었다. 실제로는:

| 규칙 | §10.4의 주장 | 실측 |
|---|---|---|
| §9-2 로더 거부 | 구현 | 키 **존재**만 검사 → `{k: "" for k in REQUIRED_PROVENANCE}`가 **통과**. `git_state()`는 실패 시 `("","")`, `_sha256`는 OSError에 `""`를 돌려주도록 설계돼 있어 **프로덕션 freeze가 만들 수 있는 산물**이다 |
| §9-2 (꿈 축) | `ingest-meta.json`이 진다 | 그쪽엔 로더도 거부도 없음. 실물 `fix-drm-r0/ingest-meta.json`이 `git_commit`·`dream_trigger` 없이 남아 있었고 `run --variants dreaming`이 그대로 통과했다 |
| §9-3 네임스페이스 | 구현 | 기본 경로에만 걸림. `--out`/`--out-dir`은 검증 0줄이라 `--out-dir <…>/dreaming_data --id fix-drm-r0`이면 **원 세션 디렉터리**에 `dreaming/live`를 쓰고 `--force`면 그 안에서 `rmtree`까지 돈다 |
| §9-1 "이미 충족" | 세션 디렉터리 미접근 | **런 JSON에 대해서만** 참. preset·card는 동결 파일에 안 실려 매 런·매 인제스천마다 디스크에서 새로 읽힌다 |
| §9-4 "이미 충족" | `source_store_reused=false`로 박제 | 값 검증이 없어 손편집 `true`도 통과 — "기계 판독 가능"이지 "기계 검증"이 아니었다 |

교정: 로더가 키가 아니라 **값**을 본다(`PROVENANCE_MAY_BE_UNKNOWN`만 `""` 허용, 비-dict는
`SystemExit`, `source_store_reused is False` 강제), `require_namespace`가 세 CLI 전부에 걸리고,
`ingest.require_meta`가 꿈 축 메타를 동결 파일 로더와 같은 규약으로 거부하며,
`verify_inputs`가 측정 시작마다 preset/card 지문을 다시 잰다.
**교훈: 게이트를 넣었으면 게이트를 깨 보는 테스트를 같이 넣을 것.** "필드를 채웠다"는 freeze 쪽
테스트만 있었고 "빈 값이 거부되는가"라는 로더 쪽 테스트가 없어서 7건이 통과했다.

**D2. 프로젝트 자신의 테스트가 결함을 성문화하고 있었다. (심각)**

- `_prov()` = `{k: "" for k in REQUIRED_PROVENANCE}` — 전부 빈 값 스텁을 `load`가 받아 준다는
  사실이 테스트에 **단언으로** 박혀 있었다. 게이트가 "파이프라인이 얼린 것"과 "손으로 만든 스텁"을
  구분 못 한다는 증거가 테스트 자산이 된 꼴이다.
- 빈 `snap/t2`·`snap/t4`를 `mkdir`만 하고 "스냅샷 이미 있음 — 스킵" 2회를 단언 — 내용 없는
  스냅샷이 완주로 통과하는 동작을 정상 경로로 못박았다.

§10.3은 "건초더미가 바뀌면 같이 움직여야 하는 수치는 테스트에 상수로 적지 말 것"을 남겼는데,
이번 건은 그 확장판이다. **교훈: 테스트가 통과시키는 fake는 프로덕션이 만들 수 있는 것이어야 한다.**
프로덕션이 절대 못 만드는 스텁을 통과시키면, 테스트가 게이트의 강도를 조용히 낮춘다. 반대로
프로덕션이 **만들 수 있는** 병리(전부 빈 provenance, 빈 스냅샷)는 반드시 거부 테스트로 박제한다.

**D3. 소급 규칙은 코드뿐 아니라 이미 디스크에 있는 산출물도 위반 상태로 만든다.**

실물 타임라인이 `ingest-meta.json`(14:04) → 동결 파일(14:15) → 스펙 §9 순이라, 코드를 고친 뒤에도
**실물 인제스천 산출물은 여전히 §9-2 미충족**이다. 게다가 둘을 잇는 식별자가 하나도 없어
"14:15 동결 파일 + 14:04 스냅샷" 조합이 조용히 측정에 나갈 수 있었다.

- 교정: 인제스천 메타가 `source_run_sha256`(동결 원문 해시)을 싣고, `run.py`가 dreaming 변형일 때
  그 값을 동결 파일과 대조한다. `frozen_at`은 재freeze마다 바뀌어 못박는 자로 못 쓴다.
- **옛 메타를 지금 값으로 메우지 않았다.** git commit·트리거 정책을 사후에 채우면 "그때 그 조건"이
  아니라 "지금 추측"이 기록된다(§10.4와 같은 계열) → 재인제스천을 요구한다. **실물
  `fix-drm-r0`의 dreaming 인제스천은 `--force` 재실행이 필요하다(LLM 비용 발생, 이번 라운드에서
  실행하지 않음).** 현재 그 스냅샷은 `snap/t79` 하나만 남았고 그마저 파일 0개라 어차피 못 쓴다.
- **교훈: 소급 규칙을 반영할 땐 "코드를 고쳤다"에서 멈추지 말고, 이미 만들어진 산출물이 그 규칙을
  통과하는지 실물로 확인할 것.** 통과 못 하면 재생성하거나, 최소한 게이트가 그걸 막는지 확인한다.

**D4. 대조할 진실이 없는 값은 값 대신 출처를 적는다.**

`run_params`는 측정값이 아니라 상수 가정이다 — 원 런 JSON의 top-level엔 `max_context`·`max_tokens`·
`reasoning`이 **없어서** freeze엔 대조할 진실이 아예 없다. 조작자가 `--max-context`를 빠뜨리면
하드코딩 `RUN_PARAMS`가 사실처럼 박히고, 그 위에서 trim 예산·오버플로 판정이 전부 계산되는데
어떤 게이트도 안 걸린다(§2.2-3 와이어 게이트는 preset+card만 본다).

교정: `provenance.run_params_source`에 `"cli"`(조작자 명시) / `"assumed-default"`(상수 가정)를 적고
freeze 출력에 찍는다. **교훈: 검증할 수 없는 값을 provenance에 실을 땐, 값 옆에 "이걸 누가
말했는가"를 같이 적을 것** — §10.4의 `git_dirty` 3상태와 같은 계열이다. 실물 두 건은 지금
`assumed-default`이며, 원 런이 정말 그 세팅이었는지는 `fix-run.log` 밖에서 확인해야 한다.

**D5. 판단 기록 — 네임스페이스 가드를 어디까지 좁혔나.**

§9-3의 문면은 "전용 네임스페이스에만 쓴다"이지만, 가드는 **`dreaming_data/` 안에서
`fixed_haystack/` 밖**만 막는다(데이터 루트 바깥의 tmp·스크래치는 통과). 근거: §9-3의 목적절이
"기존 세션 디렉터리에 쓰기 금지"이고, 실제 사고 경로가 그것 하나다. 데이터 루트 밖 경로까지
막으면 `--out`이 사실상 무용지물이 되면서 가드를 끄고 싶은 동기만 만든다.
**이 판단이 틀렸다면 조여야 할 곳은 `require_namespace` 한 곳이다.**

**D6. 안 고친 것과 사유.**

- **preset/card 내용 자체를 동결 파일에 싣기** — 스키마 변경(파일 크기 증가 + 조립 경로 재작성)이라
  이번 라운드 범위를 넘는다. 대신 경로를 ROOT 상대로 바꿔 이식성을 확보하고(워크트리를 지워도
  실행 가능), 지문 대조를 측정 시작 지점에 넣어 "바뀌었는데 모른다"는 상태를 없앴다.
- **hypa 스냅샷의 내용 검사** — dreaming과 달리 빈 `t{n}.json`은 조립 시점에 `json.loads`가
  던져 **시끄럽게** 죽는다. dreaming의 빈 디렉터리는 조용히 기억 0으로 측정되는 게 문제였다.
  같은 종류의 방어를 대칭으로 넣기보다 실패가 조용한 쪽만 막았다.
- **재freeze 3회차(누적).** `card_path` 상대화 + `run_params_source` 추가로 두 건 다시 얼렸다.
  `frozen_at`·`card_path`·`run_params_source` 외 **전 필드 바이트 동일** — validation 결론(A 누출
  `[99]`·오버플로 `[89,99]`·ledger 9/9, B 최악 T23 −0.32%·충돌쌍 2)은 그대로다.
- **관측만 하고 손대지 않은 것**: `dreaming_data/fixed_haystack/fix-drm-r0/` 안에 이 코드가 만들지
  않은 `.omc/state/...`와 `dreaming/live 2`·`dreaming/snap 2`(공백+2) 디렉터리가 있다. 벤치
  네임스페이스에 외부 산출물이 섞인 상태라 정리가 필요하지만 이 브랜치의 소관이 아니다.

### §10.6 값 생존 감사 + 기억 퍼널 (2026-08-13)

헌터스 카드 run0(v2-hunters-drm-r0)에서 프로브 결함 3건이 확인돼(스킬 획득으로 "유일 스킬"
사실이 사망, 프로브 질문이 자기 정답과 모순, 스토리가 지명을 갱신) 우리 A/B 18프로브에도
같은 감사를 돌렸다 — 각 프로브의 값이 fact_turn부터 프로브 턴까지 갱신·대체·모순되지
않았는지 원문 대조.

**결과: A 9/9 생존, B 8/9 생존 + 1건 표기 결함 교정.**

- **B T49 (fid 08f2b4c1) value 교정**: `"칼날을 다듬는 법"` → `"칼날을 다스리는 법"` (probe+ledger 쌍).
  턴 41 원문은 "내가, 다듬는 법을 가르쳐 줄 수 있어. 칼날을 다스리는 법—— 내가 아는 전부를"이고,
  fact문도 "다스리는"인데 value만 "다듬는"이었다. "칼날을 다듬는"의 유일한 완전 일치는
  "칼날을 다듬는 것까지가 한계지"라는 **반대 진술**이라 오라클이 정답 표현("다스리는")을
  놓치는 구조. 누출·값길이 게이트는 신구 값 모두 통과 확인 후 치환. 로더 게이트 통과 재검.
- **경계 관찰 (교정 불요)**: A T29 미끼 "준"이 턴 30부터 실제 등장인물이 됨 — 프로브 턴이 29라
  히스토리(0..28)는 깨끗하다. B T79 미끼 "도라지"는 동결 대본의 턴 79 원응답이 미끼를 승인해
  턴 80·95에서 canon화 — 역시 프로브 시점 히스토리(0..78)는 깨끗하고, dreaming 스냅샷도
  "꿈 소진 → 스냅샷 → 소비" 순서라 오염 없음. **둘 다 "미끼가 스토리에 편입되는" 헌터스 T80과
  같은 병리의 씨앗이며, 프로브 턴이 편입 시점보다 앞이라 살았을 뿐이다.** 다음 freeze부터
  값 생존 감사를 freeze 게이트에 넣는 게 맞다.
- **B 데이터 위생 관찰**: 턴 55·93·99 assistant 서두에 컴플라이언스 프리앰블 잔재("rendering
  standards accepted…" 류)가 섞여 있다. 전 변형이 같은 대본을 공유하므로 공정성엔 무해.

**기억 퍼널 계측 추가** (유저 지시: "추출·전달이 핵심, 캐릭터 표현 편차는 RP 허용"):
- `run.py`: dreaming 저장소 체크를 judge 실패 시에만 → **무조건**으로 (`value_in_store`),
  주입 블록 원문 값 포함 여부 `value_in_knowledge` 추가 (dreaming+retrieval).
- `assemble.py`: `_knowledge_chars` → `_knowledge_text` (블록 원문 노출), retrieval도
  `extra["knowledge_text"]`.
- `report.py`: "기억 퍼널 — 추출 → 전달 → 활용" 표 (추출=저장소, 전달=주입 블록,
  활용=전달분 judge). judge는 참고 지표로 강등 — 추출·전달이 시스템 책임 몫.

### §10.7 배치 결정론 — dreaming/ 본체 개정 (2026-08-13, 유저 승인)

유휴(5분) 트리거는 "유저가 언제 쉬나"가 배치 내용(에피소드 경계·사실 밀도)을
정한다 — 같은 100턴이 턴당 꿈 353사실/98에피소드 vs 10턴 배치 39/18로 실측
9배 갈림. 유저 지적("30턴 하다 쉴지 5턴 하다 쉴지")으로 채택.

- `dreaming/dreamer.py`: `BATCH_TURNS=10` 양자 도입. `_cycle`은 가장 오래된
  완전 양자만 소화, 잔량 <10턴은 원문 유지(부분 배치 금지). `dream()`은
  완전 양자를 내부 루프로 전부 소화하고 report에 `cycles` 합산.
  `ready()` 추가. 10은 `chunks.BOUNDARY_STEP`과 정렬 — 꿈 주기와 플랜
  계단이 한 리듬. (HypaV3 선례: max_chats_per_summary=8=4턴 고정 배치.)
- `dreaming/proxy.py`: `_finish`에 턴 기반 트리거 — `ready()`면 즉시 꿈.
  유휴·압박·캐치업은 보조 폴백으로 강등. 트리거는 이제 "언제"만 정하고
  "무엇을"은 양자 규칙이 정한다.
- `benchmarks/fixed_haystack/ingest.py`: `_drain_dreams` 사이클 계수를 호출
  횟수 → `report["cycles"]`로. `batch_turns` 파라미터(테스트 심, 기본
  프로덕션 양자). 메타에 `dream_batch_turns` 추가 + REQUIRED_META 편입 —
  옛 스냅샷은 §9 게이트가 자동 거부(원하던 강제 재인제스천).
  `DREAM_TRIGGER = "probe-drain-quanta"`.
- 테스트: 사이클 역학 테스트는 `batch_turns=1·2` 심으로 유지, 양자 절단·부분
  배치 금지·프롬프트 구간 정합은 신규 `test_dream_batches_fixed_turn_quanta`,
  프록시 턴 트리거는 `test_turn_batch_forces_dream_without_idle`(구 압박
  테스트 개작 — 턴 트리거가 압박보다 먼저 서므로).

환경 사고 2건 (코드 무관): annyeong 워크트리 삭제로 `external`·
`dreaming_data/eval/card-soyeon-v2.json` 심링크가 죽어 있었다 — sha 대조
(ab4a76fe4c61/a7a0c8605344) 후 silly-pare 사본으로 재연결. **교훈: 워크트리
간 심링크는 원본 워크트리 수명에 종속 — 지우기 전에 걸린 링크를 확인할 것.**

### §10.8 리플레이 전항화 — 재인제스천이 잡은 프로덕션 버그 (2026-08-13)

배치 결정론 후 첫 재인제스천이 `worldstate.replay`의 `TypeError: cannot add
to non-numeric slot 'time'`으로 사망. 원인: 꿈 추출 LLM이 서술 슬롯에
`op=add` 커밋을 냈고(실측: set time="…" 뒤 add time), replay가 raise —
**오염 커밋 1건이 이후 모든 동기 경로 지식 렌더를 영구 사살**하는 구조였다.
프록시는 fail-open이라 채팅은 살지만 기억이 조용히 죽는다. 벤치의 시끄러운
실패가 프로덕션의 조용한 사망을 노출한 사례.

이중 방어로 교정:
- `worldstate.replay` 전항화 — 타입 어긋난 add는 건너뛰고 경고 (원장은
  LLM이 쓴 비신뢰 데이터, 폴드는 raise하면 안 된다).
- `dreamer.apply_extraction` 쓰기 게이트 — op=add에 비수치 value면
  pending_contradiction 강등.
- 테스트: raise 고정 테스트 2건을 skip 계약으로 교체 + 비수치 add 신규.
