# HypaPlus 동작 인스펙터 Implementation Plan

작성 2026-08-28. 선행 플랜 [2026-08-27-hypa-inspector.md](2026-08-27-hypa-inspector.md)의 후속 —
같은 고정 건초더미·같은 하네스에 **두 번째 기억 시스템**을 얹어 대조군을 만든다.

## 목표

HypaPlus v1.0.9(아카 플러그인, 내장 HypaV3의 드롭인 후계)를 Python으로 포팅해
`fix-drm-r0` / `fix-van-r0` 100턴을 전진 재생하고, 턴별 내부 상태를 기록한다.
HypaV3 인스펙터와 **같은 타임라인 축**(요약 개수 / 원문시작 / wire 토큰)에 겹쳐
읽을 수 있어야 한다.

채점 LLM 없음. 판정은 전부 결정론적 문자열 검사.

## 근거 소스 (동결 대상)

`~/Documents/arca-modules-2026/hypaplus/HypaPlus v1_0_9.ts` — 4,320줄, 단일 IIFE.
sha256 `81b6c4e3f6c395811ff0950f548833cd809d5e4f8dbe7adbde9ee4f692c7da26`.
아래 인용의 줄번호는 전부 이 판본 기준이며, 본 플랜 작성 시 18개 전부
`sed -n '<L>p'`로 함수 시작 줄임을 확인했다.

해부 요약본: `~/Documents/arca-modules-2026/_analysis/memory-hypaplus.json`
(참고용 — 본 플랜의 인터페이스 목록은 **소스 직접 열람**으로 재확인했다.)

## Global Constraints

- `benchmarks/eval/{hypa,hypa_main}.py`와 `run.py`·`report.py`·채점 경로는 **무접촉**.
  HypaPlus는 신규 모듈로만 들어간다.
- 출력은 `dreaming_data/fixed_haystack/` 네임스페이스에만.
- 요약 LLM은 디스크 캐시 우선. `--allow-llm` 없으면 미스를 열거만 하고 종료코드 1.
- 요약 모델·임베딩은 하이파 관찰 런과 동일 (`google/gemini-2.5-flash-lite`,
  MiniLM) — 두 시스템 차이가 모델 차이로 오염되지 않게.
- 브랜치 `claude/hypa-inspector` 위에서 계속. main 머지 금지 (main엔
  `benchmarks/fixed_haystack/`가 없다).

## File Structure

```
benchmarks/eval/hypaplus.py            (신규) 알고리즘 포트
benchmarks/fixed_haystack/needle.py    (신규) 니들 생존 오라클 — 두 시스템 공용
benchmarks/fixed_haystack/inspect_hypa.py  (수정) --system 축 추가
tests/test_hypaplus.py                 (신규)
tests/test_needle.py                   (신규)
docs/superpowers/specs/2026-08-28-hypaplus-port-spec.md  (신규) 포트 대응표
```

---

## HypaV3와의 구조적 차이 (포트 설계의 근거)

| | HypaV3 (기존 인스펙터) | HypaPlus (이번) |
|---|---|---|
| 압축 트리거 | `currentTokens > maxContext` (토큰 압력) | 미요약 메시지 수 ≥ 40 (**컨텍스트 무관**) |
| 배치 단위 | 토큰 예산까지 채움 / 고정 창(main) | 메시지 **정확히 30개** |
| 요약 산출물 | 자유 텍스트 1덩이 | `<event>`/`<facts>` 구조화 **노드 N개** |
| 검색 단위 | 요약 1장 (청크 분할 후 RRF로 부모 복원) | 노드 1개 (분할 없음) |
| 랭킹 | important→recent→similar→random 4버킷 | 즐겨찾기 → `combined` 단일 정렬 |
| 유사도 | 내적(비정규화 가능) | **코사인** (정규화 내장) |
| 시간축 | 없음 | **작중 시각** 감쇠 `1/√(1+Δd/T)` |
| 주입 위치 | 새 system 메시지를 index 0에 unshift | 플레이스홀더 치환 → 없으면 **첫 system에 append** |
| 예산 초과 | 버킷별 비율 분배 | 낮은 점수 노드를 **자르지 않고 skip** |

핵심: HypaPlus는 maxContext를 **한 번도 읽지 않는다**. 압축도 주입도 전부
메시지 개수와 자체 문자 가중 토큰 추정으로만 결정된다.

### 이 건초더미에서의 사전 계산 (검증 가능한 예측)

히스토리 규약은 인스펙터와 동일 — `m0`=인사말, 턴 t가 `m(2t-1)`(user)·`m(2t)`(assistant).
`beforeRequest` 시점 pending = `2t − S`, `afterRequest` 시점 = `2t+1 − S` (S=누적 요약 메시지 수).

- 트리거는 **turn 20, 35, 50, 65, 80, 95** 여섯 번. 즉 청크 6개, 요약 LLM 6콜/건초더미.
- 청크 경계는 `m0–m29 / m30–m59 / … / m150–m179` (전부 30개 고정).
- `trackAssistantMessage`는 **한 번도 발동하지 않는다** — 증명: 발동 조건은
  `2t−S < 40 ≤ 2t+1−S` ⟺ `2t−S = 39`. S는 항상 30의 배수(짝수)고 `2t`도 짝수라
  `2t−S`는 짝수 → 39 불가능. (그래도 충실성을 위해 훅은 구현하고, 테스트로 no-op을 박제한다.)
- 결과적으로 원문 창은 **10~40 메시지 사이 톱니**로 유지된다. maxContext가
  45K든 100K든 동일.

두 개의 반증 가능한 예측 — 인스펙터가 확인하거나 깬다:

**P1. 창 과소사용.** 하이파는 45K를 꽉 채우는데 HypaPlus의 wire는 10~40 메시지
+노드로 훨씬 작을 것이다. 컨텍스트를 남기고도 원문을 버린다.

**P2. 검색 무작동.** 노드 총량(6청크 × 노드 5~8 ≈ 30~50개, 개당 200~400자)이
`maxMemoryTokens=20000`(휴리스틱 기준 한글 약 25,000자)에 **미달**해 전량
주입될 가능성이 높다. 그러면 임베딩도 시간감쇠도 순위에 영향을 못 준다 —
하이파에서 관측한 "similar 아사"의 정반대 형태의 무작동.

두 예측 다 사실이면 "HypaPlus의 두 발명(노드 단위 검색 + 작중 시간 감쇠)은
100턴 규모에선 발동하지 않는다"가 이번 런의 소득이 된다.

---

## Task 1: 소스 동결 + 포트 대응표

**작업**

1. `HypaPlus v1_0_9.ts`의 sha256과 줄 수를 스펙 문서에 기록. 원본은 복사하지
   않는다(레포에 성인향 프롬프트 전문이 남으면 안 됨 — 요약 프롬프트는
   `DEFAULT_SUMMARY_PROMPT`에 `Rating: NC-21+` 지시가 있다).
   포트는 프롬프트를 **런타임에 원본 파일에서 추출**한다(Task 3).
2. `docs/superpowers/specs/2026-08-28-hypaplus-port-spec.md` 작성 — 위 차이표 +
   함수 대응표(아래) + 결정론 위험 3건 + 의도적 미포팅 목록.

**함수 대응표 (소스 실측)**

| TS | 줄 | Python |
|---|---|---|
| `heuristicEstimateTokens` | 520–533 | `est_tokens` |
| `parseNodesFromSummary` | 682–714 | `parse_nodes` |
| `parseEventBlock` / `parseFactsBlock` | 716–787 | `_parse_event` / `_parse_facts` |
| `normalizeTimeString` | 647–680 | `normalize_time` |
| `parseTimeString` | 1693–1750 | `parse_time` |
| `compareNodesByTime` | 1751–1760 | `_time_key` |
| `computeTimeScore` | 1673–1691 | `time_score` |
| `buildNodeText` | 1651–1658 | `node_text` |
| `formatNodeForContext` | 1940–1967 | `format_node` |
| `buildEmbeddingQuery` | 1429–1451 | `build_query` |
| `parseChatMLPrompt` | 1462–1479 | `hypa._parse_chatml` (재사용) |
| `summarizeChunk` | 1481–1529 | `summarize_chunk` |
| `runSummarization` | 1531–1603 | `run_summarization` |
| `retrieveRelevantNodes` | 1763–1938 | `retrieve` |
| `buildMemoryContext` | 1969–2003 | `build_memory` |
| `trackUserMessage` / `trackAssistantMessage` | 2005–2041 | `track` |
| `beforeRequestHandler` 주입부 | 2136–2189 | `inject` |
| `process` 핸들러 (원문 제거) | 4284–4297 | `kept_start_msg` 계산 |

**의도적 미포팅** (스펙에 사유와 함께 명시)

- regex 전처리(`applyRegexToMessages`, 279–443) — 프리셋 미적용 상태가 기본,
  건초더미엔 적용할 regex 프리셋이 없다.
- `recentNodeCache` (기본 false) — `[HypaPlus.cached]` 플레이스홀더도 프리셋에 없음.
- GUI·백업·프리셋 임포트 전량 (2228~4270).
- `favorite` 노드 — 사람이 GUI로 찍는 값. 자동 런에선 항상 공집합.
  단 코드 경로는 남긴다(공집합이면 no-op).

**결정론 위험 3건 — 포트에서 반드시 고정**

1. `createdAt: Date.now()` (752, 782, 1508) → `compareNodesByTime` 폴백 정렬에 쓰임.
   포트는 **삽입 순 단조 카운터**로 대체.
2. `parseFactsBlock`이 `시간:` 줄이 없으면 `nowTimeString()`(실시각) 폴백 (777).
   포트는 고정 상수로 대체하고, **폴백 발동 횟수를 관측치로 기록**.
3. `nextNodeId`/`nextChunkId`는 기존 최대 id 기반 → 이미 결정론적. 그대로.

**Verify**: 스펙 문서 존재, 대응표의 모든 TS 줄번호가 실제 함수 시작 줄과 일치
(`sed -n '<L>p'` 스팟 체크 5건).

---

## Task 2: `hypaplus.py` — 순수 함수층

**구현**: 설정 dataclass + 노드 타입 + 토크나이저 + 시간 파서 + 노드 파서 + 포매터.

```python
@dataclass
class PlusSettings:
    chunk_size: int = 30                    # DEFAULT_CHUNK_SIZE (90)
    summarize_threshold: int = 40           # DEFAULT_SUMMARIZE_THRESHOLD (91)
    max_memory_tokens: int = 20000          # DEFAULT_MAX_MEMORY_TOKENS (92)
    include_user_messages: bool = True      # (257)
    embedding_context_messages: int = 5     # (259)
    time_decay_days: float = 15.0           # (262)
    time_scoring_mode: str = "minute"       # (263) minute|day|none
    time_weight: float = 1.0                # (264)
    similarity_weight: float = 1.0          # (265)
    event_format: str = DEFAULT_EVENT_FORMAT   # (95-96)
    facts_format: str = DEFAULT_FACTS_FORMAT   # (97)
    summary_prompt: str = ""                # Task 3에서 원본에서 추출 주입
```

토크나이저는 문자 코드 분기를 그대로 (520–533):
한글 `AC00–D7AF` 0.8 / CJK `4E00–9FFF` 0.9 / 일본어 `3040–30FF` 0.9 / 그 외 0.25,
합계 `ceil`. **`sum()` 부동소수 누적 순서까지 원본과 같게** 문자 순회로 더한다.

시간 파서는 3형식만 통과 — `YYYY-MM-DD HH:mm`, `MM-DD HH:mm`, `HH:mm`(범위 끝,
시작값의 날짜 상속). 범위는 `→`/`->`로 분리. 그 밖은 `None`.
`normalize_time`은 파싱 실패 시 **원문 문자열을 그대로 반환**한다(680) —
"3일차 저녁" 같은 값이 `node.time`에 저장은 되고 점수 계산에서만 죽는다.

**테스트** (`tests/test_hypaplus.py`)

- `est_tokens("한글")` == 2, `est_tokens("ab")` == 1 (0.25×2=0.5→ceil 1), 혼합 케이스.
- `parse_time("2024-03-05 14:30")` 성공 / `parse_time("3일차 저녁") is None` /
  `normalize_time("3일차 저녁") == "3일차 저녁"` (원문 보존).
- `parse_time("2024-03-05 14:30 → 16:00")` 끝값이 같은 날짜를 상속.
- 날짜 롤오버 거부: `"2024-02-30 10:00"` → `None` (TS의 필드 재확인 로직 1725–1733).
- `parse_nodes`: `<event>` 3개 + `<facts>` 2개 혼합 문자열 → 노드 5개, id는
  1..5 순차, event가 먼저(파서가 event 루프를 먼저 돈다).
- `_parse_event`가 `시간:` 없으면 `None` (724), `_parse_facts`는 `시간:` 없어도
  살아남고 고정 상수 시각을 갖는다 + 폴백 카운터가 증가.
- 대사 파싱: `이름: "대사"` 만 매치, 따옴표 없는 줄은 무시.
- `format_node` 정리 규칙 5종(`" | | "` 축약 등, 1955–1963)이 순서대로 적용.

---

## Task 3: `hypaplus.py` — 요약 루프

**프롬프트 추출.** 레포에 프롬프트 전문을 커밋하지 않는다. 원본 `.ts`에서
`const DEFAULT_SUMMARY_PROMPT = \`` ~ 첫 백틱 종료까지를 런타임에 잘라 쓴다.

```python
def load_default_prompt(ts_path: str) -> str:
    src = pathlib.Path(ts_path).read_text(encoding="utf-8")
    i = src.index("const DEFAULT_SUMMARY_PROMPT = `") + len("const DEFAULT_SUMMARY_PROMPT = `")
    j = src.index("`;", i)
    return src[i:j].replace("\\`", "`").replace("\\\\", "\\")
```

경로는 CLI `--plugin-src` (기본 `~/Documents/arca-modules-2026/hypaplus/HypaPlus v1_0_9.ts`).
파일이 없으면 **조용한 폴백 금지** — 즉시 에러.

**청킹.** `run_summarization` (1531–1603) 그대로:

```python
def run_summarization(msgs, state, S, send) -> bool:
    """한 청크 소비. 소비했으면 True."""
    start = last_summarized_index(state) + 1
    window = msgs[start:start + S.chunk_size]
    if len(window) < S.chunk_size:      # 부분 청크는 대기 (1552-1556)
        return False
    batch = window if S.include_user_messages else [m for m in window if m["role"] != "user"]
    if not batch:
        return False
    chunk = {"id": next_chunk_id(state), "msg_indices": list(range(start, start + len(window))),
             "nodes": [], "summarized": "pending"}
    state["chunks"].append(chunk)
    raw = summarize_chunk(batch, S, send)          # 캐시 경유
    if raw is None:
        chunk["summarized"] = "failed"             # 실패해도 청크는 남는다 (1596-1599)
        return True                                 # ← 인덱스는 이미 소비됨
    chunk["nodes"] = parse_nodes(raw, state)
    chunk["summarized"] = "done"
    return True
```

주의 — 원본은 LLM 실패 시에도 `pending` 청크를 `chatIds`째로 남기므로
**그 30개 메시지는 영구히 요약 대상에서 빠지고 원문 창에서도 빠진다**
(`getLastSummarizedMsgIndex`가 실패 청크도 센다, 888–902). 기억 유실 경로다.
포트는 이 동작을 그대로 재현하고 인스펙터가 `summarized == "failed"`를 이벤트로 기록한다.

**트리거 루프** (`track`, 2005–2041): pending ≥ threshold 인 동안 반복하되,
줄어들지 않으면 중단. 코드 그대로.

**캐시.** hypa와 충돌 없게 별도 키:

```python
def summary_cache_key(batch, S) -> str:
    payload = json.dumps({"system": "hypaplus",
                          "chats": [{"role": c["role"], "content": c["content"]} for c in batch],
                          "prompt": S.summary_prompt, "model": hypa._lucid_model(),
                          "max_tokens": SUMMARY_MAX_TOKENS, "temp": SUMMARY_TEMPERATURE},
                         ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

캐시 디렉터리는 하이파와 공유(`dreaming_data/fixed_haystack/hypa-cache`) — 키에
`system` 판별자가 있어 안전하다. 잘린 응답(`finish_reason=length`)은 하이파와
같은 규약으로 **캐시하지 않는다** (`hypa.mark_truncated` 재사용).

`summarize_chunk`는 `hypa.summarize`가 아니라 자체 구현이지만 전송은
`hypa._summarize_call`을 **점 접근**으로 부른다 (CLAUDE.md §5 — 인스펙터의
캐시전용 패치가 도달해야 함).

**메시지 텍스트 결합**: 원본은 `messages.map(m => m.content).join("\n\n")` (1485) —
**role 접두어가 없다**. 하이파(`f"{role}: {content}"`)와 다르다. 그대로 옮긴다.

**테스트**

- 201개 더미 메시지를 순차 투입 → 트리거 턴이 `[20, 35, 50, 65, 80, 95]`,
  청크 6개, 경계가 `m0–m29 … m150–m179`. (사전 계산 검증)
- `trackAssistant`가 no-op임을 100턴 시뮬로 확인.
- 부분 청크(29개)는 요약되지 않음.
- `send`가 예외 → 청크가 `failed`로 남고 `last_summarized_index`가 그대로 전진.
- 캐시 키가 `system` 판별자 때문에 hypa 키와 절대 충돌하지 않음.
- 프롬프트 추출: 실제 `.ts`에서 뽑은 문자열이 `{{slot}}`을 포함하고
  `<|im_start|>` 블록 3개로 파싱됨.
- **캐시 격리 픽스처 필수** — `tests/test_hypa_main.py`의 autouse
  `_isolated_summary_cache`와 같은 방식으로 tmp_path에 묶는다.
  (선행 플랜 결함 재발 방지: 테스트가 실 캐시를 오염시킨 전례가 있다.)

---

## Task 4: `hypaplus.py` — 검색 · 주입 · step

**점수** (1868–1874) — 부동소수 순서까지 원문 그대로:

```python
sim = sim_scores.get(node["id"], 0.5)          # 임베딩 없으면 전부 0.5
ts  = time_score(node, last_time, S)
tf  = 1.0 if S.time_scoring_mode == "none" else (1.0 - ts) ** S.time_weight
combined = 1.0 - tf * (1.0 - sim) ** S.similarity_weight
```

**선택** (1895–1928): 즐겨찾기 먼저(시간순) → 나머지는 `combined` 내림차순,
`combined < 0.05`면 스킵, 예산 초과 노드는 **자르지 않고 continue**(뒤의 작은
노드가 대신 들어갈 수 있다) → 최종 시간순 재정렬.

**쿼리** (1429–1451): 마지막 메시지 + 그 앞 `embedding_context_messages-1`개를
`"\n"`으로 결합. 우리 루프에선 user append 직후 시점이라 마지막 = 이번 턴 user.

**기준 시각** (`getLatestNodeTime`, 849–864): 전체 노드 중 `parse_time`이
성공하는 것들만 비교해 최댓값. 전부 실패하면 `None` → `time_score`가 일괄 0.5.

**임베딩 축 2종**

- `none` (플러그인 기본값 재현): `embeddingUrl == ""` → `getBatchEmbeddings`가
  전부 `None` → 유사도 맵 비어 있음 → `sim = 0.5` 고정. **이게 배포 기본값이다.**
- `minilm`: `hypa.embed`를 그대로 써서 쿼리·노드 벡터 생성 후 `cosine`.
  (원본은 OpenAI 호환 엔드포인트 + 노드에 벡터 영속 캐시. 포트는 프로세스 내
  dict 캐시로 대체 — 관측 대상이 아니다.)

**주입** (2136–2189): 프리셋에 `[HypaPlus.memory]`가 없으므로 **Strategy 2** —
첫 `role == "system"` 메시지 끝에 `"\n\n" + memory` append. 없으면 index 0에 unshift.
`memory`가 빈 문자열이면 아무 것도 하지 않는다(2185–2189).

**step 시그니처** (인스펙터가 하이파 step과 같은 모양으로 호출할 수 있게):

```python
def hypaplus_step(history, S, state, send, *, embed=None, trace=None):
    """반환 (memory_text, kept, kept_start_msg, state, error) — hypa_step과 동형.

    kept_start_msg = last_summarized_index + 1  (process 핸들러 4293-4296)
    error는 항상 None — HypaPlus엔 하드 실패 경로가 없다(fail-open, 2196-2199).
    """
```

`memory_text`는 노드 0개면 `""` (하이파의 `None`과 달리 빈 문자열).
인스펙터 기록에서 둘 다 falsy로 취급되므로 기존 HTML이 그대로 동작한다.

**trace 이벤트** (하이파 인스펙터와 같은 `_tr` 규약)

| ev | 필드 |
|---|---|
| `chunk` | `chunk_id`, `start`, `end`, `n_msgs`, `status`, `n_nodes`, `cache_hit` |
| `retrieve` | `n_nodes`, `query_len`, `ref_time`, `ref_time_parsed`(bool), `budget`, `used_tokens`, `n_selected`, `n_skipped_budget`, `n_skipped_floor` |
| `scores` | 노드별 `[id, sim, time, combined, picked]` 전량 (노드 수가 수십 규모라 부담 없음) |
| `time_diag` | `n_nodes`, `n_time_parsed`, `n_time_failed`, `n_facts_fallback` |

`time_diag`가 P2/시간축 사망 여부를 직접 답한다.

**테스트**

- 점수 공식: 손계산 케이스 2개(sim 0.82/time 0.953 → 0.99154, sim 0.5/time 0.5 → 0.75).
- `time_scoring_mode="none"` → `tf=1` → `combined == sim`.
- 예산 초과 시 큰 노드를 건너뛰고 **뒤의 작은 노드가 선택**됨.
- `combined < 0.05` 바닥 컷.
- 선택 결과가 시간순 정렬 (파싱 실패 노드는 삽입순 폴백).
- 주입: system 있는 wire → 첫 system 끝에 append. system 없는 wire → index 0 unshift.
  memory 빈 문자열 → 원본 무변경.
- `kept_start_msg`가 실패 청크도 포함해 전진.

---

## Task 5: 인스펙터 `--system` 축

`inspect_hypa.py`의 `run_inspection`에 `system: str = "hypav3"` 추가.
`hypav3`면 기존 경로 그대로(회귀 0), `hypaplus`면 `hypaplus.hypaplus_step`.

- 캐시전용 대역은 `_CacheOnlySummarize`를 재사용하되 패치 대상이
  `hypaplus.summarize_chunk`로 갈린다 → 클래스에 `key_fn` 주입.
- `wire_fn`: hypaplus는 memory를 별도 system으로 넣지 않고 프리셋 system에
  append하므로 전용 함수가 필요하다.
  ```python
  def _plus_wire(preset, card, kept, memory):
      msgs = assemble._wire(preset, card, kept, memory="")   # memory 슬롯 미사용
      return hypaplus.inject(msgs, memory or "")
  ```
- `_slug`: `hypa-inspect-{hsid}-{system}[-{embed}]…`. 기존 hypav3 파일명은
  유지(회귀 방지) — `system == "hypav3"`일 때만 기존 규칙 그대로.
- 스냅샷 대조는 hypav3 전용 — hypaplus면 건너뛴다.
- 새 CLI: `--system hypav3|hypaplus`, `--embed none|minilm`, `--plugin-src PATH`,
  `--chunk-size`, `--memory-tokens`, `--time-decay-days`, `--time-mode`.

**Verify**: `--system hypav3` 재실행 결과 JSON이 기존 파일과 `turns` 배열
바이트 동일 (회귀 0 증명).

---

## Task 6: 니들 생존 오라클 (`needle.py`)

두 시스템 공용, LLM 0콜. 건초더미의 `ledger`(365 팩트, 각 `{fid, value, turn, probed}`)를 쓴다.

```python
def survival(result: dict, hs) -> list[dict]:
    """턴별 기억 생존. value 부분문자열 검사만."""
    for rec in result["turns"]:
        elig = [f for f in hs.ledger if f["turn"] < rec["turn"] and len(f["value"]) >= MIN_VALUE_LEN]
        raw  = "".join(m["content"] for m in msgs[rec["kept_start_msg"]:])
        in_mem = {f["fid"] for f in elig if f["value"] in rec["memory_text"]}
        in_win = {f["fid"] for f in elig if f["value"] in raw}
        ...
```

산출: `n_eligible`, `in_memory`, `in_window`, `covered`(합집합), `lost`(둘 다 아님).

**기저율 주의를 코드로 강제**: 1차 지표는 `probed: true` 팩트만(건초더미당 9개,
프로브와 1:1이라 의미가 명확). 전체 365개는 보조 지표로 별도 필드에 담고
HTML에 "짧은 값은 우연 일치 가능"을 명기한다.

기존 hypav3 관찰 JSON 4개에 소급 적용해 **같은 축의 대조표**를 만든다.

**테스트**: 인공 결과 dict로 in_memory/in_window/lost 분류, `turn` 경계
(`f.turn < rec.turn` 엄격 부등호), `MIN_VALUE_LEN` 컷.

---

## Task 7: HTML 뷰 확장

기존 `render_html`을 분기 없이 확장 (hypav3 파일도 같은 템플릿으로 재렌더).

- 타임라인에 **커버리지 라인** 추가 (초록: covered / 전체, 우측 축).
- hypaplus 턴 상세: 청크 카드(경계·상태·노드 수), 노드 점수표
  (`sim / time / combined / 선택 여부`), 시간 진단 배지
  (`작중 시각 파싱 N/M — 실패 시 시간축 무효` 경고 포함).
- 예산 게이지: `used_tokens / max_memory_tokens` — P2 확인용.
- 파일 설명 박스(`#filedesc`)에 시스템·임베딩 축 설명 추가.
- 용어 박스에 신규 항목: 노드, 작중 시각 감쇠, `combined` 공식, 30개 고정 청킹,
  "임베딩 미설정 시 유사도 0.5 고정" 경고.

**Verify**: 4개 hypav3 HTML 재렌더 후에도 기존 섹션 전부 표시.

---

## Task 8: 런

**8-1 드라이 (₩0).** 캐시전용으로 `drm/van × {none, minilm}` 4런.
미스를 열거하고 종료코드 1. 예상 미스 = 건초더미당 청크 6개 = **총 12콜**
(임베딩 축은 요약에 영향 없으므로 캐시 공유).

**8-2 승인 게이트.** 미스 목록·예상 비용(flash-lite, 입력 ≈30메시지·출력 ≤8192tok
→ 총 $0.05 미만 추정)을 사용자에게 제시하고 **명시적 승인 후에만** `--allow-llm`.

**8-3 관찰 런.** 승인 후 4런 완주 → JSON+HTML 4쌍.
이어서 hypav3 관찰 JSON 4개에 Task 6 오라클 소급 → 대조표 생성.

**8-4 결정론 확인.** 8-3 직후 캐시전용으로 재실행 → 미스 0, `turns` 바이트 동일.

**전체 테스트**: `pytest` 전량 통과 (기존 882 + 신규).

---

## Task 9: 실행 후기

플랜 문서 하단에 다음을 기록 (CLAUDE.md §5):

- P1/P2 예측의 적중·기각 여부와 실측치.
- 사전 계산(트리거 6회, 청크 경계, trackAssistant no-op)이 맞았는지.
- 작중 시각 파싱 성공률 — flash-lite가 `YYYY-MM-DD HH:mm`을 실제로 지어내는가,
  단조증가하는가, 아니면 시간축이 붕괴하는가.
- 발견된 플랜 결함 전량.

---

## 성공 기준

1. `pytest` 전량 통과.
2. `--system hypav3` 출력이 기존 JSON과 바이트 동일 (회귀 0).
3. 건초더미당 청크 6개, 캐시 재실행 시 미스 0.
4. 관찰 HTML 4개 + hypav3 재렌더 4개, 전부 커버리지 라인 포함.
5. P1·P2가 수치로 확인되거나 기각되고, 그 근거가 후기에 남는다.

## 알려진 한계 (착수 전 수용)

- **충실도 앵커 없음.** `hypa.py`는 뮈토스 export + 스냅샷 대조로 실물과 맞춰
  검증했지만, HypaPlus 포트에는 대조할 실행 기록이 없다. 소스 대조로만 담보한다.
  실물 대조는 RisuAI에 플러그인을 실제로 로드해야 하고 그건 사용자 승인 게이트
  (RP_Backend 핸드오프의 라이브 런 항목)에 걸린다 — 이번 범위 밖.
- 임베딩이 원본(BYO OpenAI 호환 엔드포인트)이 아니라 MiniLM이다. 하이파 런과
  **축을 맞추기 위한 의도적 선택**이며, 원본 기본값은 임베딩 자체가 없다.
- 노드 벡터 영속 캐시(원본은 노드에 저장)를 프로세스 내 dict로 대체 — 재실행
  비용에만 영향, 결과 불변.
