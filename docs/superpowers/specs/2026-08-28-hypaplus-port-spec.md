# HypaPlus v1.0.9 → Python 포트 대응표

플랜: [2026-08-28-hypaplus-inspector.md](../plans/2026-08-28-hypaplus-inspector.md)

## §0 동결 대상

| | |
|---|---|
| 파일 | `~/Documents/arca-modules-2026/hypaplus/HypaPlus v1_0_9.ts` |
| 줄 수 | 4,320 |
| sha256 | `81b6c4e3f6c395811ff0950f548833cd809d5e4f8dbe7adbde9ee4f692c7da26` |
| 형태 | 단일 IIFE, RisuAI 플러그인 API 3.0 |

아래 모든 줄번호는 이 판본 기준. 18건 전부 `sed -n '<L>p'`로 함수 시작 줄임을
확인했다(플랜 작성 시점). 레포에는 원본을 복사하지 않는다 — 요약 프롬프트에
`Rating: NC-21+` 지시가 있어 프로젝트 상시 제약(성인 콘텐츠 흔적 금지)에 걸린다.
포트는 프롬프트를 런타임에 원본 파일에서 잘라 쓴다(`load_default_prompt`).

## §1 함수 대응표

| TS | 줄 | Python (`benchmarks/eval/hypaplus.py`) |
|---|---|---|
| `heuristicEstimateTokens` | 520 | `est_tokens` |
| `normalizeTimeString` | 647 | `normalize_time` |
| `parseNodesFromSummary` | 682 | `parse_nodes` |
| `parseEventBlock` | 716 | `_parse_event` |
| `parseFactsBlock` | 761 | `_parse_facts` |
| `getLastSummarizedMsgIndex` | 888 | `last_summarized_index` |
| `buildEmbeddingQuery` | 1429 | `build_query` |
| `parseChatMLPrompt` | 1462 | `parse_chatml` |
| `summarizeChunk` | 1481 | `summarize_chunk` |
| `runSummarization` | 1531 | `run_summarization` |
| `buildNodeText` | 1651 | `node_text` |
| `computeTimeScore` | 1673 | `time_score` |
| `parseTimeString` | 1693 | `parse_time` |
| `compareNodesByTime` | 1751 | `_time_key` |
| `retrieveRelevantNodes` | 1763 | `retrieve` |
| `formatNodeForContext` | 1940 | `format_node` |
| `buildMemoryContext` | 1969 | `build_memory` |
| `trackUserMessage` | 2005 | `track` |
| `beforeRequestHandler` 주입부 | 2136–2189 | `inject` |
| `process` 핸들러 | 4284 | `hypaplus_step`의 `kept_start_msg` |

## §2 결정론 위험 — 포트에서 고정

| # | 원본 | 줄 | 포트 |
|---|---|---|---|
| 1 | `createdAt: Date.now()` (노드) | 752, 782 | 삽입 순 단조 카운터 `_seq` |
| 2 | `createdAt: Date.now()` (청크) | 1508, 1573 | 미사용 — 포트는 청크 순서를 리스트 순으로만 씀 |
| 3 | `parseFactsBlock`의 `nowTimeString()` 폴백 | 777 | 상수 `FACTS_FALLBACK_TIME`, 발동 횟수를 관측치로 기록 |

`nextNodeId`/`nextChunkId`(553, 590)는 기존 최대 id 기반이라 이미 결정론적 — 그대로.

## §3 의도적 미포팅

| 대상 | 줄 | 사유 |
|---|---|---|
| regex 전처리 | 279–443 | 적용할 regex 프리셋이 건초더미에 없다 (기본 미적용) |
| `recentNodeCache` | 1793–1799, 2110–2132 | 기본 false, `[HypaPlus.cached]` 플레이스홀더도 프리셋에 없음 |
| GUI·백업·프리셋 임포트 | 2228–4270 | 관측 대상 아님 |
| `reSummarizeChunk` | 1605 | 사람이 GUI로 누르는 재요약 |
| 노드 벡터 영속 캐시 | 1806–1863 | 프로세스 내 dict로 대체 — 재실행 비용에만 영향, 결과 불변 |

`favorite` 노드는 코드 경로를 남긴다. 자동 런에선 항상 공집합이라 no-op.

## §4 원본 그대로 옮긴 비직관 동작

1. **요약 입력에 role 접두어 없음** — `messages.map(m => m.content).join("\n\n")`
   (1485). hypa는 `f"{role}: {content}"`를 쓴다. 다르다.
2. **실패 청크도 인덱스를 소비** — LLM 실패 시 `summarized: "failed"` 청크가
   남고 `getLastSummarizedMsgIndex`(888)가 그것도 센다. 그 30개 메시지는 영구히
   요약에서 빠지고 원문 창에서도 빠진다. 기억 유실 경로.
3. **부분 청크는 대기** — `window.length < take`면 요약 안 함 (1552).
4. **예산 초과 노드는 skip, truncate 아님** — 뒤의 작은 노드가 대신 들어간다 (1916).
5. **`normalizeTimeString`은 파싱 실패 시 원문 반환** (680). "3일차 저녁"이
   `node.time`에 저장은 되고 `parseTimeString`에서만 죽어 `timeScore` 0.5 고정.
6. **`parseEventBlock`은 `시간:` 없으면 노드를 버린다** (724). `_parse_facts`는 살린다.
7. **`maxContext`를 한 번도 읽지 않는다** — 압축 트리거도 주입 예산도 전부
   메시지 개수와 자체 문자 가중 토큰 추정으로만 결정된다.

## §5 주입 위치

프리셋에 `[HypaPlus.memory]` 플레이스홀더가 없으므로 항상 Strategy 2 (2166–2186):
첫 `role == "system"` 메시지 끝에 `"\n\n" + memory` append. 없으면 index 0 unshift.
`memory`가 빈 문자열이면 무변경 (2185–2189).

뮈토스 와이어는 `msgs[0]`이 system이라 `check_wire_shape`를 계속 통과한다.
