# HypaV3 동작 인스펙터 — 설계 (2026-08-27)

핸드오프: `/var/folders/.../handoff-hypav3-inspector.md` (2026-08-24 작성).
목표 = HypaV3가 켜져서 압축되고 RAG가 도는 과정을 **눈으로 보는 도구**.
채점(judge/probe 통과율)·성능 비교는 범위 밖. 기존 run/report 경로 무변경.

## 0. 핸드오프 전제 교정 (이 세션의 발견)

핸드오프 §4는 "기본값 `useExperimentalImpl: false` → 실사용 경로는
`hypaMemoryV3Main`이니 포트를 main으로 고쳐라"였다. 그러나:

- 벤치가 고정한 export(`뮈토스 하이파 V5.json`)에 **`useExperimentalImpl: true`가
  박혀 있다.** 병합 로직(hypav3.ts:1819~1823)은 "키 존재 + typeof 일치"면
  받아들이고, 진입 분기(hypav3.ts:129)가 이 값으로 exp를 태운다.
- 즉 **뮈토스 V5 실사용자는 exp 경로를 탄다.** 현 `hypa.py`가 exp를 모델링한
  것은 실수가 아니라 export에 충실한 것 (파일 독스트링 "exp 경로 단일" 명시).

결정(사용자): **둘 다 관찰한다.** exp = 뮈토스 V5 실사용 경로,
main = 프리셋 없는 기본값 유저 경로. 인스펙터에 impl 축(`--impl exp|main`).

## 1. 관찰 조건 (사용자 확정)

| 항목 | 검증 런 | 관찰 런 |
|---|---|---|
| 목적 | 포트·인스펙터 자체 검증 (기존 스냅샷 대조) | 실제 관찰 대상 |
| impl | exp | exp / main 각각 |
| memoryTokensRatio | 0.39 (export 원본) | **0.2** (오버라이드) |
| extraSummarizationRatio | 0.01 (export 원본) | **0.1** (오버라이드) |
| maxContext | **100000** (건초더미 run_params 동결값 — 실측 확인) | **45000** (사용자 지정) |
| maxResponse | 30000 (run_params) — hypa_step에선 가감이 상쇄돼 동작 무영향 | 동일 |
| 요약 모델 | google/gemini-3-flash-preview (캐시 원본) | **google/gemini-2.5-flash-lite** |
| 임베딩 | MiniLM (all-MiniLM-L6-v2, 로컬) | MiniLM — RisuAI 기본값 재현. 한국어 약점 자체가 관찰 대상 |
| 비용 | ₩0 (캐시 전부 히트) | 첫 실행만 캐시 미스분 실콜 (< $0.5 예상), 이후 ₩0 |

export의 나머지 값 그대로: maxChatsPerSummary 8, queryChatCount 3,
recent 0.6 / similar 0.4 / random **0** (도달 불가 가드 유지),
preserveOrphanedMemory true, enableSimilarityCorrection false,
summaryChunkSeparator `\n\n`.

오버라이드는 export 파일 무수정 — 인스펙터 CLI 플래그
(`--memory-ratio`, `--extra-ratio`, `--summary-model`, `--embed-model`)로만.
출력 파일명과 json 헤더에 유효 세팅을 박아 어떤 조건의 관찰인지 남긴다.

검증 앵커 주의: 기존 프로브 스냅샷(t19~t99)은 0.39/0.01/gemini-3-flash로
만들어진 것 → **오버라이드 없는 exp 실행에서만** 대조 가능.

**실측 완료 (2026-08-27, 설계 단계에서 선행 확인)**: send를 예외 폭탄으로 바꾼
100턴 재생을 두 건초더미 모두 완주 — 캐시 미스 0, LLM 콜 0, 프로브 9턴
스냅샷 chatMemos 전부 일치 (fix-drm-r0 요약 18개, fix-van-r0 11개).
검증 런 ₩0은 가정이 아니라 확인된 사실. 첫 압축이 70턴대에야 발동한 이유 =
run_params maxContext 100000. 관찰 런 45000에선 훨씬 일찍 발동할 것.

## 2. 모듈 배치

```
benchmarks/eval/hypa.py                    기존 exp 포트. 로직 무변경.
                                           select_summaries·hypa_step에
                                           trace=None 키워드만 추가.
benchmarks/eval/hypa_main.py               신규. hypaMemoryV3Main 포트.
                                           공용 부품은 hypa.py에서 import.
benchmarks/fixed_haystack/inspect_hypa.py  신규. 인스펙터 CLI + HTML 생성.
tests/test_hypa_main.py                    신규.
tests/test_inspect_hypa.py                 신규.
```

플래그가 아닌 **별도 모듈**인 이유: exp↔main 차이가 배치 루프 구조·similar
쿼리 구성·빈 카드 삽입·metrics 4곳에 퍼져 있어 분기 플래그면 양쪽 다 읽기
어려워진다. `hypa.py` 무변경(훅 제외)으로 기존 run 재현 보장이 공짜.

함정 1개(박제): `hypa_main`은 `summarize`를 **`hypa.summarize` 점 접근**으로
호출한다. `assemble.bench_summary_cache()`가 `hypa_mod.summarize`를 패치하는데
import-시점 별칭은 스냅샷이라 패치가 안 닿는다 (CLAUDE.md §5 사례).

## 3. main 포트 — exp와 갈라지는 지점 (파일:라인 전부 확인됨)

| # | 항목 | exp (`hypaMemoryV3MainExp`) | main (`hypaMemoryV3Main`) |
|---|---|---|---|
| 1 | 배치 담기 | 찰 때까지 스캔 — `toSummarize.length < maxChats` (hypav3.ts:293~296) | **고정 창** — `endIdx = min(startIdx+maxChats, len−queryChatCount)` (1059~1062). 스킵되면 배치가 작아짐 |
| 2 | 요약 시점 | 배치 전부 모은 뒤 일괄 + rate limiter (256, 373, 382) | **배치마다 즉시** summarize→push (1151~1159). 중간 실패 시 앞 요약 보존 |
| 3 | similar 쿼리 | 최근 챗 `\n\n` 문단 분할 서브쿼리, 가중치 `(i+1)/(n(n+1)/2)/문단수` (680~695) | 챗 **원문 통짜** 쿼리 (1369), simpleCC 콜백 가중치 `(i+1)/(n(n+1)/2)` (1409~1414) |
| 4 | 요약 0개 턴 | 메모리 카드 미삽입 (현 포트 §7 #9) | **빈 카드라도 삽입** (1188~1199). 빈 카드 토큰 예약 조건 `summaries==0 && current+empty≤max` (1020~1022) |
| 5 | metrics | 포트: selectedCount+memos만 | `lastImportant/Recent/Similar/Random` 인덱스 4종 (1578~1591) |
| 6 | random 패스 | 도달 불가 가드 (export ratio 0) | 동일 가드 유지, 주석만 main 라인(1470~1520)으로 |
| 7 | 과요약 직전 break | 배치 미수집·startIdx 유지 (352~362) | **동일 위치·동일 조건** (1129~1139) — 차이 아님 |

미확인 잔여 1건: `HypaProcesserEx.similaritySearchScoredEx`(1918~2042)의
코사인 계산이 포트 `_cosine`과 동치인지 — 구현 중 원본 읽고 확정.

벤치 데이터엔 스킵 대상(NewChat/빈 메시지/example)이 없어 **배치 경계는 두
구현이 동일할 전망** → 같은 세팅이면 요약 캐시 공유.

## 4. 관찰 훅

`trace: Optional[list]` 키워드 인자. `None`(기본) = no-op, 기존 동작 그대로.
리스트를 주면 이벤트 dict append만 — 선택/배치 로직 무접촉.

이벤트 스키마 (exp·main 공통 — HTML 뷰 하나로 양쪽 렌더):

```python
{"ev": "reserve",  "should_reserve": bool, "available": int, "current": int}
{"ev": "batch",    "idx": [모지수...], "tokens": int,
                   "stop": None | "over_summarize_break" | "target_reached" | ...}
{"ev": "important","picked": [인덱스...], "available_after": int}
{"ev": "recent",   "picked": [...], "reserved": int, "used": int}
{"ev": "similar",  "queries": [{"content", "weight"}...],
                   "chunk_scores": [[청크id, 점수]...],  # 쿼리별
                   "cc_rank": [...], "rrf_rank": [...],
                   "picked": [...], "reserved": int, "used": int}
```

## 5. 인스펙터

```bash
python3 -m benchmarks.fixed_haystack.inspect_hypa \
    --hs fix-drm-r0 --impl exp \
    [--memory-ratio 0.2 --extra-ratio 0.1] \
    [--summary-model google/gemini-2.5-flash-lite] [--embed-model ...] \
    [--allow-llm]
```

- **전진 루프**: `data={}`에서 시작, 턴 1~100을 `hypa_step`(exp) 또는
  `hypa_main.hypa_step`(main)으로 이어달리기 — 실물 RisuAI가 채팅마다 상태를
  누적하는 것 재현. 스냅샷 소비 아님(스냅샷은 프로브 9턴짜리 검증 앵커).
- **검증 앵커**: 오버라이드 없는 exp 실행이면 프로브 턴(t19~t99)마다 기존
  스냅샷과 요약 개수·chatMemos 대조. 불일치 = 인스펙터/포트 버그.
- **조립 재사용(읽기 전용)**: `assemble.hypa_fixed_tokens`, `_wire`+`_tokens`
  (wire_tokens 산출), `haystack` 로더. `benchmarks/eval/**`·`dreaming/**`엔
  쓰지 않는다.
- **기본 `--no-llm` 동작**: summarize 캐시 미스 → 즉시 중단 + 미스 배치 목록
  출력(몇 콜이 필요한지). `--allow-llm` 있을 때만 실콜 (flash-lite).
- 턴별 기록: `memory_text` 전문, `kept_start_msg`, `len(summaries)`,
  배치 이벤트, 선택 4단계 이벤트, wire_tokens, main이면 metrics 4종.

## 6. 출력

- `dreaming_data/fixed_haystack/hypa-inspect-<hsid>-<impl>[-mtr<x>-esr<y>].json`
  — 헤더에 유효 세팅 전문 + 턴별 레코드 배열.
- 같은 이름 `.html` — 자급자족 단일 파일. 턴 타임라인: 압축 발동 마커,
  `kept_start_msg` 계단, 요약 개수 곡선, 턴 클릭 → memory_text·similar 점수
  패널. exp/main json을 나란히 로드해 비교 가능.
- HTML 선례 참고: `rag-inspect-r0.html` (형식만).

## 7. 데이터 준비

현 워크트리엔 `dreaming_data/fixed_haystack/` 없음. `hypa-inspector` 워크트리
(브랜치 `claude/hypa-inspector`, 같은 커밋 8d63646, 코드 무변경)에 44M 실물
존재 → **복사**해온다 (심링크는 원 워크트리 삭제 시 끊겨 배제).

## 8. 테스트 (TDD 순서)

1. main 배치 고정 창: 스킵 메시지 픽스처에서 exp와 배치가 갈라지는 케이스 고정
2. 과요약 break 위치·미요약 배치 보존 (양 구현 동일 조건 확인)
3. 빈 카드 삽입 + 빈 카드 토큰 예약 조건 (main)
4. similar 통짜 쿼리·simpleCC 콜백 가중치 (main, fake embed 주입)
5. metrics 인덱스 4종 (main)
6. 무회귀: 기존 exp 테스트 전부 통과 + `trace=None` 경로 무변경
7. 인스펙터 e2e: fake summarize·미니 건초더미 3턴, `--no-llm` 중단 경로

## 9. 하지 말 것 (핸드오프 §6 승계)

- 채점·프로브·judge 경로 수정
- 기존 exp 포트 로직 변경·삭제
- `benchmarks/eval/**`·`dreaming/**`에 인스펙터가 쓰기
- 스냅샷/건초더미 원본 오염 (전진 루프는 자체 data라 해당 없음, 대조는 읽기만)
- 캐시 미스 상태로 `--allow-llm` 없이 실콜

## 10. 완료 기준

1. main 포트 ↔ exp 포트 갈라지는 지점이 §3 표 + 코드 주석으로 문서화됨
2. 검증 런(원본 세팅·exp)이 LLM 0콜로 100턴 완주 + 스냅샷 대조 통과
3. 관찰 런(0.2/0.1/flash-lite, exp·main)이 완주하고, 출력에서 눈으로 확인:
   첫 압축 발동 턴 / 요약 개수 증가 / kept_start_msg 점프 / memory_text 변화 /
   similar가 어떤 요약을 왜 골랐는지 (청크 점수까지)
4. 기존 `run.py` 경로 무변경 — 기존 테스트 전부 통과
5. 실행 후기에 핸드오프 전제 오류(export exp=true) 박제
