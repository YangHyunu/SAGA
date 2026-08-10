# Dreaming — 캐시를 깨지 않는 장기기억 + 월드 상태 엔진

> **상태**: 스펙 v0.1 (2026-08-04). 구현 전 설계 확정본.
> **선행**: SAGA(본 리포)의 후속. SAGA의 검증된 자산(pair ledger, 캐시 마킹, 프록시 골격)은 승계하고, 파이프라인(Sub-A/Sub-B/Curator)은 폐기한다.
> **설계 아티팩트**: 본편 / 워크플로우 / 경쟁 비교 3부작 (claude.ai artifact, memory/dreaming_design_2026-08.md 참조)

---

## 0. 구현 참조 체계

### 0.1 RisuAI 소스 — external/ 심볼릭 링크 (필수 참조)

구현 시 RisuAI 동작은 추측하지 말고 **소스를 직접 확인**한다.
링크 재생성·전체 진입점 지도는 [AGENTS.md](../../AGENTS.md).

```
external/risuai   -> RisuAI 소스 (읽기 전용, gitignore — 로컬 절대경로 링크)
external/modules  -> 서드파티 플러그인 실전 예시 (읽기 전용)
```

이 스펙이 의존하는 확정 사실과 소스 근거 (`external/risuai/` 기준):

| 스펙 의존 사실 | 소스 근거 |
|---|---|
| chat.id는 존재하나 **와이어 미전송** (전송 직전 내부 필드 제거) | `src/ts/storage/database.svelte.ts:1815`, `src/ts/process/request/openAI/requests.ts:141` |
| 리롤 = **플래그 없는 pop+재전송** (프록시 인지 불가) | `src/lib/ChatScreens/DefaultChatScreen.svelte:218` |
| cachePoint도 전송 직전 제거 → **캐시 마킹은 프록시/프로바이더 몫** | `src/ts/process/request/openAI/requests.ts:141` |
| Anthropic 변환은 선두 연속 system만 병합 (BP1 경계 조건) | `src/ts/process/request/anthropic.ts:209` |
| hypaV3 캐시 병리 4개소 (§1.1의 근거) | `src/ts/process/memory/hypav3.ts:1498`(시드 없는 매턴 셔플) `:1363`(similar 쿼리 슬라이딩) `:1593`(선두 system 삽입) `:1542`(단일 블록 join) |
| 메모리 활성화 게이트·우선순위 (hanurai > hypaV2 > hypaV3 > supa) | `src/ts/process/index.svelte.ts:1068` |
| 로어북 활성화 + @@데코레이터 처리 (§5 inject_lore) | `src/ts/process/lorebook.svelte.ts:75` |
| 저장 히스토리는 매 전송 매크로 재실행으로 재작성 → **전체 해시 비교 휴리스틱 금지** | AGENTS.md 핵심 사실 (risuai-source-deep-analysis) |
| 플러그인 v3 API 표면 (2단계 addProvider 경로) | `src/ts/plugins/apiV3/risuai.d.ts`, 호출부 `src/ts/process/request/request.ts:239` |
| pluginStorage = 세이브 동기화 KV (getItem/setItem, JSON blob, 쿼리 없음) / 기기 로컬은 getLocalPluginStorage | `src/ts/plugins/apiV3/risuai.d.ts:1015`, `:1065` |
| chat.id 스코핑 실전 패턴 | `external/modules/RisuAI-Agent-plugin-5.2.5/` |

### 0.2 리서치 산출물 (research/ — gitignore, 로컬 전용)

- `research/analysis/yumi-provider-manager.md` — addProvider 경로 실증 (§8 2단계 근거)
- `research/analysis/wyglore.md` — keyExcerpts·fossil 계층·pluginStorage 32-shard+gzip (§6.2, §8)
- `research/analysis/openclaw-dreaming.md` — 선별 승격 게이트, sleep-time compute 인용 (§3.2)
- `research/downloads/` — 원본 번들 (INVENTORY.md에 sha256·출처)
- **라이선스 경고**: AGPL(puding)·CC-NC(올인원) — 코드 차용 금지, 아이디어만 (AGENTS.md)

### 0.3 시장·커뮤니티 근거 (프로젝트 메모리)

`~/.claude/projects/-Users-yanghyeon-u-Desktop-RISU-ENE/memory/`:
- `market_research_2026-08.md` — 니치 판정, 하이파×캐시 조합 실태 (arca 원글 번호 포함)
- `game_chat_research_2026-08.md` — The Seed 분석, 게임형 채팅 수렴 패턴 4개
- `papers_dreaming_schema_2026-08.md` — 논문 조사 50편 (arXiv ID 색인)
- `dreaming_design_2026-08.md` — 설계 확정 이력 + 아티팩트 3부작 링크

### 0.4 표기 규약

본문의 구체 사실엔 근거를 병기한다 — 소스는 `파일:라인`, 커뮤니티는 arca 글 번호,
논문은 arXiv ID. **근거가 애매하면 추측으로 구현하지 말고 §0.1~0.3 자료에서 확인 후 진행.**

---

## 1. 포지셔닝

### 1.1 문제

RP 장기 세션에서 유저는 **기억 품질과 캐시 비용 중 하나를 포기**해야 한다.

- HypaMemory V3(RisuAI 내장): 검색 품질은 확보하지만 매턴 프리픽스를 변경한다
  (시드 없는 셔플 `hypav3.ts:1498`, similar 쿼리 슬라이딩 `:1363`,
  system 선두 삽입 `:1593`, 단일 블록 join `:1542` — §0.1)
  → 프롬프트 캐시가 매턴 깨진다.
- 캐싱 플러그인(캐시키퍼/유미/블레싱): BP 마킹은 자동화했지만 메모리가 없다.
- 두 개를 같이 켜면: 커뮤니티 실측 6건 일치
  (arca 177389225 · 176617754 · 178166829 · 166999085 · 161012502 · 178258723) —
  자동으로는 예외 없이 깨지고, 숙련자의 수동 `@position` 재배치로도
  성공률 60~70%가 상한 (166999085 실측). 슬라이딩 윈도우가 근본 원인
  (65턴째 10~60번 → 66턴째 11~61번 로드 → 프리픽스 매턴 이동).
  암묵 컨센서스는 "하이파 끄면 캐싱 안정" = 양자택일.

### 1.2 니치

**캐싱 × 압축 × 기억의 3중 조합은 비어 있다** (2026-08 시장조사 6건 + 소스 분석 3건).

- 캐싱 단독: 포화 (2026.3~5 캐시키퍼/유미/블레싱 확산)
- 기억+월드: The Seed(코히바블랙, 8월 공개 예정)가 근접 — 단 캐시율 0.6%, 외부 런타임 노선
  (arca 177672953 · 177850233, 2026-08-04 실화면 확인. 벤치: Seed 55/55 vs Hypa 40/55,
  $2.11/80턴, 턴 표시 45s)
- 커뮤니티가 수동으로 하려는 것(동적 콘텐츠는 캐시 뒤로, 정적은 앞으로 고정)이
  정확히 이 엔진이 자동화하는 것.

### 1.3 타겟

Claude/OpenRouter 유료 유저 (Gemini 무료 티어에는 캐시 가치가 없음).
프론트엔드는 RisuAI 표준.

---

## 2. 설계 원칙

1. **캐시 계층과 지식 계층의 분리** — 이 프로젝트의 존재 이유.
   - 캐시 계층: immutable 프리픽스 (system + 청크 + 원문 꼬리). byte-stable.
   - 지식 계층: mutable (Fact/WorldState/Actor). 마지막 user 메시지 prepend로만 주입.
   - 지식 계층은 아무리 바뀌어도 캐시를 깨지 않는다.
2. **동기 경로 = 턴당 LLM 0콜.** 기록·검색·조립·주입만 한다.
   프록시는 매 요청 전체 대화를 받으므로 매턴 추출이 필요 없다.
3. **이해는 전부 꿈(Dreamer)에서.** 세션 유휴 시 비동기 1콜/사이클(Flash급).
   근거: Sleep-time Compute(2504.13171), Generative Agents reflection 계보.
4. **압축은 결정론적 조립.** 이해(요약문 생성)는 꿈에서 끝내고,
   압축 시점에는 템플릿 조립만 한다 → LLM 0콜, byte-stable.
5. **서술은 상태를 못 바꾼다.** WorldState는 typed commit으로만 변한다.
   근거: RPGBench(LLM 단독 게임 메커닉 실패), Orchestrated Reality PDVA.
6. **fail-open.** 정체성 판정 불확실·꿈 실패·저장 오류 — 어떤 경우에도
   채팅을 막지 않는다 (HypaV3는 요약 실패 시 채팅 중단 — 반면교사).
7. **유저가 ground truth.** 유저 편집은 Dreamer가 덮어쓸 수 없다.
8. **Dreamer 코어는 저장소·실행환경 중립.** 프록시(1단계)와 플러그인(2단계)이
   같은 코어를 공유한다. 저장 모델은 KV 문서 샤드 단일 — SQL·외부 DB 금지 (§8).

---

## 3. 아키텍처 개요

```
[RisuAI] ──요청──▶ [동기 경로: 0콜]                    [비동기: Dreamer]
                    1. 정체성 판정 (pair ledger)          유휴 감지 (세션 타이머)
                    2. 지식 검색 (hot zone 예산)             │
                    3. 프리픽스 조립 (청크, byte-stable)     ▼
                    4. BP 마킹 (3-BP)                     B-0 스냅샷·락
                    5. 프로바이더 전달                     B-1 에피소드 경계
                    6. 응답 passthrough + 원장 기록        B-2 추출 (1콜)
                                                          B-3 검증·모순 스윕
                                                          B-4 재압축 (조립)
```

### 3.1 동기 경로 (요청마다)

1. **정체성 판정** — pair ledger(content-hash) 기반. 판정 5종:
   `new_session` / `next_turn` / `continuation` / `reroll` / `diverged`.
   왜 content-hash인가: chat.id가 와이어에 안 실리고, 리롤은 무플래그 재전송이며,
   저장 히스토리는 매 전송 매크로 재실행으로 재작성되므로 전체 해시 비교도 불가 (§0.1).
   구현체는 `saga/services/pair_ledger.py` 승계.
   - reroll: 마지막 pair pop 후 재전송 감지 → 이전 턴 기록을 잠정 무효화(삭제 아님).
   - diverged: 중간 편집 감지 → 분기점 이후 Fact를 `provisional`로 강등.
   - 판정 불확실 → fail-open (주입 없이 passthrough, 기록은 격리 버퍼에).
   - 2단계 플러그인에서는 chat.id ground truth로 대체 (판정 로직 자체가 불필요해짐).
2. **지식 검색** — 현재 장면 기준으로 Fact/WorldState/Actor에서 관련분만 선별.
   hot zone 예산 **~2K tokens** 상한. 검색 변동은 캐시 밖(last-user prepend)이므로 무해.
3. **프리픽스 조립** — 청크(§6)는 저장된 조립본 그대로. 재생성 없음.
4. **캐시 마킹** — Anthropic 3-BP: BP1 system 끝 / BP2 첫 청크 assistant /
   BP3 마지막 assistant. 최소 1024 tokens 규칙 준수. TTL "5m" 기본.
   RisuAI가 cachePoint를 전송 직전 제거하므로 마킹은 반드시 이쪽에서 (`requests.ts:141`, §0.1).
   BP1 경계는 Anthropic 변환의 선두 system 병합 규칙과 맞물림 (`anthropic.ts:209`).
5. **전달·기록** — 응답 무가공 passthrough. pair ledger에 원문 기록.

### 3.2 Dreamer (세션 유휴 시)

- **트리거**: 세션별 유휴 타이머. cron 아님. 유휴 기준 = 캐시 TTL 경과(기본 5m)
  → 캐시가 이미 죽은 시점이므로 재압축이 공짜다 (§6.3).
- **사이클** (LLM 1콜 + 임베딩):
  - B-0: 미처리 구간 스냅샷, 세션 락 (락 중 요청 오면 꿈 중단, 동기 경로 우선)
  - B-1: 에피소드 경계 판정 — LLM judge (EM-LLM 서프라이즈 신호 참고)
  - B-2: 추출 — Fact(원자 명제 + typed numbers) / WorldState commit /
    Actor 갱신 / open_threads. 단일 구조화 출력 1콜.
  - B-3: 검증 — 숫자 정규식 재검증(원문 대조), 모순은 즉시 반영 금지
    `pending_contradiction`으로 관찰 (In Praise of Stubbornness 2502.04390),
    mem0 4분류(ADD/UPDATE/DELETE/NOOP)를 갱신 액션 taxonomy로.
  - B-4: 재압축 — 새 에피소드를 청크로 조립, Tier 승격 판정 (§6.2).
- **실패 시**: 해당 사이클 폐기, 다음 유휴에 재시도. 채팅 영향 0.
- **캐치업 드림**: 유휴 창을 놓친 경우(챗 직후 프로세스/기기 종료) 다음 기동 때
  밀린 구간을 꿈꾼다. 복귀 시점엔 TTL이 이미 만료라(캐시 죽음) 비용 논리는 §6.3과
  동일하게 0. 첫 요청은 기존 프리픽스로 즉시 통과시키고 백그라운드로 꿈
  → 2턴째부터 새 프리픽스 적용 (첫 응답 지연 금지).

---

## 4. 데이터 스키마 v2 — 레코드 4종

### 4.1 Fact

원자 명제 1개 = 레코드 1개 (Dense X Retrieval 2312.06648).

| 필드 | 설명 |
|---|---|
| `claim` | 원자 명제 텍스트 |
| `entities[]` | 관련 Actor/사물 참조 |
| `numbers[]` | typed: `{name, value, unit}` — 가격/금액/수량. 문헌 공백 지점, 선행 없음 |
| `evidence` | 원문 포인터: `pair_hash + offset` (HippoRAG2 교훈 — 순수 KG는 정밀도 상실) |
| `valid_time` / `recorded_at` | bi-temporal (Zep 2501.13956) |
| `learned_by[]` | POV: 누가/언제/어떤 경로로 알았나 (RoleFact, REVERIEMEM) |
| `status` | `provisional → confirmed → pending_contradiction → superseded` |
| `supersedes` | 버전 체인. 덮어쓰기 금지, invalidate-and-append (WISE) |
| `user_edited` | 유저 편집 마킹 — Dreamer 덮어쓰기 금지, 게이트 면제 |
| `pinned` | 항상 주입 (HypaV3 Important 대응) |

### 4.2 Episode

| 필드 | 설명 |
|---|---|
| `range` | pair_hash 시작~끝 |
| `title` / `summary` | 꿈에서 생성된 요약 (청크 조립 재료) |
| `causes[]` | 선행 에피소드 인과 링크 |
| `open_threads[]` | 미회수 복선 (CFPG — LLM은 컨텍스트에 있어도 복선 회수 못 함) |
| `embedding` | 검색용 |

### 4.3 WorldState

- **slot 레지스트리** + **typed commit 원장**: `{slot, op, delta|value, turn, evidence, actor}`
- 현재값은 리플레이로 도출. 소지금이 왜 450인지 커밋 히스토리로 추적 가능.
- slot-level 검증 (Behavior Consistency — decision-critical slot 보존).
- 모순 발견 시 `pending_contradiction` — 관찰 후 확정.

### 4.4 Actor

| 필드 | 설명 |
|---|---|
| `names[]` | 한/영 별칭 통합 (SAGA alias 추출 + LLM dedup 방식 승계) |
| `profile` | 요약 프로필 |
| `knows[]` | visibility-gated Fact 참조 — POV 격리 (REVERIEMEM) |
| `tier` | 주역/조연/엑스트라 — 엑스트라는 주입 제외 (NPC 비대화 방지) |
| `last_seen` | 콜드 판정용 |

---

## 5. 프롬프트 레이아웃

```
┌─ system (봇 정의 + 정적 로어)                  ← BP1  ┐
├─ Tier3 시놉시스 (있으면)                              │ 캐시 계층
├─ Tier2 챕터 청크들                                    │ (immutable,
├─ Tier1 에피소드 청크들                        ← BP2  │  byte-stable)
├─ 원문 꼬리 (최근 N pair 무압축)               ← BP3  ┘
├─ [지식 주입] Fact/State/Actor 선별분 ≤2K     ┐ 지식 계층
└─ 마지막 user 메시지                           ┘ (mutable, 캐시 밖)
```

- 동적 로어북 델타도 지식 계층으로 (SystemStabilizer 사상 승계, `saga/system_stabilizer.py`).
- `@@inject_lore` 매크로 처리 포함 (SAGA 미해결 과제 승계 —
  RisuAI 쪽 처리 로직은 `lorebook.svelte.ts:75`에서 확인, §0.1).

- 주입 예산 실측 (fix-drm-r0, 100턴): confirmed fact 329개 = 약 14K자(≈7~9K tok).
  "선별 ≤2K"는 비용 상한이지 물리 한계가 아니다 — 예산은 `HOT_ZONE_CHAR_BUDGET`
  파라미터로 두고, 상향 여부는 benchmarks/retrieval_lab.py의 프로브별 필요예산
  곡선으로 결정한다 (T59류 패러프레이즈 포함 여부가 관건).

---

## 6. 청크 압축

### 6.1 조립

- 청크 = 에피소드 summary의 **결정론적 템플릿 조립**. 압축 시점 LLM 0콜.
- 같은 입력 → 같은 바이트. 캐시 안정성의 전제.

### 6.2 계층 (고도화 압축 — HypaV3 대응)

| Tier | 단위 | 압축률 | 승격 조건 |
|---|---|---|---|
| 원문 꼬리 | 최근 pair | 무압축 | — |
| Tier1 | 에피소드 | ~70% | 에피소드 확정 시 |
| Tier2 | 챕터(에피소드 묶음) | ~90% | 서사 시간 경과 + 참조 빈도 하락 |
| Tier3 | 시놉시스 | ~97% | 챕터 다수 누적 |

- 총량 상한: 컨텍스트의 **~30%**.
- 결정적 원문은 비압축 보존: keyExcerpts 방식 (WygLore 이식 — 유닛당 3개, 병합 시 5,
  400자 게이트. `research/analysis/wyglore.md`, §0.2 — 아이디어만, 코드 차용 금지).
  숫자 모순 검증의 근거 자료.
- 중요도 차등: pinned Fact 관련 에피소드는 승격 지연 (천천히 압축).

### 6.3 TTL 재압축 창구 — 핵심 트릭

- 유휴 = 캐시 TTL 경과 = **캐시가 어차피 죽은 시점**.
- 꿈이 이 시점에 재압축 → 돌아온 첫 요청(어차피 cache miss)에 새 프리픽스 적용.
- 재압축의 한계 캐시 비용 = **0**.
- 유휴 전 예산 임계 도달 시에만 캐시 파괴를 감수 (턴당 상각 ~2%).

- **프로바이더 한정 주의** (fix-drm-r0 실측, FINDINGS §2): "유휴 재압축 = 공짜"는
  Anthropic처럼 TTL 만료로 캐시가 소멸하는 프로바이더에서만 성립한다. DeepSeek의
  자동 프리픽스 캐싱은 유휴와 무관하게 바이트가 바뀌면 그대로 miss — 재압축 비용이
  0이 아니다. 비-Anthropic에서는 재압축 빈도 자체를 낮추는 것(BOUNDARY_STEP)이 방어선.

---

## 7. 유저 메모리 UI — 열람·수정

HypaV3의 편집 UI(textarea 수정, Important 토글, 재요약 diff 리뷰)를 구조화로 상회.

### 7.1 열람 (1단계: 로컬 웹 대시보드)

- **연대기 뷰**: Episode 시간순 → 클릭 시 소속 Fact + 원문(evidence).
- **상태판**: WorldState 현재값 + 커밋 히스토리 드릴다운.
- **주입 로그**: 이번 턴에 뭐가 주입됐나 — 캐시 히트 관측 공백(미해결 니치 ③) 해소.

### 7.2 수정 — 전부 지식 계층이라 캐시 무손상

- Fact 수정/삭제/핀: 즉시 반영. `user_edited` 마킹 → Dreamer 보호.
- 상태값 직접 수정: 수동 커밋으로 기록 (감사 체인 유지 = "OOC 정정"의 UI 버전).
- **청크만 예외**: 프리픽스라 즉시 수정 시 캐시 파괴 → 읽기 전용 + **수정 예약**.
  편집하면 다음 TTL 창구에서 반영. UI에 "다음 꿈에서 반영됨" 표시.

### 7.3 2단계

대시보드를 RisuAI 모달로 포팅 (WygLore unit-inspector 패턴).

---

## 8. 배포 전략 — 2단계

| | 1단계: 프록시 | 2단계: 플러그인 프로바이더 |
|---|---|---|
| 목적 | 알고리즘 검증 (코드 자산·벤치) | 배포 폼팩터 (커뮤니티 채택 형태) |
| 정체성 | pair ledger (content-hash) | **chat.id ground truth — ledger 불필요** |
| 경로 | reverse proxy (SAGA 골격 승계: `saga/server.py`, `saga/services/`) | `addProvider` (유미가 실증 — `research/analysis/yumi-provider-manager.md`; API 표면 `risuai.d.ts`, §0.1) |
| 저장 | KV/JSON 샤드 디렉터리 (write-temp+rename로 crash 안전) | **pluginStorage** (세이브 동기화 KV, `risuai.d.ts:1015`) + WygLore 32-shard+gzip 패턴 |
| 대형 세션 | — | 서버 모드 옵션 (원클릭 페어) |

- 서버 단독 배포 금지 (Letta+PostgreSQL 소개 글 1227뷰/10댓글 — "읽었지만 못 씀".
  근거: `market_research_2026-08.md` 배포 폼팩터 판정, §0.3).
- **저장 모델은 처음부터 KV 문서 샤드로 고정. SQL·외부 DB(PostgreSQL/Honcho/Mem0류) 금지.**
  근거: 플러그인 세계의 유일한 영속 수단이 pluginStorage(key→JSON blob, 쿼리 없음)이고,
  클라이언트 생태계 전례가 전부 KV/JSON(HypaV3=챗 내장, WygLore=샤드, 유미=KV),
  SQL/서버 DB 쓴 쪽(Letta, The Seed)은 전부 채택 실패. Phase 1과 2가 같은 저장
  모델을 쓰면 포팅 = 백엔드 교체뿐. 데이터량(세션당 수천 레코드)상 검색은
  메모리 로드 후 임베딩+필터로 충분 — SQL 쿼리력 불필요.
- Dreamer 코어 인터페이스: `Storage`(get/put/scan(namespace) — pluginStorage 표면과 1:1) /
  `IdleTrigger` / `LLMClient` 추상화 — 처음부터 양쪽 환경 공용.

---

## 9. 평가

- **드라이버**: 유저 시뮬레이터 (대본에 지뢰 = 검증 대상 사실 심기).
  코히바블랙 디렉터 방식 참고 (10막 80비트, 거리 기반 재질문 프로브).
- **채점**:
  - 숫자/상태: **결정론 오라클** — typed WorldState 덕에 채점 0콜.
  - 서사: LLM judge (NARRA-Gym 3-judge 평균 참고).
  - 10~20% 수동 감사 (챈 신뢰 포맷).
- **대조군 (의무)**: Risu 순정 / HypaV3 / **단순 turn-retrieval 베이스라인**
  (2604.11628 "Back to Basics" — TIR(Turn Isolation Retrieval)은 논문 내 기법명.
  4타입 스키마는 이 베이스라인을 ablation으로 이겨야 정당화. 단순-베이스라인
  논지의 원 인용이던 2511.17208은 EMem/EDU 표현법 논문으로 확인돼 교정) / Dreaming.
- **병기 지표**: $ / 캐시율 / 턴 준비시간 (The Seed 표 포맷 — 55점 수동채점과 비교 가능하게).
- **상세 프로토콜**: [EVAL2.md](EVAL2.md) — 80턴 3단계 대본, 프로브 5유형 ~40체크,
  이중 오라클(결정론+LLM judge), 3회 반복 mean±std (2026-08-06 딥리서치 종합).

---

## 10. 비스코프 (v1에서 안 함)

- 그룹챗 turn-taking (MultiLIGHT — 별도 난제)
- 프리셋/탈옥 경쟁, 번역
- 무압축 노선 (불가능 확인됨)
- 페르소나 드리프트 완전 방지 (문헌상 불가 — 재주입 완화만)

## 11. 리스크

| 리스크 | 대응 |
|---|---|
| The Seed 8월 공개 — 창 닫힘 | 차별축이 다름 (캐시 0.6% vs 캐시 중심). 벤치에서 정면 비교 |
| RisuAI 내부 변경 (hypa/프롬 조립) | addProvider 경계만 의존, 내부 훅 미사용 |
| 이름 미정 ("캐시키퍼" 선점됨) | 스펙과 무관, 공개 전 결정 |
| 검색 트리거링 실패 (MREval) | 주입 로그로 관측 가능하게 — 실패를 보이게 만드는 게 1차 방어 |
