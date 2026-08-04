# AGENTS.md — SAGA 작업 가이드

SAGA는 RisuAI 앞단에 두는 OpenAI-compatible 컨텍스트 미들웨어(reverse proxy)다.
프로젝트 개요와 설계 결정은 [README.md](README.md), 행동 규칙은 [CLAUDE.md](CLAUDE.md) 참조.

## 외부 참조 (external/)

RisuAI 소스와 서드파티 모듈은 심볼릭 링크로 참조한다. **읽기 전용** — 절대 수정하지 말 것.

```
external/risuai   -> /Users/yanghyeon-u/Desktop/Risuai   (RisuAI 소스, 384 files / 71K LOC)
external/modules  -> /Users/yanghyeon-u/Desktop/Modules  (서드파티 플러그인/모듈 실전 예시)
```

링크가 깨져 있으면 재생성:

```bash
ln -sfn /Users/yanghyeon-u/Desktop/Risuai external/risuai
ln -sfn /Users/yanghyeon-u/Desktop/Modules external/modules
```

`external/`은 gitignore 대상이다 (로컬 절대경로 링크라 커밋 금지).

## RisuAI 소스 지도 (2026-08-04 분석 기준)

전체 분석은 프로젝트 메모리 `risuai-source-deep-analysis.md` 참조. 자주 쓰는 진입점:

### 채팅/세션/리롤
| 무엇 | 어디 |
|---|---|
| Chat 타입 (`id?: string` — v4 uuid, 옵셔널) | `src/ts/storage/database.svelte.ts:1815` |
| sendChat 메인 플로우 | `src/ts/process/index.svelte.ts:99` |
| 리롤 (꼬리 assistant pop → 재전송, 플래그 없음) | `src/lib/ChatScreens/DefaultChatScreen.svelte:218` |
| 새 채팅 생성 (`id: v4()`) | `src/lib/SideBars/SideChatList.svelte:139` |
| 전송 직전 내부 필드 제거 (memo/cachePoint 등) | `src/ts/process/request/openAI/requests.ts:141` |
| additionalParams (`header::X` 정적 헤더) | `src/ts/process/request/shared.ts:47` |

### 프롬프트 조립
| 무엇 | 어디 |
|---|---|
| PromptItem 타입 (템플릿 카드) | `src/ts/process/prompt.ts:7` |
| 최종 조립 (버킷 → pushPrompts) | `src/ts/process/index.svelte.ts:1271` |
| Anthropic 변환 (선두 연속 system만 병합) | `src/ts/process/request/anthropic.ts:209` |
| 로어북 활성화 + @@데코레이터 | `src/ts/process/lorebook.svelte.ts:75` |
| CBS 매크로 ({{time}}/{{random}} 등) | `src/ts/cbs.ts` |
| regex script (editinput/editprocess/…) | `src/ts/process/scripts.ts` |
| 트리거 (start/input/output/display/request/manual) | `src/ts/process/triggers.ts:20` |

### 메모리 시스템
| 무엇 | 어디 |
|---|---|
| 활성화 게이트 + 우선순위 (hanurai > hypaV2 > hypaV3 > supa) | `src/ts/process/index.svelte.ts:1068` |
| hypaV3 (Similar+Random 매턴 변동 → 캐시 파괴) | `src/ts/process/memory/hypav3.ts` |
| alwaysToggleOn 함정 (토글 강제 재활성) | `src/ts/stores.svelte.ts:189` |

### 플러그인 API
| 무엇 | 어디 |
|---|---|
| v2 로더 + 훅 레지스트리 | `src/ts/plugins/plugins.svelte.ts` |
| v3 API (샌드박스 iframe + 퍼미션) | `src/ts/plugins/apiV3/v3.svelte.ts` |
| v3 타입 선언 (API 표면) | `src/ts/plugins/apiV3/risuai.d.ts` |
| replacerbeforeRequest 호출부 | `src/ts/process/request/request.ts:239` |
| bodyIntercepter 소비부 | `src/ts/globalApi.svelte.ts:744` |
| 실전 플러그인 예시 (chat.id 스코핑 패턴) | `external/modules/RisuAI-Agent-plugin-5.2.5/RisuAI Agent v5.2.5.js` |
| 인밴드 마커 기법 예시 ([LP_AX]) | `external/modules/라이프 프롬프트/life_prompt_plugin_v3.5.js` |

## 핵심 사실 (분석으로 확정)

- 기본 상태에서 **채팅방 식별자는 와이어에 안 실린다** — chat.id는 존재하지만 전송 직전 제거됨
- **리롤은 플래그 없는 재전송**, unreroll은 100% 로컬 (프록시 인지 불가)
- **헤더 동적 추가는 플러그인으로 불가능** — 해법은 replacerbeforeRequest 인밴드 마커
- RisuAI 저장 히스토리는 매 전송마다 매크로 재실행으로 재작성됨 — 전체 해시 비교 휴리스틱 금지
- hypaV3와 SAGA 캐시 설계는 구조적 양립 불가 — HypaMemory OFF + SAGA ON이 결론

## research/ — 리서치 산출물 (git 추적 제외)

구조와 인덱스는 `research/README.md`. 규약:

- **분석 에이전트 산출물은 `research/analysis/<주제>.md`에 직접 Write** (채팅 반환과 별도)
- 다운로드 원본은 `research/downloads/` + INVENTORY.md에 sha256·출처 기록
- 서드파티 코드는 **실행 금지, 정적 분석만**. AGPL(puding)·CC-NC(올인원) 라이선스 — 코드 차용 금지, 아이디어만
- `research/` 전체가 gitignore — 공개 레포에 절대 커밋 금지

## arca.live 접근 플레이북 (2026-08-04 실측)

| 상황 | 방법 |
|---|---|
| 일반 글 읽기 (자동화/서브에이전트) | WebFetch + `https://r.jina.ai/https://arca.live/...` 래핑. 직접 fetch는 403 (Cloudflare). 키 없이 분당 ~20건 |
| 성인글/자료·프롬 카테고리 (HTTP 451) | 로그인 세션 필요 — 인앱 브라우저(로그인됨) 또는 Claude in Chrome |
| 글 제목/존재 확인만 | WebSearch "arca.live characterai <키워드>" |
| 파일 다운로드 | 배포처가 대부분 외부 (GitHub releases, Proton Drive). Proton은 E2E라 curl 불가 → 브라우저 다운로드 → ~/Downloads에서 이동 |
| 수집류 에이전트 모델 | sonnet 충분. 분석·종합만 상위 모델 |

## 작업 규칙

- `external/` 이하는 참조만. 분석 결과는 `research/analysis/` 또는 프로젝트 메모리에 기록
- RisuAI 소스 대량 탐색은 서브에이전트로 격리 (스코프를 파일 단위로 지정)
- SAGA 코드 수정 규칙은 [CLAUDE.md](CLAUDE.md) 우선
