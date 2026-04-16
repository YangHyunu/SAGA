# HypaMemory V3 vs SAGA — 소스코드 기반 비교

RisuAI 기본 메모리 시스템(HypaMemory V3)과 SAGA의 구조를 소스코드 레벨에서 비교한다.

---

## 공통점: 검색 구조가 거의 같다

둘 다 오래된 턴을 요약하고, 벡터 검색으로 관련 기억을 골라서 주입한다. 랭킹 알고리즘도 같다.

| | HypaMemory V3 | SAGA |
|--|---------------|------|
| 요약 | 보조 AI에 6개씩 묶어서 요약 | Sub-B Flash 4필드 요약 |
| 임베딩 | MiniLM / BGE / OpenAI / Voyage | text-embedding-3-small |
| 유사도 | 코사인, chunk 단위 | 코사인, 에피소드 단위 |
| 랭킹 | `childToParentRRF`, k=60 | RRF, k=60 |
| 검색 풀 | Important → Recent(40%) → Similar(40%) → Random(20%) | Recent(1.2) → Important(1.0) → Similar(0.8) |

RRF 상수(k=60)까지 동일. 검색 자체의 품질 차이는 크지 않다.

---

## 차이점 1: 요약 비용

HypaMemory는 토큰 초과 시마다 보조 AI를 호출해서 요약을 만든다. SAGA는 Sub-B가 매 턴 이미 생성한 turn_log 요약을 재활용한다.

```
HypaMemory: 토큰 초과 → 보조 AI 호출 → 요약 생성 (비용 발생)
SAGA:       매 턴 Sub-B가 Flash 요약 생성 → turn_log 저장 → 압축 시 재활용 (추가 0회)
```

---

## 차이점 2: 주입 위치와 캐시 (가장 큰 차이)

### HypaMemory V3가 보내는 메시지 구조

```
[system] 캐릭터 카드 + 로어북              ← 캐싱 안 됨 (string 형식으로 전송)
[user]   "System: <Past Events Summary>    ← 매 턴 Similar 결과가 달라서 내용 변동
          Turn 1-3 요약, Turn 7 요약..."       → 이후 전체 prefix 캐시 미스
[assistant] Turn N 응답
[user]   Turn N+1                          ← 여기에 자동 캐싱 (마지막 user 3개)
...
```

소스코드 근거 (`hypav3.ts`):
```typescript
const newChats = [{
    role: "system",
    content: memory,        // 매 턴 재생성됨
    memo: "supaMemory",
}, ...chats.slice(startIdx)]
```

Anthropic 핸들러에서 이 system 메시지가 user로 변환됨 (`anthropic.ts`):
```typescript
case 'system':
    if (claudeChat.length === 0) {
        systemPrompt += chat.content        // 첫 번째만 system
    } else {
        addClaudeChat({
            role: 'user',
            content: "System: " + chat.content   // 나머지는 user로 변환
        })
    }
```

### SAGA가 보내는 메시지 구조

```
[system] 캐릭터 카드 (SystemStabilizer 고정)     ← BP1, cache_control 적용
[user+asst] chunk: Turn 1-8 요약                  ← 불변 (immutable), 캐싱됨
[user+asst] chunk: Turn 9-16 요약                 ← 불변, BP2
[user] Turn 17
[assistant] Turn 17 응답
...
[user] Turn 20 + [SAGA Dynamic: 에피소드 + 상태]  ← 동적 컨텍스트는 맨 끝 (prefix 밖)
```

### 비교

| | HypaMemory V3 | SAGA |
|--|---------------|------|
| system 프롬프트 캐싱 | 안 됨 (string 형식) | BP1로 캐싱 (content array + cache_control) |
| 메모리 주입 위치 | system 바로 뒤 (prefix 깨뜨림) | 마지막 user (prefix 밖) |
| 요약 chunk 안정성 | 매 턴 검색 결과에 따라 변동 | 한번 만들면 불변 |
| 캐시 히트율 (50턴) | 12.1% | **85.7%** |
| 비용 효과 | -11.4% (손해) | **43.5% 절감** |

---

## 왜 이런 차이가 나는가

Anthropic 프롬프트 캐싱은 **prefix가 동일해야** 히트한다. 메시지 배열 앞쪽이 바뀌면 그 이후 전체가 캐시 미스.

HypaMemory V3는 **압축과 검색이 합쳐져 있다.** 요약 chunk를 만들고, 매 턴 그중 관련 있는 걸 골라서 system 뒤에 주입한다. 검색 결과가 매 턴 달라지니까 주입 내용도 달라지고, prefix가 깨진다.

SAGA는 **압축과 검색을 분리했다:**
- MessageCompressor: 압축만 담당. chunk는 대화 안에 고정 위치에 남음 (불변 → 캐시용)
- Sub-A Context Builder: 검색만 담당. 결과는 마지막 user에 붙임 (동적 → prefix 밖)

이 분리 덕분에 prefix가 안정적이고 캐시가 유지된다.

---

## 차이점 3: RisuAI 자체 캐싱의 한계

RisuAI에도 `automaticCachePoint` 설정이 있다. 켜면 마지막 user 메시지 3개에 `cache_control: ephemeral`을 붙인다.

문제:
- **system 프롬프트를 캐싱 안 함** — 캐릭터 카드 + 로어북(수천 토큰)을 매번 새로 읽음
- **마지막 user 3개에만 캐시** — 앞쪽 대화는 캐싱 대상이 아님
- 턴 12 이후 Anthropic의 **20-block lookback 제한**에 걸려서 캐시 무효화
- 1시간 TTL 지원은 있지만 prefix 불안정하면 의미 없음

소스코드 (`index.svelte.ts`):
```typescript
// 뒤에서부터 user 3개에만 cachePoint 할당
if (DBState.db.automaticCachePoint && !hasCachePoint) {
    let pointer = formated.length - 1
    let depthRemaining = 3
    while (pointer >= 0) {
        if (depthRemaining === 0) break
        if (formated[pointer].role === 'user') {
            formated[pointer].cachePoint = true
            depthRemaining--
        }
        pointer--
    }
}
```

system 캐싱이 빠진 이유 — system을 string으로 보냄 (`anthropic.ts`):
```typescript
body.system = systemPrompt   // string, cache_control 붙일 수 없음
// Anthropic은 system을 content array로 보내야 cache_control 적용 가능
```

---

## 차이점 4: SAGA에만 있는 기능

| 기능 | 설명 |
|------|------|
| 모순 탐지 | 죽은 NPC 재등장, 타임라인 불일치, 캐릭터 중복 감지 (Curator) |
| 복선 추적 | 서사 구간 분리, 복선 태깅, 회수 감지 (Curator) |
| 로어 자동생성 | 로어가 없는 엔티티에 Flash LLM으로 자동 생성 |
| NPC 3-Layer dedup | 정규화 → fuzzy(88%) → cross-script romanization(72%) |
| 구조화 요약 | 텍스트 덩어리가 아닌 { summary, npcs, scene_type, key_event } 4필드 |
| 슬라이딩 윈도우 복구 | hash 비교로 윈도우 이동 감지 → 잘린 턴 요약 주입 |
| 비용 추적 | 모델별 단가, 캐시 절감, 세션별 집계 |

---

## HypaMemory V3가 나은 점

| 항목 | 설명 |
|------|------|
| 설치 불필요 | RisuAI 내장, 서버 없이 동작 |
| Random 풀 | 예상 못한 기억을 불러올 수 있음 (SAGA는 없음) |
| Voyage Context 3 | 문맥 인식 임베딩 지원 (chunk 간 관계 반영) |
| 로컬 임베딩 | 브라우저에서 MiniLM/BGE 실행 (API 호출 없음) |

---

## 같이 쓸 때 문제

HypaMemory V3와 SAGA를 동시에 켜면:

1. **이중 요약** — HypaMemory가 턴 1-5 요약 → MessageCompressor도 같은 턴 압축 시도
2. **SAGA 주입물 피드백 루프** — SAGA가 user에 주입한 동적 컨텍스트를 다음 턴에 HypaMemory가 요약 → SAGA 메타데이터가 기억에 들어감
3. **윈도우 감지 오탐** — HypaMemory가 메시지를 요약으로 바꾸면 WindowRecovery가 매번 "윈도우 이동됨"으로 오탐
4. **에피소드 이중 주입** — HypaMemory 유사 기억 + SAGA 에피소드가 비슷한 내용으로 중복

→ **HypaMemory OFF + SAGA ON** (대체)이 가장 깔끔한 구성.

---

## 벤치마크

동일 모델, 동일 조건 A/B 비교. HypaMemory 공식 벤치마크는 없음.

### LOCOMO (ACL 2024) — 304 QA

| | Baseline (최근 60턴 truncation) | SAGA |
|--|--------------------------------|------|
| Overall | 2.02/5 | **3.12/5** (+54%) |
| multi-hop | 3.08/5 | **4.71/5** (+53%) |

### LongMemEval (ICLR 2025) — 499 QA

| | Baseline (마지막 10세션 truncation) | SAGA |
|--|-----------------------------------|------|
| Overall | 21.2% | **63.5%** (+42.3%p) |

### 프롬프트 캐싱 (50턴 E2E)

| 모드 | 히트율 | 비용 효과 |
|------|:------:|:---------:|
| SAGA 3-BP + 1h TTL | **85.7%** | **43.5% 절감** |
| RisuAI 자동 캐싱 | 12.1% | -11.4% (손해) |
| 캐시 없음 | 0% | 기준선 |

---

## 참고 소스

- [`src/ts/process/memory/hypav3.ts`](https://github.com/kwaroran/RisuAI/blob/main/src/ts/process/memory/hypav3.ts) — HypaMemory V3 메인 엔진 (2,067줄)
- [`src/ts/process/memory/hypamemory.ts`](https://github.com/kwaroran/RisuAI/blob/main/src/ts/process/memory/hypamemory.ts) — 임베딩 기반 클래스, similarity()
- [`src/ts/process/request/anthropic.ts`](https://github.com/kwaroran/RisuAI/blob/main/src/ts/process/request/anthropic.ts) — Anthropic 요청 빌더, cache_control 적용
- [`src/ts/process/index.svelte.ts`](https://github.com/kwaroran/RisuAI/blob/main/src/ts/process/index.svelte.ts) — automaticCachePoint 로직
- [SupaMemory Wiki](https://github.com/kwaroran/RisuAI/wiki/SupaMemory)
- [HypaMemory V3 Issue #1051](https://github.com/kwaroran/RisuAI/issues/1051) — 설정값 + SystemMessageOrderError
