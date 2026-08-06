# 프리픽스 캐시 아키텍처 — 로어북 delta 문제

> 작성 2026-08-05. 대상: `dreaming/` 프록시의 캐시 계층.
> 관련: [SPEC.md](SPEC.md) §3.1(3-BP), §5(SystemStabilizer 승계)
>
> 이 문서는 **실측 기록 + 설계 미결 사항**이다. 결론이 아직 안 난 부분은
> "열린 결정"으로 표시했다. 실측값과 추정값을 섞지 않으려 애썼으니
> 인용할 때 구분해서 볼 것.

---

## 0. 한 줄 요약

RisuAI의 keyed 로어북은 매 턴 활성 집합이 바뀌고, 그 블록이 프롬프트
**프리픽스 안**에 앉는다. 프리픽스가 1바이트라도 바뀌면 프리픽스 캐시는
**그 뒤 전부**를 무효화한다.

**진짜 RisuAI 14턴 캡처 실측 (§9)**: 13턴 중 9턴에서 프리픽스 파손,
**9/9 전부 로어북 구획**. 파손 턴 프리픽스 생존율 10~33%.
keyed를 프리픽스 밖으로 빼면 히트율 **38.4% → 85.2%**, 비용 **3~4배 절감**.

> §9가 실측이다. §4·§5는 자체 시뮬(`benchmarks/cardsim`) 기반이고,
> 그 시뮬이 실물과 상당히 다르다는 게 §9에서 드러났다 — §9를 우선할 것.

---

## 1. 왜 이 문서가 필요한가

Dreaming은 RisuAI 앞에 서는 리버스 프록시다. 두 가지를 한다.

1. 기억을 주입한다 (지식 계층)
2. 캐시 브레이크포인트를 찍는다 (캐시 계층)

2번이 성립하려면 **프리픽스가 턴 사이에 바이트 단위로 안정**해야 한다.
그런데 우리가 프롬프트를 만드는 게 아니다 — RisuAI가 만든 걸 받는다.
RisuAI가 매 턴 프리픽스를 흔들면 우리 캐시 설계는 무의미하다.

그래서 **실제 카드로 RisuAI 조립을 재현**해서 얼마나 흔들리는지 쟀다.

---

## 2. 재현 대상 — RisuAI 프롬프트 조립

읽기 전용 소스: `/Users/yanghyeon-u/Desktop/Risuai` (심링크 `external/risuai`)

### 2.1 기본 promptTemplate 순서

`process/prompt.ts:427`

```
plain(main) → description → persona → lorebook → chat → authornote → plain(globalNote)
```

핵심: **globalNote가 히스토리 뒤에 온다.** charx의 `post_history_instructions`(PHI)가
globalNote로 들어간다 (`ts/characterCards.ts:992`). 즉 메시지 배열의 **꼬리에
system 역할이 하나 더 붙는다.**

### 2.2 로어북 활성화

`process/lorebook.svelte.ts`

| 단계 | 동작 | 위치 |
|---|---|---|
| 스캔 | 최근 `scan_depth` 메시지만 본다 (기본 5) | `database.svelte.ts:77` |
| 매칭 | lowercase + **공백 전부 제거** + substring | `:206` |
| 컷 | `insertion_order` 내림차순 정렬 → `token_budget` 초과분 버림 | `:608` |
| 배치 | order 정렬 → reverse | — |

기본 `loreBookToken = 800` (`database.svelte.ts:80`) 이지만 **카드가 덮어쓴다.**
실측 두 카드 모두 `token_budget: 100000` — 사실상 컷 없음.

결과 블록은 `normalActives → unformated.lorebook` 으로 들어가
(`process/index.svelte.ts:529-539`) description 바로 뒤, 즉 **프리픽스 index 1**.

### 2.3 Anthropic 변환 규칙

`process/request/anthropic.ts:226-238`

- **선두 연속 system** → 병합해서 `systemPrompt`로
- **그 외 system** → `role:'user', content: "System: " + content` 으로 강등

즉 꼬리 PHI는 애초에 system 블록이 아니다. BP1 후보가 될 수 없다.

### 2.4 RisuAI 자체 캐시 마킹

`process/index.svelte.ts:1413` — `automaticCachePoint`는 **마지막 user 3개**에만
찍는다. 로어북 블록은 대상이 아니다. 그리고 RisuAI는 리버스 프록시로 보낼 때
`cachePoint`를 전송 직전에 제거한다(`requests.ts:141`) — **마킹 주체는 우리다.**

### 2.5 슬라이딩 윈도우

`process/index.svelte.ts:1143`

```js
while (currentTokens > maxContextTokens) { chats.splice(0, 1) }
```

히스토리 **앞에서만** 자른다. 카드/로어북은 별도 버킷이라 안 잘린다.
→ 로어북이 크면 히스토리가 먼저 증발한다.

---

## 3. 발견한 문제 3개

### 3.1 Bug A — BP1이 꼬리 system에 찍혔다 (수정 완료)

`dreaming/marking.py`가 **전체에서 마지막 system**을 BP1으로 잡았다.
RisuAI 레이아웃에서 그건 꼬리 PHI다. 그 결과:

```
[system desc][system lore][…chat…][user + 주입지식][system PHI ← BP1]
                                        ^^^^^^^^^^
                                   매 턴 바뀌는 게 캐시 span 안
```

주입한 지식이 캐시 구간 **안**으로 들어간다. 스펙 §3.1 위반 —
"지식 계층은 캐시 밖" 원칙이 깨진다.

**수정**: BP1 후보를 **선두 연속 system 구간**으로 한정.

```python
last_system = None
for i, m in enumerate(out):
    if m.get("role") != "system":
        break
    last_system = i
```

회귀 테스트 2개 추가 (`tests/test_dreaming_marking.py`):
`test_bp1_stays_in_leading_system_run`, `test_no_bp_after_injected_last_user`.
수정 전 실패 → 수정 후 통과. 전체 457 passed.

**단, 비용 효과는 없었다.** 아래 §4.2 A/B 참조. 처음에 "턴당 55K 재작성"이라
추정했는데 틀렸다 — BP3(마지막 assistant)가 이미 재사용 가능한 프리픽스를
제공하고 있었고, Anthropic이 증분으로 쓰기 때문에 비-churn 턴은 ~370토큰만
쓴다. **성능 수정이 아니라 스펙 정합성 수정으로 재분류.**

### 3.2 로어북 delta — 본체 문제 (미해결)

keyed 엔트리가 매 턴 켜지고 꺼진다. 위치는 프리픽스 index 1.
→ 그 뒤 전부(description 이후 모든 것, 즉 히스토리 전체) 캐시 무효.

실측(§4.1): churn 턴 캐시 히트 **16.9%**, 쓰기 **36,062토큰**.
비-churn 턴 히트 **97.8%**, 쓰기 **377토큰**. **96배 차이.**

델타 자체는 작다 — 중앙값 607토큰, 최대 1,378토큰. **원인은 작고 피해는 크다.**

### 3.3 벤치 하네스 오염 (수정 완료)

첫 realm 실행에서 30턴을 돌렸는데 원장/raw는 12행뿐이었고 index 11~19에 구멍,
전부 `provisional`, index 20은 mtime이 다른 stale `superseded`.

원인: 같은 세션 ID로 두 번 돌려서 두 실행이 원장을 공유.

**수정**: `benchmarks/cardsim/bench.py`에 가드 추가.

```python
def _prepare_data_dir(session: str, reset: bool) -> None:
    """세션 디렉터리 오염 방지 — 이전 실행 잔재가 원장에 구멍을 낸다."""
    d = DATA_ROOT / session
    if not d.exists():
        return
    if not reset:
        raise SystemExit(f"{d} 이미 있음. 같은 세션에 두 번 돌리면 원장이 섞인다.…")
    shutil.rmtree(d)
```

파괴적 삭제는 명시적 `--reset` 에서만.

### 3.4 (별건) Dreamer 크래시

`ValueError: invalid storage path segment: '개선문은 하늘을 가로지르는…'`
— LLM이 `target_fact_id`에 주장 문장을 넣어 지어냄. 수정 커밋 `203cc3b`이
`origin/main`에 있었으나 워크트리에 없었음. `git merge --ff-only`로 해결.

---

## 4. 실측

### 4.1 하네스

`benchmarks/cardsim/` (신규, 아직 미커밋)

| 파일 | 역할 |
|---|---|
| `lorebook.py` | RisuAI 로어북 활성화 부분 포팅 + charx 로더 + `build_messages` |
| `bench.py` | 30턴 대본, 유저 발화는 Gemini Flash가 세계관 맞춰 생성, 리롤 1회, 유휴 2회 |

```bash
python -m benchmarks.cardsim.bench "<charx>" <세션ID> [--no-keyed-lore] [--reset]
```

**포팅 생략분** (docstring에 명시): recursive_scanning(두 카드 모두 False),
`@@` 데코레이터, selective/secondary_keys, use_regex, 폴더 모드,
keep/dontActivateAfterMatch.

카드: `THE AMOROUS REALM Ⅱ.charx`. 모델 `anthropic/claude-haiku-4.5`,
`max_tokens=300`, 30턴, 유휴 12초×2 (꿈 사이클 유발), 턴20 리롤.

### 4.2 A/B — BP1 수정 전후

| | before (Bug A) | after (수정) |
|---|---|---|
| 총 비용 | **$0.917** | **$0.914** |
| 평균 히트 (t2+) | 53.7% | 52.0% |
| churn 턴 | 17/30 | 22/30 |
| churn 평균 히트 | 16.9% | 37.6% |
| churn 평균 쓰기 | 36,062t | 27,722t |
| churn 소계 | $0.821 | $0.849 |
| 비-churn 평균 히트 | 97.8% | 97.5% |
| 비-churn 평균 쓰기 | 377t | 373t |
| 비-churn 소계 | $0.096 (n=13) | $0.065 (n=8) |

**판정: 비용 차이 없음** ($0.003). churn 턴 수가 17 vs 22로 다른 건
유저 발화를 LLM이 생성해서 대화 내용이 달라졌기 때문 — 통제된 비교가 아니다.
같은 대본이 아니라 같은 *비트 지시*를 줬을 뿐.

**측정 주의**: after 실행의 T01이 `hit=100% write=0`이다. before 실행이 5분 전에
같은 프리픽스를 캐시에 올려놔서 그걸 주워 먹었다. 그래서 t2+ 평균만 봐야 한다.
(SUMMARY의 히트율은 t2+ 기준. 전 턴 포함하면 before 52.0% / after 53.6%로 뒤집힌다.)

### 4.3 비용 분해 (before 실행, 30턴)

| 항목 | 토큰 | 단가 | 비용 |
|---|---|---|---|
| 캐시 읽기 | 737,578 | $0.10/M | $0.074 |
| **캐시 쓰기** | **617,955** | **$1.25/M** | **$0.772** |
| 일반 입력 | 18,989 | $1.00/M | $0.019 |
| 출력 | 9,000 | $5.00/M | $0.045 |
| 합계(추정) | | | **$0.910** |
| 실제 청구 | | | $0.917 |

**캐시 쓰기가 84%.** 프롬프트 토큰 총 1,374,522 — 30턴에 137만 토큰을 밀어넣은
셈인데, 그중 45%가 재작성이다.

### 4.4 카드 통계

| | AMOROUS REALM Ⅱ | MORTAL REALM |
|---|---|---|
| description | 11,235자 | 5,385자 |
| PHI | 14,395자 | 12,334자 |
| constant 엔트리 | 33개 / **39,097t** | 27개 / 21,936t |
| keyed 엔트리 | 98개 / 93,516t | 123개 / **134,192t** |
| scan_depth | 5 | 5 |
| token_budget | 100000 | 100000 |

MORTAL이 keyed 총량 1.4배. **아직 벤치 미실행.**

### 4.5 A/U — 동시 활성 vs 누적 합집합 (⚠️ 정정)

**이전 세션에서 "A/U ≈ 1/7"이라고 말했는데 틀렸다.** 재계산 결과:

| 실행 | keyed 합집합 U | 최대 동시 A | **A/U** | 마지막 신규 |
|---|---|---|---|---|
| realm-before | 9개 / 7,323t | 4,796t | **0.65** | T27 |
| realm-after | 7개 / 4,885t | 3,667t | **0.75** | T13 |

카드 전체 keyed는 98개 / 93,516t인데 30턴 동안 실제로 켜진 건 **7~9개(5~8%)** 뿐.
그리고 **동시에 켜져 있는 양이 누적 합집합의 2/3**다 — 즉 켜진 건 잘 안 꺼진다.

이 값이 §5의 경제성 판단을 뒤집는다.

### 4.6 대조군

`위지소연` 카드 (keyed 엔트리 0개) 30턴 = **$0.124, 히트 92.9%**.

⚠️ 이건 다른(작은) 카드다. **제대로 된 대조군은 `realm --no-keyed-lore`인데
아직 안 돌렸다.** 지금 상태로는 "$0.917 중 얼마가 delta 탓인가"를 직접 못 잘라
말한다 — §4.2의 churn/비-churn 분리가 간접 근거일 뿐.

---

## 5. 설계 선택지

기호:
- **A** = 동시 활성 keyed 로어 토큰 (실측 3,667~4,796t)
- **U** = 누적 합집합 (실측 4,885~7,323t)
- **D** = 프리픽스 변경 시 재작성되는 다운스트림 크기 (실측 38,000~51,000t)

### 1안 — keyed 로어를 프리픽스에서 빼서 매 턴 마지막 user에 prepend

```
[system: desc + constant lore ← BP1 고정][…chat…][BP3][user: keyed lore + 발화]
```

- 프리픽스 완전 안정. churn 이벤트 **0**.
- 대가: A를 매 턴 일반 입력으로 재전송.
- 비용 ≈ `30 × A × $1.00/M` ≈ 30 × 4,000 × 1e-6 = **$0.12** (추정)

### 4안 — keyed 로어를 프리픽스에 append-only 누적

```
[system: desc + constant + 지금까지 켜진 적 있는 keyed 전부][…chat…]
```

- 한 번 켜진 건 안 끈다 → churn 이벤트가 **신규 등장 시점에만** 발생.
- 실측 신규 등장: 9회(before) / 7회(after). churn 17~22회 대비 2~3배 감소.
- T13(after) / T27(before) 이후로는 완전 안정.
- 대가: 각 append가 **다운스트림 전체 재작성**.
- 비용 ≈ `9 × D × $1.25/M` ≈ 9 × 45,000 × 1.25e-6 = **$0.51** (추정)

### ⚠️ 결론 정정

이전 세션에서 "실측상 4안이 압도적"이라고 했다. **A/U를 잘못 봐서 나온 결론이다.
정정하면 1안이 4~5배 싸다.**

이유는 로어 크기가 아니다. **어떤 프리픽스 변경이든 다운스트림 전체(D≈45,000t)를
재작성시키는데, 델타(600t)는 D의 1.3%밖에 안 된다.** 즉 append를 아무리 아껴도
한 번 할 때마다 45,000t를 문다. 반면 1안은 A(4,000t)를 매 턴 무는 대신
프리픽스를 절대 안 건드린다.

손익분기: 1안 턴당 `A × $1/M` = $0.004. 4안 append 1회 `D × $1.25/M` = $0.056.
→ **append가 14턴에 1번보다 잦으면 1안 승.** 실측은 3.3턴에 1번.

장기적으로는 4안이 포화(T13~T27)되므로 유리해질 수 있으나, 그때쯤 히스토리가
커져 D도 같이 커진다. 슬라이딩 윈도우로 D가 상한에 묶이면 다시 계산해야 한다.

⚠️ 위 비용은 **실측 단가 × 실측 카운트로 계산한 추정치**다. 1안을 실제로 구현해
A/B를 돌리기 전에는 확정이 아니다.

### 변형안 (미분석)

누적 로어 블록을 히스토리 **뒤**, 마지막 user **앞**에 두고 BP를 찍는 안.
프리픽스(desc+chat)는 안정, 누적 블록만 꼬리. 다만 히스토리가 매 턴 자라므로
블록 위치가 계속 밀려 결국 매 턴 재작성 — 1안과 같은 양을 쓰기 단가($1.25)로
무는 셈이라 더 나쁠 가능성. **계산 안 해봄.**

---

## 6. 경쟁 모듈은 어떻게 하는가

조사 방법: `research/downloads/` 7개 파일 + `Modules/` 중 캐시/로어 관련 3개를
grep으로 지점 특정 후 해당 구간 직접 읽음. (`.risup`/`.risum`은 캐릭터 프리셋
데이터라 제외.)

| 모듈 | 주입 위치 | 캐시 마킹 | 압축 | 우리 안과 대응 |
|---|---|---|---|---|
| **WygLore Leaf 3.0.4** | 마지막 user **prepend** (fallback) / 템플릿 플레이스홀더 | **있음** — `markCachePointCounted`, 최대 4 BP | 종류별 글자수 하드캡 | **= 1안** |
| Flashback.Memory 0.9.20 | 마지막 user 직전 별도 system | **철회됨** | 문장 창 발췌 | 1안 형태, 캐시 보호 없음 |
| GRADIA | RisuAI 네이티브 `position`/`depth` 존중 | 없음 (0건) | 글자수 절삭 | 제3안 |
| RisuAI Agent 5.2.5 | `chat.localLore`, `alwaysActive` | 없음 (0건) | `retention_keep` | 4안에 형태만 근접 |
| hayaku locator | 로어북 미사용 | 없음 | 자체 packet | 비교 불가 |
| gemini-cache-keeper 2.10.4 | 로어북 안 건드림 | 마커 심고 다음 턴 경계 감지 | — | 관측 레이어 |
| LBI 0.35.0-pre31 | — | `u[-1] u[-2]` DSL 수동 지정 | — | 로어 무관 |

### 6.1 WygLore Leaf = 1안, 코드로 확정

`research/downloads/WygLoreLeaf-3.0.4.js`

| 근거 | 위치 |
|---|---|
| `DEFAULT_ENGINE_PARAMS`, `cacheMarkEnabled` | L17 |
| Anthropic `cache_control:{type:"ephemeral"}` | L711 |
| `useCaching` 설정 | L720 |
| `formatInjection` | L992 |
| `markCachePointCounted`, `injectIntoMessages` | L999 |
| **fallback: last-user prepend 직후 `markCachePointCounted(messages, lastUserIdx-1)`** | L1007 |
| `memory` 플레이스홀더 모드도 동일 패턴 | L1013 |
| `ANTHROPIC_MAX_BREAKPOINTS = 4` | — |
| 종류별 캡 (desc 4000, alwaysActive 4000, selective 600, handSummary 2000, persona 2000자) | L17 |

`v2` 빌드도 해당 함수 바이트 동일 (diff 확인). `pocket`/`legacy`는 `coldStartBlocking`
1개만 다른 동일 빌드.

**활성화 모델**은 우리와 다르다 — 스프레딩 액티베이션(`decayPerTurn .02`,
`reinforceDelta .05`, `propagationDecay .6`, `maxHops 4`)으로 연속 점수를 매기고
상위 N개(`injectTopN` 기본 8)만 주입. 완전 sticky도 매턴 재계산도 아닌 중간.

즉 **1안은 실전에서 도는 검증된 패턴이다. 추측 아님.**

또 하나: 종류별 글자수 하드캡은 우리 `HOT_ZONE_CHAR_BUDGET = 6000`
(`dreaming/assembly.py:15`)과 같은 발상이다. 다만 WygLore는 종류별로 쪼갰고
우리는 통짜다. 우리 6,000자에는 스펙상 근거가 없다 (SPEC.md §3.1 line 144
"~2K tokens 상한"에도 출처 없음) — **휴리스틱이 맞다.** 참고로 realm 실측
로어 delta는 21턴 중 **0번**만 6,000자를 넘었다.

### 6.2 4안은 선례 없음

"로어북 활성 집합을 프리픽스에 무한 append-only 누적"하는 모듈은 **없다.**

가장 가까운 건 RisuAI Agent plugin의 `write_mode:"append"` (L9945-9984) —
로어 엔트리 하나에 `### Turn N` 블록을 이어붙인다. 하지만
(i) WI 활성 집합 관리가 아니라 자체 서사 로그(`ra_arc_memory` 등) 전용,
(ii) `retention_after`/`retention_keep`으로 결국 트리밍되는 유한 누적,
(iii) 캐시 고려 0 — `alwaysActive` 엔트리를 매 턴 재작성하므로 프리픽스를
오히려 스스로 깬다.

### 6.3 Flashback의 철회 이력 — 읽을 만한 신호

`Flashback.Memory.js` v0.8.8에 "캐시 안전화 — 응답 모델 프롬프트 prefix 캐시
보호" 목적의 정적/동적 분리가 있었다 (L543-548, 리비전 번호 + `stableHash`).

현재 0.9.20에서 `buildFlashbackStaticEvidenceContract`가
`body:'', disabled:true, reason:'diegetic_memory_only'` 를 반환한다
(L11640-11664). 근거였던 `findStableSystemPrefixEnd`(L12871-12891)는
**호출부 없는 죽은 코드.**

캐시 보호를 넣었다가 뺐다. **이유는 코드에 안 적혀 있다.** `diegetic_memory_only`
라는 사유만 보면 "정적 지시문 자체를 안 내보내기로 했다"는 서사적 결정이지
캐시가 안 통해서는 아닌 걸로 읽힌다. 단정 불가.

---

## 7. 열린 결정

### D-1. 1안 vs 4안 — **1안으로 확정 (§9.4 실측)**

진짜 RisuAI 14턴에서 1안이 히트율 38.4% → 85.2%, 비용 3~4배 절감.
keyed 재전송 비용(4,092t/턴)을 포함한 수치다. WygLore Leaf가 쓰는
검증된 패턴이기도 하다(§6.1). 4안은 선례도 없고 실측 근거도 없다.

남은 확인: 1안을 실제 구현해 캡처 재생으로 A/B (§9.7).

### D-2. MORTAL 벤치를 돌릴까 (~$1)

keyed 총량 1.4배 카드에서 A/U가 어떻게 나오는지. 1안으로 간다면 **A값 상한**을
잡는 데 쓰인다 (hot zone 예산의 실증 근거). 안 돌리면 예산은 감이다.

### D-3. hot zone 예산 재설계

현 `HOT_ZONE_CHAR_BUDGET = 6000`은 근거 없는 통짜값.
WygLore처럼 **종류별로 쪼갤지** 결정 필요. realm 실측으로는 6,000자를
한 번도 안 넘었으므로 당장 아프진 않다.

---

## 8. 이후 작업 (기승인 순서)

1. ~~로어북 delta 재현~~ ✅
2. ~~dreaming 캐시 깨지는 것 실측~~ ✅
3. **SystemStabilizer 상당 기능 포팅** ← 여기 (= 1안 구현)
   - SPEC §5 line 233 "동적 로어북 델타도 지식 계층으로" — `dreaming/`에 미구현
   - 원본 사상: `saga/system_stabilizer.py`
4. 슬라이딩 윈도우 얹어서 합산 검증

---

## 9. 실측 — 진짜 RisuAI 캡처 (2026-08-05, 최우선 근거)

§4·§5는 자체 시뮬 기반 추정이다. 여기는 **실제 RisuAI가 와이어로 보낸 요청**이다.

### 9.1 방법

RisuAI 소스 복사본을 `pnpm dev`로 띄우고(원본 `/Users/yanghyeon-u/Desktop/Risuai`
무수정), 캡처 프록시를 물려 요청 원문을 통째로 저장했다.

```
RisuAI(:5174) → capture(:8788) → dreaming(:8787) → OpenRouter
```

- 카드 `THE AMOROUS REALM Ⅱ.charx`, 페르소나 `반S2`, 모델 `deepseek/deepseek-v4-pro`
- 프리셋은 커뮤니티 프리셋(`<SYSTEM_RULE>`/`<CONFIGURATION>`/`<RESPONSE_INSTRUCTION>` 구조)
- 14턴. 일부러 로어북 키워드를 심은 턴과 안 심은 턴을 섞음
- 캡처: `scratchpad/captures/req-001~014.json`

**웹판 제약 2개** (재현 시 필요):
- `knownHostes = ["localhost","127.0.0.1","0.0.0.0"]`는 하드 차단
  (`globalApi.svelte.ts:598,740`) → `lvh.me` 같은 루프백 DNS 별칭을 써야 한다
- "직접 요청 보내기"(`usePlainFetch`, `advancedSettingsData.ts:243`)를 켜야
  브라우저 fetch로 나간다. 안 켜면 RisuAI 원격 프록시를 경유해 로컬에 못 닿는다
- `maxContext` 기본값 4000 (`database.svelte.ts:56`) — 이 카드 고정분이 54K라 필히 상향

### 9.2 실제 와이어 레이아웃

```
msg[0] system  119,894자   ← 전부 하나로 병합
   offset       0   프리셋 head
   offset   2,084   description
   offset  13,309   로어북 (constant 31 + keyed 활성분)   ← 파손 지점
   offset 103,503   PHI (post_history_instructions)
   offset ~117,800  프리셋 tail
msg[1] assistant  4,501자   ← alternate_greetings[1] (first_mes 아님)
msg[2] user          12자
msg[3] system     1,159자   ← @@depth 0 로어 엔트리, 14턴 내내 고정
```

턴1 총 125,566자 ≈ 50,226토큰.

**CBS는 전부 해석돼서 온다** — `{{getglobalvar}}` 0, `{{?}}` 0, `{{user}}` 0,
`@@depth` 0. (`{{Annotation::…}}` 5건만 남는데 그건 LLM 출력 포맷 지시라 의도된 것.)

### 9.3 프리픽스 파손 실측

| 턴 | 파손 오프셋 | 구획 | 생존율 | 바뀐 로어 엔트리 |
|---|---|---|---|---|
| 2 | 13,346 | 로어북 | 11.1% | Ordo Sacra → Hrafn |
| 3 | 13,346 | 로어북 | 10.8% | Hrafn → Sir Percival |
| 4 | — | — | 100% | (동일) |
| 5 | 42,053 | 로어북 | 32.0% | Via Magna → Forum Primaris |
| 6 | — | — | 100% | (동일) |
| 7 | 42,053 | 로어북 | 32.6% | Forum Primaris → Colosseum |
| 8 | — | — | 100% | (동일) |
| 9 | — | — | 100% | (동일) |
| 10 | 43,356 | 로어북 | 32.9% | Arena Maximus → Portus Caliga |
| 11 | 13,346 | 로어북 | 10.4% | Sir Percival → **Hrafn** |
| 12 | 13,346 | 로어북 | 10.3% | Hrafn → **Sir Percival** |
| 13 | 20,458 | 로어북 | 15.4% | Ordo Sacra → Nectar |
| 14 | 16,791 | 로어북 | 12.1% | Tiberius Aquilius → Nectar |

**파손 9/9 전부 로어북 구획.** 프리셋 head·description·PHI·꼬리 system은
14턴 내내 1바이트도 안 움직였다.

**T11 ↔ T12**: Hrafn과 Sir Percival이 로어북 선두에서 교대로 깜빡인다.
정보량 변화 0인데 매번 12만 자 중 89%가 재작성된다.

### 9.4 비용 — 현재 vs 1안

constant 31개(88,939자)는 전 턴 동일. keyed는 평균 **10,229자 = 4,092토큰/턴**.
1안 계산에는 keyed를 마지막 user로 재전송하는 비용을 포함했다.

| | 현재 | 1안 (keyed만 프리픽스 밖) |
|---|---|---|
| 캐시 히트율 | 38.4% | **85.2%** |
| 캐시 읽기 | 319,388t | 707,621t |
| 미스 | 511,850t | 65,894t |
| keyed 재전송 | 0 | 57,279t |
| **deepseek-v4-pro** | $0.224 | **$0.056** (4.0배↓) |
| **haiku-4.5** | $0.672 | **$0.210** (3.2배↓) |

1안 적용 시 `msg[0]`이 전 턴 **119,825자로 정확히 동일**함을 확인했다.

> 계산 시행착오 2건 (기록): ① 로어북을 통째로 제거해 constant까지 날림 → $0.014로
> 과장. ② keyed만 뺐으나 **구분자 개행이 잔류**해 프리픽스가 여전히 흔들림 →
> 개선 없음으로 오판. 위 표는 `msg[0]` 완전 동일을 검증한 뒤의 값이다.

### 9.5 §3.1(Bug A) 정정

§3.1에 "RisuAI 기본 템플릿은 PHI를 히스토리 뒤에 두므로 마지막 system을 잡으면
**꼬리 PHI**에 찍힌다"고 썼다. **PHI는 꼬리에 없다** — 병합된 `msg[0]` 안
offset 103,503이다. 기본 템플릿 기준 추론이었고 실제 프리셋에서는 다르다.

**단, 수정 자체는 맞았고 실측이 더 강하게 뒷받침한다.** 꼬리 system이 실제로
존재하기 때문이다 — `msg[3]`, `@@depth 0` 로어 엔트리(1,159자). 옛 코드였으면
BP1이 거기 찍혀 `msg[2]`(유저 발화 + 주입할 지식)가 캐시 span 안으로 들어갔다.
원인이 PHI가 아니라 depth-0 로어였을 뿐 병은 동일하다.

### 9.6 시뮬(`benchmarks/cardsim`) 격차 — 확정

| | |
|---|---|
| ✓ 로어북 reverse 순서 | 일치 |
| ✓ 로어북이 description 뒤 | 일치 |
| ✗ 프리셋 main prompt | 없음 (실제 head 2,084자 + tail ~2,046자) |
| ✗ 단일 system 병합 | 시뮬은 2개로 분리 |
| ✗ greeting 선택 | `first_mes` 465자 vs 실제 `alternate_greetings[1]` 4,501자 |
| ✗ PHI 위치 | 시뮬 꼬리 vs 실제 병합 안 |
| ✗ `@@depth 0` | 시뮬 프리픽스 vs 실제 꼬리 system |
| ✗ CBS 해석 | 원문 그대로 남김 (실물은 전부 해석됨) |

**재구현으로 메우기엔 너무 많다.** RisuAI의 CBS 엔진·데코레이터·프리셋 템플릿을
파이썬으로 다시 짜는 건 토끼굴이다.

### 9.7 전환 — 캡처 & 재생

```
1회 (비쌈)  진짜 RisuAI N턴 → 요청 원문 N개 캡처
이후 (공짜)  그 N개를 프록시에 재전송하며 A/B
```

캡처는 CBS 다 풀리고 데코레이터 다 적용된 실물이다. 재생엔 브라우저도 LLM 변덕도
필요 없고, 매번 같은 바이트가 들어가니 **A/B가 진짜 통제된다** — §4.2에서 churn이
17 vs 22로 갈려 비교가 무의미해졌던 문제가 원천 해결된다.

`benchmarks/cardsim`의 역할은 **RisuAI 재현 → 캡처 재생·변형 하네스**로 축소된다.

---

## 부록 A. 미커밋 자산

| 경로 | 상태 |
|---|---|
| `benchmarks/cardsim/__init__.py` | 신규 (untracked) |
| `benchmarks/cardsim/lorebook.py` | 신규 (untracked) |
| `benchmarks/cardsim/bench.py` | 신규 (untracked) |
| `dreaming/marking.py` | 수정 (§3.1) |
| `tests/test_dreaming_marking.py` | 수정 (회귀 테스트 2개) |

실행 데이터: `dreaming_data/realm-before`, `dreaming_data/realm-after` (gitignore됨)

## 부록 B. RisuAI 소스 인용 목록

읽기 전용. `external/risuai` → `/Users/yanghyeon-u/Desktop/Risuai`. **수정 금지.**

| 파일:라인 | 내용 |
|---|---|
| `process/prompt.ts:427` | 기본 promptTemplate 순서 |
| `process/index.svelte.ts:529-539` | `normalActives → unformated.lorebook` |
| `process/index.svelte.ts:1143` | 슬라이딩 윈도우 `chats.splice(0,1)` |
| `process/index.svelte.ts:1413` | `automaticCachePoint` — 마지막 user 3개만 |
| `process/index.svelte.ts:1507` | 하드 트림 |
| `process/lorebook.svelte.ts:206` | 매칭 (lowercase + 공백 제거 + substring) |
| `process/lorebook.svelte.ts:608` | `token_budget` 컷 |
| `process/request/anthropic.ts:226-238` | 선두 system 병합 / 그 외 user 강등 |
| `process/request/requests.ts:141` | 전송 직전 `cachePoint` 제거 |
| `ts/characterCards.ts:992` | charx PHI → globalNote |
| `ts/storage/database.svelte.ts:77,80` | `loreBookDepth=5`, `loreBookToken=800` |
