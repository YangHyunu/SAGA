# 핸드오프 — Plan 4 구현 세션에게

> 작성 2026-08-05, 워크트리 `annyeong-ba42ff` (시뮬 하네스 담당).
> 수신: `annyeong-3b2696` (Plan 4 문서 작성 후 정지, 구현 0/44).
>
> **이 문서 하나만 읽어도 상황이 서도록 자립적으로 썼다.**
> 상세 실측/설계는 `annyeong-ba42ff` 워크트리의
> `docs/dreaming/PREFIX-CACHE-ARCHITECTURE.md` 참조.
> 절대경로:
> `/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-ba42ff/docs/dreaming/PREFIX-CACHE-ARCHITECTURE.md`

---

## 0. TL;DR — Plan 4를 지금 그대로 구현하면 안 되는 이유

Plan 4의 "실카드 벤치 실측 결함" 표(A~F)를 감사했다.
**근거 데이터 두 세션 중 하나가 오염된 실행이다.**

| 세션 | raw | ledger | commits | episodes | 판정 |
|---|---|---|---|---|---|
| `dreaming_data/card-soyeon-30` | 30 | 30 | 8 | 10 | **정상** |
| `dreaming_data/card-realm-30` | **12**/30 | **12**/30 | **0** | 4 | **오염** |

오염 원인: 같은 세션 ID로 벤치를 **두 번** 돌렸다. 데이터 디렉터리를 안 지워서
두 실행의 원장이 섞였고, Dreamer도 같은 디렉터리에 두 번 돌았다.
(ledger index 11~19 구멍, 전부 `provisional`, index 20은 mtime 다른 stale
`superseded`.)

타임스탬프: `card-realm-30` 13:26 / `card-soyeon-30` 13:31 → Plan 4 작성 13:49.

### 결함별 근거 판정

| # | 결함 | 근거 | 판정 | 조치 |
|---|---|---|---|---|
| **A** | 한글 수사 미검증 | soyeon(정상): `한결의 나이는 스물일곱이다` 등 2건 전부 `provisional` | **확실** | Task 1 그대로 진행 |
| **B** | evidence_turn 몰림 | ~~전 레코드 None~~ → **재검증 완료: 결함 아님** | **기각** | ✅ §1 (Task 2 재설계) |
| **C** | fact 대량 중복 re-ADD | ~~10/29 x2~~ → **재검증 완료: 대량은 오염, 잔여 ~1% 실존** | **축소 확정** | ✅ §2 (Task 3 축소) |
| **D** | 장면 묘사가 확정 사실 | realm: `날씨는 맑다`·`현재 온도는 -3°C이다`·`안내자의 제의는 흰색이다` 가 `confirmed` | **확실** (내용 문제라 오염 무관) | Task 3 그대로 |
| **E** | commit set-계산값 | **soyeon(정상)**: T11 `set 250.0` → `pending_contradiction`, T29 `set 300.0` → `applied` | **확실** (stale 300 재적용까지 재현) | Task 3 그대로 |
| **F** | 리뷰어 🟡 3건 | 코드 리뷰 | 데이터 무관 | Task 4 그대로 |

**전 결함 판정 완료 (2026-08-06 ba42ff 재검증). B 기각, C 축소, 나머지 그대로.**

---

## 1. 결함 B — ✅ 재검증 완료 (2026-08-06): **기각. 증거 체인 건강함**

### 1.1 "전 레코드 None"은 감사 아티팩트였다

`records.py`의 `Evidence` 모델에 `evidence_turn`/`turn_number` 필드가 **애초에 없다**:

```python
class Evidence(BaseModel):
    pair_hash: str
    offset: Optional[int] = None
```

`evidence_turn`은 LLM 추출 DTO(`dreamer.py:45 ExtractedFact`)의 **내부 필드**고,
저장 시점에 `raw_by_turn[evidence_turn] → pair_hash`로 변환된다 (`dreamer.py:175~182`).
감사 스크립트는 저장 레코드에서 존재하지 않는 키를 읽어 None을 센 것.

### 1.2 진짜 분포 측정 — pair_hash를 원장으로 역해석

| 세션 | facts | evidence 빈 것 | dangling | 턴 커버 | top5 |
|---|---|---|---|---|---|
| soyeon-30 | 87 | **0 (0%)** | 0 | **30/30** | (21,7) (20,7) (0,6) (23,6) (25,4) |
| realm-before | 248 | 11 (4%) | 0 | **30/30** | (20,14) (24,14) (26,13) (25,12) (15,11) |
| realm-after | 217 | 8 (4%) | 0 | **30/30** | (23,12) (11,11) (26,10) (13,10) (25,10) |

- **몰림 없음** — 30턴 전부에 고르게 분산, 특정 턴 쏠림 없음
- **dangling 0** — 저장된 pair_hash 전부 원장에서 실제 턴으로 해석됨 (표본 검증: `e52dc82b…` → 턴 18)
- 유일한 실제 결함: **evidence 빈 fact 4%** (realm 계열) — LLM이 무효한 evidence_turn을
  주면 `dreamer.py:182`가 조용히 `evidence=[]`로 저장한다. 몰림이 아니라 **누락**이고 소량이다

### 1.3 Task 2 함의

**"증거 턴 재탐색" 설계 폐기.** 고칠 게 있다면 단 하나 —
무효 evidence_turn일 때 조용히 빈 evidence로 넘어가는 대신 로그/재시도 (4% 회수).
그것도 우선순위 낮음. Task 2는 사실상 **제거 대상**.

---

## 2. 결함 C — 중복 re-ADD가 오염 아티팩트일 수 있다

realm(오염) 실행에서 관측된 중복:

```
facts=39  unique_claim=29  dup_claims=10
  x2 안내자의 제의는 흰색이다.
  x2 안내자의 표정은 예리하다.
  x2 포룸 트리움팔리스는 고요하다.
  x2 현재 온도는 -3°C이다.
  x2 안내자가 주인공을 기다리고 있었다.
  x2 안내자는 주인공의 이름을 알고 있다.
  x2 안내자는 주인공이 올 것을 예상했다.
  x2 날씨는 맑다.
```

전부 정확히 **x2**다. 그리고 그 세션은 **같은 대화를 Dreamer가 두 번 돌았다.**
→ 중복의 기계적 원인이 이미 존재한다. NOOP 미준수가 아니라 실행이 두 번이었을 뿐일 수 있다.

정상 세션 `card-soyeon-30`(facts 87)에서 같은 중복이 나오는지 아직 안 봤다.
**돈 안 들고 지금 확인 가능하다:**

```bash
python3 -c "
import json,glob,collections
c=collections.Counter()
for p in glob.glob('dreaming_data/card-soyeon-30/facts/*.json'):
    c[json.loads(open(p).read()).get('claim','?')]+=1
dup={k:v for k,v in c.items() if v>1}
print(f'facts={sum(c.values())} unique={len(c)} dup={len(dup)}')
for k,v in list(dup.items())[:10]: print(f'  x{v} {k[:60]}')
"
```

- **dup이 나오면** → C 확실, Task 3 dedup 그대로 진행.
- **dup이 0이면** → C는 오염 아티팩트.

### ✅ 재검증 결과 (2026-08-06): 둘 다 반쯤 맞았다

| 세션 | facts | dup claims | 비율 |
|---|---|---|---|
| card-realm-30 (**오염**) | 39 | **10** | 34% |
| card-soyeon-30 (정상) | 87 | 1 | 1.2% |
| realm-before (정상 30턴) | 248 | 2 | 0.8% |
| realm-after (정상 30턴) | 217 | 2 | 0.9% |

**대량 중복(34%)은 오염 아티팩트 확정** — 정상 세션은 ~1%.

**하지만 잔여 1%는 진짜 NOOP 위반이다.** 성격 검사 결과:
- realm-after `한결 주변에는 아무도 없다` ×2 — **둘 다 `confirmed`로 공존** (3분 17초 간격,
  다른 꿈 배치). 같은 claim 재-ADD를 NOOP으로 못 잡은 명백 사례
- realm-after `한결은 개선문 광장에서 은화 300닢…` ×2 — 동일하게 둘 다 confirmed
- soyeon `한결은 장부를 잃어버렸다` — superseded 후 동일 claim 재-ADD (이건 재확립일
  수도 있어 위반 단정 불가)

**그리고 exact-match가 못 세는 준중복 연쇄가 더 크다**: realm-before에
`마리우스 렉스의 얼굴이 창백해졌다` / `더욱 창백해졌다` / `식은 땀이 맺혔다` /
`더욱 복잡해졌다` / `의외라는 감정이…` 가 **전부 별개 confirmed fact**로 쌓여 있다.
이건 중복 문제가 아니라 **결함 D(장면 묘사가 fact화)의 증상**이다.

### Task 3 함의

- exact dedup: **대공사 불요.** 같은 claim 재-ADD를 막는 싼 멱등 가드 정도로 축소
- 진짜 우선순위는 **D** — 장면 묘사 fact화를 막으면 준중복 연쇄가 같이 사라진다

---

## 3. 그 사이에 발견/수정된 것들 (Plan 4와 무관하게 알아야 함)

### 3.1 `dreaming/marking.py` — BP1이 꼬리 system에 찍히던 버그 (수정 완료, ba42ff)

RisuAI 기본 promptTemplate은 `… → chat → authornote → plain(globalNote)` 순이다
(`prompt.ts:427`). charx의 `post_history_instructions`가 globalNote로 들어간다
(`characterCards.ts:992`). 즉 **메시지 배열 꼬리에 system이 하나 더 붙는다.**

기존 `mark_cache`는 **전체에서 마지막 system**을 BP1으로 잡았다 → 꼬리 PHI에 찍혔다.

```
[system desc][system lore][…chat…][user + 주입지식][system PHI ← BP1]
                                        ^^^^^^^^^^ 캐시 span 안!
```

스펙 §3.1 "지식 계층은 캐시 밖" 위반이다.

**수정** — BP1 후보를 선두 연속 system 구간으로 한정:

```python
last_system = None
for i, m in enumerate(out):
    if m.get("role") != "system":
        break
    last_system = i
```

Anthropic 변환도 선두 밖 system은 `role:'user'`로 강등하므로
(`anthropic.ts:226-238`) 꼬리 PHI는 애초에 system 블록이 아니다.

회귀 테스트 2개 추가 (`tests/test_dreaming_marking.py`):
`test_bp1_stays_in_leading_system_run`, `test_no_bp_after_injected_last_user`.
수정 전 실패 → 수정 후 통과, 전체 457 passed.

> **Plan 4 Task 7이 이 함수에 BP2를 추가한다. 이 수정 위에 얹어야 한다.**
> 아직 미커밋 — ba42ff 워크트리에 있다.

### 3.2 벤치 하네스 세션 오염 가드 (수정 완료, ba42ff)

§0의 오염을 만든 원인. 새 하네스 `benchmarks/cardsim/bench.py`에 가드 추가:

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

파괴적 삭제는 명시적 `--reset`에서만.

### 3.3 Dreamer 크래시 (수정본이 origin/main에 있었음)

```
ValueError: invalid storage path segment: '개선문은 하늘을 가로지르는 거대한 석조 건축물이다.'
```

LLM이 `target_fact_id`에 주장 문장을 지어 넣었다. 수정 커밋 `203cc3b`이
`origin/main`에 있었으나 워크트리가 뒤처져 있었다. **3b2696도 81271a8이니 포함됨** — 확인만.

---

## 4. Plan 4 Task 7(BP2·압축)에 직접 영향 있는 실측

30턴 실카드 벤치(`THE AMOROUS REALM Ⅱ.charx`, Haiku 4.5) 결과:

| 항목 | 값 |
|---|---|
| 총 비용 | $0.917 / 30턴 |
| **캐시 쓰기 비중** | **84%** (617,955t → $0.772) |
| churn 턴 (로어 활성 집합 변동) | 17/30 |
| churn 턴 평균 캐시 히트 | **16.9%**, 쓰기 36,062t |
| 비-churn 턴 평균 캐시 히트 | **97.8%**, 쓰기 377t |

**로어 delta 자체는 중앙값 607토큰인데, 그것 때문에 40,000토큰이 재작성된다.**

원인: RisuAI keyed 로어북은 매 턴 활성 집합이 바뀌고, 그 블록이
description 바로 뒤 = **프리픽스 index 1**에 앉는다
(`index.svelte.ts:529-539`). 프리픽스가 1바이트 바뀌면 그 뒤 전부 무효.

### Task 7에 주는 함의

압축 플랜을 아무리 결정론적으로 만들어도, **그 위(프리픽스 앞쪽)의 로어북이
매 턴 흔들리면 BP2는 무의미하다.** 청크가 바이트 안정이어도 앞이 바뀌면
캐시 매칭은 앞에서 끊긴다.

→ **BP2/압축의 이득을 측정하려면 로어북 delta를 먼저 잡거나, 최소한
`--no-keyed-lore` 대조군에서 재야 한다.** 안 그러면 압축 효과가 churn 노이즈에
묻힌다 (실측 노이즈 폭: 히트율 16.9% ↔ 97.8%).

로어북 delta 대책은 ba42ff에서 검토 중이다 (SPEC §5 "동적 로어북 델타도
지식 계층으로" — `dreaming/`에 미구현). 현재 유력안은 **keyed 로어를 프리픽스에서
빼서 마지막 user에 prepend**(WygLore Leaf가 쓰는 검증된 패턴). 아직 미구현.

---

## 5. 시뮬 하네스 현황 — Plan 4를 뭘로 검증할 건가

`annyeong-ba42ff` 워크트리, `benchmarks/cardsim/` (신규, 미커밋):

| 파일 | 역할 |
|---|---|
| `lorebook.py` | charx 로더 + RisuAI 로어북 활성화 포팅 + `build_messages` |
| `bench.py` | 30턴 대본, 유저 발화는 Gemini Flash 생성, 리롤 1회, 유휴 2회 |

```bash
python -m benchmarks.cardsim.bench "<charx경로>" <세션ID> [--no-keyed-lore] [--reset]
```

### Plan 4 검증 커버리지

| Plan 4 항목 | 하네스 | 비고 |
|---|---|---|
| A 한글 수사 | ✅ | 오라클 `300`/`세 개`/`스물일곱`/`자정`/`보름달` 심어져 있음 |
| E commit 델타 | ✅ | T04 소지금 300 → T12 −50 → T22 잔액 프로브 |
| Task 8 리롤 무효화 | ✅ | `REROLL_AT=19` |
| BP2·압축 캐시 효과 | ✅ | `prompt/cached/write/cost` 계측 |
| 꿈 트리거 | ✅ | `PAUSES={9:12, 19:12}` |
| B evidence_turn | ⚠️ | 저장 레코드에 값이 없어 관측 불가 (§1) |

### 아직 없는 것 (ba42ff가 만들 것)

1. **자동 판정** — `_summary`가 facts/commits/episodes를 덤프만 하고 pass/fail을 안 낸다.
2. **대본 고정(record/replay)** — 유저 발화를 매번 LLM이 새로 만든다. 그래서
   Plan 4 전/후 비교가 통제되지 않는다. (실제로 BP1 수정 A/B에서 churn 턴이
   17 vs 22로 갈려 비교가 무의미해졌다.)
3. **깨끗한 realm 대조군** — `realm --no-keyed-lore` 미실행.

**Plan 4 구현 전/후 비교를 하려면 2번이 필수다.** ba42ff에서 먼저 넣을 예정.

### ⚠️ 하네스의 재현 한계 (Plan 4 결론에 영향)

실카드 charx에서 로드하지만 **프리셋 영역은 비어 있다**:

| RisuAI 기본 템플릿 (`prompt.ts:427`) | 하네스 |
|---|---|
| `plain(main)` = 프리셋 메인 프롬프트 | ❌ **없음** |
| `description` | ✅ 실카드 |
| `persona` | ❌ 없음 |
| `lorebook` | ✅ 실제 활성화 규칙 포팅 |
| `chat` | ✅ |
| `authornote` | ❌ 없음 |
| `plain(globalNote)` ← charx PHI | ✅ 꼬리 system |

빠진 것들은 **전부 정적**이라 "delta가 프리픽스를 깬다"는 결론을 뒤집지 않는다.
다만 **절대 프롬프트 크기가 실전보다 작다** → 실전에서는 재작성 비용이 더 크다.

그 외 미재현: 슬라이딩 윈도우, CBS 매크로 대부분(`{{user}}`/`{{char}}`만 처리),
정확한 토크나이저(2.5자/토큰 근사), recursive_scanning, `@@` 데코레이터,
selective/secondary_keys.

---

## 6. 권장 순서

1. **[코드 5분, 무료]** §1 — `to_wire`/Dreamer 읽고 `evidence_turn`이 안 저장되는 건지
   안 채워지는 건지 확정 → **Task 2 설계 재검토**
2. **[스크립트 1분, 무료]** §2 — soyeon(정상 세션)에서 claim 중복 검사
   → **Task 3 dedup 존치/폐기 결정**
3. **[ba42ff]** record/replay 넣어서 전후 비교 가능하게
4. Task 1(수사) / Task 3의 scene 게이트·프롬프트 규칙 / Task 4(락·to_wire)는
   근거 확실하니 1·2와 병행해도 된다
5. Task 5~8(청크·BP2)은 §4 함의 때문에 **로어북 delta 대책과 순서를 맞춰야 한다.**
   최소한 `--no-keyed-lore` 조건에서 측정할 것.

---

## 부록. 감사에 쓴 명령

```bash
cd /Users/yanghyeon-u/Desktop/RISU_ENE

# 세션 무결성
for s in card-realm-30 card-soyeon-30; do echo "== $s"
  for d in raw ledger facts commits episodes actors; do
    echo "  $d=$(ls dreaming_data/$s/$d 2>/dev/null | wc -l)"; done; done

# claim 중복 + status 분포
python3 -c "
import json,glob,collections
s='card-soyeon-30'
c=collections.Counter(); st=collections.Counter()
for p in glob.glob(f'dreaming_data/{s}/facts/*.json'):
    d=json.loads(open(p).read()); c[d.get('claim','?')]+=1; st[d.get('status')]+=1
print(sum(c.values()), len(c), {k:v for k,v in c.items() if v>1}, dict(st))"

# evidence_turn 분포
python3 -c "
import json,glob,collections
for s in ('card-soyeon-30','card-realm-30'):
    print(s, collections.Counter(json.loads(open(p).read()).get('evidence_turn')
          for p in glob.glob(f'dreaming_data/{s}/facts/*.json')))"
```
