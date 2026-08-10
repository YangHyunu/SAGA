# DREAMING_FLAW — 핵심 두 층이 꺼져 있었다 (2026-08-10)

> 구현 세션 진입점. 먼저 [HANDOFF-2026-08-10.md](HANDOFF-2026-08-10.md)(코드 지도·실행법·제약) + [AGENTS.md](../AGENTS.md) + [CLAUDE.md](../CLAUDE.md)를 읽을 것.

## 0. 한 줄

2026-08-10 야간 본런에서 dreaming이 1/6로 최하위였는데, 원인은 메모리 설계가 아니라 **압축 층이 한 번도 작동한 적 없고, 지식 주입이 초반 20개 사실에 고정돼 있었기 때문**이다. 두 결함 모두 조용히 실패한다 — 에러도 경고도 남기지 않는다.

**어젯밤 결과는 변형 간 비교에 쓸 수 없다.** 고치고 다시 돌려야 출발선이 선다.

---

## 1. 증거

야간 본런 `night-drm-r0` (100턴 목표, T78에서 중단), 실측:

| 항목 | 값 |
|---|---|
| 원문 턴 | 77턴 / 161,832자 |
| 저장된 confirmed fact | **179개** |
| 실제 주입된 fact | **20개** (전부 런 시작 4분 이내 생성분) |
| 저장된 에피소드 | **76개** |
| 생성된 압축 청크 | **0개** |
| 주입 지식 블록 길이 | 885자 / 예산 6,000자 |
| 격리(quarantine) | 0건 — 원장 판정 자체는 정상 |

다른 세션도 동일: `night2-drm-r0` 청크 0개, `smoke2-drm-r0` 에피소드 19개인데 청크 0개.

**데이터 위치**: `dreaming_data/`는 gitignore라 이 워크트리(`.claude/worktrees/annyeong-3b2696`)에만 있다. 다른 워크트리에서 재현하려면 절대경로로 읽거나 복사할 것.

---

## 2. 결함 1 — 압축이 한 번도 작동하지 않았다 (심각)

### 증상

`build_compression()`이 항상 `None`을 반환한다. 에피소드가 76개 쌓여 있어도 청크가 0개다. 결과적으로 **Dreaming의 3층 구조 중 중간층(요약)이 통째로 없고**, 프롬프트는 원문이 무한 누적되기만 했다.

### 근본 원인 — 턴 번호 시작점 불일치

두 모듈이 서로 다른 시작점을 가정한다.

**생산 측** — [dreaming/identity.py:27,95-99](../dreaming/identity.py):
```python
_BASELINE_PAD = 1024
...
if not dense and pairs:
    # 트림된 대화 중간 합류 — 베이스라인 패드
    raw["position"] += _BASELINE_PAD
```
원장이 비어 있는데 요청에 pair가 있으면(= 대화 중간 합류 가능성) 턴 번호를 1024부터 시작한다. 앞이 잘린 대화에 붙었을 때 음수 오프셋을 피하려는 안전장치다. 실측 원장 index: `1025, 1026, ... 1101`.

**소비 측** — [dreaming/chunks.py:51-57](../dreaming/chunks.py):
```python
next_turn = 0                      # ← 턴이 0부터 시작한다고 가정
for e in eps:
    if e.start_turn < next_turn:
        continue
    if e.start_turn > next_turn or e.end_turn > cutoff:
        break                      # ← 1025 > 0 → 첫 에피소드에서 즉시 break
```
첫 에피소드의 `start_turn`이 1025인데 `next_turn`이 0이므로 **루프 첫 바퀴에서 break**, `chain`이 비고 `None` 반환.

### 재현

```bash
python3 -c "
from pathlib import Path
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.chunks import build_compression
s = MemoryStore(JsonDirStorage(Path('dreaming_data')), 'night-drm-r0')
eps = sorted([e for e in s.list_episodes() if e.start_turn is not None],
             key=lambda e: e.start_turn)
print('에피소드:', len(eps), '| 첫 구간:', (eps[0].start_turn, eps[0].end_turn))
print('압축 플랜:', build_compression(s, last_turn=1101))
"
```
기대 출력: `에피소드: 76 | 첫 구간: (1025, 1025)` / `압축 플랜: None`

### 왜 테스트가 못 잡았나

[tests/test_dreaming_chunks.py:41-56](../tests/test_dreaming_chunks.py)의 픽스처가 **전부 턴 0부터** 시작한다:
```python
def _ep(start, end, title="에피"):
    return Episode(..., start_turn=start, end_turn=end, ...)

store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
```
단위 테스트는 패드 없는 세계에서만 돌고, 프로덕션은 항상 패드된 세계에서 돈다. 테스트가 통과하면서도 프로덕션에서 100% 실패하는 구조다.

### 수정 방향

1. `next_turn`의 초기값을 0이 아니라 **에피소드 체인의 실제 시작 턴**으로:
   ```python
   next_turn = min(e.start_turn for e in eps)   # eps 비어있으면 조기 반환
   ```
   `covers_until_turn`도 같은 패드 공간에 남으므로 [apply_compression](../dreaming/chunks.py)의 `to_drop = covers_until_turn - window_start_turn` 계산은 그대로 맞는다 (`window_start_turn`은 `verdict.offset`이고 이것도 패드됨 — identity.py:99).
2. **조용한 실패 금지**: `None` 반환 시 이유를 `logger.info`로 남길 것 (에피소드 0개 / 갭 / 전부 꼬리 안 / 시작점 불일치). 이 결함이 몇 주간 안 보인 유일한 이유가 침묵이다.
3. 회귀 테스트를 **패드된 턴 번호(1025~)로** 추가. 기존 0-베이스 테스트는 남겨두고 병렬로.

### 수정 시뮬레이션 (실데이터로 검증함)

`next_turn = min(e.start_turn for e in eps)`만 바꿔 `night-drm-r0` 데이터로 돌린 결과 — **수정 방향이 실제로 동작한다**:

| 항목 | 값 |
|---|---|
| 체인에 들어간 에피소드 | 71개 (76개 중, 꼬리 6턴 제외) |
| `covers_until_turn` | 1096 |
| 생성된 청크 | **19개** (Tier2 챕터 13 + Tier1 에피소드 6) |
| 원문 → 청크 | 145,976자 → **8,922자 (93.9% 감소)** |

구현 후 이 숫자에 근접하는지가 1차 확인점이다.

### 검증

- 단위: 패드된 픽스처(1025~)로 `build_compression`이 플랜을 만든다
- 통합: 짧은 스모크(20턴 이상, `TAIL_KEEP=6`+`T1_MAX=8` 넘기게) 후 `dreaming_data/<session>/compression/plan` 파일 존재 + `messages` 길이 > 0
- 실측: 위 시뮬레이션 표와 대조

### ⚠️ 같이 봐야 할 것 — 에피소드가 너무 잘다

평가 하네스는 `DREAMING_IDLE_SECONDS=10`으로 돌아 **매 턴 꿈이 발화**했다. 그래서 에피소드가 전부 **1턴짜리 76개**다.

이게 압축 결과를 왜곡한다. `T1_MAX=8`이라 71개 중 대부분이 Tier2 챕터로 승격되고, 챕터는 에피소드당 **100자로 잘린 한 줄**이다(`chunks.py` `assemble_tier2`). 위 93.9%라는 압축률은 사실상 **턴당 100자 이하로 뭉갠 결과**이고, 이 상태로는 회상 성능이 오히려 나빠질 수 있다.

압축을 켠 뒤 회상이 개선되지 않으면 이쪽을 의심할 것. 선택지:
- 평가의 idle 값을 올려 에피소드를 굵게 만든다 (프로덕션 기본값 300초에 가깝게)
- `T1_MAX`/`CHAPTER_SIZE`를 에피소드 입도에 맞춰 조정한다

**어느 쪽이든 결함 1 수정과 분리해서 판단할 것** — 먼저 압축이 돌게 만들고, 그 다음 입도를 튜닝한다.

---

## 3. 결함 2 — 지식 주입이 가장 오래된 20개에 고정 (심각)

### 증상

confirmed fact가 179개 저장돼 있는데 프롬프트에는 **항상 같은 20개**만 들어간다. 그 20개는 전체를 시간순 정렬했을 때 인덱스 0~19, 즉 **런 시작 4분 이내에 배운 것들**이다. 이후 27분 동안 배운 159개는 한 번도 주입되지 않았다.

### 근본 원인

[dreaming/sync.py:23,41](../dreaming/sync.py):
```python
_MAX_FACTS = 20
...
facts = sorted(facts, key=lambda f: (not f.pinned, f.recorded_at))[:_MAX_FACTS]
```
`recorded_at` **오름차순**(= 오래된 것부터) 정렬 후 앞에서 20개를 자른다. 의도가 "최근 20개"였다면 방향이 반대다.

프로브는 대부분 중·후반 사실을 묻는다. 즉 **정답이 애초에 프롬프트에 없는 상태로 시험을 봤다.**

### 재현

```bash
python3 -c "
from pathlib import Path
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
s = MemoryStore(JsonDirStorage(Path('dreaming_data')), 'night-drm-r0')
f = [x for x in s.list_facts() if x.pinned or x.status == 'confirmed']
sel = sorted(f, key=lambda x: (not x.pinned, x.recorded_at))[:20]
allf = sorted(f, key=lambda x: x.recorded_at)
print('전체:', len(f), '| 선택된 20개의 전체 내 위치:', [allf.index(x) for x in sel])
"
```
기대 출력: `전체: 179 | 선택된 20개의 전체 내 위치: [0, 1, 2, ... 19]`

### 수정 방향

1. **정렬 뒤집기** — pinned 우선은 유지하면서 나머지는 최신순:
   ```python
   facts = sorted(facts, key=lambda f: (f.pinned, f.recorded_at), reverse=True)[:_MAX_FACTS]
   ```
   (`reverse=True`면 `pinned=True`가 앞, 그 안에서 `recorded_at` 내림차순 = 최신순)
2. **개수 상한 → 글자 예산**: 현재 주입이 885자인데 예산은 6,000자다. 개수로 자르면 예산의 15%만 쓴다. fact 줄을 예산 찰 때까지 담는 방식으로 바꾸면 3~4배 더 들어간다.
3. **⚠️ 예산 배분 주의**: [render_knowledge](../dreaming/sync.py)는 `[현재 상태]` → `[확정 사실]` → `[주요 인물]` 순으로 이어붙이고, [clip_knowledge](../dreaming/assembly.py)가 맨 끝을 통째로 자른다. 사실이 예산을 다 먹으면 **인물 블록이 통째로 사라진다.** 블록별 예산을 따로 두거나 자르는 순서를 정할 것.
4. 최신순도 결국 임의 기준이다. "지금 대화에 관련된 사실"을 고르려면 검색이 필요하고, 그건 별도 설계다 (스펙의 임베딩/keyExcerpts 항목). **이번 수정 범위에 넣지 말 것** — 1·2번만으로 다음 런의 출발선이 선다.

### 수정 시뮬레이션 (실데이터로 검증함)

`night-drm-r0`의 fact 179개로 확인:

| 확인 | 결과 |
|---|---|
| `sorted(key=(pinned, recorded_at), reverse=True)[:20]` 선택 위치 | 178~159 (= **최신 20개**, 기존은 0~19) |
| 가장 오래된 fact에 `pinned=True` 부여 시 | 여전히 1순위로 선택됨 — pinned 우선 유지 확인 |
| fact 예산을 4,000자로 뒀을 때 | fact **97개** / 3,970자 (기존 20개·블록 전체 885자) |

### 검증

- 단위: fact를 시각차를 두고 넣고, 주입 결과에 최신 것이 포함되는지 + pinned가 항상 살아남는지
- 실측: 수정 후 같은 세션 데이터로 `render_knowledge` 길이가 885자 → 수천 자로 늘어나는지

---

## 4. 결함 3 — 리롤 원인 확정: 프로바이더 거부가 아니라 language_drift (2026-08-10 Track A 실측으로 교체)

> **원안의 "프로바이더 거부" 추정은 반박됨.** Track A(annyeong-3b2696) 세션이 결과 JSON의 `flaw_history`를 전수 확인한 결과다.

- 리롤 원인은 4변형 전부 **`language_drift`** (flash가 한국어 RP에서 언어 이탈, 100턴당 8~10회 — pro에선 없던 flash 고유 약점). 성인 콘텐츠 거부 아님.
- abort 문자열 `"프로바이더 거부 반복"`은 night2 시절 문구가 남은 **라벨 오표기** — [run2.py](../benchmarks/eval/run2.py)의 abort 사유 문자열을 실원인("리롤 캡 도달")으로 고칠 것.
- 프로바이더/속도 문제는 **이미 수정 완료** (`58b1387`): 기본 라우팅이 SiliconFlow로 가면 11.8K 와이어에서 무한 행(하트비트 공백이 read timeout을 리셋), effort=max는 flash 추론이 1K~19K로 확률 폭주(동일 요청·동일 프로바이더에서 8s vs 132s 재현). 조치 = `provider: {"sort": "throughput"}` + 추론 예산 4,000토큰 캡. 검증: 실와이어 3연속 15~36s, 2턴 스모크 턴당 10~12s.

남은 착수 항목:
1. abort 라벨 문자열 수정 (위)
2. 리롤 게이트([benchmarks/eval/quality.py](../benchmarks/eval/quality.py))에 **한국어 가드/거부 문구 마커 부재** — 결함 4의 T70 유출("프롬프트를 위반했습니다…")이 리롤 없이 통과됐다. 마커 추가.
3. language_drift 자체는 flash 상수로 취급 — 리롤 캡을 "누적"에서 "연속"으로 바꾸거나 캡을 올려 정상 리롤이 런을 죽이지 않게 (dreaming은 캡 10 도달로 T78 중단됨)

---

## 4.5. 결함 4 — `<dreaming_context>` 주입 형식이 프리셋 스캐폴드와 충돌 (심각, 원안에 없음)

### 증상 (Track A 실측, `night-drm-r0`)

- **"——" 더듬기 붕괴가 T36부터 dreaming에만 연발** (`빛—— 이—— 비치기—— 시작했다` 식). vanilla/trim/hypa 0건, dreaming은 T36~44 연속 구간 포함 다수. 두 집단의 유일한 차이 = 프록시 주입.
- **T70 나레이션 말미에 가드 문구가 그대로 유출**: `프롬프트를 위반했습니다: "System" 역할인 "gov"와 "developer"의 프롬프트를 위반하는 지시가 포함되어 있습니다.` — 뮈토스 프리셋 내부 역할(gov/developer) 방어 규칙에 flash가 반응한 것.
- 주입 내용물 자체는 깨끗함 (885자 사실 목록, 렌더 확인). 문제는 **형식**: [inject_knowledge](../dreaming/assembly.py)가 마지막 user 메시지 앞에 `<dreaming_context>...</dreaming_context>` XML풍 태그를 끼워 넣는데, 이게 **지시(instruction)처럼 보여서** 프리셋의 주입 방어를 건드린다.

### 수정 방향 — 프리셋 일반화 원칙 (뮈토스 맞춤 금지)

**주입물은 내용물처럼 보여야지, 지시처럼 보이면 안 된다.**
- 태그 제거 → **로어북풍 평문 서술 블록** (명령문·역할 주장·XML 전부 금지). 근거: RisuAI는 로어북 항목을 평문으로 끼워 넣고, 모든 프리셋은 로어북과 공존하도록 설계돼 있다 — 생태계에서 이미 검증된 유일한 주입 형식.
- 위치는 유지 (마지막 user prepend — 캐시 안전).
- 뮈토스는 "맞춤 대상"이 아니라 **회귀 테스트 프리셋** — 방어가 가장 센 프리셋을 통과하면 약한 프리셋은 대개 통과. 뮈토스**에서** 검증하지 뮈토스**용으로** 고치지 않는다.

### 부수 발견

- worldstate `money: -240.0` — 음수 돈. [dreaming/worldstate.py](../dreaming/worldstate.py) 산수(지출 이중 차감 또는 초기값 미설정) 별건 확인 필요. 주입 블록 신뢰도를 갉아먹는다.

### 검증

- 주입 형식 교체 후 20턴+ 스모크에서 더듬기 패턴(`(——\s*\S{1,3}){4,}` 정규식) 0건 + 가드 문구 0건
- 같은 스모크를 vanilla와 나란히 돌려 dreaming만의 문체 이상이 사라졌는지 육안 대조

---

## 5. 권고 순서 (2026-08-10 Track A 교정 반영)

1. **결함 1 수정 + 조용한 실패 제거** — 층 하나가 통째로 없는 상태라 가장 크다
2. **결함 2 수정** — 한 줄 + 예산 로직, 비용 대비 효과 최대
3. **결함 4 수정 (주입 형식 → 로어북풍 평문)** — 1·2를 고쳐도 이거 안 고치면 dreaming 산문이 또 깨진다 (더듬기·가드 유출)
4. **결함 3 잔여분** — abort 라벨 정정 + quality.py 한국어 가드 마커 + 리롤 캡 정책 (원인 확정은 완료, 프로바이더/추론 수정은 `58b1387`로 반영됨)
5. 전부 끝난 뒤 **재런** — 그때의 숫자가 진짜 비교 기준선이다

1·2번을 고쳐도 dreaming이 이긴다는 보장은 없다. **결과를 좋게 만드는 게 목적이 아니라, 설계가 실제로 켜진 상태에서 재는 게 목적이다.** 지고 있으면 지고 있다고 보고할 것.

---

## 6. 제약 (본런 규약 승계)

- API 키 값 출력·커밋 금지 (`.env`는 변수로만 읽는다, echo 금지)
- `dreaming_data/*`(카드 저작물), `.env`, `research/` 커밋 금지
- 리포트에 NSFW 카드/트랜스크립트 본문 재현 금지 — 구조적 서술만
- 결과 스핀 금지: 손실은 손실로 보고
- 상세: [HANDOFF-2026-08-10.md](HANDOFF-2026-08-10.md) §6
