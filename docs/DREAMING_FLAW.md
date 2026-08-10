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

## 4. 결함 3 — 프로바이더 거부로 런 오염 (원인 미확정)

야간 본런 리롤 횟수: dreaming 10회(T78 중단), vanilla 8회, hypa 8회, trim 2회. dreaming 중단 사유 문자열은 `"누적 리롤 10회 (T78) — 프로바이더 거부 반복"`.

나레이터를 `deepseek/deepseek-v4-flash-0731`로 바꾼 런이고, 프로브 시점에 OpenRouter가 SiliconFlow로 라우팅한 것을 확인했다. **성인 콘텐츠 거부가 리롤을 유발하고 그것이 런을 오염시킨 구조로 보이나, 실제 거부 응답 본문을 확인하지 않았다 — 추정이다.**

착수 순서:
1. `dreaming_data/eval/night-*.log`와 결과 JSON의 리롤 턴에서 **실제 응답 본문/사유**를 확인 (NSFW 본문을 리포트에 재현하지 말 것 — 거부 여부와 형태만)
2. 거부가 맞으면 프로바이더 고정(`provider.order` + `allow_fallbacks: false`) 또는 pro 복귀
3. 리롤 게이트([benchmarks/eval/quality.py](../benchmarks/eval/quality.py))가 "프로바이더 거부"와 "품질 미달"을 구분해 집계하는지 확인 — 지금은 섞여서 진단이 어렵다

---

## 5. 권고 순서

1. **결함 1 수정 + 조용한 실패 제거** — 층 하나가 통째로 없는 상태라 가장 크다
2. **결함 2 수정** — 한 줄 + 예산 로직, 비용 대비 효과 최대
3. **결함 3 원인 확정** — 안 고치면 다음 런도 같은 이유로 오염된다
4. 셋 다 끝난 뒤 **재런** — 그때의 숫자가 진짜 비교 기준선이다

1·2번을 고쳐도 dreaming이 이긴다는 보장은 없다. **결과를 좋게 만드는 게 목적이 아니라, 설계가 실제로 켜진 상태에서 재는 게 목적이다.** 지고 있으면 지고 있다고 보고할 것.

---

## 6. 제약 (본런 규약 승계)

- API 키 값 출력·커밋 금지 (`.env`는 변수로만 읽는다, echo 금지)
- `dreaming_data/*`(카드 저작물), `.env`, `research/` 커밋 금지
- 리포트에 NSFW 카드/트랜스크립트 본문 재현 금지 — 구조적 서술만
- 결과 스핀 금지: 손실은 손실로 보고
- 상세: [HANDOFF-2026-08-10.md](HANDOFF-2026-08-10.md) §6
