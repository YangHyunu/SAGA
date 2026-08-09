# 야간 벤치 루프 (B구현 → 4변형 발사 → 모니터링 → 아침 리포트) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Task 6~7은 `/loop` 반복분 — 매 wakeup마다 Task 7 체크리스트를 다시 돈다.

**Goal:** 거리 기반 프로브 게이팅(B) 구현 후 vanilla/trim/retrieval/dreaming 4변형을 100턴@32K로 밤새 병렬 실행하고, 크래시를 자동 복구하며, 아침에 비교 리포트를 완성한다.

**Architecture:** (1) `director.py`의 evict-전용 게이팅을 사실 나이(distance) 기반으로 교체하고 프로브 기록에 `in_window`를 남겨 LITM/eviction 실패를 분리한다. (2) T40–45 당채련 고정 등장 이벤트로 로어북 인물을 유도한다 (주역은 위지소연 유지). (3) night 스크립트가 4변형을 병렬 발사(dreaming은 프록시+스모크 게이트 뒤), /loop가 30분 주기로 생존/크래시/완료를 점검한다.

**Tech Stack:** Python 3 (pip), pytest, zsh, OpenRouter (DeepSeek V4 Pro 나레이터 / Gemini 3 Flash 디렉터 / Sonnet 4.5 judge), dreaming 프록시(uvicorn, 8790).

## Global Constraints

- API 키 값을 stdout/커밋/로그에 절대 출력하지 않는다 — `.env`에서 `grep '^DREAMING_UPSTREAM_KEY=' .env | cut -d= -f2-`로 변수에만 담는다.
- `dreaming_data/*`(코퍼스·결과·카드)와 `.env`는 커밋 금지 (gitignore 유지).
- 리포트에 카드 NSFW 원문 재현 금지 — 구조 서술만.
- 결과 스핀 금지 — 진 것은 진 것으로 기록.
- 테스트: `python3 -m pytest -q tests/test_eval_v2.py` / 린트: `ruff check benchmarks/eval/`
- 예산 $16.45. 야간 추정 총비용 ~$3.5–5 (아래 비용 절 참조). 만약 중간 점검에서 부분 JSON 합산이 $10을 넘으면 남은 런 중단.
- 프리셋: `/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup`
- 카드: `dreaming_data/eval/card-soyeon-v2.json`
- 작업트리: `/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696` (브랜치 `dreaming/spec`)

## 비용·시간 추정

실측 기준: 80턴 vanilla@12K 총 $0.369 (나레이터 $0.167 + 디렉터 $0.164/154콜 + judge $0.038/6콜).

| 항목 | 근거 | 추정 |
|---|---|---|
| trim/retrieval/dreaming 각 | 100턴(×1.25) · 창 12K→32K(나레이터 프롬프트 ~2.5×) · 프로브 ~10개 | ~$0.8–1.0/런 |
| vanilla | 풀 히스토리 말미 ~100K, 캐시 히트 높음(append-only) | ~$0.9–1.2 |
| dreaming 스모크 6턴 | | ~$0.05 |
| **야간 총합** | 4런+스모크 | **~$3.5–5.0** |
| 소요 시간 | 턴당 40–60s(디렉터+나레이터+추출), 병렬 | **~1.5–2.5h** |

---

### Task 1: 거리 기반 프로브 게이팅 (director.py)

**Files:**
- Modify: `benchmarks/eval/director.py:89-93` (eligible), `benchmarks/eval/director.py:127-146` (probe_plan)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `MIN_PROBE_AGE = 15`, `RECENT_MAX_AGE = 8` (모듈 상수),
  `eligible(ledger, turn_now: int, kind=None, min_age=MIN_PROBE_AGE) -> List[DirFact]`,
  `probe_plan(ledger, turn_now: int, want: Dict[str,int], min_age=MIN_PROBE_AGE) -> List[Tuple[str, DirFact]]`
- 의미 변화: 두 번째 인자가 `window_start_turn`(창 시작 턴)에서 `turn_now`(현재 턴)로 바뀐다. Task 2에서 run2 호출부를 같이 고친다.

- [ ] **Step 1: 실패하는 테스트 작성** — `tests/test_eval_v2.py`의 기존 `eligible`/`probe_plan` 창 기반 테스트를 아래로 교체:

```python
def test_eligible_distance_gated():
    led = Ledger()
    led.add([DirFact(turn=0, kind="exact", value="사과", text="사과 삼")])
    assert director.eligible(led, 14) == []          # 나이 14 < 15 → 미달
    assert len(director.eligible(led, 15)) == 1      # 나이 15 → 적격

def test_probe_plan_recent_pool_is_young():
    led = Ledger()
    led.add([DirFact(turn=0, kind="exact", value="배", text="배 삼"),
             DirFact(turn=30, kind="exact", value="감", text="감 삼")])
    plan = director.probe_plan(led, 35, {"recent": 1})
    assert plan and plan[0][1].value == "감"         # 나이 5 ≤ 8만 recent

def test_probe_plan_old_facts_for_recall():
    led = Ledger()
    led.add([DirFact(turn=0, kind="exact", value="배", text="배 삼"),
             DirFact(turn=30, kind="exact", value="감", text="감 삼")])
    plan = director.probe_plan(led, 35, {"recall": 1})
    assert plan and plan[0][1].value == "배"         # 나이 35 ≥ 15만 recall
```

- [ ] **Step 2: 실패 확인** — `python3 -m pytest -q tests/test_eval_v2.py -k "distance or recent_pool or old_facts"` → FAIL (TypeError 또는 assert).

- [ ] **Step 3: 구현** — `director.py`에서:

```python
# LITM 분리 게이팅: 창 밖 여부가 아니라 사실 나이로 출제한다.
# 파일럿 실측 — 풀컨텍스트 vanilla가 dist 19+에서 실패(순수 LITM), dist 9는 통과.
MIN_PROBE_AGE = 15   # 이 나이부터 원거리 프로브 출제
RECENT_MAX_AGE = 8   # 단기 대조군(recent) 상한


def eligible(ledger: Ledger, turn_now: int, kind: Optional[str] = None,
             min_age: int = MIN_PROBE_AGE) -> List[DirFact]:
    """나이(현재턴-기록턴)가 min_age 이상인 미출제 사실."""
    return [f for f in ledger.unprobed(kind)
            if turn_now - f.turn >= min_age]
```

probe_plan은 시그니처를 `(ledger, turn_now, want, min_age=MIN_PROBE_AGE)`로 바꾸고, recent 분기만 교체:

```python
        if ptype == "recent":
            pool = [f for f in ledger.unprobed(_PTYPE_KIND[ptype])
                    if turn_now - f.turn <= RECENT_MAX_AGE]
        else:
            pool = eligible(ledger, turn_now, kind=_PTYPE_KIND[ptype],
                            min_age=min_age)
```

docstring의 "창 밖(evict)" 문구를 "나이 기반 — evict 여부는 run2가 in_window로 별도 기록"으로 수정.

- [ ] **Step 4: 통과 확인** — 같은 pytest 명령 → PASS. (커밋은 Task 3에서 일괄.)

---

### Task 2: run2 호출부 + in_window 기록

**Files:**
- Modify: `benchmarks/eval/run2.py:315-330` (probe_plan 호출), `benchmarks/eval/run2.py:395-405` (프로브 기록)

**Interfaces:**
- Consumes: Task 1의 새 `probe_plan(ledger, turn_now, want)`.
- Produces: 프로브 JSON 레코드에 `"in_window": bool` — 나레이터가 실제로 본 컨텍스트에 사실 턴이 남아 있었는가. vanilla는 항상 True(풀 히스토리). report2/viewer가 이 키를 읽는다(Task 4).

- [ ] **Step 1: 호출부 교체** — `run2.py:320` `plan = probe_plan(ledger, win_start, {ptype: 1})` → `plan = probe_plan(ledger, i, {ptype: 1})`. 그 위 315행의 사전 `token_trim` 호출(`_, win_start = ...`)은 이제 게이팅에 안 쓰이므로 삭제 (338행에서 실제 창 계산이 다시 이뤄져 `win_start`가 잡힌다).

- [ ] **Step 2: 기록 추가** — 프로브 append(`probes.append({...})`)에 두 키 추가:

```python
                           "in_window": (variant == "vanilla"
                                         or fact.turn >= win_start),
```

`distance_turns` 바로 옆에 둔다. 여기의 `win_start`는 338행(`window, win_start = token_trim(...)`)이 남긴 실제 전송 창 시작 턴.

- [ ] **Step 3: 스모크 실행으로 확인** — 실런 없이 임포트·시그니처만: `python3 -c "from benchmarks.eval.run2 import *; from benchmarks.eval.director import probe_plan, MIN_PROBE_AGE; print(MIN_PROBE_AGE)"` → `15` 출력, 예외 없음.

---

### Task 3: 당채련 등장 이벤트 (T40–45 고정) + 테스트 + 커밋

50/80턴 파일럿에서 로어북 NPC 0명 등장 — 단조로움. 결정: 범용 NPC 초대가 아니라 **모든 런에서 NPC 등장은 "당채련" 하나로 고정**, T40–45 사이 자연 합류. 런 간 비교가능성(같은 사건 축). 로어 7엔트리는 preset2wire가 항상 주입하므로(키워드 게이팅 없음, 당채련 프로필 = 엔트리 6) 활성화 문제가 아니라 장면 유도 문제 — 이름을 직접 불러 나레이터가 꺼내게 한다. **주의: 메인 캐릭터는 어디까지나 위지소연** — 당채련은 곁가지, 장면을 뺏으면 안 된다.

**Files:**
- Modify: `benchmarks/eval/run2.py` (상수 + pick_beat + run_once 호출부)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Produces: `NPC_NAME = "당채련"`, `NPC_EVENT_TURN = 40`(0-기준, 표시 T41), `NPC_EVENT_RETRY = 44`, `_NPC_BEAT`(지시문), `pick_beat(i, npc_due=False)` — `npc_due=True`면 다른 비트 대신 _NPC_BEAT 반환.

- [ ] **Step 1: 실패 테스트**:

```python
def test_npc_event_introduces_dangchaeryun_only():
    # NPC 등장은 당채련 하나로 고정 (T40-45 자연 합류), 주인공은 위지소연
    assert run2.NPC_NAME == "당채련"
    assert run2.NPC_EVENT_TURN == 40 and run2.NPC_EVENT_RETRY == 44
    beat = run2.pick_beat(40, npc_due=True)
    assert "당채련" in beat and "위지소연" in beat      # 이름 명시 + 주역 고정
    assert "당채련" not in run2.pick_beat(40)           # npc_due 없으면 평소 비트

def test_beats_have_no_generic_npc_invite():
    # 범용 NPC 초대 비트 없음 — 당채련 외 인물 유입 차단
    assert all("인물" not in b for b in run2._BEATS)
```

- [ ] **Step 2: 실패 확인** — `python3 -m pytest -q tests/test_eval_v2.py -k "dangchae or generic_npc"` → FAIL.

- [ ] **Step 3: 구현** — `run2.py` 상수(UPDATE_EVENTS 근처):

```python
# NPC 등장은 당채련 하나로 고정 — 런 간 같은 사건 축이라 비교 가능하고,
# 이름을 직접 불러 로어북 키워드가 확실히 발화된다. T41~T45 사이 자연 합류.
# 디렉터의 카드 지식 선취 금지의 유일한 예외가 이 이름이다.
NPC_NAME = "당채련"
NPC_EVENT_TURN = 40           # 0-기준 (표시 T41)에 첫 유도
NPC_EVENT_RETRY = 44          # 이때까지 미등장이면 한 번 더 강하게

_NPC_BEAT = (f"{NPC_NAME}(이)가 자연스럽게 장면에 합류할 상황을 만든다 — "
             f"찾아가거나, 우연히 마주치거나, 이름을 언급하며 소식을 묻는다. "
             f"'{NPC_NAME}'이라는 이름은 말해도 되지만 그 외 설정은 지어내지 "
             f"마라. 장면의 중심은 계속 위지소연이다 — {NPC_NAME}은 곁가지로만.")
```

`pick_beat` 교체 (우선순위: NPC 이벤트 > UPDATE > 5턴 비트):

```python
def pick_beat(i: int, npc_due: bool = False) -> str:
    """턴 i의 필러 지시. NPC 이벤트 > UPDATE_EVENTS > 5턴 주기 비트 > 평서."""
    if npc_due:
        return _NPC_BEAT
    if i in UPDATE_EVENTS:
        return _UPDATE_BEAT
    if i % 5 == 4:
        return _BEATS[(i // 5) % len(_BEATS)]
    return "자연스럽게 이어간다."
```

run_once 필러 분기(`pick_beat(i)` 호출부)를:

```python
            npc_due = (i == NPC_EVENT_TURN
                       or (NPC_EVENT_TURN < i <= NPC_EVENT_RETRY
                           and not any(NPC_NAME in m["content"]
                                       for m in history)))
            utext = director(
                dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
                f"[최근 대화]\n{ctx}\n[지시]\n{pick_beat(i, npc_due)}")
```

(T41에 무조건 한 번 유도, T42~T45엔 히스토리에 이름이 아직 없을 때만 재시도 — 등장했으면 평소 비트로 복귀. T41이 프로브 턴이면 프로브가 우선이고 다음 필러 턴에서 npc_due 조건이 자동으로 잡는다.)

- [ ] **Step 4: 전체 테스트** — `python3 -m pytest -q tests/test_eval_v2.py` → 전부 PASS. `python3 -m ruff check benchmarks/eval/run2.py` — 기존 E402 5건 외 신규 위반 없음.

- [ ] **Step 5: 커밋**:

```bash
git add benchmarks/eval/director.py benchmarks/eval/run2.py tests/test_eval_v2.py
git commit -m "feat(eval): 거리 기반 프로브 게이팅 + in_window 기록 + 당채련 등장 이벤트

evict-전용 게이팅은 LITM(창 안 원거리 실패)을 못 재고 창이 클수록
벤치가 관대해진다. 사실 나이 >= 15턴이면 출제하고, 나레이터가 실제로
본 창 기준 in_window를 기록해 LITM/eviction 실패를 리포트에서 분리한다.
recent 대조군은 나이 <= 8턴. 로어북 NPC 미등장(단조로움)은 T41~T45
당채련 고정 합류 이벤트로 해소 — 이름 언급은 카드 지식 선취 금지의
유일한 예외, 장면 주역은 위지소연 유지."
```

---

### Task 4: report2 창 분리(2×2) 집계

**Files:**
- Modify: `benchmarks/eval/report2.py` (변형별 집계 dict + 렌더)
- Test: `tests/test_eval_v2.py`

**Interfaces:**
- Consumes: 프로브 레코드의 `in_window`(Task 2), `judge`.
- Produces: 변형별 `창내 LITM x/y · 창밖 evict x/y` 줄. `in_window` 키 없는 구 JSON은 창밖으로 집계(기존 게이팅이 evict-전용이었으므로 사실과 일치).

- [ ] **Step 1: 실패 테스트**:

```python
def test_report_splits_by_window():
    probes = [{"judge": True, "in_window": True}, {"judge": False, "in_window": True},
              {"judge": False, "in_window": False}]
    inw, out = report2.window_split(probes)
    assert inw == (1, 2) and out == (0, 1)
```

- [ ] **Step 2: 실패 확인** — `-k window_split` FAIL.

- [ ] **Step 3: 구현** — `report2.py`에 헬퍼 + 변형별 표에 열 추가:

```python
def window_split(probes):
    """(창내 pass, 창내 n), (창밖 pass, 창밖 n) — 구 JSON은 in_window 부재=창밖."""
    inw = [p for p in probes if p.get("in_window")]
    out = [p for p in probes if not p.get("in_window")]
    def _p(xs):
        return (sum(1 for p in xs if p["judge"] is True), len(xs))
    return _p(inw), _p(out)
```

렌더 줄 예: `창내(LITM) 1/2 · 창밖(evict) 0/1`. viewer.py 프로브 카드에도 배지 한 개: `in_window`면 `창내`, 아니면 `창밖` (기존 distance 칩 옆).

- [ ] **Step 4: 전체 테스트 PASS 후 커밋**:

```bash
git add benchmarks/eval/report2.py benchmarks/eval/viewer.py tests/test_eval_v2.py
git commit -m "feat(eval): 리포트 LITM/eviction 2x2 분리 — in_window 기준 집계"
```

---

### Task 5: night 스크립트 v2 (100턴 · 32K · 4변형)

**Files:**
- Create: `/private/tmp/claude-501/-Users-yanghyeon-u-Desktop-RISU-ENE--claude-worktrees-annyeong-3b2696/20640edd-234c-4f27-9afa-6a712cdfe9ce/scratchpad/night_run2.sh`

기존 `night_run.sh`에서 바뀌는 것: 세션 `night2-{van,trim,ret,drm}`, `--turns 100 --trim-tokens 32000`, vanilla 포함 4변형(pilot80b 죽음), 프록시 재기동. 나머지 골격(키 비노출, 스모크 게이트, 리포트·비용 합산) 유지.

- [ ] **Step 1: 스크립트 작성**:

```bash
#!/bin/zsh
# 야간 비교런 v2: vanilla/trim/retrieval/dreaming 각 1런 100턴 @ trim 32K.
# 목적: 아침에 변형별 실패 지점(LITM vs evict) 비교. 로그: night2-run.log
set -u
WT="/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696"
cd "$WT" || exit 1
PRESET="/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup"
CARD="dreaming_data/eval/card-soyeon-v2.json"
LOG="dreaming_data/eval/night2-run.log"
say() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

say "=== 야간 비교런 v2: 4변형 각 1런 100턴 @32K ==="

run_variant() {
  python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" "$1" \
    --session "night2-$2" --runs 1 --turns 100 --trim-tokens 32000 --reset \
    >> "dreaming_data/eval/night2-$2.log" 2>&1
  say "변형 $1 종료 (exit $?)"
}
run_variant vanilla   van  & PID_V=$!
run_variant trim      trim & PID_T=$!
run_variant retrieval ret  & PID_R=$!

# ── Dreaming 프록시 (8790) — 키는 변수로만, 절대 echo 금지 ──
export DREAMING_DREAM_BASE="https://openrouter.ai/api/v1"
export DREAMING_DREAM_KEY="$(grep '^DREAMING_UPSTREAM_KEY=' .env | cut -d= -f2-)"
export DREAMING_DREAM_MODEL="google/gemini-3-flash-preview"
export DREAMING_IDLE_SECONDS=10
export DREAMING_CARD_PATH="/Users/yanghyeon-u/Downloads/위지소연 (1).charx"
export DREAMING_CARD_USER="렌"
python3 -c "import uvicorn; from dreaming.proxy import Settings, create_app; \
uvicorn.run(create_app(Settings.from_env()), host='127.0.0.1', port=8790)" \
  >> dreaming_data/eval/proxy-8790.log 2>&1 &
PROXY_PID=$!
sleep 3
kill -0 $PROXY_PID 2>/dev/null || { say "프록시 기동 실패 — 나머지 3변형은 계속"; PROXY_PID=""; }

PID_D=""
if [ -n "$PROXY_PID" ]; then
  if python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" dreaming \
       --session smoke2 --turns 6 --trim-tokens 32000 --reset >> "$LOG" 2>&1; then
    say "dreaming 스모크 통과 — 100턴 시작"
    run_variant dreaming drm & PID_D=$!
  else
    say "dreaming 스모크 실패 — 본런 건너뜀 (아침에 proxy-8790.log 확인)"
  fi
fi

wait $PID_V $PID_T $PID_R ${PID_D:+$PID_D}
say "런 전부 종료"
[ -n "$PROXY_PID" ] && kill $PROXY_PID 2>/dev/null

FILES=$(ls dreaming_data/eval/v2-night2-*-r0-run0.json 2>/dev/null)
if [ -n "$FILES" ]; then
  python3 -m benchmarks.eval.report2 ${=FILES} \
    > dreaming_data/eval/night2-report.md 2>>"$LOG"
  python3 -m benchmarks.eval.viewer ${=FILES} >> "$LOG" 2>&1
  python3 - <<'EOF' >> "$LOG" 2>&1
import json, glob
tot = 0.0
for p in sorted(glob.glob("dreaming_data/eval/v2-night2-*-r0-run0.json")):
    r = json.load(open(p)); t = r["totals"]
    c = t["cost"] + t.get("cost_director", 0) + t.get("cost_judge", 0)
    tot += c
    litm = [x for x in r["probes"] if x.get("in_window")]
    ev = [x for x in r["probes"] if not x.get("in_window")]
    print(f'{r["variant"]:<10} {t["judge_pass"]}/{t["probes"]} '
          f'창내 {sum(1 for x in litm if x["judge"] is True)}/{len(litm)} '
          f'창밖 {sum(1 for x in ev if x["judge"] is True)}/{len(ev)} '
          f'rerolls={t.get("rerolls")} flawed={t.get("flawed")} ${round(c, 3)}')
print(f"야간 총비용(스모크 제외): ${round(tot, 2)}")
EOF
fi
say "=== 완료 — night2-report.md ==="
```

- [ ] **Step 2: 문법 검사** — `zsh -n .../night_run2.sh` → 출력 없음(통과).

---

### Task 6: 발사 + 생존 확인

- [ ] **Step 1: 잔여 프로세스 정리** — `pgrep -fl "benchmarks.eval.run2|uvicorn.*8790"` 확인, 있으면 kill (이전 세션 잔재만; 다른 사용자 프로세스 건드리지 않음).
- [ ] **Step 2: 발사** — `chmod +x night_run2.sh` 후 Bash `run_in_background`로 실행 (nohup 불필요 — 백그라운드 태스크가 세션에 추적됨).
- [ ] **Step 3: 3분 뒤 생존 확인** — `pgrep -fl benchmarks.eval.run2` 4개(스모크 중엔 3+1) + 각 `night2-*.log`에 "턴 1" 진행 흔적 + `proxy-8790.log`에 리슨 로그. 실패 변형 있으면 로그 tail로 원인 확인 후 즉시 수정·재발사.

---

### Task 7: /loop 모니터링 반복분 (매 wakeup ~30분 주기)

매 반복 이 체크리스트를 돈다. 전부 통과 전엔 루프 유지(`ScheduleWakeup delaySeconds=1800`), 완료 조건 충족 시 아침 요약 쓰고 루프 종료(`stop: true`).

- [ ] **Step 1: 생존 점검** — `pgrep -fl benchmarks.eval.run2` + `ls dreaming_data/eval/v2-night2-*-run0.json`. 완료 변형은 프로세스 죽고 JSON 존재 = 정상.
- [ ] **Step 2: 크래시 스캔** — `grep -l Traceback dreaming_data/eval/night2-*.log`. 크래시 발견 시: 원인 읽고 (a) 코드 버그면 수정 + 테스트 + 커밋 후 해당 변형만 `--session night2-<arm>b`로 재발사, (b) 프로바이더 5xx 연속이면 그 변형 포기하고 로그에 기록. 진행 중 파일 삭제 금지. **예외**: `런 중단: 누적 리롤` 메시지는 리롤 중단 게이트(`MAX_RUN_REROLLS=10`, 이미 구현·커밋됨)의 정상 동작 — NSFW 거부 반복이므로 재발사하지 말고 부분 JSON(저장돼 있음)을 그대로 리포트에 포함, 요약에 중단 사실 기록.
- [ ] **Step 3: 예산 가드** — 완료된 JSON의 totals 비용 합산. $10 초과 시 남은 런 중단하고 그 시점까지로 리포트.
- [ ] **Step 4: 완료 판정** — night2-run.log에 "완료 — night2-report.md" 또는 (모든 run2 프로세스 종료 ∧ JSON 존재). 스크립트가 리포트 못 만들고 죽었으면 Task 5의 리포트 블록을 수동 실행.
- [ ] **Step 5: 종료 시 아침 요약** — `dreaming_data/eval/night2-summary.md` 작성: 변형별 프로브 성적(창내/창밖 분리), 리롤·잔존 병리, 비용 합계, 크래시·재발사 이력, 내일 할 일(하이파V5 변형·salience 태그·NPC 이름 해금·3반복 본런). 결과가 나빠도 그대로 기록. 그 후 `ScheduleWakeup stop`.

## Self-Review 결과

- 스펙 커버: 게이팅 교체(T1-2), 당채련 등장 이벤트(T3), 2×2 리포트(T4), 100턴@32K 4변형(T5), 자동 발사(T6), 야간 무인 운영(T7) — 전부 태스크에 매핑.
- 타입 일관성: `probe_plan(ledger, turn_now, want)` — T1 정의 = T2 호출. `in_window` — T2 기록 = T4/T5 소비.
- 구 JSON 호환: report2는 `in_window` 부재를 창밖으로 처리 (구 게이팅이 evict-전용이었으므로 정확).
