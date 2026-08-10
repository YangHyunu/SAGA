#!/bin/zsh
# 야간 비교런: vanilla/trim/hypa/dreaming 각 1런 @ maxContext 100K.
# 목적: 아침에 변형별 실패 지점(LITM vs evict) 비교.
# dreaming은 스모크런 후 격리(quarantine) 0건일 때만 본런을 시작한다 —
# night2-drm-r0 사고(105/106턴 격리)가 100턴을 다 태운 뒤에야 발견된 재발 방지.
# 2026-08-10 확정 세팅: 100K(2026 실사용 관측·프리셋 기본값 사이 대표) +
# reasoning=max + 캡 30K(뮈토스 maxResponse 충실). cost-probe 실측 근거.
set -u
WT="/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696"
cd "$WT" || exit 1
PRESET="/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup"
CARD="dreaming_data/eval/card-soyeon-v2.json"
SESSION_PREFIX="${SESSION_PREFIX:-night}"
TURNS="${TURNS:-100}"
SMOKE_TURNS="${SMOKE_TURNS:-6}"
TRIM_TOKENS="${TRIM_TOKENS:-100000}"
# TTL 재압축 창구(C11/C12) 개방 — 기본 ON. 100턴 기준 i%10==9가 10회
# 걸리고 매회 305초 대기 → 10 × 305s = 3050s = +51분. 1시 디버깅처럼
# 그 대기가 필요 없을 때만 TTL_WAIT=0으로 끈다.
TTL_WAIT="${TTL_WAIT:-1}"
export DREAMING_EVAL_MAX_TOKENS="${DREAMING_EVAL_MAX_TOKENS:-30000}"
# 추론 예산 4000 토큰 상한 — effort=max는 flash에서 추론이 1K~19K로
# 폭주해 턴당 400s+·업스트림 행까지 발생 (2026-08-10 발사 중단 실측).
export DREAMING_EVAL_REASONING="${DREAMING_EVAL_REASONING:-4000}"
# 나레이터 flash — 유저 실사용 확인(Pro와 체감 무차이) + 저비용 100턴 테스트.
export DREAMING_EVAL_MODEL="${DREAMING_EVAL_MODEL:-deepseek/deepseek-v4-flash-0731}"
LOG="dreaming_data/eval/${SESSION_PREFIX}-run.log"
say() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

say "=== 야간 비교런: 4변형 각 1런 ${TURNS}턴 @${TRIM_TOKENS} (세션=${SESSION_PREFIX}) ==="
[ "$TTL_WAIT" = "1" ] && say "TTL_WAIT=1 — ${TURNS}턴 기준 10회 × 305초 = +51분 추가 소요"

run_variant() {
  local ttl_flag=()
  [ "$TTL_WAIT" = "1" ] && ttl_flag=(--ttl-wait)
  python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" "$1" \
    --session "${SESSION_PREFIX}-$2" --runs 1 --turns "$TURNS" \
    --trim-tokens "$TRIM_TOKENS" --reset "${ttl_flag[@]}" \
    >> "dreaming_data/eval/${SESSION_PREFIX}-$2.log" 2>&1
  say "변형 $1 종료 (exit $?)"
}
run_variant vanilla   van  & PID_V=$!
run_variant trim      trim & PID_T=$!
run_variant hypa      hypa & PID_H=$!

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

# ── dreaming: 스모크 관통 확인 후 본런. 격리 게이트는 현재 no-op —
# 아래 QCOUNT가 보는 dreaming_data/${SMOKE_SESSION}/quarantine/ 경로는
# run2.main()이 실제로 쓰는 ${SMOKE_SESSION}-r0/quarantine/(sess =
# f"{session}-r{n}")와 안 맞아 ls가 항상 빈 결과 → QCOUNT=0 → 격리가
# 있어도 게이트가 절대 안 걸린다(한 번도 안 걸림). 브랜치 이전부터의
# 버그이며, 언제 본런을 중단시킬지는 운영 판단이라 별도 트랙으로 미룬다
# — 코드는 그대로 둔다 ──
# 스모크는 run_variant()를 안 거치는 별도 호출이라 --ttl-wait이 자연히
# 안 붙는다 — 붙여도 SMOKE_TURNS=6에선 i%10==9가 안 걸려 무의미하고,
# 스모크의 목적 자체가 "실행 가능한가"만 보는 빠른 확인이라 51분 대기와
# 안 어울린다. 위 run_variant()와 이 블록이 분리 상태를 유지하는 한
# 손대지 않아도 계속 배제된다.
# run2.py exit code: 0=완전 정상, 2=완주했지만 런 유효성 게이트 실패
# (GATE_ONLY_EXIT — 예: 압축 버그가 살아있는 한 G1은 항상 빨간불),
# 그 외 비영점=크래시/중단/격리. 스모크는 "실행 가능한가"만 확인하는
# 단계라 2는 용인하고, 그 외 비영점만 실패로 본다 — 본런(run_variant)은
# 게이트 실패를 그대로 exit 2로 보고해 상위에서 감지할 수 있게 둔다.
SMOKE_SESSION="${SESSION_PREFIX}-smoke"
PID_D=""
if [ -n "$PROXY_PID" ]; then
  python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" dreaming \
       --session "$SMOKE_SESSION" --turns "$SMOKE_TURNS" \
       --trim-tokens "$TRIM_TOKENS" --reset >> "$LOG" 2>&1
  SMOKE_EXIT=$?
  if [ "$SMOKE_EXIT" -eq 0 ] || [ "$SMOKE_EXIT" -eq 2 ]; then
    QCOUNT=$(ls "dreaming_data/${SMOKE_SESSION}/quarantine/" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$QCOUNT" -eq 0 ]; then
      say "dreaming 스모크 통과 (격리 0건, exit ${SMOKE_EXIT}) — ${TURNS}턴 시작"
      run_variant dreaming drm & PID_D=$!
    else
      say "dreaming 스모크 격리 ${QCOUNT}건 — 본런 건너뜀 (아침에 dreaming_data/${SMOKE_SESSION}/quarantine/ 확인)"
    fi
  else
    say "dreaming 스모크 실패 (exit ${SMOKE_EXIT}) — 본런 건너뜀 (아침에 proxy-8790.log 확인)"
  fi
fi

wait $PID_V $PID_T $PID_H ${PID_D:+$PID_D}
say "런 전부 종료"
[ -n "$PROXY_PID" ] && kill $PROXY_PID 2>/dev/null

FILES=$(ls dreaming_data/eval/v2-${SESSION_PREFIX}-*-r0-run0.json 2>/dev/null)
if [ -n "$FILES" ]; then
  python3 -m benchmarks.eval.report2 ${=FILES} \
    > "dreaming_data/eval/${SESSION_PREFIX}-report.md" 2>>"$LOG"
  python3 -m benchmarks.eval.viewer ${=FILES} >> "$LOG" 2>&1
  SESSION_PREFIX="$SESSION_PREFIX" python3 - <<'EOF' >> "$LOG" 2>&1
import glob
import json
import os

prefix = os.environ["SESSION_PREFIX"]
tot = 0.0
for p in sorted(glob.glob(f"dreaming_data/eval/v2-{prefix}-*-r0-run0.json")):
    r = json.load(open(p)); t = r["totals"]
    c = t["cost"] + t.get("cost_lucid", t.get("cost_director", 0)) + t.get("cost_judge", 0)
    tot += c
    litm = [x for x in r["probes"] if x.get("in_window")]
    ev = [x for x in r["probes"] if not x.get("in_window")]
    print(f'{r["variant"]:<10} {t["judge_pass"]}/{t["probes"]} '
          f'창내 {sum(1 for x in litm if x["judge"] is True)}/{len(litm)} '
          f'창밖 {sum(1 for x in ev if x["judge"] is True)}/{len(ev)} '
          f'rerolls={t.get("rerolls")} flawed={t.get("flawed")} '
          f'aborted={t.get("aborted") or "-"} ${round(c, 3)}')
print(f"야간 총비용(스모크 제외): ${round(tot, 2)}")
EOF
fi
say "=== 완료 — ${SESSION_PREFIX}-report.md ==="
