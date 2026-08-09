#!/bin/zsh
# 야간 비교런: vanilla/trim/hypa/dreaming 각 1런 @ maxContext 32K.
# 목적: 아침에 변형별 실패 지점(LITM vs evict) 비교.
# dreaming은 스모크런 후 격리(quarantine) 0건일 때만 본런을 시작한다 —
# night2-drm-r0 사고(105/106턴 격리)가 100턴을 다 태운 뒤에야 발견된 재발 방지.
set -u
WT="/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696"
cd "$WT" || exit 1
PRESET="/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup"
CARD="dreaming_data/eval/card-soyeon-v2.json"
SESSION_PREFIX="${SESSION_PREFIX:-night}"
TURNS="${TURNS:-100}"
SMOKE_TURNS="${SMOKE_TURNS:-6}"
TRIM_TOKENS="${TRIM_TOKENS:-32000}"
LOG="dreaming_data/eval/${SESSION_PREFIX}-run.log"
say() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

say "=== 야간 비교런: 4변형 각 1런 ${TURNS}턴 @${TRIM_TOKENS} (세션=${SESSION_PREFIX}) ==="

run_variant() {
  python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" "$1" \
    --session "${SESSION_PREFIX}-$2" --runs 1 --turns "$TURNS" \
    --trim-tokens "$TRIM_TOKENS" --reset \
    >> "dreaming_data/eval/${SESSION_PREFIX}-$2.log" 2>&1
  say "변형 $1 종료 (exit $?)"
}
run_variant vanilla   van  & PID_V=$!
run_variant trim      trim & PID_T=$!
run_variant hypa      hypa & PID_R=$!

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

# ── dreaming: 스모크 관통 확인 + 격리 게이트 통과 후에만 본런 ──
SMOKE_SESSION="${SESSION_PREFIX}-smoke"
PID_D=""
if [ -n "$PROXY_PID" ]; then
  if python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" dreaming \
       --session "$SMOKE_SESSION" --turns "$SMOKE_TURNS" \
       --trim-tokens "$TRIM_TOKENS" --reset >> "$LOG" 2>&1; then
    QCOUNT=$(ls "dreaming_data/${SMOKE_SESSION}/quarantine/" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$QCOUNT" -eq 0 ]; then
      say "dreaming 스모크 통과 (격리 0건) — ${TURNS}턴 시작"
      run_variant dreaming drm & PID_D=$!
    else
      say "dreaming 스모크 격리 ${QCOUNT}건 — 본런 건너뜀 (아침에 dreaming_data/${SMOKE_SESSION}/quarantine/ 확인)"
    fi
  else
    say "dreaming 스모크 실패 — 본런 건너뜀 (아침에 proxy-8790.log 확인)"
  fi
fi

wait $PID_V $PID_T $PID_R ${PID_D:+$PID_D}
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
    c = t["cost"] + t.get("cost_director", 0) + t.get("cost_judge", 0)
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
