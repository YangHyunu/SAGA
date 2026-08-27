#!/bin/zsh
# 헌터스 카드 100턴 haystack 생성 — dreaming 변형 단독.
# 파일럿(pilot-rules-r0) 검증 조합 그대로: IDLE=100 + --ttl-wait + 포트 8791.
# 스모크 6턴(exit 0/2 용인) → 격리 0건 확인 → 본런 100턴.
set -u
WT="/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/silly-pare-d2e948"
cd "$WT" || exit 1
PRESET="/Users/yanghyeon-u/Downloads/뮈토스6.2/🏺뮈토스 프롬프트 V6.2/🏺뮈토스 프롬프트 - DeepSeek V6.2_preset.risup"
CARD="dreaming_data/eval/card-hunters-v1.json"
SESSION="hunters-drm"
export DREAMING_EVAL_MAX_TOKENS=30000
export DREAMING_EVAL_REASONING=4000
export DREAMING_EVAL_MODEL="deepseek/deepseek-v4-flash-0731"
export DREAMING_EVAL_PROXY="http://127.0.0.1:8791"
LOG="dreaming_data/eval/hunters-run.log"
say() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG"; }

say "=== 헌터스 100턴 haystack 생성 (dreaming, IDLE=100, ttl-wait) ==="

# ── Dreaming 프록시 (8791) — 키는 변수로만, 절대 echo 금지 ──
export DREAMING_DREAM_BASE="https://openrouter.ai/api/v1"
export DREAMING_DREAM_KEY="$(grep '^DREAMING_UPSTREAM_KEY=' .env | cut -d= -f2-)"
export DREAMING_DREAM_MODEL="google/gemini-3-flash-preview"
export DREAMING_IDLE_SECONDS=100
export DREAMING_CARD_PATH="/Users/yanghyeon-u/Downloads/Alternate Hunters.charx"
export DREAMING_CARD_USER="렌"
if lsof -iTCP:8791 -sTCP:LISTEN >/dev/null 2>&1; then
  say "8791 이미 점유 — 중단"; exit 1
fi
python3 -c "import uvicorn; from dreaming.proxy import Settings, create_app; \
uvicorn.run(create_app(Settings.from_env()), host='127.0.0.1', port=8791)" \
  >> dreaming_data/eval/proxy-8791.log 2>&1 &
PROXY_PID=$!
sleep 3
kill -0 $PROXY_PID 2>/dev/null || { say "프록시 기동 실패"; exit 1; }

# ── 스모크 6턴: 실행 가능 여부만 (게이트 exit 2는 6턴에선 구조적 실패라 용인) ──
python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" dreaming \
  --session hunters-smoke --turns 6 --trim-tokens 100000 --reset >> "$LOG" 2>&1
SMOKE_EXIT=$?
# night_run.sh의 격리 경로 버그 교정: run2 실제 세션 디렉터리는 <session>-r0
QCOUNT=$(ls "dreaming_data/hunters-smoke-r0/quarantine/" 2>/dev/null | wc -l | tr -d ' ')
if { [ "$SMOKE_EXIT" -ne 0 ] && [ "$SMOKE_EXIT" -ne 2 ]; } || [ "$QCOUNT" -ne 0 ]; then
  say "스모크 실패 (exit ${SMOKE_EXIT}, 격리 ${QCOUNT}건) — 본런 중단"
  kill $PROXY_PID 2>/dev/null
  exit 1
fi
say "스모크 통과 (exit ${SMOKE_EXIT}, 격리 0건) — 본런 100턴 시작"

python3 -u -m benchmarks.eval.run2 "$PRESET" "$CARD" dreaming \
  --session "$SESSION" --runs 1 --turns 100 \
  --trim-tokens 100000 --reset --ttl-wait \
  >> "dreaming_data/eval/${SESSION}.log" 2>&1
MAIN_EXIT=$?
say "본런 종료 (exit ${MAIN_EXIT} — 0=게이트 통과, 2=완주+게이트 일부 실패)"
kill $PROXY_PID 2>/dev/null

RESULT="dreaming_data/eval/v2-${SESSION}-r0-run0.json"
if [ -f "$RESULT" ]; then
  python3 - "$RESULT" <<'EOF' | tee -a "$LOG"
import json, sys
r = json.load(open(sys.argv[1])); t = r["totals"]
c = t["cost"] + t.get("cost_lucid", t.get("cost_director", 0)) + t.get("cost_judge", 0)
print(f'턴 {len(r.get("turns", []))} 프로브 {t["probes"]} judge {t["judge_pass"]}/{t["probes"]} '
      f'rerolls={t.get("rerolls")} aborted={t.get("aborted") or "-"} 총 ${round(c, 2)}')
EOF
fi
say "=== 완료 ==="
