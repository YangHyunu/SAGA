"""실카드 벤치 — RisuAI 프롬프트 조립 재현 + 로어북 delta 계측.

유저 턴은 고정 대본이 아니라 시뮬레이터가 생성: 카드 설정 요약 + 직전 캐릭터
응답을 보고 비트 지시(지뢰 심기/프로브/자유)를 세계관에 맞는 발화로 변환한다.
오라클은 숫자(300/3/50/자정/보름달)로 고정 — 단위는 시뮬레이터가 세계관에 맞춘다.

기존 스크래치패드 판과 달리 프롬프트를 RisuAI 기본 promptTemplate 순서로 조립하고
(`cardsim/lorebook.py`), 매 턴 keyed 로어북을 재활성화해 **프리픽스가 흔들리는
실제 상황**을 프록시에 먹인다. `--no-keyed-lore`가 delta 없는 대조군이다.

usage:
    python -m benchmarks.cardsim.bench <charx경로> <세션ID> [--no-keyed-lore] [--reset]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

import httpx

from benchmarks.cardsim.lorebook import activate, build_messages, load_card

BASE = "http://127.0.0.1:8787"
MODEL = "anthropic/claude-haiku-4.5"
SIM_MODEL = "google/gemini-2.5-flash"
USER_NAME = "한결"
ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "dreaming_data"

BEATS = [
    "정중히 자기소개를 한다. 이름은 '한결', 나이는 '스물일곱'이라고 명확히 밝힌다.",
    "상대에 대해 물어본다 — 어떻게 불러야 할지, 어떤 사람인지.",
    "지금 있는 장소/상황에 대해 자연스럽게 묻는다.",
    "자신의 소지금이 정확히 300(이 세계관의 화폐 단위)뿐이라고 대화 중에 언급한다. 숫자 300을 명시.",
    "상대의 취향이나 좋아하는 것을 묻는다.",
    "작은 선물로 먹을 것(세계관에 어울리는 것) '세 개'를 건넨다. 개수 '세 개'를 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "오늘은 물러가겠다며, '내일 자정'에 다시 오겠다고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "짧게 작별 인사를 한다.",
    # -- pause: dream #1 --
    "약속대로 다시 찾아왔다고 인사한다.",
    "값으로 50(화폐)을 치르고 마실 것이나 먹을 것을 산다. 숫자 50을 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "자신의 이름과 나이를 기억하고 있는지 상대에게 묻는다.",
    "직전 응답에 자연스럽게 반응한다.",
    "자신이 사실 '왼손잡이'라는 것을 고백한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "잠시 침묵하다 상대를 바라보며 짧게 한마디 한다.",
    # -- reroll turn 20, then pause: dream #2 --
    "시간이 지나 다시 찾아왔다고 인사한다.",
    "장부를 잃어버렸다며, 자신에게 남은 소지금이 얼마인지 아는지 묻는다.",
    "직전 응답에 자연스럽게 반응한다.",
    "'다음 보름달'에 함께 축제나 나들이를 가자고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "지난 며칠을 회상하며 짧게 감상을 말한다.",
    "처음 만난 날 자신이 건넨 선물이 무엇이었는지 기억하는지 묻는다.",
    "예전에 자신이 '몇 시'에 다시 오겠다고 약속했었는지 묻는다.",
    "지금까지 자신(한결)에 대해 알게 된 것을 전부 말해달라고 한다.",
]
PAUSES = {9: 12, 19: 12}
REROLL_AT = 19
PROBES = (14, 21, 27, 28, 29)


def _upstream_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def gen_user(client: httpx.Client, setting: str, last_reply: str,
             beat: str) -> str:
    sys_p = ("너는 RP에서 유저(1인칭 남성, 이름 한결) 역할을 연기하는 시뮬레이터다. "
             "작품 설정과 직전 장면에 자연스럽게 이어지는 유저의 다음 발화 하나만 출력한다. "
             "지문과 대사 혼합 가능, 3문장 이내. 메타 발언·설명 금지.\n\n[작품 설정 요약]\n"
             + setting)
    usr_p = (f"[직전 캐릭터 응답]\n{last_reply[-800:]}\n\n[이번 턴 지시]\n{beat}\n\n"
             "유저 발화:")
    try:
        r = client.post("/chat/completions", json={
            "model": SIM_MODEL, "max_tokens": 250,
            "messages": [{"role": "system", "content": sys_p},
                         {"role": "user", "content": usr_p}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip() or beat
    except Exception as e:                       # noqa: BLE001 — 벤치는 계속 돈다
        print(f"  (sim fail: {e} — 비트 원문 사용)", flush=True)
        return beat


def call(session: str, msgs: list) -> tuple:
    t0 = time.time()
    r = httpx.post(BASE + "/v1/chat/completions", timeout=300,
                   headers={"x-dreaming-session-id": session},
                   json={"model": MODEL, "max_tokens": 300, "messages": msgs})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    return d["choices"][0]["message"]["content"], {
        "prompt": u.get("prompt_tokens", 0),
        "cached": det.get("cached_tokens", 0),
        "write": det.get("cache_write_tokens", 0),
        "out": u.get("completion_tokens", 0),
        "cost": u.get("cost", 0.0),
        "sec": round(time.time() - t0, 1),
    }


def _prepare_data_dir(session: str, reset: bool) -> None:
    """세션 디렉터리 오염 방지 — 이전 실행 잔재가 원장에 구멍을 낸다."""
    d = DATA_ROOT / session
    if not d.exists():
        return
    if not reset:
        raise SystemExit(
            f"{d} 이미 있음. 같은 세션에 두 번 돌리면 원장이 섞인다.\n"
            f"지우고 돌리려면 --reset, 아니면 다른 세션 ID를 써라.")
    shutil.rmtree(d)
    print(f"[reset] {d} 삭제", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("session")
    ap.add_argument("--no-keyed-lore", action="store_true",
                    help="keyed 로어북 비활성 (delta 없는 대조군)")
    ap.add_argument("--reset", action="store_true",
                    help="기존 세션 데이터 삭제 후 시작")
    args = ap.parse_args()

    _prepare_data_dir(args.session, args.reset)

    card = load_card(args.card, USER_NAME)
    if args.no_keyed_lore:
        card.lore = [e for e in card.lore if e.constant]
    n_const = sum(1 for e in card.lore if e.constant)
    print(f"CARD={card.name} desc={len(card.description)}자 "
          f"PHI={len(card.post_history)}자 greet={len(card.greeting)}자", flush=True)
    print(f"LORE constant={n_const} keyed={len(card.lore) - n_const} "
          f"scan_depth={card.scan_depth} token_budget={card.token_budget}",
          flush=True)

    sim = httpx.Client(base_url="https://openrouter.ai/api/v1", timeout=60,
                       headers={"Authorization": f"Bearer {_upstream_key()}"})
    setting = card.description[:2500]

    history: list = []
    last_reply = card.greeting or "(첫 장면)"
    stats, lore_log, total_cost = [], [], 0.0
    prev_active: set = set()

    for i, beat in enumerate(BEATS):
        utext = gen_user(sim, setting, last_reply, beat)
        print(f"U{i+1:02d}: {utext[:110].replace(chr(10), ' ')}", flush=True)
        history.append({"role": "user", "content": utext})

        actives = activate(card, history)
        names = {e.name for e in actives}
        lore_tokens = sum(e.tokens for e in actives)
        added, dropped = names - prev_active, prev_active - names
        prev_active = names
        lore_log.append({"turn": i + 1, "active": len(names),
                         "tokens": lore_tokens,
                         "added": sorted(added), "dropped": sorted(dropped)})

        msgs = build_messages(card, actives, history)
        reply, st = call(args.session, msgs)
        if i == REROLL_AT:
            reply, st2 = call(args.session, msgs)     # 리롤: 같은 요청 재전송
            st["cost"] += st2["cost"]
            print(f"T{i+1:02d} REROLL cached={st2['cached']}", flush=True)
        history.append({"role": "assistant", "content": reply})
        last_reply = reply
        total_cost += st["cost"]

        hit = st["cached"] / st["prompt"] * 100 if st["prompt"] else 0
        delta = f" LORE+{len(added)}/-{len(dropped)}" if (added or dropped) else ""
        print(f"T{i+1:02d} prompt={st['prompt']} cached={st['cached']}"
              f" ({hit:.0f}%) write={st['write']} out={st['out']}"
              f" ${st['cost']:.4f} {st['sec']}s lore={lore_tokens}t{delta}",
              flush=True)
        stats.append(st)
        if i in PAUSES:
            print(f"-- idle {PAUSES[i]}s (dream) --", flush=True)
            time.sleep(PAUSES[i])

    print("-- final dream wait --", flush=True)
    time.sleep(10)
    cursor_p = DATA_ROOT / args.session / "dreamer" / "cursor.json"
    deadline = time.time() + 120
    while time.time() < deadline:
        if cursor_p.is_file() and \
                json.loads(cursor_p.read_text())["next_turn"] >= len(BEATS) - 1:
            break
        time.sleep(3)

    _summary(args.session, stats, lore_log, total_cost, cursor_p, history)


def _summary(session: str, stats: list, lore_log: list, total_cost: float,
             cursor_p: pathlib.Path, history: list) -> None:
    print("=" * 30, "SUMMARY", "=" * 30, flush=True)
    hits = [s["cached"] / s["prompt"] for s in stats[1:] if s["prompt"]]
    churn = [r for r in lore_log if r["added"] or r["dropped"]]
    print(f"turns={len(stats)} total_cost=${total_cost:.3f} "
          f"avg_cache_hit(t2+)={sum(hits) / len(hits) * 100:.1f}%", flush=True)
    print(f"lore churn turns={len(churn)}/{len(lore_log)} "
          f"max_active_tokens={max(r['tokens'] for r in lore_log)}", flush=True)
    for r in churn:
        st = stats[r["turn"] - 1]
        hit = st["cached"] / st["prompt"] * 100 if st["prompt"] else 0
        print(f"  T{r['turn']:02d} hit={hit:.0f}% write={st['write']} "
              f"+{r['added']} -{r['dropped']}", flush=True)

    print("CURSOR:", cursor_p.read_text() if cursor_p.is_file() else "없음")
    for kind in ("facts", "commits", "actors", "episodes"):
        d = DATA_ROOT / session / kind
        rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] \
            if d.is_dir() else []
        print(f"--- {kind} ({len(rows)}) ---", flush=True)
        for row in rows:
            print(json.dumps(row, ensure_ascii=False)[:260], flush=True)

    sys.path.insert(0, str(ROOT))
    from dreaming.storage import JsonDirStorage
    from dreaming.store import MemoryStore
    from dreaming.sync import render_knowledge
    print("--- render_knowledge ---", flush=True)
    print(render_knowledge(MemoryStore(JsonDirStorage(DATA_ROOT), session)),
          flush=True)
    for t in PROBES:
        print(f"--- PROBE T{t+1} reply ---", flush=True)
        print(history[2 * t + 1]["content"][:400], flush=True)


if __name__ == "__main__":
    main()
