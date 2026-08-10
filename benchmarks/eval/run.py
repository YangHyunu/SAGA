# [v1 하네스] 현행은 run2 계열 — 이 파일은 구 테스트·PR 호환용으로 보존.
"""평가 드라이버 — 한 변형을 대본으로 완주하고 결과 JSON을 남긴다 (스펙 §9).

usage:
    python3 -m benchmarks.eval.run <card.charx> dreaming --session ev1
    python3 -m benchmarks.eval.run <card.charx> trim --session ev1-trim \
        --script dreaming_data/eval/script-ev1.json

첫 실행(--script 없음)이 시뮬레이터로 유저 턴을 생성해 동결하고, 대조군은
--script로 같은 텍스트를 재생한다. dreaming만 프록시를 경유하며 idle pause로
꿈을 트리거한다. 결과·대본은 dreaming_data/eval/ (gitignored — 커밋 금지).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time
from typing import Dict, List

import httpx

from benchmarks.cardsim.lorebook import load_card
from benchmarks.eval.oracle import score_reply
from benchmarks.eval.script import (BEATS, PAUSES, PROBES, freeze_script,
                                    load_script)
from benchmarks.eval.variants import prepare_request

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = os.environ.get("DREAMING_EVAL_PROXY", "http://127.0.0.1:8787")
UPSTREAM = "https://openrouter.ai/api/v1"
MODEL = "anthropic/claude-haiku-4.5"
SIM_MODEL = "google/gemini-2.5-flash"
USER_NAME = "한결"
VARIANTS = ("dreaming", "vanilla", "trim", "retrieval")


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def build_result(variant: str, session: str, model: str,
                 turns: List[Dict]) -> Dict:
    probes = []
    for p in PROBES:
        reply = turns[p.turn]["reply"] if p.turn < len(turns) else ""
        probes.append({**score_reply(reply, p), "turn": p.turn,
                       "reply": reply})
    hits = [t["cached"] / t["prompt"] for t in turns[1:] if t["prompt"]]
    totals = {
        "cost": round(sum(t["cost"] for t in turns), 4),
        "avg_hit_t2": round(sum(hits) / len(hits) * 100, 1) if hits else 0.0,
        "avg_sec": round(sum(t["sec"] for t in turns) / len(turns), 1)
        if turns else 0.0,
        "oracle_full": sum(1 for p in probes
                           if p["hit"] == "full" and not _is_recall(p)),
        "oracle_partial": sum(1 for p in probes
                              if p["hit"] == "partial" and not _is_recall(p)),
        "recall": next((f'{p["matched"]}/{p["total"]}' for p in probes
                        if _is_recall(p)), "-"),
    }
    return {"variant": variant, "session": session, "model": model,
            "turns": turns, "probes": probes, "totals": totals}


def _is_recall(scored: Dict) -> bool:
    return scored["turn"] == next(p.turn for p in PROBES if p.recall)


def _gen_user(client: httpx.Client, setting: str, last_reply: str,
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
    except Exception as e:                     # noqa: BLE001 — 벤치는 계속 돈다
        print(f"  (sim fail: {e} — 비트 원문 사용)", flush=True)
        return beat


def _call(variant: str, session: str, key: str, msgs: List[Dict]) -> Dict:
    t0 = time.time()
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session},
                       json={"model": MODEL, "max_tokens": 300,
                             "messages": msgs})
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "max_tokens": 300,
                             "messages": msgs, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    return {"reply": d["choices"][0]["message"]["content"],
            "prompt": u.get("prompt_tokens", 0),
            "cached": det.get("cached_tokens", 0),
            "write": det.get("cache_write_tokens", 0),
            "cost": u.get("cost", 0.0),
            "sec": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("card")
    ap.add_argument("variant", choices=VARIANTS)
    ap.add_argument("--session", required=True)
    ap.add_argument("--script", help="동결 대본 JSON — 없으면 시뮬레이터 생성")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    sess_dir = DATA / args.session
    if sess_dir.exists():
        if not args.reset:
            raise SystemExit(f"{sess_dir} 이미 있음 — --reset 또는 다른 세션 ID")
        shutil.rmtree(sess_dir)

    card = load_card(args.card, USER_NAME)
    key = _key()
    scripted = {t["turn"]: t["user_text"]
                for t in load_script(args.script)} if args.script else {}
    sim = (None if scripted else
           httpx.Client(base_url=UPSTREAM, timeout=60,
                        headers={"Authorization": f"Bearer {key}"}))

    history: List[Dict] = []
    last_reply = card.greeting or "(첫 장면)"
    turns: List[Dict] = []
    frozen: List[Dict] = []
    for i, beat in enumerate(BEATS):
        utext = scripted.get(i) or _gen_user(
            sim, card.description[:2500], last_reply, beat)
        frozen.append({"turn": i, "user_text": utext})
        history.append({"role": "user", "content": utext})
        msgs = prepare_request(args.variant, card, history)
        st = _call(args.variant, args.session, key, msgs)
        history.append({"role": "assistant", "content": st["reply"]})
        last_reply = st["reply"]
        hit = st["cached"] / st["prompt"] * 100 if st["prompt"] else 0
        print(f"T{i + 1:02d} prompt={st['prompt']} cached={st['cached']}"
              f" ({hit:.0f}%) ${st['cost']:.4f} {st['sec']}s", flush=True)
        turns.append({"turn": i, "user": utext, **st})
        if args.variant == "dreaming" and i in PAUSES:
            print(f"-- idle {PAUSES[i]}s (dream) --", flush=True)
            time.sleep(PAUSES[i])

    if args.variant == "dreaming":
        cursor = sess_dir / "dreamer" / "cursor.json"
        deadline = time.time() + 120
        while time.time() < deadline:
            if cursor.is_file() and \
                    json.loads(cursor.read_text())["next_turn"] >= len(BEATS) - 1:
                break
            time.sleep(3)

    if not args.script:
        freeze_script(EVAL_DIR / f"script-{args.session}.json", frozen)
        print(f"대본 동결 → eval/script-{args.session}.json", flush=True)

    result = build_result(args.variant, args.session, MODEL, turns)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"result-{args.session}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    t = result["totals"]
    print(f"[{args.variant}] full={t['oracle_full']}/5 "
          f"partial={t['oracle_partial']} recall={t['recall']} "
          f"${t['cost']} hit={t['avg_hit_t2']}% {t['avg_sec']}s/turn",
          flush=True)


if __name__ == "__main__":
    main()
