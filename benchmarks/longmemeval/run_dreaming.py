"""LongMemEval(oracle)을 Dreaming으로 — 실파이프라인 관통.

기존 evaluator(ChromaDB 검색 비교)와 달리 실제 파이프라인을 관통한다:
haystack 세션을 SyncPath에 턴 단위로 기록 → Dreamer가 사실 추출 →
render_knowledge 주입 상태로 QA → 이진 judge.

usage:
    python3 -m benchmarks.longmemeval.run_dreaming --limit 50
    python3 -m benchmarks.longmemeval.run_dreaming --limit 50 --variant none
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib

import httpx

from benchmarks.longmemeval.download import load_longmemeval
from dreaming.dreamer import Dreamer
from dreaming.llm import OpenAICompatLLM
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, render_knowledge

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "dreaming_data" / "eval"
UPSTREAM = os.environ.get("DREAMING_EVAL_UPSTREAM",
                          "https://openrouter.ai/api/v1")
QA_MODEL = os.environ.get("DREAMING_EVAL_MODEL", "deepseek/deepseek-v4-pro")
DREAM_MODEL = os.environ.get("DREAMING_EVAL_DREAMER",
                             "google/gemini-3-flash-preview")
JUDGE_MODEL = os.environ.get("DREAMING_EVAL_JUDGE",
                             "anthropic/claude-sonnet-4.5")
DREAM_CAP = 30


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def _chat(client: httpx.Client, model: str, system: str, user: str,
          max_tokens: int = 400) -> str:
    r = client.post("/chat/completions", json={
        "model": model, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def _judge(client: httpx.Client, qtype: str, question: str, answer: str,
           prediction: str) -> bool:
    if qtype.endswith("_abs") or "abstention" in qtype:
        p = (f"질문: {question}\n모델 답: {prediction}\n\n이 질문은 대화에 근거가 "
             "없다. 모델이 근거 없음을 인정하거나 확답을 피했는가? Y/N만.")
    else:
        p = (f"질문: {question}\n정답: {answer}\n모델 답: {prediction}\n\n"
             "모델 답이 정답과 사실상 일치하는가? Y/N만.")
    return _chat(client, JUDGE_MODEL, "Y/N 한 글자만.",
                 p).strip().upper().startswith("Y")


async def _ingest_and_dream(data_root: pathlib.Path, session: str,
                            sessions: list) -> str:
    storage = JsonDirStorage(data_root)
    sp = SyncPath(storage, session)
    history: list = []
    for sess in sessions:
        for i in range(0, len(sess) - 1, 2):
            if sess[i].get("role") != "user":
                continue
            history.append({"role": "user", "content": sess[i]["content"]})
            _, v = sp.process(list(history))
            reply = sess[i + 1]["content"] if i + 1 < len(sess) else ""
            sp.record_response(v, list(history), reply)
            history.append({"role": "assistant", "content": reply})
    llm = OpenAICompatLLM(UPSTREAM, _key(), DREAM_MODEL)
    dreamer = Dreamer(storage, llm)
    for _ in range(DREAM_CAP):
        if not dreamer.has_backlog(session):
            return "ok"
        await dreamer.dream(session)
    return "dream_overflow" if dreamer.has_backlog(session) else "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--variant", choices=("dreaming", "none"),
                    default="dreaming")
    args = ap.parse_args()

    data = load_longmemeval("oracle")[: args.limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"lme-{args.variant}.jsonl"
    done = {json.loads(l)["qid"] for l in out.read_text().splitlines()} \
        if out.exists() else set()
    client = httpx.Client(base_url=UPSTREAM, timeout=180,
                          headers={"Authorization": f"Bearer {_key()}"})
    data_root = ROOT / "dreaming_data" / "lme"
    correct = total = 0
    for q in data:
        qid = q["question_id"]
        if qid in done:
            continue
        knowledge = ""
        status = "ok"
        if args.variant == "dreaming":
            session = f"lme-{qid}"
            status = asyncio.run(_ingest_and_dream(
                data_root, session, q["haystack_sessions"]))
            store = MemoryStore(JsonDirStorage(data_root), session)
            knowledge = render_knowledge(store)
        sys_p = "대화 상대의 과거 대화 기억을 바탕으로 질문에 짧게 답하라."
        if knowledge:
            sys_p += f"\n\n[기억]\n{knowledge}"
        pred = _chat(client, QA_MODEL, sys_p, q["question"])
        ok = _judge(client, q["question_type"], q["question"],
                    q["answer"], pred)
        correct += ok
        total += 1
        with out.open("a") as f:
            f.write(json.dumps({"qid": qid, "type": q["question_type"],
                                "ok": ok, "status": status,
                                "pred": pred[:200]},
                               ensure_ascii=False) + "\n")
        print(f"{qid} {q['question_type']:24s} {'O' if ok else 'X'} "
              f"({correct}/{total})", flush=True)
    print(f"[{args.variant}] accuracy {correct}/{total}")


if __name__ == "__main__":
    main()
