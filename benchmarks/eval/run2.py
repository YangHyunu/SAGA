"""80턴 디렉터 벤치 드라이버 (EVAL2 §2·§3·§4).

진행 40턴: 디렉터가 유저 발화를 생성하며 매 턴 사실을 추출해 원장에 쌓는다.
지식갱신 이벤트(값 변경) 2개는 디렉터 지시문으로 강제 발생.
평가 40턴: 가시 창 밖으로 evict된 사실만 프로브로 재질문 (recall/relation/
false/update/recent), 채점은 오라클+judge 이중, 미스는 저장/활용 실패 분해.
--runs N 반복 (진행·필러만 변동, report2가 mean±std 집계).

와이어는 뮈토스 6.2 프리셋을 preset2wire로 조립 — 매 요청 check_wire_shape
게이트. 모델·엔드포인트는 env로 오버라이드 (확정 설정: 나레이터 V4 Pro,
디렉터 Gemini 3 Flash, judge Sonnet 4.5).

usage: python3 -m benchmarks.eval.run2 <preset.risup> <card.json> dreaming \
           --session v2a --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple

import httpx

from benchmarks.eval.director import (Ledger, LlmFn, extract_facts,
                                      make_false_premise, make_probe,
                                      probe_plan)
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.preset2wire import assemble, decode_risup, reformat
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval.variants import retrieve_turns

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "dreaming_data"
EVAL_DIR = DATA / "eval"
PROXY = os.environ.get("DREAMING_EVAL_PROXY", "http://127.0.0.1:8790")
UPSTREAM = os.environ.get("DREAMING_EVAL_UPSTREAM",
                          "https://openrouter.ai/api/v1")
MODEL = os.environ.get("DREAMING_EVAL_MODEL", "deepseek/deepseek-v4-pro")
JUDGE_MODEL = os.environ.get("DREAMING_EVAL_JUDGE",
                             "anthropic/claude-sonnet-4.5")
DIRECTOR_MODEL = os.environ.get("DREAMING_EVAL_DIRECTOR",
                                "google/gemini-3-flash-preview")
PROGRESS_TURNS = 40
EVAL_TURNS = 40
TRIM_TOKENS = 12000
UPDATE_EVENTS = (12, 28)      # 지식갱신 강제 턴 (진행 구간)
MAX_TOKENS = 1000             # 실사용 응답 길이 근사 (EVAL2 충실도 보강)

# 확정 토글: RP 모드·한국어·성인 지침 ON·중립 렌더링 프리필 ON, 나머지 기본
# select은 옵션 인덱스 문자열: response_language 1=🇰🇷 한국어, execution_mode 0=💬 RP.
# 나머지는 불리언. 미설정 전역변수는 "null"이라(preset2wire.UNSET) 실행 모드는
# 반드시 명시해야 tis:: 분기가 걸린다.
TOGGLES = {"mythos_response_language": "1",
           "mythos_execution_mode": "0",
           "mythos_genre_ero": "1",
           "mythos_mature_content_guidance": "1",
           "mythos_domain_neutral_rendering_prefill": "1"}


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def _mk_llm(model: str, temperature: float) -> LlmFn:
    client = httpx.Client(base_url=UPSTREAM, timeout=120,
                          headers={"Authorization": f"Bearer {_key()}"})

    def call(system: str, user: str) -> str:
        r = client.post("/chat/completions", json={
            "model": model, "max_tokens": 400, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""
    return call


def make_judge_llm() -> LlmFn:
    return _mk_llm(JUDGE_MODEL, 0.0)


def make_director_llm() -> LlmFn:
    return _mk_llm(DIRECTOR_MODEL, 0.7)


def _count(text: str) -> int:
    return int(len(text) / 2.5) + 1        # cardsim 근사와 동일


def token_trim(history: List[Dict], budget: int,
               count_fn: Callable[[str], int] = _count
               ) -> Tuple[List[Dict], int]:
    """토큰 예산 기반 트림 — 실클라이언트 maxContext 절단 근사 (pair 경계).

    반환: (윈도우, 잘린 pair 수 = 창 시작 턴). greeting 등 첫 user 이전
    메시지는 트림이 시작되는 순간 통째로 떨어진다 (캡처 실측과 동일).
    """
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    if history and history[-1]["role"] == "user":
        starts.pop()
    total_pairs = len(starts)
    keep = 0
    for k in range(total_pairs, 0, -1):
        seg = history[starts[k - 1]:]
        if sum(count_fn(m["content"]) for m in seg) > budget:
            break
        keep = total_pairs - k + 1
    if keep >= total_pairs:
        return history, 0
    cut = starts[total_pairs - keep] if keep else (
        len(history) - 1 if history and history[-1]["role"] == "user"
        else len(history))
    return history[cut:], total_pairs - keep


def probe_schedule(total: int) -> List[Optional[str]]:
    """평가 구간 턴별 프로브 유형 배치 — 필러를 사이에 끼워 자연스럽게."""
    seq = ["recall", None, "recall", "relation", None, "false", "recall", None,
           "update", "recall", None, "false", "relation", None, "recall",
           "recent", None, "recall", "false", None, "update", "recall", None,
           "relation", "recall", None, "false", "recent", None, "recall"]
    return (seq * ((total // len(seq)) + 1))[:total]


def build_wire(preset: Dict, card: Dict, window: List[Dict],
               retrieval_block: str = "") -> List[Dict]:
    """뮈토스 조립 + reformater. retrieval 변형은 마지막 user에 발췌 prepend."""
    window = [dict(m) for m in window]
    if retrieval_block:
        for m in reversed(window):
            if m["role"] == "user":
                m["content"] = retrieval_block + "\n\n" + m["content"]
                break
    msgs = assemble(preset, TOGGLES, window,
                    card={"description": card.get("description", ""),
                          "persona": card.get("persona", ""),
                          "lore": card.get("lore", []),
                          "globalnote": card.get("globalnote", "")},
                    char_name=card.get("name", ""),
                    user_name=card.get("user_name", ""))
    return reformat(msgs)


def _call_upstream(variant: str, session: str, key: str,
                   msgs: List[Dict]) -> Dict:
    t0 = time.time()
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs})
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={"model": MODEL, "max_tokens": MAX_TOKENS,
                             "messages": msgs, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    cached = det.get("cached_tokens", 0) or u.get("prompt_cache_hit_tokens", 0)
    return {"reply": d["choices"][0]["message"]["content"],
            "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}


_DIRECT_SYS = ("너는 RP에서 유저(1인칭{user}) 역할을 연기한다. 작품 "
               "설정과 직전 장면에 자연스럽게 이어지는 유저 발화 하나만 출력. "
               "3문장 이내, 반말 채팅체, 메타 발언 금지.")
_UPDATE_BEAT = ("이번 발화에서 이전에 언급된 수치나 소지품 상태를 명확히 바꾸는 "
                "행동을 한다 (지불, 획득, 분실 중 하나). 새 값이 드러나게.")


def _load_json(path: str) -> Dict:
    return json.loads(pathlib.Path(path).read_text())


def run_once(preset_path: str, card_path: str, variant: str, session: str,
             run_no: int, trim_tokens: int, reroll_at: List[int],
             edit_at: List[int], ttl_wait: bool) -> Dict:
    preset = decode_risup(preset_path)
    card = _load_json(card_path)
    key = _key()
    director = make_director_llm()
    judge = make_judge_llm()
    ledger = Ledger()
    history: List[Dict] = []
    if card.get("greeting"):
        history.append({"role": "assistant", "content": card["greeting"]})
    last_reply = card.get("greeting") or "(첫 장면)"
    few_shot = "\n".join(f"- {s}" for s in card.get("style_examples", [])[:8])
    uname = card.get("user_name", "")
    dir_sys = _DIRECT_SYS.format(user=f", 이름 {uname}" if uname else "·무명")
    if few_shot:
        dir_sys += f"\n[실제 유저 발화 예시 — 문체 참고]\n{few_shot}"
    turns, probes = [], []
    sched = probe_schedule(EVAL_TURNS)

    for i in range(PROGRESS_TURNS + EVAL_TURNS):
        _, win_start = token_trim(history, trim_tokens)
        ptype = sched[i - PROGRESS_TURNS] if i >= PROGRESS_TURNS else None
        fact, wrong = None, ""
        t_dir = time.time()
        if ptype:
            plan = probe_plan(ledger, win_start, {ptype: 1})
            if plan:
                _, fact = plan[0]
        if fact is not None and ptype == "false":
            utext, wrong = make_false_premise(director, fact)
        elif fact is not None:
            utext = make_probe(director, fact)
        else:
            ptype = None                       # eligible 없으면 필러로 강등
            beat = _UPDATE_BEAT if i in UPDATE_EVENTS else "자연스럽게 이어간다."
            utext = director(
                dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
                f"[직전 캐릭터 응답]\n{last_reply[-800:]}\n[지시]\n{beat}")
        dir_sec = round(time.time() - t_dir, 1)

        history.append({"role": "user", "content": utext})
        window, win_start = token_trim(history, trim_tokens)
        block = ""
        if variant == "retrieval":
            ex = retrieve_turns(history, utext)
            if ex:
                block = "[과거 대화 발췌]\n" + "\n---\n".join(ex)
        use_window = history if variant == "vanilla" else window
        msgs = build_wire(preset, card, use_window, retrieval_block=block)
        bad = check_wire_shape(msgs)
        if bad:
            raise SystemExit(f"와이어 형태 위반 T{i + 1}: {bad}")
        st = _call_upstream(variant, session, key, msgs)
        history.append({"role": "assistant", "content": st["reply"]})
        last_reply = st["reply"]

        if i in reroll_at:                     # 리롤: 동일 요청 재전송
            st2 = _call_upstream(variant, session, key, msgs)
            history[-1] = {"role": "assistant", "content": st2["reply"]}
            last_reply = st2["reply"]
            st["cost"] += st2["cost"]
        if i in edit_at:                       # 수정: user 텍스트 바꿔 재전송
            history[-2]["content"] = utext + " (아니, 정정할게.)"
            window, _ = token_trim(history[:-1], trim_tokens)
            use_window = history[:-1] if variant == "vanilla" else window
            msgs2 = build_wire(preset, card, use_window)
            st3 = _call_upstream(variant, session, key, msgs2)
            history[-1] = {"role": "assistant", "content": st3["reply"]}
            last_reply = st3["reply"]
            st["cost"] += st3["cost"]

        t_ext = time.time()
        if ptype is None:
            ledger.add(extract_facts(director, utext, st["reply"], i))
        ext_sec = round(time.time() - t_ext, 1)

        turns.append({"turn": i, "user": utext, **st,
                      "sec_director": dir_sec, "sec_extract": ext_sec,
                      "ptype": ptype})
        if fact is not None:
            o = oracle_pass(st["reply"], fact.value)
            j = judge_pass(judge, ptype, fact.text, fact.value, utext,
                           st["reply"], wrong_value=wrong)
            miss = "-"
            if not j["pass"] and variant == "dreaming":
                miss = decompose_miss(DATA, session, fact)
            probes.append({"turn": i, "ptype": ptype, "fact": fact.text,
                           "value": fact.value, "question": utext,
                           "reply": st["reply"], "oracle": o,
                           "judge": j["pass"], "why": j["why"],
                           "miss_cause": miss,
                           "distance_turns": i - fact.turn})
        if variant == "dreaming" and i in (PROGRESS_TURNS // 2, PROGRESS_TURNS):
            time.sleep(12)                     # 꿈 트리거 (유휴 Dreamer)
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)

    passed = sum(1 for p in probes if p["judge"])
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes), "judge_pass": passed,
                         "cost": round(sum(t["cost"] for t in turns), 4)}}
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"v2-{session}-run{run_no}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("card")
    ap.add_argument("variant",
                    choices=("dreaming", "vanilla", "trim", "retrieval"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--trim-tokens", type=int, default=TRIM_TOKENS)
    ap.add_argument("--reroll-at", default="18,33")
    ap.add_argument("--edit-at", default="25")
    ap.add_argument("--ttl-wait", action="store_true")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    reroll = [int(x) for x in args.reroll_at.split(",") if x]
    edit = [int(x) for x in args.edit_at.split(",") if x]
    for n in range(args.runs):
        sess = f"{args.session}-r{n}"
        d = DATA / sess
        if d.exists():
            if not args.reset:
                raise SystemExit(f"{d} 이미 있음 — --reset")
            shutil.rmtree(d)
        r = run_once(args.preset, args.card, args.variant, sess, n,
                     args.trim_tokens, reroll, edit, args.ttl_wait)
        t = r["totals"]
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} ${t['cost']}",
              flush=True)


if __name__ == "__main__":
    main()
