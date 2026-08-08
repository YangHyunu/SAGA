"""디렉터 벤치 드라이버 (EVAL2 §2·§3·§4). 기본 80턴, 파일럿은 짧게.

디렉터가 유저(렌) 역할로 전 구간을 자연스럽게 진행하며 매 턴 사실을 추출해
원장에 쌓는다. 지식갱신 이벤트(값 변경) 2개는 디렉터 지시문으로 강제 발생.
--probe-every(기본 10)턴마다 발화 하나가 가시 창 밖으로 evict된 과거 사실을
슬며시 화제로 되짚는다 — 시험조 금지, 유형은 회전
(recall/relation/false/update/recent). 채점은 오라클+judge 이중, 미스는
저장/활용 실패 분해. --turns로 줄여 파일럿을 돌린다.
--runs N 반복 (진행·필러만 변동, report2가 mean±std 집계).

vanilla 변형만 트림 없이 전체 히스토리를 보낸다 — 프로브는 트림 창 기준으로
뽑히므로, vanilla가 틀리면 그건 정보 부재가 아니라 lost-in-the-middle이다.

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
import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")

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
TURNS = 80
PROBE_EVERY = 10              # 이 간격마다 발화 하나가 과거를 슬며시 되짚는다
TRIM_TOKENS = 12000
UPDATE_EVENTS = (12, 28)      # 지식갱신 강제 턴
# 캡처에서 RisuAI가 실제로 보낸 값이 4000이다 (capture-mythos req-001).
# 실측 완성 평균은 771토큰이라 캡이 물리지 않는다 — 절단은 기억 실패로
# 오인되는 교란이라 finish_reason을 턴마다 기록해 0%임을 증명한다.
MAX_TOKENS = 4000

# 확정 토글: RP 모드·한국어·성인 지침 ON·중립 렌더링 프리필 ON, 나머지 기본
# select은 옵션 인덱스 문자열: response_language 1=🇰🇷 한국어, execution_mode 0=💬 RP.
# 나머지는 불리언. 미설정 전역변수는 "null"이라(preset2wire.UNSET) 실행 모드는
# 반드시 명시해야 tis:: 분기가 걸린다.
TOGGLES = {"mythos_response_language": "1",           # 🇰🇷 한국어
           "mythos_execution_mode": "0",              # 💬 RP
           "mythos_user_persona_usage": "0",          # 🙋 사용 — 안 켜면 슬롯이 통째로 빈다
           "mythos_bot_structure": "0",               # 💬 캐릭터 중심
           # select 토글은 미설정이면 사이드바 SelectInput이 바인딩하며 첫 옵션
           # 인덱스를 써 넣는다. 프리셋의 templateDefaultVariables는 여기 안
           # 먹는다 — getChatVar 전용이고 tis는 getGlobalChatVar를 본다
           # (chatVar.svelte.ts:15 vs 35, parser.svelte.ts:1284).
           "mythos_user_character_authorship": "0",   # 🛡️ 보호 — 캡처 req-005 확인
           "mythos_input_authority": "0",             # 🔨 사실 확정
           "mythos_prose_register": "0",              # 🤷 미지정
           "mythos_narrative_pov": "0",               # 🤷 자율
           "mythos_narrative_pacing": "0",            # 🤷 자율
           "mythos_response_length_band": "0",        # 🤷 미지정
           "mythos_size_scenario": "0",               # 🤷 미지정
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
        data = r.json()
        u = data.get("usage") or {}
        call.cost += u.get("cost") or 0.0      # 부대비용도 잰다 — 나레이터만
        call.calls += 1                        # 재면 총비용을 과소보고한다
        return data["choices"][0]["message"]["content"] or ""
    call.cost, call.calls = 0.0, 0
    return call


def make_judge_llm() -> LlmFn:
    return _mk_llm(JUDGE_MODEL, 0.0)


def make_director_llm() -> LlmFn:
    return _mk_llm(DIRECTOR_MODEL, 0.7)


def _count(text: str) -> int:
    # RisuAI reverse_proxy 기본 토크나이저와 동일 (tokenizer.ts:105-133 →
    # o200k_base). len/2.5 근사는 한국어를 ~40% 과소평가해 12K 예산에서
    # eviction이 아예 안 일어났다 (파일럿 실측 18,917 vs 근사 11,816).
    return len(_ENC.encode(text))


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


def probe_schedule(total: int, every: int = PROBE_EVERY) -> List[Optional[str]]:
    """턴별 프로브 배치 — every턴마다 1번, 나머지는 전부 자연 진행.

    진행/평가 구간 구분 없이 렌이 total턴을 자연스럽게 보내고, every턴째
    발화만 과거를 슬며시 화제로 끌어들인다. 유형은 회전 — 30턴 파일럿이면
    recall·relation·false 3개, 80턴 본런이면 회전 두 바퀴째까지 돈다.
    """
    rotation = ["recall", "relation", "false", "update", "recent"]
    out: List[Optional[str]] = [None] * total
    k = 0
    for i in range(every - 1, total, every):
        out[i] = rotation[k % len(rotation)]
        k += 1
    return out


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
                          "system_prompt": card.get("system_prompt", ""),
                          "replace_globalnote":
                              card.get("replace_globalnote", ""),
                          "authornote": card.get("authornote", ""),
                          "post_everything": card.get("post_everything", "")},
                    char_name=card.get("name", ""),
                    user_name=card.get("user_name", ""))
    # 캡처 확인: Custom API 경로는 hasFullSystemPrompt + requiresAlternateRole
    # 해제로 동작한다 — 중간 system(req-005의 2534자)도, 연속 user(req-006)도
    # 그대로 실린다. 둘 다 끄지 않으면 우리 와이어만 다른 모양이 된다.
    return reformat(msgs, fold_mid_system=False, alternate=False)


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
    choice = d["choices"][0]
    return {"reply": choice["message"]["content"],
            "finish": choice.get("finish_reason", ""),
            "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "completion": u.get("completion_tokens", 0),
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}


_DIRECT_SYS = ("너는 RP에서 유저(1인칭{user}) 역할을 연기한다. 작품 "
               "설정과 직전 장면에 자연스럽게 이어지는 유저 발화 하나만 출력. "
               "3문장 이내, 반말 채팅체, 메타 발언 금지.\n"
               "[작품 설정]은 배경 이해용이다 — 대화에서 아직 드러나지 않은 "
               "정보(호칭·직함·이름·과거사·신체 특징)를 네가 먼저 입에 올리지 "
               "마라. 상대가 말해주기 전까지 모르는 사람으로 산다. "
               "(파일럿 실측: '신녀님' 호칭을 대화에 나온 적 없는데 선취했다)")
_UPDATE_BEAT = ("이번 발화에서 이전에 언급된 수치나 소지품 상태를 명확히 바꾸는 "
                "행동을 한다 (지불, 획득, 분실 중 하나). 새 값이 드러나게.")
# 5턴마다 이야기를 미는 지시 — 없으면 디렉터가 같은 장면을 맴돈다
# (30턴 파일럿 실측: 한 장소 하룻밤에서 정체). 회전이라 런마다 결이 달라진다.
_BEATS = ("장면이나 장소를 바꾸는 행동을 한다 — 밖으로 나가자고 하거나, "
          "다른 공간으로 옮기거나, 산책을 청한다.",
          "새로운 화제나 작은 사건을 꺼낸다 — 마을, 소문, 상대의 과거, "
          "앞으로의 계획 중 하나.",
          "시간을 흘려보낸다 — 다음 날이나 몇 시간 뒤로 넘어갔음이 발화에 "
          "드러나게 한다.",
          "가벼운 갈등이나 의견 차이를 만든다 — 금방 풀 수 있는 수준으로.",
          "구체적인 수치·이름·약속이 나올 만한 행동을 한다 — 거래, 날짜 잡기, "
          "무언가를 세거나 값을 치르기.")


def pick_beat(i: int) -> str:
    """턴 i의 필러 지시. UPDATE_EVENTS > 5턴 주기 비트 > 평서 진행."""
    if i in UPDATE_EVENTS:
        return _UPDATE_BEAT
    if i % 5 == 4:
        return _BEATS[(i // 5) % len(_BEATS)]
    return "자연스럽게 이어간다."


def recent_dialogue(history: List[Dict], pairs: int = 3) -> str:
    """디렉터에게 주는 최근 대화 — 직전 응답 하나만 주면 맥락 없이 맴돈다."""
    tail = history[-(pairs * 2):]
    return "\n\n".join(
        f"[{'렌' if m['role'] == 'user' else '캐릭터'}]\n{m['content'][-600:]}"
        for m in tail)


def _load_json(path: str) -> Dict:
    return json.loads(pathlib.Path(path).read_text())


def run_once(preset_path: str, card_path: str, variant: str, session: str,
             run_no: int, trim_tokens: int, reroll_at: List[int],
             edit_at: List[int], ttl_wait: bool,
             total_turns: int = TURNS,
             probe_every: int = PROBE_EVERY) -> Dict:
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
    sched = probe_schedule(total_turns, probe_every)

    for i in range(total_turns):
        _, win_start = token_trim(history, trim_tokens)
        ptype = sched[i]
        fact, wrong = None, ""
        t_dir = time.time()
        if ptype:
            plan = probe_plan(ledger, win_start, {ptype: 1})
            if plan:
                _, fact = plan[0]
        if fact is not None and ptype == "false":
            utext, wrong = make_false_premise(director, fact,
                                              scene=last_reply, style=few_shot)
        elif fact is not None:
            utext = make_probe(director, fact,
                               scene=last_reply, style=few_shot)
        else:
            ptype = None                       # eligible 없으면 필러로 강등
            ctx = recent_dialogue(history) or f"[캐릭터]\n{last_reply[-600:]}"
            utext = director(
                dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
                f"[최근 대화]\n{ctx}\n[지시]\n{pick_beat(i)}")
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
            o = oracle_pass(st["reply"], fact.value, wrong_value=wrong,
                            char_name=card.get("name", ""))
            j = judge_pass(judge, ptype, fact.text, fact.value, utext,
                           st["reply"], wrong_value=wrong)
            miss = "-"
            # judge 파싱 실패(None)는 미스가 아니다 — 원인 분해에서 뺀다
            if j["pass"] is False and variant == "dreaming":
                miss = decompose_miss(DATA, session, fact)
            probes.append({"turn": i, "ptype": ptype, "fact": fact.text,
                           "value": fact.value, "wrong": wrong,
                           "question": utext,
                           "reply": st["reply"], "oracle": o,
                           "judge": j["pass"], "why": j["why"],
                           "miss_cause": miss,
                           "distance_turns": i - fact.turn})
        if variant == "dreaming" and i in (total_turns // 3,
                                           2 * total_turns // 3):
            time.sleep(12)                     # 꿈 트리거 (유휴 Dreamer)
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)

    passed = sum(1 for p in probes if p["judge"] is True)
    unparsed = sum(1 for p in probes if p["judge"] is None)
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes), "judge_pass": passed,
                         "judge_unparsed": unparsed,
                         "oracle_pass": sum(1 for p in probes if p["oracle"]),
                         # 절단은 기억 실패로 오인된다 — 0인지 매 런 확인한다
                         "truncated": sum(1 for t in turns
                                          if t.get("finish") == "length"),
                         "cost": round(sum(t["cost"] for t in turns), 4),
                         "cost_director": round(director.cost, 4),
                         "director_calls": director.calls,
                         "cost_judge": round(judge.cost, 4),
                         "judge_calls": judge.calls}}
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
    ap.add_argument("--turns", type=int, default=TURNS)
    ap.add_argument("--probe-every", type=int, default=PROBE_EVERY)
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
                     args.trim_tokens, reroll, edit, args.ttl_wait,
                     args.turns, args.probe_every)
        t = r["totals"]
        grand = t["cost"] + t.get("cost_director", 0) + t.get("cost_judge", 0)
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} "
              f"나레이터 ${t['cost']} + 디렉터 ${t.get('cost_director', 0)} "
              f"+ judge ${t.get('cost_judge', 0)} = ${round(grand, 4)}",
              flush=True)


if __name__ == "__main__":
    main()
