"""디렉터 벤치 드라이버 (EVAL2 §2·§3·§4). 기본 80턴, 파일럿은 짧게.

디렉터가 유저(렌) 역할로 전 구간을 자연스럽게 진행하며 매 턴 사실을 추출해
원장에 쌓는다. 지식갱신 이벤트(값 변경) 2개는 디렉터 지시문으로 강제 발생.
--probe-every(기본 10)턴마다 발화 하나가 가시 창 밖으로 evict된 과거 사실을
슬며시 화제로 되짚는다 — 시험조 금지, 유형은 회전
(recall/relation/false/update/recent). 채점은 오라클+judge 이중, 미스는
저장/활용 실패 분해. --turns로 줄여 파일럿을 돌린다.
--runs N 반복 (진행·필러만 변동, report2가 mean±std 집계).

vanilla·dreaming 변형은 트림 없이 전체 히스토리를 보낸다 — 프로브는 트림 창
기준으로 뽑히므로, vanilla가 틀리면 그건 정보 부재가 아니라
lost-in-the-middle이다. dreaming은 창 관리(압축)가 프록시 책임이라 벤치가
미리 자르지 않는다. hypa는 RisuAI HypaV3(뮈토스 하이파 V5 설정)를 그대로
돌려 스스로 요약·절단하므로 벤치의 token_trim을 쓰지 않는다 — 요약 블록은
프리셋 memory 카드(시스템 프롬프트 한가운데) 자리에 실린다.

와이어는 뮈토스 6.2 프리셋을 preset2wire로 조립 — 매 요청 check_wire_shape
게이트. 모델·엔드포인트는 env로 오버라이드 (확정 설정: 나레이터 V4 Pro,
디렉터 Gemini 3 Flash, judge Sonnet 4.5).

usage: python3 -m benchmarks.eval.run2 <preset.risup> <card.json> dreaming \
           --session v2a --runs 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval.config import (DATA, EVAL_DIR, HYPA_EXPORT, MAX_CONTEXT,
                                    MAX_RUN_REROLLS, MAX_TOKENS, MODEL,
                                    NPC_EVENT_RETRY, NPC_EVENT_TURN, NPC_NAME,
                                    PROBE_EVERY, TOGGLES, TURNS, UPDATE_EVENTS)
from benchmarks.eval import prompts
# 별칭 재노출(이 파일 안에서는 안 쓰임) — 기존 테스트(run2._DIRECT_SYS 등)가
# 이 이름으로 내용을 검증한다. 실제 호출부는 override_from 반영을 위해
# prompts.X(점 접근)를 쓴다.
from benchmarks.eval.prompts import (BEATS as _BEATS,  # noqa: F401
                                     DIRECT_SYS as _DIRECT_SYS,  # noqa: F401
                                     NPC_BEAT as _NPC_BEAT,  # noqa: F401
                                     UPDATE_BEAT as _UPDATE_BEAT)  # noqa: F401
# transport 모듈 자체를 쓴다 — run_once의 call_fn 심은
# transport.call_upstream을 기본값으로 참조한다(점 접근, 모듈 속성이라
# monkeypatch.setattr(transport, "call_upstream", ...)가 걸린다). 나머지는
# 별칭 재노출(이 파일 안에서는 일부만 쓰임) — 기존 테스트·스크립트가
# run2._key 등으로 참조한다.
from benchmarks.eval import transport
from benchmarks.eval.transport import (call_upstream as _call_upstream,  # noqa: F401
                                       call_upstream_once as _call_upstream_once,  # noqa: F401
                                       key as _key, make_director_llm,
                                       make_judge_llm, mk_llm as _mk_llm)  # noqa: F401
# 별칭 재노출 — 기존 테스트가 run2._count, run2._FULL_HISTORY 등으로 참조.
from benchmarks.eval.windowing import (FULL_HISTORY as _FULL_HISTORY,
                                       count as _count, hypa_in_window,
                                       token_trim, wire_history)
# reply_flaw는 이 파일 안에서는 안 쓰임(quality.reroll_until_clean 내부용) —
# 기존 테스트가 run2.reply_flaw로 직접 참조한다.
from benchmarks.eval.quality import (abort_reroll_count,
                                     reply_flaw,  # noqa: F401
                                     reroll_until_clean)

from benchmarks.eval.director import (DirFact, Ledger, LlmFn, extract_facts,
                                      make_false_premise, make_probe,
                                      _probe_mentions_fact_object,
                                      probe_plan)
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.preset2wire import assemble, decode_risup, reformat
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval import hypa


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
               memory: str = "") -> List[Dict]:
    """뮈토스 조립 + reformater.

    memory는 hypa 요약 블록 — 프리셋의 memory 카드(promptTemplate[35] 'Past
    Summary') 자리에 들어간다. 즉 chat 히스토리보다 앞, 시스템 프롬프트
    한가운데다 (index.svelte.ts:1429-1443). 캐시 파괴 병리의 구조적 원인이다.
    """
    msgs = assemble(preset, TOGGLES, window, memory=memory,
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


def pick_beat(i: int, npc_due: bool = False) -> str:
    """턴 i의 필러 지시. NPC 이벤트 > UPDATE_EVENTS > 5턴 주기 비트 > 평서."""
    if npc_due:
        return prompts.NPC_BEAT
    if i in UPDATE_EVENTS:
        return prompts.UPDATE_BEAT
    if i % 5 == 4:
        return prompts.BEATS[(i // 5) % len(prompts.BEATS)]
    return "자연스럽게 이어간다."


def recent_dialogue(history: List[Dict], pairs: int = 3) -> str:
    """디렉터에게 주는 최근 대화 — 직전 응답 하나만 주면 맥락 없이 맴돈다."""
    tail = history[-(pairs * 2):]
    return "\n\n".join(
        f"[{'렌' if m['role'] == 'user' else '캐릭터'}]\n{m['content'][-600:]}"
        for m in tail)


def _load_json(path: str) -> Dict:
    return json.loads(pathlib.Path(path).read_text())


def _play_turn(i: int, utext: str, variant: str, history: List[Dict],
               session: str, key: str,
               call: Callable[[str, str, str, List[Dict]], Dict],
               hypa_S: Optional[hypa.HypaSettings], hypa_data: Dict,
               hypa_state: pathlib.Path, fixed_tokens: int, max_context: int,
               trim_budget: int, kept_start_msg: int, preset: Dict,
               card: Dict, reroll_at: List[int], edit_at: List[int],
               turns: List[Dict], total_rerolls: int
               ) -> Tuple[Optional[Dict], List[Dict], Optional[int], int,
                         Dict, int, Optional[str], str]:
    # 턴 1회 실행+리롤: 와이어 조립 → 품질 리롤 → reroll_at/edit_at 재전송.
    # hypa 요약 실패(herr)면 원래 run_once의 break 지점에서 조기 반환한다 —
    # aborted 메시지를 채워 돌려주면 호출부가 루프를 끊는다(break 승격).
    # 반환: (st, use_window, win_start, kept_start_msg, hypa_data,
    #        total_rerolls, last_reply, aborted) — aborted가 있으면
    #        st/last_reply는 None, 호출부는 즉시 break해야 한다.
    history.append({"role": "user", "content": utext})
    memory_text = None
    win_start = None
    if variant == "hypa":
        # hypa는 token_trim이 아니라 자기 slice로 자른다 — window/win_start
        # 는 이 분기에서 안 읽힌다. 매 턴 전체 히스토리 토큰 카운트를
        # 태우는 데드워크라 건너뛴다.
        (memory_text, use_window, kept_start_msg, hypa_data,
         herr) = hypa.hypa_step(history, fixed_tokens, hypa_S, hypa_data,
                                hypa._summarize_call, max_context,
                                MAX_TOKENS)
        hypa_state.parent.mkdir(parents=True, exist_ok=True)
        hypa_state.write_text(json.dumps(hypa_data, ensure_ascii=False,
                                         indent=1))
        if herr:
            # 원본도 요청 자체를 실패시킨다 (hypav3.ts:263-274) — 병리 재현이지
            # 하네스 버그가 아니다. 부분 결과를 저장하고 중단한다.
            turns.append({"turn": i, "user": utext, "hypa_error": herr,
                          "cost": 0.0})
            aborted = f"hypa 요약 불가 T{i + 1}: {herr}"
            return (None, use_window, win_start, kept_start_msg, hypa_data,
                    total_rerolls, None, aborted)
    else:
        window, win_start = token_trim(history, trim_budget)
        use_window = wire_history(variant, history, window)
    msgs = build_wire(preset, card, use_window, memory=memory_text or "")
    bad = check_wire_shape(msgs)
    if bad:
        raise SystemExit(f"와이어 형태 위반 T{i + 1}: {bad}")
    # 품질 리롤: 거부·언어 드리프트·중복 응답(loop)은 실유저가 리롤로
    # 걷어내는 응답이다. 남기면 이후 턴 전체가 오염된다 (디렉터 사칭·
    # 영어 고착·자기표절).
    prior_replies = [m["content"] for m in history[-6:]
                      if m["role"] == "assistant"]
    st, flaw_history = reroll_until_clean(
        lambda: call(variant, session, key, msgs), prior_replies)
    total_rerolls += abort_reroll_count(flaw_history)
    history.append({"role": "assistant", "content": st["reply"]})
    last_reply = st["reply"]

    if i in reroll_at:                     # 리롤: 동일 요청 재전송
        st2 = call(variant, session, key, msgs)
        history[-1] = {"role": "assistant", "content": st2["reply"]}
        last_reply = st2["reply"]
        st["cost"] += st2["cost"]
        st["reply"] = st2["reply"]         # 기록·추출은 히스토리와 같게
    if i in edit_at:                       # 수정: user 텍스트 바꿔 재전송
        history[-2]["content"] = utext + " (아니, 정정할게.)"
        if variant == "hypa":
            # 같은 턴의 재전송 — 요약을 다시 돌리지 않는다 (memo는 인덱스
            # 기반이라 편집에도 불변, 창도 그대로다). token_trim은 여기서
            # 안 쓰이는 데드워크라 건너뛴다.
            use_window = history[:-1][kept_start_msg:]
        else:
            window, _ = token_trim(history[:-1], trim_budget)
            use_window = wire_history(variant, history[:-1], window)
        msgs2 = build_wire(preset, card, use_window,
                           memory=memory_text or "")
        st3 = call(variant, session, key, msgs2)
        history[-1] = {"role": "assistant", "content": st3["reply"]}
        last_reply = st3["reply"]
        st["cost"] += st3["cost"]
        st["reply"] = st3["reply"]

    return (st, use_window, win_start, kept_start_msg, hypa_data,
            total_rerolls, last_reply, "")


def _record_probe(i: int, ptype: Optional[str], fact: Optional[DirFact],
                  wrong: str, utext: str, st: Dict, dir_sec: float,
                  director: LlmFn, ledger: Ledger, judge: LlmFn, card: Dict,
                  variant: str, session: str, kept_start_msg: int,
                  win_start: Optional[int], use_window: List[Dict],
                  turns: List[Dict], probes: List[Dict]) -> None:
    # 프로브 계획+기록: extract_facts로 원장을 키워 미래 프로브 재료를 대고,
    # 이번 턴이 프로브면 오라클+judge 이중 채점을 기록한다.
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
                       "distance_turns": i - fact.turn,
                       # 나레이터가 실제 본 창 기준 — 창내 실패=LITM,
                       # 창밖 실패=eviction. 풀 히스토리 변형은 항상 창내
                       # (dreaming은 프록시가 압축했을 수 있어 상한값이다).
                       # hypa는 token_trim이 아니라 자기 slice로 자른다.
                       "in_window": (
                           variant in _FULL_HISTORY
                           or (hypa_in_window(fact.turn, kept_start_msg,
                                              bool(card.get("greeting")))
                               if variant == "hypa"
                               else fact.turn >= win_start)),
                       # 원본 턴은 evict돼도 서사 반복으로 값이 창 안에
                       # 남을 수 있다 (실측: retrieval "렌" 452회,
                       # "15년" 10회 — night2-deep-analysis.md). 문자열
                       # 완전일치만 본다 — "250"/"이백오십" 같은 한글
                       # 표기 변형은 놓친다 (알려진 한계, 과소탐지 방향).
                       "value_in_window": any(
                           fact.value in m["content"]
                           for m in use_window),
                       # 하드 차단·재시도는 무한루프 위험 — 로깅만.
                       "drift_suspected": not _probe_mentions_fact_object(
                           fact, utext)})


def _collect_totals(variant: str, session: str, run_no: int,
                    prompt_set: str, turns: List[Dict], probes: List[Dict],
                    ledger: Ledger, director: LlmFn, judge: LlmFn,
                    hypa_cost0: float, hypa_truncated0: int,
                    aborted: str) -> Dict:
    # totals 집계: probes/turns에서 판정·비용을 합산해 최종 result를 조립.
    passed = sum(1 for p in probes if p["judge"] is True)
    unparsed = sum(1 for p in probes if p["judge"] is None)
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "prompt_set": prompt_set,
              "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes), "judge_pass": passed,
                         "judge_unparsed": unparsed,
                         "oracle_pass": sum(1 for p in probes if p["oracle"]),
                         # 절단은 기억 실패로 오인된다 — 0인지 매 런 확인한다
                         "truncated": sum(1 for t in turns
                                          if t.get("finish") == "length"),
                         "rerolls": sum(t.get("rerolls", 0) for t in turns),
                         # 리롤 2회로도 못 걷어낸 병리 턴 — 0이어야 한다
                         "flawed": sum(1 for t in turns if t.get("flaw")),
                         "cost": round(sum(t["cost"] for t in turns), 4),
                         "cost_director": round(director.cost, 4),
                         "director_calls": director.calls,
                         "cost_judge": round(judge.cost, 4),
                         "judge_calls": judge.calls,
                         # hypa 요약 콜 — 모듈 전역 누적이라 런 시작 대비 증분
                         "cost_hypa": round(hypa.SUMMARY_COST - hypa_cost0, 4),
                         "hypa_truncated": hypa.SUMMARY_TRUNCATED - hypa_truncated0,
                         "aborted": aborted}}
    return result


def run_once(preset_path: str, card_path: str, variant: str, session: str,
             run_no: int, max_context: int, reroll_at: List[int],
             edit_at: List[int], ttl_wait: bool,
             total_turns: int = TURNS,
             probe_every: int = PROBE_EVERY,
             call_fn: Optional[Callable[[str, str, str, List[Dict]], Dict]]
             = None) -> Dict:
    call = call_fn or transport.call_upstream  # 심(seam) — 오프라인 테스트가 주입
    prompt_set = prompts.active()          # A/B 추적 — 이 런에 실제 쓰인 프롬프트 세트
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
    dir_sys = prompts.DIRECT_SYS.format(user=f", 이름 {uname}" if uname else "·무명")
    if few_shot:
        dir_sys += f"\n[실제 유저 발화 예시 — 문체 참고]\n{few_shot}"
    turns, probes = [], []
    total_rerolls, aborted = 0, ""
    sched = probe_schedule(total_turns, probe_every)
    # hypa 상태 — 요약은 세션 안에서 누적된다 (data.summaries).
    hypa_S = hypa.load_hypa_settings(HYPA_EXPORT) if variant == "hypa" else None
    hypa_data: Dict = {"summaries": []}
    hypa_state = EVAL_DIR / f"hypa-state-{session}.json"
    hypa_cost0 = hypa.SUMMARY_COST
    hypa_truncated0 = hypa.SUMMARY_TRUNCATED
    # 프리셋+카드 고정 비용 — RisuAI가 히스토리 앞에 먼저 태우는 몫
    # (index.svelte.ts:614-618). 빈 히스토리 와이어로 실측한다.
    fixed_tokens = (sum(hypa.tok_chat(m) for m in build_wire(preset, card, []))
                    if variant == "hypa" else 0)
    # trim 예산 = 공유 풀 - 프리셋/카드 고정 비용 - 응답 예약(maxResponse+50)
    # (index.svelte.ts:614,618) — hypa와 같은 실측 패턴, 카운터만 o200k.
    trim_budget = (max_context
                   - sum(_count(m["content"])
                        for m in build_wire(preset, card, []))
                   - MAX_TOKENS - 50) if variant == "trim" else max_context
    kept_start_msg = 0

    for i in range(total_turns):
        ptype = sched[i]
        fact, wrong = None, ""
        t_dir = time.time()
        if ptype:
            plan = probe_plan(ledger, i, {ptype: 1})
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
            # T41에 무조건 한 번, T45까지는 히스토리에 이름 없을 때만 재유도
            npc_due = (i == NPC_EVENT_TURN
                       or (NPC_EVENT_TURN < i <= NPC_EVENT_RETRY
                           and not any(NPC_NAME in m["content"]
                                       for m in history)))
            utext = director(
                dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
                f"[최근 대화]\n{ctx}\n[지시]\n{pick_beat(i, npc_due)}")
        dir_sec = round(time.time() - t_dir, 1)

        (st, use_window, win_start, kept_start_msg, hypa_data,
         total_rerolls, new_last_reply, turn_aborted) = _play_turn(
            i=i, utext=utext, variant=variant, history=history,
            session=session, key=key, call=call, hypa_S=hypa_S,
            hypa_data=hypa_data, hypa_state=hypa_state,
            fixed_tokens=fixed_tokens, max_context=max_context,
            trim_budget=trim_budget, kept_start_msg=kept_start_msg,
            preset=preset, card=card, reroll_at=reroll_at,
            edit_at=edit_at, turns=turns, total_rerolls=total_rerolls)
        if turn_aborted:
            aborted = turn_aborted
            break
        last_reply = new_last_reply

        _record_probe(i=i, ptype=ptype, fact=fact, wrong=wrong, utext=utext,
                      st=st, dir_sec=dir_sec, director=director,
                      ledger=ledger, judge=judge, card=card, variant=variant,
                      session=session, kept_start_msg=kept_start_msg,
                      win_start=win_start, use_window=use_window,
                      turns=turns, probes=probes)
        if variant == "dreaming" and i in (total_turns // 3,
                                           2 * total_turns // 3):
            time.sleep(12)                     # 꿈 트리거 (유휴 Dreamer)
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)
        if total_rerolls >= MAX_RUN_REROLLS:
            aborted = (f"누적 리롤 {total_rerolls}회 (T{i + 1}) — "
                       f"프로바이더 거부 반복, 런 중단")
            break

    result = _collect_totals(variant=variant, session=session, run_no=run_no,
                             prompt_set=prompt_set, turns=turns,
                             probes=probes, ledger=ledger, director=director,
                             judge=judge, hypa_cost0=hypa_cost0,
                             hypa_truncated0=hypa_truncated0,
                             aborted=aborted)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"v2-{session}-run{run_no}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preset")
    ap.add_argument("card")
    ap.add_argument("variant",
                    choices=("dreaming", "vanilla", "trim", "hypa"))
    ap.add_argument("--session", required=True)
    ap.add_argument("--runs", type=int, default=1)
    # --trim-tokens는 하위호환 별칭 — 같은 단일 풀(maxContext)이다
    ap.add_argument("--max-context", "--trim-tokens", dest="max_context",
                    type=int, default=MAX_CONTEXT)
    ap.add_argument("--turns", type=int, default=TURNS)
    ap.add_argument("--probe-every", type=int, default=PROBE_EVERY)
    ap.add_argument("--reroll-at", default="18,33")
    ap.add_argument("--edit-at", default="25")
    ap.add_argument("--ttl-wait", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--prompts", default="", help="프롬프트 오버라이드 JSON (A/B)")
    args = ap.parse_args()
    if args.prompts:
        prompts.override_from(args.prompts)
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
                     args.max_context, reroll, edit, args.ttl_wait,
                     args.turns, args.probe_every)
        t = r["totals"]
        grand = (t["cost"] + t.get("cost_director", 0) + t.get("cost_judge", 0)
                 + t.get("cost_hypa", 0))
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} "
              f"나레이터 ${t['cost']} + 디렉터 ${t.get('cost_director', 0)} "
              f"+ judge ${t.get('cost_judge', 0)} "
              f"+ hypa ${t.get('cost_hypa', 0)} = ${round(grand, 4)}",
              flush=True)
        if t.get("aborted"):
            # 부분 결과는 이미 저장됨 — 비정상 종료로 상위 스크립트에 알린다
            raise SystemExit(f"런 중단: {t['aborted']}")


if __name__ == "__main__":
    main()
