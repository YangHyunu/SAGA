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
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval.config import (DATA, EVAL_DIR, HYPA_EXPORT, MAX_CONTEXT,
                                    MAX_REROLL_STREAK, MAX_TOKENS, MODEL,
                                    NPC_EVENT_RETRY, NPC_EVENT_TURN, NPC_NAME,
                                    PROBE_EVERY, TOGGLES, TURNS, UPDATE_EVENTS)
from benchmarks.eval import config
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
# monkeypatch.setattr(transport, "call_upstream", ...)가 걸린다). 패치 표면은
# 둘로 갈린다: call은 transport. 점 접근이라 transport 패치가 먹힌다;
# _key/make_*_llm은 run2 별칭 경유(import 시점에 값이 고정)라
# monkeypatch.setattr(transport, "key", ...)는 조용한 no-op이고 run2. 쪽을
# 패치해야 실제로 걸린다. 나머지는 별칭 재노출(이 파일 안에서는 일부만
# 쓰임) — 기존 테스트·스크립트가 run2._key 등으로 참조한다.
from benchmarks.eval import transport
from benchmarks.eval.transport import (call_upstream as _call_upstream,  # noqa: F401
                                       key as _key, make_lucid_llm,
                                       make_judge_llm)
# 별칭 재노출 — 기존 테스트가 run2._count, run2._FULL_HISTORY 등으로 참조.
from benchmarks.eval.windowing import (FULL_HISTORY as _FULL_HISTORY,
                                       count as _count, hypa_in_window,
                                       token_trim, wire_history)
# reply_flaw는 이 파일 안에서는 안 쓰임(quality.reroll_until_clean 내부용) —
# 기존 테스트가 run2.reply_flaw로 직접 참조한다.
from benchmarks.eval.quality import (abort_reroll_count,
                                     reply_flaw,  # noqa: F401
                                     reroll_until_clean)

from benchmarks.eval.lucid import (DirFact, Ledger, LlmFn, extract_facts,
                                   make_false_premise, make_probe,
                                   _probe_mentions_fact_object,
                                   probe_leaks_value, probe_plan)
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.preset2wire import assemble, decode_risup, reformat
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval import gates, hypa


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
               memory: str = "", toggles: Optional[Dict[str, str]] = None
               ) -> List[Dict]:
    """뮈토스 조립 + reformater.

    memory는 hypa 요약 블록 — 프리셋의 memory 카드(promptTemplate[35] 'Past
    Summary') 자리에 들어간다. 즉 chat 히스토리보다 앞, 시스템 프롬프트
    한가운데다 (index.svelte.ts:1429-1443). 캐시 파괴 병리의 구조적 원인이다.

    toggles: 기본은 config.TOGGLES(현행 시나리오). 실캡처 재현 테스트처럼
    캡처 당시 토글로 조립해야 할 때만 명시한다.
    """
    msgs = assemble(preset, toggles if toggles is not None else TOGGLES,
                    window, memory=memory,
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
    """턴 i의 필러 지시. NPC 이벤트 > UPDATE_EVENTS > 5턴 주기 비트 > 평서.

    비트 위상은 i%5==2 — 프로브 그리드(i%10==9)와 겹치면 프로브가 비트를
    먹어서 i%5==4였을 때 비트 절반(특히 갈등 비트)이 소실됐다 (20턴 스모크
    실측: 갈등 0회, 소꿉놀이 정체). UPDATE_EVENTS(12)와 1회 겹침은 감수.
    """
    if npc_due:
        return prompts.NPC_BEAT
    if i in UPDATE_EVENTS:
        return prompts.UPDATE_BEAT
    if i % 5 == 2:
        return prompts.BEATS[(i // 5) % len(prompts.BEATS)]
    return "자연스럽게 이어간다."


def recent_dialogue(history: List[Dict], pairs: int = 3) -> str:
    """디렉터에게 주는 최근 대화 — 직전 응답 하나만 주면 맥락 없이 맴돈다."""
    tail = history[-(pairs * 2):]
    return "\n\n".join(
        f"[{'렌' if m['role'] == 'user' else '캐릭터'}]\n{m['content'][-600:]}"
        for m in tail)


def _filler_turn(i: int, history: List[Dict], last_reply: str, dir_sys: str,
                 card: Dict, lucid: LlmFn) -> str:
    """프로브 자격이 없는(또는 누출로 강등된) 턴의 필러 유저 발화 생성.

    run_once 본문 else 분기에서 추출 — probe_plan 자격이 없는 턴뿐 아니라
    누출 2연속으로 강등된 프로브 턴도 이걸 재사용해야 해서 별도 함수로 뺐다.
    """
    ctx = recent_dialogue(history) or f"[캐릭터]\n{last_reply[-600:]}"
    # T41에 무조건 한 번, T45까지는 **캐릭터가** 아직 안 등장시켰을
    # 때만 재유도 — 유저 발화까지 검사하면 T41 디렉터가 이름을 말한
    # 순간 retry가 영구 봉쇄된다 (실측: retry 죽은 코드였음).
    # 등장 후 유지는 스케줄이 아니라 DIRECT_SYS 조연 조항에 맡긴다.
    npc_in_reply = any(NPC_NAME in m["content"] for m in history
                       if m["role"] == "assistant")
    npc_due = (i == NPC_EVENT_TURN
              or (NPC_EVENT_TURN < i <= NPC_EVENT_RETRY
                  and not npc_in_reply))
    return lucid(
        dir_sys + f"\n[작품 설정]\n{card.get('description', '')[:2000]}",
        f"[최근 대화]\n{ctx}\n[지시]\n{pick_beat(i, npc_due)}")


def _probe_text(lucid: LlmFn, fact: DirFact, ptype: Optional[str],
                last_reply: str, few_shot: str) -> Tuple[str, str]:
    """프로브/오염 발화 생성 한 번. false는 (발화, 오염값), 나머지는 오염값 없음."""
    if ptype == "false":
        return make_false_premise(lucid, fact, scene=last_reply, style=few_shot)
    return make_probe(lucid, fact, scene=last_reply, style=few_shot), ""


def _resolve_probe_turn(i: int, ptype: Optional[str], fact: Optional[DirFact],
                        lucid: LlmFn, last_reply: str, few_shot: str,
                        history: List[Dict], dir_sys: str, card: Dict
                        ) -> Tuple[Optional[str], Optional[DirFact], str, str,
                                  int, int]:
    """이번 턴의 유저 발화를 정한다 — 프로브 생성 + 누출 하드 게이트(D5/I1·I2).

    fact가 있으면 프로브/false 발화를 만들고 probe_leaks_value로 정답 유출을
    검사한다. 새면 1회만 재생성한다. 재생성도 새면(그리고 자격 있는 fact가
    애초에 없었으면) 이 턴을 _filler_turn으로 강등한다 — 이때 fact.probed를
    되돌려 사실을 태우지 않고 미출제 풀에 남긴다. false 프로브도 참값
    기준으로 검사한다 — 발화에는 오염값만 실려야 하니까(I2).

    반환: (ptype, fact, utext, wrong, leak_retries, leak_dropped). 마지막
    둘은 이번 턴에 벌어진 재시도/강등 건수(0 또는 1) — 호출부가 누적한다.
    """
    retries, dropped = 0, 0
    wrong = ""
    if fact is not None:
        utext, wrong = _probe_text(lucid, fact, ptype, last_reply, few_shot)
        if probe_leaks_value(utext, fact):
            retries = 1
            utext, wrong = _probe_text(lucid, fact, ptype, last_reply, few_shot)
            if probe_leaks_value(utext, fact):
                dropped = 1
                fact.probed = False        # 태우지 않고 미출제 풀로 되돌림
                fact = None
    if fact is None:
        ptype = None                       # eligible 없거나 누출로 강등 — 필러로
        utext = _filler_turn(i, history, last_reply, dir_sys, card, lucid)
    return ptype, fact, utext, wrong, retries, dropped


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
                  lucid: LlmFn, ledger: Ledger, judge: LlmFn, card: Dict,
                  variant: str, session: str, kept_start_msg: int,
                  win_start: Optional[int], use_window: List[Dict],
                  turns: List[Dict], probes: List[Dict]) -> None:
    # 프로브 계획+기록: extract_facts로 원장을 키워 미래 프로브 재료를 대고,
    # 이번 턴이 프로브면 오라클+judge 이중 채점을 기록한다.
    t_ext = time.time()
    if ptype is None:
        ledger.add(extract_facts(lucid, utext, st["reply"], i))
    ext_sec = round(time.time() - t_ext, 1)

    turns.append({"turn": i, "user": utext, **st,
                  "sec_lucid": dir_sec, "sec_extract": ext_sec,
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


def _dream_metrics(variant: str, session: str
                   ) -> Tuple[Optional[bool], Optional[int], Optional[bool]]:
    """dreaming 저장소 계측 — (dream_ran, episodes_written, compression_planned).

    dreaming 외 변형은 (None, None, None) — 게이트 G1이 dreaming만 본다.
    경로 규약은 scoring.decompose_miss의 base/{kind}/*.json과 동일
    (JsonDirStorage 네임스페이스: DATA/{session}/{kind}/{key}.json).
    dream_ran은 dreamer/cursor.json 존재, episodes_written은 episodes/*.json
    개수, compression_planned은 compression/plan.json 존재 — plan이 None이면
    dreamer.py:345-346이 그 파일 자체를 쓰지 않으므로 부재가 모호함 없는
    "압축 미성립" 신호다(A7·C6).
    """
    if variant != "dreaming":
        return None, None, None
    base = DATA / session
    dream_ran = (base / "dreamer" / "cursor.json").is_file()
    ep_dir = base / "episodes"
    episodes_written = len(list(ep_dir.glob("*.json"))) if ep_dir.is_dir() else 0
    compression_planned = (base / "compression" / "plan.json").is_file()
    return dream_ran, episodes_written, compression_planned


def _collect_totals(variant: str, session: str, run_no: int,
                    prompt_set: Dict[str, object], turns: List[Dict],
                    probes: List[Dict], ledger: Ledger, lucid: LlmFn,
                    judge: LlmFn, hypa_cost0: float, hypa_truncated0: int,
                    aborted: str, probe_leak_retries: int = 0,
                    probe_leak_dropped: int = 0, probes_scheduled: int = 0,
                    dream_ran: Optional[bool] = None,
                    episodes_written: Optional[int] = None,
                    compression_planned: Optional[bool] = None,
                    prompt_hashes: Optional[Dict[str, str]] = None) -> Dict:
    # totals 집계: probes/turns에서 판정·비용을 합산해 최종 result를 조립.
    passed = sum(1 for p in probes if p["judge"] is True)
    unparsed = sum(1 for p in probes if p["judge"] is None)
    result = {"variant": variant, "session": session, "run": run_no,
              "model": MODEL, "prompt_set": prompt_set,
              "prompt_hashes": prompt_hashes or {},
              "turns": turns, "probes": probes,
              "ledger": ledger.to_rows(),
              "totals": {"probes": len(probes),
                         "probes_scheduled": probes_scheduled,
                         "judge_pass": passed,
                         "judge_unparsed": unparsed,
                         "oracle_pass": sum(1 for p in probes if p["oracle"]),
                         # 절단은 기억 실패로 오인된다 — 0인지 매 런 확인한다
                         "truncated": sum(1 for t in turns
                                          if t.get("finish") == "length"),
                         "rerolls": sum(t.get("rerolls", 0) for t in turns),
                         # 리롤 2회로도 못 걷어낸 병리 턴 — 0이어야 한다
                         "flawed": sum(1 for t in turns if t.get("flaw")),
                         "cost": round(sum(t["cost"] for t in turns), 4),
                         "cost_lucid": round(lucid.cost, 4),
                         "lucid_calls": lucid.calls,
                         "lucid_model": config.LUCID_MODEL,
                         "cost_judge": round(judge.cost, 4),
                         "judge_calls": judge.calls,
                         # hypa 요약 콜 — 모듈 전역 누적이라 런 시작 대비 증분
                         "cost_hypa": round(hypa.SUMMARY_COST - hypa_cost0, 4),
                         "hypa_truncated": hypa.SUMMARY_TRUNCATED - hypa_truncated0,
                         "aborted": aborted,
                         # 0이 정상 — D5 누출 하드 게이트가 걸린 횟수.
                         "probe_leak_retries": probe_leak_retries,
                         "probe_leak_dropped": probe_leak_dropped,
                         # dreaming 전용 계측(A7·C6) — 다른 변형은 None
                         "dream_ran": dream_ran,
                         "episodes_written": episodes_written,
                         "compression_planned": compression_planned}}
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
    lucid = make_lucid_llm()
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
    total_rerolls, reroll_streak, aborted = 0, 0, ""
    probe_leak_retries, probe_leak_dropped = 0, 0
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
        ptype, fact, utext, wrong, leak_retries, leak_dropped = (
            _resolve_probe_turn(i, ptype, fact, lucid, last_reply, few_shot,
                               history, dir_sys, card))
        probe_leak_retries += leak_retries
        probe_leak_dropped += leak_dropped
        dir_sec = round(time.time() - t_dir, 1)

        rerolls_before = total_rerolls
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
        # 흩어진 정상 리롤은 스트릭을 끊는다 — 연속 실패만 런의 파손이다
        reroll_streak = (reroll_streak + 1 if total_rerolls > rerolls_before
                         else 0)

        _record_probe(i=i, ptype=ptype, fact=fact, wrong=wrong, utext=utext,
                      st=st, dir_sec=dir_sec, lucid=lucid,
                      ledger=ledger, judge=judge, card=card, variant=variant,
                      session=session, kept_start_msg=kept_start_msg,
                      win_start=win_start, use_window=use_window,
                      turns=turns, probes=probes)
        if variant == "dreaming" and i in (total_turns // 3,
                                           2 * total_turns // 3):
            time.sleep(12)                     # 꿈 트리거 (유휴 Dreamer)
        if ttl_wait and i % 10 == 9:
            time.sleep(305)                    # TTL 5m 만료 재현 (옵션)
        if reroll_streak >= MAX_REROLL_STREAK:
            aborted = (f"연속 {reroll_streak}턴 품질 게이트 실패 (T{i + 1}, "
                       f"누적 리롤 {total_rerolls}회) — 런 중단")
            break

    probes_scheduled = sum(1 for x in sched if x is not None)
    dream_ran, episodes_written, compression_planned = _dream_metrics(
        variant, session)
    result = _collect_totals(variant=variant, session=session, run_no=run_no,
                             prompt_set=prompt_set, turns=turns,
                             probes=probes, ledger=ledger, lucid=lucid,
                             judge=judge, hypa_cost0=hypa_cost0,
                             hypa_truncated0=hypa_truncated0,
                             aborted=aborted,
                             probe_leak_retries=probe_leak_retries,
                             probe_leak_dropped=probe_leak_dropped,
                             probes_scheduled=probes_scheduled,
                             dream_ran=dream_ran,
                             episodes_written=episodes_written,
                             compression_planned=compression_planned,
                             prompt_hashes=prompts.layer_hashes())
    # 런 유효성 판정 — 부분/무효 결과도 감사 가치가 있으므로 런을 죽이지
    # 않는다. main()이 실패 게이트를 보고 비영점 종료한다.
    result["gates"] = gates.evaluate(result)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"v2-{session}-run{run_no}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


# 크래시/중단(SystemExit 문자열 → exit 1)과 "완주했지만 게이트 실패"를 exit
# 코드로 구분한다. G1(compression_planned)은 업스트림 압축 버그
# (DREAMING_FLAW.md)가 살아있는 한 dreaming 런마다 계속 빨간불이라, 스모크
# 단계(night_run.sh)가 "실행 가능한가"만 확인하려면 이 코드를 용인해야
# 한다 — exit 1(진짜 크래시·abort·격리)까지 삼키면 안 된다.
GATE_ONLY_EXIT = 2


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
    gate_failed = False
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
        grand = (t["cost"] + t.get("cost_lucid", 0) + t.get("cost_judge", 0)
                 + t.get("cost_hypa", 0))
        print(f"[run{n}] {t['judge_pass']}/{t['probes']} "
              f"나레이터 ${t['cost']} + Lucid ${t.get('cost_lucid', 0)} "
              f"+ judge ${t.get('cost_judge', 0)} "
              f"+ hypa ${t.get('cost_hypa', 0)} = ${round(grand, 4)}",
              flush=True)
        failed_gates = r["gates"]["failed"]
        if failed_gates:
            # 런은 죽이지 않는다 — 부분/무효 결과도 감사 가치가 있다. 대신
            # 결과 JSON(gates)에 이미 기록됐고, 아래에서 프로세스를
            # 비영점 종료해 night_run.sh 등 상위 스크립트가 감지하게 한다.
            gate_failed = True
            print(f"[run{n}] 게이트 실패: "
                  + ", ".join(gid for gid, _ in failed_gates), flush=True)
        if t.get("aborted"):
            # 부분 결과는 이미 저장됨 — 비정상 종료(exit 1)로 상위
            # 스크립트에 알린다. 크래시/중단은 게이트 실패보다 심각하므로
            # GATE_ONLY_EXIT이 아니라 기본 exit 1로 구분해서 알린다.
            raise SystemExit(f"런 중단: {t['aborted']}")
    if gate_failed:
        # 여기 도달했다는 건 위에서 abort로 죽지 않았다는 뜻 — 런은
        # 완주했지만 유효성 게이트가 하나 이상 실패했다는 신호를 exit
        # 1과 구분되는 코드로 낸다(GATE_ONLY_EXIT). night_run.sh 스모크
        # 단계는 이 코드를 용인하고, 본런은 여전히 비영점으로 보고 감지한다.
        print("런 유효성 게이트 실패 — 결과 JSON의 gates.failed 참고",
              flush=True)
        sys.exit(GATE_ONLY_EXIT)


if __name__ == "__main__":
    main()
