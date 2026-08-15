"""프로브 루프 + 이중 채점 (구현 계약 §6).

얼린 건초더미의 프로브 턴에서만 라이브로 나레이터를 부르고, 그 응답을
오라클(결정론)과 judge(이진 LLM)로 이중 채점한다.

**프로브 응답은 어디에도 누적되지 않는다** (스펙 §3.2, 계약 §8-1). 히스토리는
매 프로브마다 `messages_upto`가 얼린 원문에서 새로 만들고, 유상태 변형은
인제스천 스냅샷의 *복사본*에서 돈다. 이 파일 어디에도 `st["reply"]`를
히스토리·저장소·hypa data에 되먹이는 경로가 없다 — 있으면 "고정 건초더미"가
프로브 순서에 따라 갈라져 변형 간 비교가 무의미해진다.

나레이터·judge는 주입 심(seam)이다 (run2.py:436-438의 `call_fn`과 같은 패턴) —
오프라인 스모크가 LLM 0콜로 전 경로를 관통한다.

`benchmarks/eval/**`·`dreaming/**`는 읽기 전용 — 전부 import 재사용한다.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval import hypa as hypa_mod
from benchmarks.eval import transport
from benchmarks.eval.config import HYPA_EXPORT
from benchmarks.eval.lucid import DirFact, LlmFn
from benchmarks.eval.preset2wire import decode_risup
from benchmarks.eval.quality import LOOP_LOOKBACK, reroll_until_clean
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
from benchmarks.eval.windowing import hypa_in_window
from benchmarks.fixed_haystack import assemble, ingest, narrator
from benchmarks.fixed_haystack.haystack import (DEFAULT_OUT_DIR,
                                                PROVIDER_CTX_LIMIT,
                                                FixedHaystack, FrozenProbe,
                                                load, require_namespace,
                                                resolve_path, verify_inputs)

NarrateFn = Callable[[List[Dict]], Dict]   # 와이어 -> transport 스키마 응답
VARIANTS = ("vanilla", "trim", "retrieval", "hypa", "dreaming")
# 조립 계측 중 임시 경로 — 채점이 끝나면 지워지므로 결과에 남기지 않는다
_EPHEMERAL = ("storage_root", "workspace", "workspace_temp")


# ── 나레이터 심 ────────────────────────────────────────────────────────────

def default_narrate(hs: FixedHaystack) -> NarrateFn:
    """얼린 원문을 쓴 모델·파라미터 그대로 (계약 §8-5: `config.MODEL` 참조 금지).

    키는 변수로만 다룬다 — 값을 로그·결과 JSON 어디에도 남기지 않는다.
    """
    key = transport.key()

    def narrate(msgs: List[Dict]) -> Dict:
        return narrator.call(hs.narrator_model, msgs, key,
                             max_tokens=hs.run_params["max_tokens"],
                             reasoning=hs.run_params.get("reasoning", ""))
    return narrate


def _prior_replies(hs: FixedHaystack, probe_turn: int) -> List[str]:
    """얼린 히스토리의 직전 assistant 3개 — **프로브 응답이 아니다**.

    loop 게이트는 "방금 대본에 나온 문장을 그대로 반복했는가"를 본다. 그
    대본은 얼린 원문이고, 프로브 응답은 서로 다른 변형·반복의 산물이라
    여기에 섞이면 변형 간 리롤 확률이 달라진다.
    """
    return [hs.turns[t].assistant
            for t in range(max(0, probe_turn - LOOP_LOOKBACK), probe_turn)]


# ── 프로브 1건 ─────────────────────────────────────────────────────────────

def _skip(p: FrozenProbe, reason: str, **extra) -> Dict:
    """채점 제외 기록. 조용히 버리지 않고 사유를 남긴다 (계약 §2.2-1, §4.5)."""
    return {"turn": p.turn, "ptype": p.ptype, "fact": p.fact, "value": p.value,
            "wrong": p.wrong, "question": p.question, "reply": "",
            "skipped": reason, "oracle": None, "judge": None, "why": "",
            "miss_cause": "-", "value_in_store": None,
            "value_in_knowledge": None,
            "distance_turns": p.turn - p.fact_turn, **extra}


def _in_window(variant: str, p: FrozenProbe, a: assemble.Assembly,
               card: Dict) -> bool:
    """나레이터가 실제 본 창에 원 사실의 턴이 남아 있는가 (run2.py:337-342).

    창내 실패=LITM, 창밖 실패=eviction. 풀 히스토리 변형은 항상 창내이고,
    dreaming은 프록시가 압축했을 수 있어 **상한값**이다.
    """
    if variant in ("vanilla", "dreaming"):
        return True
    if variant == "hypa":
        # hypa는 턴이 아니라 메시지 인덱스로 자른다 (hypav3.ts:934)
        return hypa_in_window(p.fact_turn, a.kept_start_msg,
                              bool(card.get("greeting")))
    return p.fact_turn >= (a.win_start or 0)


def _score(variant: str, p: FrozenProbe, a: assemble.Assembly, reply: str,
           card: Dict, judge_fn: LlmFn, session: str) -> Tuple:
    """(oracle, judge, in_store, miss_cause). judge 파싱 실패(None)는 미스가 아니다.

    저장소 체크는 judge 결과와 무관하게 **무조건** 잰다 — 캐릭터가 표현을
    바꿔 말해 judge가 갈려도, 추출(저장소)·전달(주입)은 시스템 책임 몫이라
    독립 지표여야 한다. 임시 복사본 저장소를 본다 — 지우기 전에(§6-4).
    """
    o = oracle_pass(reply, p.value, wrong_value=p.wrong,
                    char_name=card.get("name", ""))
    j = judge_pass(judge_fn, p.ptype, p.fact, p.value, p.question, reply,
                   wrong_value=p.wrong)
    in_store, miss = None, "-"
    if variant == "dreaming":
        in_store = decompose_miss(
            a.extra["storage_root"], session,
            DirFact(fid=p.fid, kind="", value=p.value,
                    text=p.fact, turn=p.fact_turn)) == "utilization_fail"
        if j["pass"] is False:
            miss = "utilization_fail" if in_store else "storage_fail"
    return o, j, in_store, miss


def _cleanup(a: assemble.Assembly) -> None:
    """우리가 만든 임시 저장소만 지운다 — workdir을 준 호출자는 소유자다."""
    if a.extra.get("workspace_temp"):
        shutil.rmtree(a.extra["workspace"], ignore_errors=True)


def run_probe(hs: FixedHaystack, card: Dict, preset: Dict, variant: str,
              p: FrozenProbe, *, root: str, session: str, fixed_tokens: int,
              trim_budget: int, narrate_fn: NarrateFn, judge_fn: LlmFn,
              hypa_settings=None, summarize=None,
              provider_ctx_limit: int = PROVIDER_CTX_LIMIT,
              workdir: Optional[str] = None) -> Tuple[Dict, Dict]:
    """프로브 1건 = 계약 §6의 1~6단계. 반환 (프로브 기록, 조립 계측)."""
    # 1. 조립 — 못 보내는 상태면 나레이터를 아예 안 부른다
    a = assemble.build(variant, hs, card, preset, p.turn,
                       fixed_tokens=fixed_tokens, trim_budget=trim_budget,
                       root=root, session=session, hypa_settings=hypa_settings,
                       summarize=summarize,
                       provider_ctx_limit=provider_ctx_limit, workdir=workdir)
    if a.overflow:
        # dreaming도 오버플로할 수 있고 그때 임시 저장소가 이미 만들어져 있다
        _cleanup(a)
        return _skip(p, "context_overflow",
                     wire_tokens=a.extra.get("wire_tokens", 0),
                     provider_tokens=a.extra.get("provider_tokens", 0)), a.extra
    if a.extra.get("hypa_error"):
        # 원본도 요청 자체를 실패시킨다 (hypav3.ts:263-274) — 병리 재현이다
        _cleanup(a)
        return _skip(p, "hypa_error",
                     hypa_error=a.extra["hypa_error"]), a.extra
    try:
        # 2. 나레이터 — 품질 리롤은 얼린 직전 응답 3개를 기준으로 한다.
        #    st["reply"]는 아래 채점에만 쓰인다. 히스토리·저장소·hypa data
        #    어디에도 되먹이지 않는다 (§3.2/§8-1).
        st, _flaws = reroll_until_clean(lambda: narrate_fn(a.msgs),
                                        _prior_replies(hs, p.turn))
        # 3~4. 이중 채점 + dreaming 미스 분해
        o, j, in_store, miss = _score(variant, p, a, st["reply"], card,
                                      judge_fn, session)
    finally:
        # 임시 저장소는 미스 분해 뒤에 지운다.
        _cleanup(a)
    # 5. 기록
    rec = {
        "turn": p.turn, "ptype": p.ptype, "fact": p.fact, "value": p.value,
        "wrong": p.wrong, "question": p.question, "reply": st["reply"],
        "oracle": o, "judge": j["pass"], "why": j["why"], "miss_cause": miss,
        "skipped": "", "distance_turns": p.turn - p.fact_turn,
        "in_window": _in_window(variant, p, a, card),
        # 원본 턴이 evict돼도 서사 반복으로 값이 창 안에 남을 수 있다 —
        # 통과가 진짜 기억 때문인지 가늠하는 오염 지표 (run2.py:348-350).
        # dreaming/retrieval은 주입 블록이 창에 포함되므로 "전달"과 겹친다 —
        # 분리 지표는 아래 value_in_store / value_in_knowledge.
        "value_in_window": any(p.value in m["content"] for m in a.use_window
                               if isinstance(m.get("content"), str)),
        # 기억 퍼널: 추출(저장소에 값 존재) → 전달(주입 블록에 값 포함).
        # 캐릭터 표현 편차(RP 허용)와 무관한 시스템 책임 지표다.
        "value_in_store": in_store,
        "value_in_knowledge": (p.value in a.extra["knowledge_text"]
                               if a.extra.get("knowledge_text") else None),
        "rerolls": st.get("rerolls", 0), "flaw": st.get("flaw", ""),
        "finish": st.get("finish", ""), "prompt": st.get("prompt", 0),
        "cached": st.get("cached", 0), "completion": st.get("completion", 0),
        "cost": st.get("cost", 0.0), "sec": st.get("sec", 0.0),
        "wire_tokens": a.extra.get("wire_tokens", 0),
    }
    return rec, a.extra


# ── 변형 1회 ───────────────────────────────────────────────────────────────

def run_variant(hs: FixedHaystack, card: Dict, preset: Dict, variant: str,
                run_no: int, *, root: str, fixed_tokens: int, trim_budget: int,
                narrate_fn: Optional[NarrateFn] = None,
                judge_fn: Optional[LlmFn] = None,
                hypa_settings: Optional[hypa_mod.HypaSettings] = None,
                summarize=None,
                provider_ctx_limit: int = PROVIDER_CTX_LIMIT,
                session: str = ingest.SESSION,
                workdir: Optional[str] = None) -> Dict:
    """한 변형 × 전 프로브 1회분. 결과 JSON 스키마는 계약 §6 마지막 문단."""
    narrate_fn = narrate_fn or default_narrate(hs)
    judge_fn = judge_fn or transport.make_judge_llm()
    judge_cost0 = float(getattr(judge_fn, "cost", 0.0))
    # 프로브 루프가 태우는 hypa 요약 콜도 유상태 변형의 비용이다 — 인제스천
    # 메타만 싣고 여기를 빼면 hypa가 실제보다 싸 보인다(report.py:184-197).
    # 모듈 전역이라 `hypa_mod.NAME` 점 접근으로 읽는다(별칭은 스냅샷).
    hypa_cost0, hypa_calls0 = hypa_mod.SUMMARY_COST, hypa_mod.SUMMARY_CALLS
    # freeze가 잡아낸 누출 프로브 — 채택은 하되 채점에서 뺀다 (§2.2-1)
    leaky = set(hs.validation.get("leaky_probes", []))
    # 값이 1글자라 마스킹·게이트로 보호 못 하는 프로브 — 같은 대접(§2.2-1b)
    unprotectable = set(hs.validation.get("unprotectable_probes", []))

    probes: List[Dict] = []
    meta: Dict[str, Dict] = {}
    t0 = time.time()
    for p in hs.probes:
        if p.turn in leaky:
            # 발화가 정답을 그대로 담은 프로브 — 조립·나레이터도 안 부른다
            probes.append(_skip(p, "probe_leak"))
            continue
        if p.turn in unprotectable:
            # 값이 얼린 원문 전반에 흩어져 있어 오라클도 value_in_window도
            # 무력하다 — 채점하면 노이즈가 분모에 섞인다
            probes.append(_skip(p, "probe_unprotectable"))
            continue
        rec, extra = run_probe(
            hs, card, preset, variant, p, root=root, session=session,
            fixed_tokens=fixed_tokens, trim_budget=trim_budget,
            narrate_fn=narrate_fn, judge_fn=judge_fn,
            hypa_settings=hypa_settings, summarize=summarize,
            provider_ctx_limit=provider_ctx_limit, workdir=workdir)
        probes.append(rec)
        meta[str(p.turn)] = {k: v for k, v in extra.items()
                             if k not in _EPHEMERAL}

    def _n(reason: str) -> int:
        return sum(1 for r in probes if r["skipped"] == reason)

    live = [r for r in probes if not r["skipped"]]
    return {
        "haystack_id": hs.haystack_id, "variant": variant, "run": run_no,
        "narrator_model": hs.narrator_model,
        "origin_variant": hs.origin_variant,
        # §9-2가 provenance를 요구하는 최종 목적은 **측정 결과를 조건에
        # 귀속**시키는 것이다. 결과 JSON만 손에 남았을 때 어느 git commit·어느
        # 카드 해시·어느 재freeze로 잰 것인지 복원할 수 없으면 그 사슬이
        # 마지막 칸에서 끊긴다 — 동결 파일은 하루에 두 번 다시 얼렸다(§10.3·§10.4).
        "provenance": dict(hs.provenance),
        "probes": probes,
        "totals": {
            "probes": len(probes),
            # 분모 — judge 파싱 실패는 뺀다 (scoring.py:15-18 관례)
            "probes_scored": sum(1 for r in live if r["judge"] is not None),
            "judge_pass": sum(1 for r in live if r["judge"] is True),
            "judge_unparsed": sum(1 for r in live if r["judge"] is None),
            "oracle_pass": sum(1 for r in live if r["oracle"] is True),
            "oracle_na": sum(1 for r in live if r["oracle"] is None),
            "skipped_overflow": _n("context_overflow"),
            "skipped_leak": _n("probe_leak"),
            "skipped_unprotectable": _n("probe_unprotectable"),
            "skipped_hypa_error": _n("hypa_error"),
            # 절단은 기억 실패로 오인된다 — 0인지 매 런 확인한다
            "truncated": sum(1 for r in live if r.get("finish") == "length"),
            "rerolls": sum(r.get("rerolls", 0) for r in live),
            "flawed": sum(1 for r in live if r.get("flaw")),
            "cost": round(sum(r.get("cost", 0.0) for r in live), 4),
            "cost_judge": round(float(getattr(judge_fn, "cost", 0.0))
                                - judge_cost0, 4),
            # 프로브 루프가 태운 hypa 요약(캐시 미스분) — 보통 0이지만
            # 요약 모델을 바꾸거나 캐시를 비우면 여기로 청구된다
            "cost_hypa_summary": round(hypa_mod.SUMMARY_COST - hypa_cost0, 6),
            "hypa_summary_calls": hypa_mod.SUMMARY_CALLS - hypa_calls0,
            "sec": round(time.time() - t0, 1),
        },
        "assembly_meta": {
            "fixed_tokens": fixed_tokens, "trim_budget": trim_budget,
            "provider_ctx_limit": provider_ctx_limit, "session": session,
            "run_params": dict(hs.run_params), "by_probe": meta,
        },
    }


def save_result(result: Dict, out_dir) -> pathlib.Path:
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{result['variant']}-run{result['run']}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


# ── CLI ───────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="고정 건초더미 프로브 루프")
    ap.add_argument("--haystack", required=True, help="haystack_id")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--provider-ctx-limit", type=int,
                    default=PROVIDER_CTX_LIMIT)
    ap.add_argument("--out", default="",
                    help=f"결과 디렉터리 (기본 {DEFAULT_OUT_DIR}/<id>/runs)")
    a = ap.parse_args(argv)

    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"알 수 없는 변형: {bad} (가능: {list(VARIANTS)})")

    base = pathlib.Path(DEFAULT_OUT_DIR)
    hs = load(str(base / f"{a.haystack}.json"))
    # 동결 파일에 실리지 않는 조립 입력 2개(preset·card)를 매 측정 시작 때
    # 지문과 대조한다 — provenance는 freeze 시점 스냅샷일 뿐이라 아무도 다시
    # 재보지 않으면 A는 어제 프리셋, B는 오늘 프리셋으로 돌아도 모른다(§9-2).
    verify_inputs(hs)
    root = base / a.haystack
    # 스펙 §9-3의 전용 네임스페이스 — `dreaming_data/fixed_haystack/{id}/runs/`.
    # 기존 세션 디렉터리에는 아무것도 쓰지 않는다(--out도 같은 가드).
    out_dir = (require_namespace(a.out, "결과 디렉터리") if a.out
               else root / "runs")
    if "dreaming" in variants:
        # 유상태 축의 §9-2 — 동결 파일 로더만으론 "옛 스냅샷 + 새 동결 파일"
        # 조합을 못 잡는다(로더는 동결 파일만 본다)
        ingest.require_meta(root, hs)
    preset = decode_risup(resolve_path(hs.preset_path))
    card = json.loads(pathlib.Path(resolve_path(hs.card_path))
                      .read_text(encoding="utf-8"))
    fixed_tokens, trim_budget = assemble.budgets(preset, card, hs.run_params)
    # 세션 id는 인제스천이 쓴 값을 그대로 — 스냅샷 경로가 여기 달려 있다
    session = ingest.SESSION
    meta_path = root / "ingest-meta.json"
    if meta_path.is_file():
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        session = doc.get("dreaming", {}).get("session", session)
    S = (hypa_mod.load_hypa_settings(HYPA_EXPORT)
         if "hypa" in variants else None)

    print(f"{hs.haystack_id}: 고정부 {fixed_tokens}tok · trim 예산 "
          f"{trim_budget}tok · 나레이터 {hs.narrator_model} · "
          f"누출제외 {hs.validation.get('leaky_probes', [])} · "
          f"값길이제외 {hs.validation.get('unprotectable_probes', [])}")
    judge_fn = transport.make_judge_llm()
    for variant in variants:
        for n in range(a.runs):
            r = run_variant(hs, card, preset, variant, n, root=str(root),
                            fixed_tokens=fixed_tokens,
                            trim_budget=trim_budget, judge_fn=judge_fn,
                            hypa_settings=S,
                            provider_ctx_limit=a.provider_ctx_limit,
                            session=session)
            t = r["totals"]
            path = save_result(r, out_dir)
            print(f"[{variant} run{n}] judge {t['judge_pass']}"
                  f"/{t['probes_scored']} · 오라클 {t['oracle_pass']} · "
                  f"제외 누출{t['skipped_leak']}/값{t['skipped_unprotectable']}"
                  f"/오버플로{t['skipped_overflow']}"
                  f"/hypa{t['skipped_hypa_error']} · "
                  f"미파싱 {t['judge_unparsed']} · ${t['cost']} "
                  f"(judge ${t['cost_judge']}, hypa요약 "
                  f"${t['cost_hypa_summary']}) · {t['sec']}s → {path.name}")
    print(f"  → {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
