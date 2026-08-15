"""프로브 턴 컨텍스트 조립 — 4+1변형 (구현 계약 §4).

같은 얼린 히스토리를 변형별 조립 알고리즘으로만 다르게 만든다. 무상태 3종
(vanilla/trim/retrieval)은 `(haystack, probe_turn)`의 순수 함수라 LLM 0콜이고,
유상태 2종(hypa/dreaming)은 `ingest.py`가 만든 스냅샷을 **복사본에서** 소비한다
— 원본을 건드리면 반복(runs)·다음 프로브 사이로 상태가 새어 비교가 깨진다
(계약 §8-2).

와이어 형태 검사는 **항상 `build_wire` 출력(=클라이언트 요청)** 에 건다.
dreaming의 최종 와이어는 프록시가 cache_control을 찍고 `to_wire`가 content를
parts로 바꾼 뒤라 `check_wire_shape`의 "평문 문자열" 규칙에 걸린다 — 그건
프록시가 정상 동작한 증거지 조립 실수가 아니다.

`benchmarks/eval/**`·`dreaming/**`는 읽기 전용 — 전부 import 재사용한다.
"""

from __future__ import annotations

import contextlib
import copy
import functools
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval import hypa as hypa_mod
from benchmarks.eval.fidelity import check_wire_shape
from benchmarks.eval.run2 import build_wire
from benchmarks.eval.variants import retrieve_turns
from benchmarks.eval.windowing import count, token_trim
from benchmarks.fixed_haystack.haystack import (DEFAULT_OUT_DIR,
                                                PROVIDER_CTX_LIMIT,
                                                FixedHaystack, messages_upto,
                                                provider_tokens)
from dreaming.assembly import (HOT_ZONE_CHAR_BUDGET, KNOWLEDGE_SEP,
                               inject_knowledge)
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath
from dreaming.upstream import to_wire

# variants.prepare_request:81과 같은 블록 형태 — 발췌는 원문 그대로 옮긴다.
EXCERPT_HEADER = "[과거 대화 발췌]\n"
EXCERPT_SEP = "\n---\n"
EXCERPT_K = 3

STATELESS = ("vanilla", "trim", "retrieval")


@dataclass
class Assembly:
    msgs: List[Dict]                     # 나레이터에 보낼 최종 와이어
    use_window: List[Dict]               # 나레이터가 실제로 본 히스토리
    win_start: Optional[int] = None      # 창 시작 턴 (trim/retrieval), 그 외 None
    kept_start_msg: int = 0              # hypa 전용
    memory_text: str = ""                # hypa 요약 블록
    overflow: bool = False               # vanilla 컨텍스트 초과 (§4.5)
    extra: Dict = field(default_factory=dict)


# ── 요약 캐시 네임스페이스 (스펙 §9-3) ────────────────────────────────────

BENCH_HYPA_CACHE = Path(DEFAULT_OUT_DIR) / "hypa-cache"


@contextlib.contextmanager
def bench_summary_cache(cache_dir: Optional[Path] = None):
    """hypa 요약 캐시를 벤치 전용 네임스페이스로 돌린다 (§9-3, 되돌린다).

    `hypa.summarize`의 `cache_dir` 기본값은 **def 시점에 바인딩**된
    `hypa.CACHE_DIR`(= `dreaming_data/eval/hypa-cache`)이라, 모듈 속성을
    갈아끼워도 no-op이다 — 2026-08-10 후기의 "import 시점 별칭은 스냅샷"과
    같은 계열의 함정이고, 그래서 **호출부를 감싸 인자로 넘긴다**.
    `hypa_step`은 `summarize`를 모듈 전역으로 조회하므로 이 교체가 도달한다.

    §9-3은 값이 아니라 **쓰기 위치**에 대한 규칙이다: 캐시가 내용 키 기반이라
    값 오염은 없어도, 벤치 산출물이 `fixed_haystack/{id}/` 밖 오염 지대에
    떨어지고 eval v2 하네스와 같은 디렉터리를 공유하는 것 자체가 위반이다.
    `benchmarks/eval/**`는 읽기 전용이라 이 자리가 유일한 손잡이다.
    """
    target = cache_dir or BENCH_HYPA_CACHE
    orig = hypa_mod.summarize
    hypa_mod.summarize = functools.partial(orig, cache_dir=target)
    try:
        yield target
    finally:
        hypa_mod.summarize = orig


# ── 예산 (계약 §4, run2.py:466-474 공식) ───────────────────────────────────

def budgets(preset: Dict, card: Dict, run_params: Dict) -> Tuple[int, int]:
    """(fixed_tokens, trim_budget).

    고정부는 빈 히스토리 와이어로 실측한다 — RisuAI가 히스토리 앞에 먼저
    태우는 몫(index.svelte.ts:614-618). trim 예산은 거기서 응답 예약
    (maxResponse+50)을 더 뺀 나머지다.

    5변형이 **같은 풀**을 쓴다는 계약 §8-6에 따라 `windowing.count` 하나로
    통일한다. 단 hypa만은 예외다 — `hypa_fixed_tokens` 주석 참고.
    """
    fixed_tokens = sum(count(m["content"]) for m in build_wire(preset, card, []))
    trim_budget = (run_params["max_context"] - fixed_tokens
                   - run_params["max_tokens"] - 50)
    return fixed_tokens, trim_budget


def hypa_fixed_tokens(preset: Dict, card: Dict) -> int:
    """hypa 전용 고정부 — `hypa.tok_chat`(메시지당 +3, run2.py:467).

    hypa는 `current > max_ctx` 경계에서 요약 배치 유무가 갈린다. 인제스천과
    프로브 조립이 서로 다른 자로 재면(실물 11,830 vs 11,806, 메시지 8개×3)
    그 경계에 걸린 턴에서 프로브가 인제스천과 **다른 배치**를 만들고, 그때
    `hypa.summarize`가 계획에 없던 실제 요약 콜을 태운다. 원본의 발동 시점을
    재현하는 쪽이 단일 자보다 중요해 hypa만 이 카운터를 쓴다.
    """
    return sum(hypa_mod.tok_chat(m) for m in build_wire(preset, card, []))


def _wire(preset: Dict, card: Dict, window: List[Dict],
          memory: str = "") -> List[Dict]:
    """build_wire + 형태 검사. 위반이면 즉시 중단 (run2.py:259-261과 동일)."""
    msgs = build_wire(preset, card, window, memory=memory)
    bad = check_wire_shape(msgs)
    if bad:
        raise SystemExit(f"와이어 형태 위반: {bad}")
    return msgs


def _tokens(msgs: List[Dict]) -> int:
    return sum(count(m["content"]) for m in msgs)


# ── 무상태 3종 (LLM 0콜, 순수 함수) ────────────────────────────────────────

def assemble_vanilla(hs: FixedHaystack, card: Dict, preset: Dict,
                     probe_turn: int, fixed_tokens: int, *,
                     provider_ctx_limit: int = PROVIDER_CTX_LIMIT) -> Assembly:
    """전체 히스토리, 트림 없음 (§4.1).

    프로바이더 창을 넘으면 **나레이터를 부르지 않고** overflow로 돌려준다
    (§4.5) — 이건 하네스 에러가 아니라 "순정이 이 대본에선 물리적으로 못
    돈다"는 관측 결과다. run.py가 분모에서 뺀다.
    """
    window = messages_upto(hs, probe_turn, card)
    # 고정부 + 히스토리 = 와이어 토큰. 오버플로 판정은 조립 **전에** 한다.
    # 판정은 반드시 프로바이더 토크나이저 공간에서 — o200k 값을 128K와 직접
    # 비교하면 한국어 히스토리가 최대 12% 과소계상돼 실제로는 창을 넘는 턴이
    # 가드를 통과한다(haystack.provider_tokens 주석의 T89 실측).
    wire_tokens = fixed_tokens + _tokens(window)
    est = provider_tokens(fixed_tokens, _tokens(window))
    extra = {"wire_tokens": wire_tokens, "provider_tokens": est,
             "provider_ctx_limit": provider_ctx_limit}
    if est > provider_ctx_limit:
        return Assembly(msgs=[], use_window=window, win_start=0,
                        overflow=True, extra=extra)
    return Assembly(msgs=_wire(preset, card, window), use_window=window,
                    win_start=0, extra=extra)


def assemble_trim(hs: FixedHaystack, card: Dict, preset: Dict, probe_turn: int,
                  trim_budget: int) -> Assembly:
    """클라이언트 트림 베이스라인 (§4.2)."""
    full = messages_upto(hs, probe_turn, card)
    window, win_start = token_trim(full, trim_budget)
    msgs = _wire(preset, card, window)
    return Assembly(msgs=msgs, use_window=window, win_start=win_start,
                    extra={"wire_tokens": _tokens(msgs),
                           "trim_budget": trim_budget,
                           "dropped_msgs": len(full) - len(window)})


def _excerpt_block(excerpts: List[str], budget: int) -> Tuple[str, int]:
    """발췌를 **청크 단위**로 예산 안에 담는다 — 문장 중간 절단 금지(§4.3).

    누적 길이가 예산을 넘는 발췌부터 버린다(그 뒤도 함께). retrieve_turns가
    관련도 내림차순으로 주므로 버려지는 쪽이 항상 덜 관련된 발췌다.
    한 발췌가 혼자 예산을 넘으면 블록은 빈다 — 자르지 않는 게 계약이다.
    """
    kept: List[str] = []
    used = len(EXCERPT_HEADER)
    for e in excerpts:
        add = len(e) + (len(EXCERPT_SEP) if kept else 0)
        if used + add > budget:
            break
        kept.append(e)
        used += add
    if not kept:
        return "", 0
    return EXCERPT_HEADER + EXCERPT_SEP.join(kept), len(kept)


def assemble_retrieval(hs: FixedHaystack, card: Dict, preset: Dict,
                       probe_turn: int, trim_budget: int, *, k: int = EXCERPT_K,
                       excerpt_budget: int = HOT_ZONE_CHAR_BUDGET) -> Assembly:
    """trim + 잘린 구간 top-k 발췌 prepend (§4.3, 2511.17208 단순 baseline).

    발췌 블록은 `dreaming.assembly.inject_knowledge`로 **마지막 user에**
    붙인다 — dreaming 지식 블록과 같은 위치·같은 6000자 상한이라야 예산이
    통제된다(§8-6). 창(use_window)에 넣고 조립하므로 `value_in_window` 판정도
    발췌로 본 값을 정직하게 센다.

    주입은 조립 **전에**, 즉 히스토리 창에 한다. 뮈토스 6.2 실물 와이어는
    chat 블록 뒤에 프리필 왕복 5개(user/bot/user/bot/user)를 달아서 와이어의
    마지막 user가 프로브 발화가 아니라 영문 프리필이다(실측: 106메시지 중
    index 105). 조립 후 와이어의 마지막 user에 붙이면 발췌가 그 프리필 앞에
    박힌다. dreaming은 `scaffold.split`으로 그 꼬리를 벗기고 주입하므로
    (sync.py:163,200) 창에 넣는 쪽이 dreaming과 같은 자리다.
    """
    full = messages_upto(hs, probe_turn, card)
    window, win_start = token_trim(full, trim_budget)
    # full[-1]은 프로브 발화 자체(messages_upto 계약) = probe.question
    excerpts = retrieve_turns(full, full[-1]["content"], k=k, window=window)
    block, kept = _excerpt_block(excerpts, excerpt_budget)
    use_window = inject_knowledge(window, block) if block else window
    msgs = _wire(preset, card, use_window)
    return Assembly(msgs=msgs, use_window=use_window, win_start=win_start,
                    extra={"wire_tokens": _tokens(msgs),
                           "trim_budget": trim_budget,
                           "excerpts_found": len(excerpts),
                           "excerpts_kept": kept,
                           "excerpt_chars": len(block),
                           "excerpt_budget": excerpt_budget,
                           # 퍼널 "전달" 판정용 — run.py가 값 포함 여부를 잰다
                           "knowledge_text": block})


# ── 스냅샷 경로 (계약 §1 레이아웃 — ingest.py가 쓰고 여기서 읽는다) ────────

def hypa_snapshot_path(root: str, probe_turn: int) -> Path:
    return Path(root) / "hypa" / "snap" / f"t{probe_turn}.json"


def dreaming_snapshot_dir(root: str, probe_turn: int) -> Path:
    return Path(root) / "dreaming" / "snap" / f"t{probe_turn}"


def load_hypa_snapshot(root: str, probe_turn: int) -> Dict:
    """없으면 조용히 폴백하지 않고 실패한다 — 인제스천 미실행은 결과가 아니다."""
    path = hypa_snapshot_path(root, probe_turn)
    if not path.is_file():
        raise SystemExit(
            f"hypa 스냅샷 없음: {path} — "
            f"`python3 -m benchmarks.fixed_haystack.ingest --variants hypa`를 "
            f"먼저 돌려라")
    return json.loads(path.read_text(encoding="utf-8"))


# ── 유상태 2종 (스냅샷 복사본에서만 실행) ──────────────────────────────────

def assemble_hypa(hs: FixedHaystack, card: Dict, preset: Dict, probe_turn: int,
                  fixed_tokens: int, data: Dict, *,
                  settings: hypa_mod.HypaSettings, max_context: int,
                  max_tokens: int,
                  summarize: Optional[Callable[[List[Dict]], str]] = None
                  ) -> Assembly:
    """스냅샷 data를 deepcopy해 프로브 턴 1회분만 소비한다 (§4.4).

    **deepcopy 필수** — hypa_step은 `data["summaries"]`를 제자리에서 키운다.
    원본을 넘기면 반복(runs) 사이로 요약이 새어 2회차부터 다른 상태가 된다.

    summarize는 주입 심 — 기본은 실제 요약 콜(`hypa._summarize_call`)이지만
    테스트는 fake를 넣어 LLM 0콜로 돈다.
    """
    full = messages_upto(hs, probe_turn, card)
    # 프로브 루프도 캐시 미스면 실제 요약을 태운다 — 그 쓰기가 §9-3의 전용
    # 네임스페이스에 떨어지도록 여기도 감싼다(인제스천과 같은 캐시를 공유해야
    # 재실행이 싸다는 계약 §5.2도 이 경로에서 유지된다).
    with bench_summary_cache():
        memory, kept, kept_start_msg, _data, err = hypa_mod.hypa_step(
            full, fixed_tokens, settings, copy.deepcopy(data),
            summarize or hypa_mod._summarize_call, max_context, max_tokens)
    extra = {"kept_start_msg": kept_start_msg,
             "summaries": len(_data.get("summaries", []))}
    if err:
        # 원본도 요청 자체를 실패시킨다 (hypav3.ts:263-274). 병리 재현이지
        # 하네스 버그가 아니다 — 조립 실패로 남기고 그 프로브는 채점 제외.
        extra["hypa_error"] = err
        return Assembly(msgs=[], use_window=kept, kept_start_msg=kept_start_msg,
                        extra=extra)
    memory_text = memory or ""
    msgs = _wire(preset, card, kept, memory=memory_text)
    extra["wire_tokens"] = _tokens(msgs)
    extra["memory_chars"] = len(memory_text)
    return Assembly(msgs=msgs, use_window=kept, kept_start_msg=kept_start_msg,
                    memory_text=memory_text, extra=extra)


def assemble_dreaming(hs: FixedHaystack, card: Dict, preset: Dict,
                      probe_turn: int, root: str, session: str, *,
                      fixed_tokens: int = 0,
                      provider_ctx_limit: int = PROVIDER_CTX_LIMIT,
                      workdir: Optional[str] = None) -> Assembly:
    """스냅샷을 임시 복사본으로 옮긴 뒤 프로덕션 동기 경로를 그대로 태운다
    (§4.6, proxy.py:173-179).

    **임시 복사 필수** — `sp.process`는 저장소에 쓴다(원장·scaffold·격리).
    스냅샷 원본에 직접 돌리면 다음 프로브와 다음 반복이 오염된다.

    복사본은 함수가 끝나도 남긴다 — run.py가 미스 분해
    (`scoring.decompose_miss(extra["storage_root"], session, fact)`, 계약 §6-4)에
    쓰기 때문이다. 채점이 끝나면 호출부가 `extra["workspace"]`를 지운다.
    """
    full = messages_upto(hs, probe_turn, card)
    snap = dreaming_snapshot_dir(root, probe_turn)
    if not snap.is_dir():
        raise SystemExit(
            f"dreaming 스냅샷 없음: {snap} — "
            f"`python3 -m benchmarks.fixed_haystack.ingest --variants dreaming`을 "
            f"먼저 돌려라")
    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="fixhay-drm-"))
    live = work / f"drm-t{probe_turn}"
    # workdir을 준 재호출(반복 실행)에서 copytree가 FileExistsError로 죽는다.
    # dirs_exist_ok로 덮으면 지난 회차 쓰기가 남아 스냅샷이 아니게 되므로
    # 지우고 새로 뜬다 — 복사본은 항상 스냅샷과 동일해야 한다(§8-2).
    shutil.rmtree(live, ignore_errors=True)
    shutil.copytree(snap, live)

    try:
        storage = JsonDirStorage(live)
        sp = SyncPath(storage, session, mark_cache_enabled=True)
        raw = _wire(preset, card, full)
        out, verdict = sp.process(raw)
        msgs = to_wire(out)
        plan = storage.get(f"{session}/compression", "plan")
    except BaseException:
        # process가 던지면(원장 손상·shift_keyed 예외 등) 스냅샷 통째 복사본이
        # /var/folders에 영구 잔류한다 — 100턴 저장소 × 9프로브 × 반복.
        # 우리가 만든 임시 디렉터리만 치운다(호출자 workdir는 호출자 소유).
        if workdir is None:
            shutil.rmtree(work, ignore_errors=True)
        raise
    # 프록시 압축은 턴 경계와 1:1이 아니다 → win_start 없음. 창은 out 기준.
    use_window = [m for m in out if m.get("role") in ("user", "assistant")]
    # 프록시 압축이 안 물면 dreaming도 풀 히스토리를 그대로 보낸다 —
    # vanilla와 같은 창 초과 위험이 있고, 넘으면 §4.5의 "관측 결과 기록"이
    # 아니라 4xx 전파(narrator.py:34-37)로 런이 죽는다. 고정부는 압축돼도
    # 그대로라 히스토리 몫만 떼어 보정한다.
    total = _tokens(out)
    est = provider_tokens(fixed_tokens, max(0, total - fixed_tokens))
    ktext = _knowledge_text(out, full[-1]["content"])
    extra = {
        # 실제로 보낼 몫은 프록시 통과 후다 — 압축이 물면 raw보다 작다.
        # (to_wire 뒤엔 content가 parts라 세기 전에 out에서 잰다)
        "wire_tokens": total,
        "wire_tokens_raw": _tokens(raw),
        "provider_tokens": est,
        "provider_ctx_limit": provider_ctx_limit,
        "verdict_kind": verdict.kind,
        "quarantine": verdict.quarantine,
        "aligned": verdict.aligned,
        # sync.py:196-199의 적용 조건 그대로 재현한 계측치
        "compression_applied": bool(plan is not None and verdict.aligned
                                    and verdict.offset is not None),
        "knowledge_chars": len(ktext),
        "knowledge_text": ktext,
        "storage_root": str(live),      # decompose_miss가 받는 세션 디렉터리 부모
        "workspace": str(work),
        "workspace_temp": workdir is None,
    }
    if est > provider_ctx_limit:
        return Assembly(msgs=[], use_window=use_window, overflow=True,
                        extra=extra)
    return Assembly(msgs=msgs, use_window=use_window, extra=extra)


def _knowledge_text(out: List[Dict], user_text: str) -> str:
    """주입된 지식 블록 원문 — 프로브 발화 앞에 붙은 몫.

    `out`의 **마지막** user를 보면 안 된다: 뮈토스 프리필 꼬리가 그 자리다.
    inject_knowledge는 꼬리를 벗긴 뒤의 마지막 user, 곧 프로브 발화에
    `지식 + KNOWLEDGE_SEP`을 prepend한다 (sync.py:163,200).
    """
    suffix = KNOWLEDGE_SEP + user_text
    for m in reversed(out):
        c = m.get("content")
        if (m.get("role") == "user" and isinstance(c, str)
                and c.endswith(suffix)):
            return c[:-len(suffix)]
    return ""


# ── 디스패처 ───────────────────────────────────────────────────────────────

def build(variant: str, hs: FixedHaystack, card: Dict, preset: Dict,
          probe_turn: int, *, fixed_tokens: int, trim_budget: int,
          root: str = "", session: str = "",
          hypa_settings: Optional[hypa_mod.HypaSettings] = None,
          summarize: Optional[Callable[[List[Dict]], str]] = None,
          provider_ctx_limit: int = PROVIDER_CTX_LIMIT,
          workdir: Optional[str] = None) -> Assembly:
    """변형 이름 → Assembly. 유상태 변형만 root/session을 요구한다."""
    if variant == "vanilla":
        return assemble_vanilla(hs, card, preset, probe_turn, fixed_tokens,
                                provider_ctx_limit=provider_ctx_limit)
    if variant == "trim":
        return assemble_trim(hs, card, preset, probe_turn, trim_budget)
    if variant == "retrieval":
        return assemble_retrieval(hs, card, preset, probe_turn, trim_budget)
    if variant == "hypa":
        if hypa_settings is None:
            raise SystemExit("hypa 변형은 hypa_settings가 필요하다 "
                             "(hypa.load_hypa_settings(config.HYPA_EXPORT))")
        return assemble_hypa(
            hs, card, preset, probe_turn, hypa_fixed_tokens(preset, card),
            load_hypa_snapshot(root, probe_turn), settings=hypa_settings,
            max_context=hs.run_params["max_context"],
            max_tokens=hs.run_params["max_tokens"], summarize=summarize)
    if variant == "dreaming":
        if not session:
            # 없으면 JsonDirStorage._check_segment까지 내려가 정규식 위반으로
            # 죽는다 — hypa_settings 누락과 같은 종류의 실수에 같은 대접을.
            raise SystemExit("dreaming 변형은 session이 필요하다 "
                             "(인제스천이 쓴 ingest.SESSION 또는 "
                             "ingest-meta.json의 dreaming.session)")
        return assemble_dreaming(hs, card, preset, probe_turn, root, session,
                                 fixed_tokens=fixed_tokens,
                                 provider_ctx_limit=provider_ctx_limit,
                                 workdir=workdir)
    raise SystemExit(f"알 수 없는 변형: {variant}")
