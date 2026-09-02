"""HypaV3 동작 인스펙터 — 턴 1~N 전진 루프, 내부 상태 턴별 기록.

스펙: docs/superpowers/specs/2026-08-27-hypa-inspector-design.md
채점 없음, run.py 경로 무접촉. 출력은 dreaming_data/fixed_haystack/ 에만.

요약 캐시 규약: 기본은 캐시-히트만 소비(미스 = 실콜 0으로 열거 후 출력
차단·종료코드 1). --allow-llm 일 때만 미스분을 실제 호출한다.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

from benchmarks.eval import hypa, hypa_main, hypaplus, transport
from benchmarks.fixed_haystack import assemble, haystack, needle
from benchmarks.fixed_haystack.ingest import (_seed_history, decode_risup,
                                              resolve_path)

OUT_DIR = pathlib.Path("dreaming_data/fixed_haystack")


class _CacheOnlySummarize:
    """hypa.summarize 대역 — 디스크 캐시만 읽는다. 미스는 기록 후
    플레이스홀더 반환(배치 경계는 요약 텍스트와 무관하니 열거는 유효,
    출력은 미스 존재 시 폐기). 디스크에 아무것도 쓰지 않는다."""

    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self.misses: List[Dict[str, Any]] = []

    def __call__(self, send, batch, S, cache_dir=None):
        key = hypa.summary_cache_key(batch, S)
        p = self.cache_dir / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        self.misses.append({"key": key, "n_msgs": len(batch),
                            "memos": [c.get("memo") for c in batch]})
        return "[cache-miss placeholder]"


class _CacheOnlyPlus:
    """hypaplus.summarize_chunk 대역 — 디스크 캐시만 읽는다.

    청크 경계는 순수 인덱스 산술이라 요약 텍스트와 무관하다 → 미스가 있어도
    전체 열거는 유효하다. 출력은 미스 존재 시 폐기한다."""

    def __init__(self, cache_dir: pathlib.Path):
        self.cache_dir = pathlib.Path(cache_dir)
        self.misses: List[Dict[str, Any]] = []

    def __call__(self, batch, S, send, cache_dir=None):
        key = hypaplus.summary_cache_key(batch, S)
        p = self.cache_dir / f"{key}.txt"
        if p.exists():
            return p.read_text(encoding="utf-8")
        self.misses.append({"key": key, "n_msgs": len(batch),
                            "memos": [f"m{i}" for i in range(len(batch))]})
        return "[cache-miss placeholder]"


@contextlib.contextmanager
def _noop_ctx():
    yield


def run_inspection(hs, preset: Dict, card: Dict, impl: str,
                   S: hypa.HypaSettings, max_ctx: int, *,
                   fixed_tokens: Optional[int] = None,
                   embed: Optional[Callable] = None,
                   allow_llm: bool = False,
                   cache_dir: Optional[pathlib.Path] = None,
                   wire_fn: Optional[Callable] = None,
                   snapshot_check_root: Optional[pathlib.Path] = None,
                   ) -> Dict[str, Any]:
    """전진 루프. ingest_hypa와 같은 히스토리 규약(유저 append → step →
    어시 append, ingest.py:400~421)을 그대로 밟는다."""
    if fixed_tokens is None:
        fixed_tokens = assemble.hypa_fixed_tokens(preset, card)
    if wire_fn is None:
        wire_fn = lambda kept, memory: assemble._wire(
            preset, card, kept, memory=memory or "")

    cache_dir = cache_dir or (OUT_DIR / "hypa-cache")
    max_tok = hs.run_params["max_tokens"]
    step = hypa.hypa_step if impl == "exp" else hypa_main.hypa_step_main

    cache_only = None
    orig_summarize = hypa.summarize
    if not allow_llm:
        cache_only = _CacheOnlySummarize(cache_dir)
        hypa.summarize = cache_only        # 모듈 점 접근 패치 (CLAUDE.md §5)
    send = hypa._summarize_call if allow_llm else None

    hist = _seed_history(card)
    data: Dict[str, Any] = {"summaries": []}
    turns: List[Dict[str, Any]] = []
    snap_mismatch: List[int] = []
    try:
        ctx = assemble.bench_summary_cache() if allow_llm else _noop_ctx()
        with ctx:
            for t in hs.turns:
                if snapshot_check_root is not None:
                    p = assemble.hypa_snapshot_path(snapshot_check_root, t.turn)
                    if p.is_file():
                        snap = json.loads(p.read_text(encoding="utf-8"))
                        ours = [s["chatMemos"] for s in data["summaries"]]
                        theirs = [s["chatMemos"] for s in snap["summaries"]]
                        if ours != theirs:
                            snap_mismatch.append(t.turn)
                hist.append({"role": "user", "content": t.user})
                tr: List[Dict[str, Any]] = []
                kwargs: Dict[str, Any] = {"trace": tr}
                if impl == "main":
                    kwargs["embed"] = embed
                mem, kept, ksm, data, err = step(
                    hist, fixed_tokens, S, data, send, max_ctx, max_tok,
                    **kwargs)
                rec = {"turn": t.turn,
                       "memory_text": mem or "",
                       "kept_start_msg": ksm,
                       "n_summaries": len(data.get("summaries", [])),
                       "events": tr,
                       "metrics": data.get("metrics"),
                       "error": err}
                if not err:
                    rec["wire_tokens"] = assemble._tokens(wire_fn(kept, mem))
                turns.append(rec)
                if err:
                    break
                hist.append({"role": "assistant", "content": t.assistant})
    finally:
        hypa.summarize = orig_summarize

    return {
        "meta": {"impl": impl, "hsid": hs.haystack_id, "max_ctx": max_ctx,
                 "summary_model": transport.SUMMARY_MODEL,
                 "settings": {k: getattr(S, k) for k in vars(S)},
                 "snapshot_mismatch": snap_mismatch},
        "turns": turns,
        # 대화 원문 — HTML에서 memo 클릭 → 접힌 메시지 확인용
        "messages": [{"memo": f"m{i}", "role": m["role"], "content": m["content"]}
                     for i, m in enumerate(hist)],
        "misses": cache_only.misses if cache_only else [],
    }


def _plus_wire(preset: Dict, card: Dict, kept: List[Dict],
                memory: str) -> List[Dict]:
    """HypaPlus 주입은 프리셋 memory 슬롯이 아니라 첫 system append다 (스펙 §5)."""
    return hypaplus.inject(assemble._wire(preset, card, kept, memory=""),
                           memory or "")


def run_inspection_plus(hs, preset: Dict, card: Dict, S: hypaplus.PlusSettings,
                        *, embed: Optional[Callable] = None,
                        allow_llm: bool = False,
                        cache_dir: Optional[pathlib.Path] = None,
                        ) -> Dict[str, Any]:
    """HypaPlus 전진 루프. hypav3 경로와 같은 히스토리 규약을 밟는다."""
    cache_dir = cache_dir or (OUT_DIR / "hypa-cache")

    cache_only = None
    orig = hypaplus.summarize_chunk
    if not allow_llm:
        cache_only = _CacheOnlyPlus(cache_dir)
        hypaplus.summarize_chunk = cache_only     # 모듈 점 접근 패치
    send = hypa._summarize_call if allow_llm else None

    hist = _seed_history(card)
    state = hypaplus.new_state()
    turns: List[Dict[str, Any]] = []
    try:
        for t in hs.turns:
            hist.append({"role": "user", "content": t.user})
            tr: List[Dict[str, Any]] = []
            mem, kept, ksm, state, err = hypaplus.hypaplus_step(
                hist, S, state, send, embed=embed, cache_dir=cache_dir,
                trace=tr)
            nodes = hypaplus.get_nodes(state)
            rec = {"turn": t.turn, "memory_text": mem or "",
                   "kept_start_msg": ksm,
                   # 주입 단위가 노드다 — hypa의 요약 카드에 대응한다
                   "n_summaries": len(nodes),
                   "n_chunks": len(state["chunks"]),
                   "events": tr,
                   "metrics": {"nodeCount": len(nodes),
                               "chunkCount": len(state["chunks"]),
                               "failedChunks": sum(
                                   1 for c in state["chunks"]
                                   if c["summarized"] == "failed")},
                   "error": err,
                   "wire_tokens": assemble._tokens(
                       _plus_wire(preset, card, kept, mem))}
            turns.append(rec)
            hist.append({"role": "assistant", "content": t.assistant})
            # afterRequest — 이 규약에선 증명상 no-op이지만 발동하면 기록된다
            tr_post: List[Dict[str, Any]] = []
            hypaplus.hypaplus_post(hist, S, state, send, cache_dir, tr_post)
            if tr_post:
                rec["events"].extend(tr_post)
    finally:
        hypaplus.summarize_chunk = orig

    settings = {k: v for k, v in vars(S).items() if k != "summary_prompt"}
    settings["summary_prompt_sha256"] = hashlib.sha256(
        S.summary_prompt.encode("utf-8")).hexdigest()[:16]
    return {
        "meta": {"system": "hypaplus", "impl": "plus", "hsid": hs.haystack_id,
                 "max_ctx": None,       # HypaPlus는 maxContext를 읽지 않는다
                 "summary_model": transport.SUMMARY_MODEL,
                 "embed": "minilm" if embed is not None else "none",
                 "settings": settings, "snapshot_mismatch": []},
        "turns": turns,
        "messages": [{"memo": f"m{i}", "role": m["role"], "content": m["content"]}
                     for i, m in enumerate(hist)],
        "misses": cache_only.misses if cache_only else [],
    }


def _slug(meta: Dict[str, Any]) -> str:
    s = meta["settings"]
    if meta.get("system") == "hypaplus":
        base = f"hypa-inspect-{meta['hsid']}-hypaplus-{meta['embed']}"
        tags = []
        if s.get("max_memory_tokens") not in (None, 20000):
            tags.append(f"mmt{s['max_memory_tokens']}")
        if s.get("chunk_size") not in (None, 30):
            tags.append(f"cs{s['chunk_size']}")
        if s.get("time_scoring_mode") not in (None, "minute"):
            tags.append(f"tsm{s['time_scoring_mode']}")
        return "-".join([base] + tags)
    base = f"hypa-inspect-{meta['hsid']}-{meta['impl']}"
    tags = []
    if s.get("memory_tokens_ratio") not in (None, 0.39):
        tags.append(f"mtr{s['memory_tokens_ratio']}")
    if s.get("extra_summarization_ratio") not in (None, 0.01):
        tags.append(f"esr{s['extra_summarization_ratio']}")
    return "-".join([base] + tags)


def write_outputs(result: Dict[str, Any], out_dir: pathlib.Path = OUT_DIR,
                  ledger: Optional[List[Dict]] = None) -> bool:
    """미스가 있으면 아무것도 쓰지 않는다 — 미완성 관찰을 남기지 않는다."""
    if result["misses"]:
        return False
    if ledger is not None:                # 니들 생존 오라클 (LLM 0콜)
        needle.annotate(result, ledger)
        result["meta"]["needle"] = needle.summarize(result)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(result["meta"])
    (out_dir / f"{slug}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"{slug}.html").write_text(render_html(result), encoding="utf-8")
    return True


def _main_plus(a, hs, preset: Dict, card: Dict) -> None:
    """HypaPlus 경로 — maxContext·메토비 같은 hypa 노브는 존재하지 않는다."""
    S = hypaplus.PlusSettings(
        summary_prompt=hypaplus.load_default_prompt(a.plugin_src))
    if a.chunk_size is not None:
        S.chunk_size = a.chunk_size
    if a.memory_tokens is not None:
        S.max_memory_tokens = a.memory_tokens
    if a.time_decay_days is not None:
        S.time_decay_days = a.time_decay_days
    if a.time_mode is not None:
        S.time_scoring_mode = a.time_mode
    if a.summary_model:
        transport.SUMMARY_MODEL = a.summary_model   # 캐시 키가 이 값을 본다

    embed = hypa.embed if a.embed == "minilm" else None

    result = run_inspection_plus(hs, preset, card, S, embed=embed,
                                 allow_llm=a.allow_llm)
    if result["misses"]:
        print(f"캐시 미스 {len(result['misses'])}건 — 출력 안 씀. "
              f"실콜하려면 --allow-llm.")
        for m in result["misses"]:
            print(f"  {m['key'][:12]}… msgs={m['n_msgs']}")
        sys.exit(1)
    # 요약 실패도 미완성 관찰이다 — 캐시 미스와 같은 등급으로 막는다.
    # (플랜 결함 교정: 키 부재로 24콜이 전부 조용히 failed 처리됐는데도
    #  misses가 0이라 "완주"로 기록될 뻔했다.)
    fails = [e for t_ in result["turns"] for e in t_["events"]
             if e["ev"] == "chunk" and e["status"] == "failed"]
    if fails:
        print(f"요약 실패 청크 {len(fails)}건 — 출력 안 씀.")
        for f in fails:
            print(f"  청크 #{f['chunk_id']} m{f['start']}~m{f['end']}: "
                  f"{f.get('fail_reason')}")
        sys.exit(3)
    # 절단본은 캐시되지 않는다 → 다음 런이 다른 요약을 받는다. 축 간 비교가
    # 오염되므로 재현 불가 관찰로 간주하고 막는다 (실측으로 당한 결함).
    trunc = [e for t_ in result["turns"] for e in t_["events"]
             if e["ev"] == "chunk" and e.get("truncated")]
    if trunc:
        print(f"절단된 요약 {len(trunc)}건 — 출력 안 씀 (재현 불가). "
              f"hypaplus.SUMMARY_MAX_TOKENS를 올릴 것.")
        for f in trunc:
            print(f"  청크 #{f['chunk_id']} m{f['start']}~m{f['end']}")
        sys.exit(4)
    write_outputs(result, pathlib.Path(a.out_dir), hs.ledger)
    last = result["turns"][-1]
    print(f"완주 {len(result['turns'])}턴, 노드 {last['n_summaries']}개, "
          f"청크 {last['n_chunks']}개, 미스 0 → {_slug(result['meta'])}.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs", required=True)
    ap.add_argument("--system", default="hypav3",
                    choices=["hypav3", "hypaplus"])
    ap.add_argument("--impl", default="exp", choices=["exp", "main"])
    ap.add_argument("--embed", default="none", choices=["none", "minilm"],
                    help="hypaplus 전용. 배포 기본값은 none(임베딩 미설정)")
    ap.add_argument("--plugin-src", default=hypaplus.DEFAULT_PLUGIN_SRC)
    ap.add_argument("--chunk-size", type=int)
    ap.add_argument("--memory-tokens", type=int)
    ap.add_argument("--time-decay-days", type=float)
    ap.add_argument("--time-mode", choices=["minute", "day", "none"])
    ap.add_argument("--memory-ratio", type=float)
    ap.add_argument("--extra-ratio", type=float)
    ap.add_argument("--max-context", type=int)
    ap.add_argument("--summary-model")
    ap.add_argument("--embed-model")
    ap.add_argument("--allow-llm", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args(argv)

    from benchmarks.eval.config import HYPA_EXPORT
    hs = haystack.load(f"dreaming_data/fixed_haystack/{a.hs}.json")
    preset = decode_risup(resolve_path(hs.preset_path))
    card = json.loads(pathlib.Path(resolve_path(hs.card_path))
                      .read_text(encoding="utf-8"))

    if a.system == "hypaplus":
        return _main_plus(a, hs, preset, card)

    S = hypa.load_hypa_settings(HYPA_EXPORT)
    overridden = False
    if a.memory_ratio is not None:
        S.memory_tokens_ratio, overridden = a.memory_ratio, True
    if a.extra_ratio is not None:
        S.extra_summarization_ratio, overridden = a.extra_ratio, True
    max_ctx = a.max_context or hs.run_params["max_context"]
    if a.max_context is not None:
        overridden = True
    if a.summary_model:
        transport.SUMMARY_MODEL = a.summary_model   # 캐시 키가 이 값을 본다
        overridden = True
    embed = None
    if a.embed_model:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(a.embed_model)
        m.max_seq_length = 512
        embed = lambda texts: m.encode(list(texts), normalize_embeddings=True)
        hypa._DEFAULT_EMBED = embed                 # exp 경로에도 도달
        overridden = True

    # 스냅샷 대조는 무오버라이드 exp에서만 유효 (스펙 §1 검증 앵커)
    snap_root = (pathlib.Path(f"dreaming_data/fixed_haystack/{a.hs}")
                 if (a.impl == "exp" and not overridden) else None)

    result = run_inspection(hs, preset, card, a.impl, S, max_ctx,
                            embed=embed, allow_llm=a.allow_llm,
                            snapshot_check_root=snap_root)

    if result["misses"]:
        print(f"캐시 미스 {len(result['misses'])}건 — 출력 안 씀. "
              f"실콜하려면 --allow-llm.")
        for m in result["misses"]:
            print(f"  {m['key'][:12]}… msgs={m['n_msgs']} memos={m['memos']}")
        sys.exit(1)
    if result["meta"]["snapshot_mismatch"]:
        print(f"스냅샷 불일치 턴: {result['meta']['snapshot_mismatch']}")
        sys.exit(2)
    write_outputs(result, pathlib.Path(a.out_dir), hs.ledger)
    errs = [t["turn"] for t in result["turns"] if t.get("error")]
    print(f"완주 {len(result['turns'])}턴, 요약 "
          f"{result['turns'][-1]['n_summaries']}개, 에러턴 {errs or '없음'}, "
          f"미스 0 → {_slug(result['meta'])}.json")



# ── HTML 뷰 (자급자족 단일 파일 — 외부 리소스 0) ──────────────────────────

_HTML_TEMPLATE = """<!doctype html>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 0; display: flex;
         height: 100vh; color: #222; }
  #left { width: 46%; overflow-y: auto; border-right: 1px solid #ccc;
          padding: 12px; box-sizing: border-box; }
  #right { flex: 1; overflow-y: auto; padding: 12px; box-sizing: border-box; }
  h1 { font-size: 16px; margin: 4px 0 8px; }
  table.meta td { font-size: 12px; padding: 1px 8px 1px 0; color: #444;
                  vertical-align: top; }
  svg { width: 100%; background: #fafafa; border: 1px solid #ddd; }
  .legend { font-size: 11px; color: #555; margin: 4px 0 10px; }
  .turnrow { padding: 3px 6px; font-size: 12px; cursor: pointer;
             border-bottom: 1px solid #eee; display: flex; gap: 8px; }
  .turnrow:hover { background: #eef; }
  .turnrow.sel { background: #dde6ff; }
  .turnrow .b { color: #c33; font-weight: bold; }
  pre { white-space: pre-wrap; background: #f6f6f6; padding: 8px;
        font-size: 12px; border: 1px solid #e0e0e0; }
  table.ev { border-collapse: collapse; font-size: 11px; margin: 6px 0; }
  table.ev td, table.ev th { border: 1px solid #ddd; padding: 2px 6px;
                             text-align: left; }
  .tag { display: inline-block; background: #e8e8e8; border-radius: 3px;
         padding: 0 5px; margin-right: 4px; font-size: 11px; }
</style>
<div id="left">
  <h1>hypa 인스펙터 — <span id="t-hsid"></span> / <span id="t-impl"></span></h1>
  <div id="filedesc" style="font-size:12px; background:#f2f5ff; border:1px solid #ccd6f2;
       border-radius:6px; padding:8px 10px; margin:6px 0; line-height:1.55"></div>
  <table class="meta" id="meta"></table>
  <svg id="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
  <div class="legend">● 압축 발동 턴 · <span style="color:#36c">요약/노드 개수</span> ·
    <span style="color:#c33">kept_start_msg</span> ·
    <span style="color:#999">wire_tokens(우축)</span> ·
    <span style="color:#2a2">기억 커버리지(0~1)</span></div>
  <details style="font-size:12px; margin:4px 0 10px; color:#444">
    <summary style="cursor:pointer">용어 설명 (펼치기)</summary>
    <ul style="margin:6px 0; padding-left:18px; line-height:1.6">
      <li><b>요약 N</b> — 이 턴 종료 시점까지 쌓인 요약 카드 수.
        카드 1장 = 옛 메시지 최대 8개를 접은 것.</li>
      <li><b>원문시작(kept_start) N</b> — N번 메시지부터만 원문이 프롬프트에
        남음. 그 앞(m0~m(N−1))은 프롬프트에서 사라지고 요약으로만 존재.</li>
      <li><b>wire N</b> — 이 턴에 LLM으로 나간 전체 프롬프트 토큰
        (프리셋+카드+메모리 카드+남은 원문). max_ctx 대비로 읽을 것.</li>
      <li><b>압축 배치</b> — 이 턴에 요약으로 접힌 메시지 묶음(memo 목록)과
        접히기 전 원문 토큰 수.</li>
      <li><b>important / recent / similar</b> — 메모리 카드에 넣을 요약을
        고르는 3단계. 숫자는 요약 인덱스(0부터). reserved=그 단계 토큰 예산,
        used=실제 사용.</li>
      <li><b>chunk / cos</b> — similar 검색 근거. chunk "a-b" = 후보 a번째
        요약의 b번째 문단, cos = 그 문단과 쿼리(최근 메시지)의 코사인 유사도
        (1에 가까울수록 닮음, 0.3 이하는 사실상 무관).</li>
      <li><b>skipped: no_chunks</b> — similar 차례에 후보 요약이 0장
        (important·recent가 전부 가져감) → 검색 자체가 일 없음.
        예산이 넉넉하면(메토비 큼) recent가 다 먹어 RAG가 아사하는 역설.</li>
      <li><b>과요약 직전 중단</b> — 다음 배치를 접기 직전 "이미 한도 안이고,
        이 배치까지 접으면 목표(max_ctx×(1−추가요약비율)) 아래로 뚫는다" 싶으면
        접지 않고 원문으로 살려둠. 과잉 압축 방지 브레이크.</li>
      <li><b>memo(m숫자)</b> — 메시지 일련번호이자 요약 카드의 출처 영수증
        (chatMemos). 카드마다 "어느 메시지들을 접었나"가 붙어 재개 지점 계산과
        원문 삭제 시 카드 무효화에 쓰임.</li>
      <li><b>캐시 관점</b> — 압축 턴엔 ①메모리 카드 내용 변경 ②원문 창 점프가
        동시에 일어나 프리픽스 캐시가 깨짐. 요약 없는 턴도 similar 선택이
        바뀌면 카드가 변해 깨질 수 있음.</li>
      <li><b>기억 커버리지</b> — 이 턴까지 이미 일어난 원장 팩트 중 몇 %가
        (주입된 기억 블록 ∪ 남은 원문 창) 안에 문자열로 실제 존재하는가.
        판정 LLM 없음. <b>memory_only</b>는 원문 창엔 없고 기억 블록에만
        있는 것 — 순수 검색 기여분.</li>
    </ul>
    <ul style="margin:6px 0; padding-left:18px; line-height:1.6" id="plus-gloss" hidden>
      <li><b>노드</b> — HypaPlus의 기억 최소 단위. 요약 LLM이 청크를
        <code>&lt;event&gt;</code>(시간·장소·인물·내용·대사)와
        <code>&lt;facts&gt;</code> 블록으로 뱉고, 그 블록 하나가 노드 1개.
        검색·주입이 전부 노드 단위다(하이파는 요약 카드 단위).</li>
      <li><b>청크</b> — 메시지 <b>정확히 30개</b> 고정 창. 미요약 메시지가
        40개 쌓이면 앞 30개를 한 번에 요약한다. <b>maxContext와 무관</b> —
        컨텍스트가 아무리 넉넉해도 같은 시점에 접는다.</li>
      <li><b>combined 점수</b> — <code>1 − (1−time)^tw × (1−sim)^sw</code>.
        sim=쿼리와의 코사인 유사도, time=작중 시각 감쇠. 둘 중 하나만 높아도
        점수가 오르는 형태(OR에 가까움).</li>
      <li><b>작중 시각 감쇠</b> — <code>1/√(1+Δ일수/반감기)</code>. 현실
        시각이 아니라 노드에 적힌 <b>작중 날짜</b>로 잰다. 단
        <code>YYYY-MM-DD HH:mm</code> 꼴만 파싱된다 — "3일차 저녁" 같은 값은
        저장은 되고 점수에서만 죽어 <b>time이 0.5로 고정</b>된다.</li>
      <li><b>유사도 0.5 고정</b> — 임베딩 엔드포인트 미설정이 배포 기본값이다.
        그러면 모든 노드의 sim이 0.5가 되어 순위가 시간축만으로 결정된다.
        (플러그인 문서가 말하는 "키워드 폴백"은 실제로 구현돼 있지 않다.)</li>
      <li><b>예산 skip</b> — 노드가 예산을 넘으면 자르지 않고 건너뛴다.
        뒤의 더 작은 노드가 대신 들어간다.</li>
      <li><b>주입 위치</b> — 프리셋에 <code>[HypaPlus.memory]</code>
        플레이스홀더가 없으므로 <b>첫 system 메시지 끝에 append</b>된다.
        프리픽스 맨 앞이 매번 바뀌는 셈.</li>
    </ul>
  </details>
  <div id="turns"></div>
</div>
<div id="right"><em style="font-size:13px">턴을 클릭하면 상세.</em></div>
<script>
const DATA = __DATA__;
const T = DATA.turns;
function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;"); }
// 대화 원문 인덱스 (memo -> 메시지)
const MSG = {};
(DATA.messages || []).forEach(m => MSG[m.memo] = m);
// meta
const IMPL_KO = {
  exp: "실험 구현(exp) — 뮈토스 하이파 V5 export가 실제로 켜는 경로(useExperimentalImpl=true). 검색 쿼리를 문단 단위로 쪼갬.",
  main: "기본 구현(main) — 프리셋 없이 쓰는 유저의 기본 경로. 검색 쿼리를 메시지 통짜로 사용, 요약 0개여도 빈 메모리 카드 삽입.",
};
const HS_KO = {
  "fix-drm-r0": "dreaming 변형이 생성한 100턴 고정 RP 대화",
  "fix-van-r0": "vanilla(무기억) 변형이 생성한 100턴 고정 RP 대화",
};
const IS_PLUS = DATA.meta.system === "hypaplus";
const EMBED_KO = {
  none: "임베딩 미설정 — 플러그인 배포 기본값. 모든 노드의 유사도가 0.5로 " +
        "고정되어 순위가 작중 시간 감쇠만으로 결정된다.",
  minilm: "MiniLM 임베딩 — 하이파 관찰 런과 같은 모델. 유사도 축이 살아난다.",
};
const S_KO = {
  memory_tokens_ratio: "메토비 — 메모리 카드 토큰 예산 = max_ctx × 이 값",
  extra_summarization_ratio: "추가요약비율 — 압축 발동 시 max_ctx×(1−값)까지 몰아서 접음(히스테리시스)",
  max_chats_per_summary: "카드 1장에 접는 메시지 수",
  query_chat_count: "검색 쿼리로 쓰는 최근 메시지 수 = 절대 요약되지 않는 보호구간",
  recent_memory_ratio: "예산 중 recent(최신 요약부터) 몫",
  similar_memory_ratio: "예산 중 similar(유사도 검색) 몫 — recent가 먼저 집으므로 남는 요약이 없으면 아사",
  do_not_summarize_user_message: "유저 메시지 요약 제외 여부",
  summarization_prompt: "요약 프롬프트(빈 값 = RisuAI 기본 프롬프트)",
  summary_chunk_separator: "요약을 검색 조각으로 쪼개는 구분자",
  max_ctx: "컨텍스트 한도 — 이걸 넘으면 압축 발동",
  summary_model: "요약을 만든 LLM",
  chunk_size: "청크 1개에 접는 메시지 수 (고정 창)",
  summarize_threshold: "미요약 메시지가 이만큼 쌓이면 압축 발동",
  max_memory_tokens: "주입 예산 — 자체 문자 가중 추정(한글 0.8/CJK 0.9/기타 0.25)",
  include_user_messages: "유저 메시지도 요약 대상에 포함",
  embedding_context_messages: "검색 쿼리로 쓰는 최근 메시지 수",
  time_decay_days: "작중 시간 감쇠 반감기(일)",
  time_scoring_mode: "시간 점수 정밀도 — minute/day/none(끔)",
  time_weight: "시간 점수 지수 가중",
  similarity_weight: "유사도 점수 지수 가중",
  event_format: "event 노드 주입 템플릿",
  facts_format: "facts 노드 주입 템플릿",
  summary_prompt_sha256: "요약 프롬프트 지문(전문은 레포에 두지 않음)",
  embed: "임베딩 축",
};
document.getElementById("t-hsid").textContent = DATA.meta.hsid;
document.getElementById("t-impl").textContent = IS_PLUS
  ? `HypaPlus / embed=${DATA.meta.embed}` : DATA.meta.impl;
document.getElementById("filedesc").innerHTML = IS_PLUS
  ? `<b>이 파일:</b> ${HS_KO[DATA.meta.hsid] || DATA.meta.hsid}를 ` +
    `<b>HypaPlus v1.0.9</b>(아카 플러그인)로 한 턴씩 재생하며 청킹·노드 검색 ` +
    `내부를 기록한 것.<br>HypaPlus는 <b>maxContext를 읽지 않는다</b> — ` +
    `압축은 메시지 개수(30개 청크 / 임계 40)로, 주입은 자체 토큰 추정 ` +
    `예산으로만 결정된다.<br>${EMBED_KO[DATA.meta.embed] || ""}`
  : `<b>이 파일:</b> ${HS_KO[DATA.meta.hsid] || DATA.meta.hsid}를 ` +
    `<b>${DATA.meta.impl}</b> 구현으로 한 턴씩 재생하며 압축·검색 내부를 기록한 것.<br>` +
    `${IMPL_KO[DATA.meta.impl] || ""}`;
if (IS_PLUS) {
  document.querySelector("h1").firstChild.textContent = "HypaPlus 인스펙터 — ";
  const g = document.getElementById("plus-gloss");
  if (g) g.hidden = false;
}
document.getElementById("meta").innerHTML =
  Object.entries({...(IS_PLUS ? {embed: DATA.meta.embed}
                              : {max_ctx: DATA.meta.max_ctx}),
                  summary_model: DATA.meta.summary_model,
                  ...DATA.meta.settings})
    .map(([k,v]) => `<tr><td><b>${esc(k)}</b> = ${esc(v)}</td>` +
                    `<td>${esc(S_KO[k] || "")}</td></tr>`).join("");
// chart
(function () {
  if (!T.length) return;
  const W = 800, H = 260, P = 28;
  const xs = t => P + (W - 2*P) * (t - T[0].turn) / Math.max(1, T[T.length-1].turn - T[0].turn);
  const maxSum = Math.max(1, ...T.map(r => r.n_summaries));
  const maxKsm = Math.max(1, ...T.map(r => r.kept_start_msg));
  const maxWire = Math.max(1, ...T.map(r => r.wire_tokens || 0));
  const y = (v, m) => H - P - (H - 2*P) * v / m;
  const line = (pts, color, w) =>
    `<polyline fill="none" stroke="${color}" stroke-width="${w}" points="${pts}"/>`;
  let svg = "";
  svg += line(T.map(r => `${xs(r.turn)},${y(r.wire_tokens||0, maxWire)}`).join(" "), "#bbb", 1);
  svg += line(T.map(r => `${xs(r.turn)},${y(r.kept_start_msg, maxKsm)}`).join(" "), "#c33", 1.5);
  svg += line(T.map(r => `${xs(r.turn)},${y(r.n_summaries, maxSum)}`).join(" "), "#36c", 2);
  const COMPRESS_EV = IS_PLUS ? "chunk" : "batch";
  const cov = T.filter(r => r.needle && r.needle.probed.coverage !== null);
  if (cov.length)
    svg += line(cov.map(r => `${xs(r.turn)},${y(r.needle.probed.coverage, 1)}`).join(" "),
                "#2a2", 1.5);
  for (const r of T)
    if (r.events.some(e => e.ev === COMPRESS_EV))
      svg += `<circle cx="${xs(r.turn)}" cy="${y(r.n_summaries, maxSum)}" r="4" fill="#c33"/>`;
  svg += `<text x="${P}" y="14" font-size="10" fill="#666">${IS_PLUS ? "노드" : "요약"} max ${maxSum} · ksm max ${maxKsm} · wire max ${maxWire}</text>`;
  document.getElementById("chart").innerHTML = svg;
})();
// turn list
const list = document.getElementById("turns");
list.innerHTML = T.map((r, i) => {
  const nb = r.events.filter(e => e.ev === (IS_PLUS ? "chunk" : "batch")).length;
  const cv = r.needle && r.needle.probed.coverage !== null
    ? `<span title="기억 커버리지 — 원장 팩트 중 기억+원문에 살아 있는 비율"
             style="color:#2a2">${(r.needle.probed.coverage*100).toFixed(0)}%</span>` : "";
  return `<div class="turnrow" data-i="${i}">
    <span>t${r.turn}</span>
    <span title="${IS_PLUS ? "누적 노드 수 (기억 최소 단위)" : "누적 요약 카드 수"}">${IS_PLUS ? "노드" : "요약"} ${r.n_summaries}</span>
    <span title="원문이 남는 시작 메시지 인덱스">원문시작 ${r.kept_start_msg}</span>
    ${cv}
    ${nb ? `<span class="b" title="이 턴에 압축 발동">압축×${nb}</span>` : ""}
    ${r.error ? `<span class="b">ERR</span>` : ""}</div>`;
}).join("");
list.addEventListener("click", ev => {
  const row = ev.target.closest(".turnrow");
  if (!row) return;
  document.querySelectorAll(".turnrow.sel").forEach(e => e.classList.remove("sel"));
  row.classList.add("sel");
  show(T[+row.dataset.i]);
});
function evTable(rows, cols) {
  return `<table class="ev"><tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>` +
    rows.map(r => `<tr>${cols.map(c=>`<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("") +
    `</table>`;
}
const STOP_KO = {target_reached: "목표 토큰 도달", over_summarize_break: "과요약 직전 중단(배치 미요약)",
                 min_messages_break: "보호구간 도달(최근 메시지 불가침)"};
function show(r) {
  let h = `<h1>턴 ${r.turn}</h1>`;
  h += `<div><span class="tag" title="${IS_PLUS ? "누적 노드 수" : "누적 요약 카드 수"}">` +
       `${IS_PLUS ? `노드 ${r.n_summaries}개` : `요약 카드 ${r.n_summaries}장`}</span>` +
       (IS_PLUS ? `<span class="tag" title="지금까지 접은 30개 청크 수">청크 ${r.n_chunks}개</span>` : "") +
       `<span class="tag" title="이 앞 메시지는 프롬프트에서 사라지고 요약으로만 존재">원문시작 m${r.kept_start_msg}</span>` +
       `<span class="tag" title="이 턴에 LLM으로 나간 전체 프롬프트 토큰">wire ${r.wire_tokens ?? "-"} tok</span></div>`;
  h += `<div class="legend">m0~m${Math.max(0, r.kept_start_msg-1)}은 접혀서 아래 memory_text로만 전달됨. ` +
       `wire = 프리셋+카드+메모리+남은 원문 합계 토큰.</div>`;
  if (r.needle) {
    const n = r.needle.probed;
    h += `<div class="legend" style="color:#2a2">기억 커버리지 ` +
         `<b>${n.coverage === null ? "-" : (n.coverage*100).toFixed(0)+"%"}</b> — ` +
         `이 턴까지의 원장 팩트 ${n.n_eligible}건 중 기억블록 ${n.in_memory} · ` +
         `원문창 ${n.in_window} · 합쳐 ${n.covered} 생존, ${n.lost} 유실. ` +
         `그중 기억블록에만 있는 것(순수 검색 기여) ${n.memory_only}건. ` +
         `[전체 원장 기준 ${r.needle.all.covered}/${r.needle.all.n_eligible}]</div>`;
  }
  // 이 턴의 대화 원문 (m0=인사, 턴 t: user=m(2t-1), assistant=m(2t))
  const um = MSG["m" + (2*r.turn - 1)], am = MSG["m" + (2*r.turn)];
  if (um || am) {
    h += `<details><summary style="cursor:pointer;font-size:13px">이 턴의 대화 원문</summary>` +
         (um ? `<pre><b>user (m${2*r.turn-1}):</b>\n${esc(um.content)}</pre>` : "") +
         (am ? `<pre><b>assistant (m${2*r.turn}):</b>\n${esc(am.content)}</pre>` : "") +
         `</details>`;
  }
  if (r.error) h += `<pre>ERROR: ${esc(r.error)}</pre>`;
  if (IS_PLUS) { h += plusDetail(r); h += memoryBlock(r);
                 document.getElementById("right").innerHTML = h; return; }
  const batches = r.events.filter(e => e.ev === "batch");
  if (batches.length) {
    h += `<h3 style="font-size:13px">압축 배치 — 이 턴에 요약으로 접힌 메시지들</h3>` +
         `<div class="legend">memo를 클릭하면 접히기 전 원문이 열림</div>`;
    batches.forEach(b => {
      h += `<div style="margin:4px 0">` +
           b.idx.map(m => `<span class="tag" style="cursor:pointer;background:#dbe7ff" data-memo="${m}">${m}</span>`).join("") +
           `<span class="tag">원문 ${b.tokens} tok → 카드 1장</span></div>`;
    });
    h += `<div id="memoview"></div>`;
  }
  const stops = r.events.filter(e => e.ev === "batch_stop");
  if (stops.length) h += `<div class="legend">압축 중단 사유: ${stops.map(s=>STOP_KO[s.reason]||s.reason).join(", ")}</div>`;
  const res = r.events.find(e => e.ev === "reserve");
  if (res) h += `<div class="legend">메모리 예산: 카드에 쓸 수 있는 토큰 ${res.available} (예약 후 current ${res.current})</div>`;
  for (const stage of ["important", "recent"]) {
    const e = r.events.find(x => x.ev === stage);
    if (e && (e.picked.length || e.ev === "recent"))
      h += `<div class="legend">${stage} 선택: 요약 [${e.picked}]` +
           (e.reserved !== undefined ? ` (예산 ${e.reserved} 중 ${e.used} 사용)` : "") + `</div>`;
  }
  const sim = r.events.find(x => x.ev === "similar");
  if (sim) {
    if (sim.skipped) {
      h += `<h3 style="font-size:13px">similar(검색) — 실행 안 됨</h3>` +
           `<div class="legend">사유: ${sim.skipped === "no_chunks"
             ? "후보 요약 0장 (important·recent가 전부 가져감)"
             : "쿼리 없음"} · 예산 ${sim.reserved}는 미사용</div>`;
    } else {
      h += `<h3 style="font-size:13px">similar(검색) — 요약 [${sim.picked}] 선택 (예산 ${sim.reserved} 중 ${sim.used} 사용)</h3>`;
      h += `<div class="legend">부모 요약 최종 순위(RRF): [${sim.rrf_rank}] — 이 순서대로 예산에 담김</div>`;
      sim.queries.forEach((q, qi) => {
        const top = (sim.chunk_scores[qi] || []).slice(0, 5);
        h += `<div class="legend">쿼리 q${qi} (가중치 ${q.weight.toFixed(3)}): "${esc(q.content.slice(0, 90))}"</div>`;
        h += evTable(top.map(([id, sc]) => ({"문단 조각": id, "코사인 유사도": sc})),
                     ["문단 조각", "코사인 유사도"]);
      });
    }
  }
  if (r.metrics) h += `<pre>${esc(JSON.stringify(r.metrics))}</pre>`;
  h += memoryBlock(r);
  document.getElementById("right").innerHTML = h;
}
function memoryBlock(r) {
  return `<h3 style="font-size:13px">memory_text — 이 턴 프롬프트에 실제로 붙은 ` +
         `${IS_PLUS ? "노드 블록" : "메모리 카드"} 전문</h3>` +
         `<pre>${esc(r.memory_text || "(없음)")}</pre>`;
}
// ── HypaPlus 턴 상세 ─────────────────────────────────────────────────────
function plusDetail(r) {
  let h = "";
  const chunks = r.events.filter(e => e.ev === "chunk");
  if (chunks.length) {
    h += `<h3 style="font-size:13px">압축 — 이 턴에 접힌 30개 고정 청크</h3>` +
         `<div class="legend">memo를 클릭하면 접히기 전 원문이 열림</div>`;
    chunks.forEach(c => {
      const memos = [];
      for (let i = c.start; i <= c.end; i++) memos.push("m" + i);
      const nodes = c.nodes || [];          // 구 JSON엔 없다 — 개수만 보인다
      const ev = nodes.filter(n => n.type === "event");
      const fa = nodes.filter(n => n.type === "facts");
      h += `<div style="margin:4px 0"><span class="tag">청크 #${c.chunk_id} ` +
           `(${c.status}) → 노드 ${c.n_nodes}개` +
           (nodes.length ? ` = 사건 ${ev.length} + 사실 ${fa.length}` : "") + `</span>` +
           memos.map(m => `<span class="tag" style="cursor:pointer;background:#dbe7ff" ` +
                          `data-memo="${m}">${m}</span>`).join("") + `</div>`;
      [["사건 &lt;event&gt;", ev, "#fff3e0"], ["사실 &lt;facts&gt;", fa, "#e8f5e9"]].forEach(([label, ns, bg]) => {
        if (!ns.length) return;
        h += `<details style="margin:2px 0 4px 8px"><summary style="cursor:pointer;font-size:12px">` +
             `${label} ${ns.length}개</summary>`;
        ns.forEach(n => {
          const meta = [n.time, n.location, (n.characters || []).join(", ")].filter(Boolean).map(esc).join(" · ");
          h += `<div style="margin:3px 0;padding:4px 6px;background:${bg};border-radius:4px;font-size:12px">` +
               `<b>#${esc(n.id)}</b> <span class="legend">${meta}</span><div>${esc(n.content)}</div>` +
               (n.dialogues || []).map(d => `<div class="legend">${esc(d.speaker)}: "${esc(d.text)}"</div>`).join("") +
               `</div>`;
        });
        h += `</details>`;
      });
      if (c.status === "failed")
        h += `<div class="legend" style="color:#b00">요약 실패 — 이 30개는 ` +
             `영구히 요약에서 빠지고 원문 창에서도 사라진다(기억 유실 경로).</div>`;
    });
    h += `<div id="memoview"></div>`;
  }
  const rt = r.events.find(e => e.ev === "retrieve");
  if (rt) {
    if (rt.skipped) {
      h += `<div class="legend">검색 실행 안 됨 — 아직 노드가 없다.</div>`;
    } else {
      const pct = rt.budget ? (rt.used_tokens / rt.budget * 100).toFixed(1) : "-";
      h += `<h3 style="font-size:13px">검색 — 노드 ${rt.n_nodes}개 중 ${rt.n_selected}개 주입</h3>`;
      h += `<div class="legend">예산 ${rt.used_tokens} / ${rt.budget} tok (<b>${pct}%</b> 사용)` +
           ` · 예산초과 건너뜀 ${rt.n_skipped_budget} · 바닥컷(0.05 미만) ${rt.n_skipped_floor}` +
           ` · 임베딩 ${rt.embedded ? "사용" : "<b>미사용 → 유사도 0.5 고정</b>"}</div>`;
      if (rt.n_skipped_budget === 0 && rt.n_selected === rt.n_nodes)
        h += `<div class="legend" style="color:#b60">예산이 남아 <b>노드 전량이 주입</b>됐다 — ` +
             `이 턴엔 순위(임베딩·시간감쇠)가 결과에 영향을 주지 못한다.</div>`;
      h += `<div class="legend">기준 시각(최신 노드) ${esc(rt.ref_time || "(없음)")}` +
           ` — 파싱 ${rt.ref_time_parsed ? "성공" : "<b>실패</b>"}</div>`;
    }
  }
  const td = r.events.find(e => e.ev === "time_diag");
  if (td) {
    h += `<div class="legend">작중 시각 진단: 노드 ${td.n_nodes}개 중 파싱 성공 ` +
         `<b>${td.n_time_parsed}</b> / 실패 ${td.n_time_failed}` +
         ` · facts 시각 폴백 ${td.n_facts_fallback}건</div>`;
    if (td.n_nodes && td.n_time_parsed === 0)
      h += `<div class="legend" style="color:#b00">전부 파싱 실패 — <b>시간축이 죽었다</b>. ` +
           `모든 time 점수가 0.5로 고정된다.</div>`;
  }
  const sc = r.events.find(e => e.ev === "scores");
  if (sc && sc.rows.length) {
    const top = sc.rows.slice(0, 12);
    h += `<h3 style="font-size:13px">노드 점수 상위 ${top.length} / ${sc.rows.length}</h3>`;
    h += evTable(top.map(([id, sim, tm, cb, pk]) => ({
      "노드": id, "유사도(cos)": sim, "시간점수": tm, "combined": cb,
      "주입": pk ? "O" : "" })),
      ["노드", "유사도(cos)", "시간점수", "combined", "주입"]);
  }
  if (r.metrics) h += `<pre>${esc(JSON.stringify(r.metrics))}</pre>`;
  return h;
}
// 압축 배치 memo 클릭 → 원문 토글
document.getElementById("right").addEventListener("click", ev => {
  const memo = ev.target.getAttribute && ev.target.getAttribute("data-memo");
  if (!memo) return;
  const box = document.getElementById("memoview");
  const m = MSG[memo];
  if (!box) return;
  box.innerHTML = m
    ? `<pre><b>${memo} (${esc(m.role)}):</b>\n${esc(m.content)}</pre>`
    : `<pre>${memo}: 원문 없음 (이 json은 원문 미포함 — 재생성 필요)</pre>`;
});
</script>
"""


def render_html(result: Dict[str, Any]) -> str:
    """자급자족 HTML — DATA 인라인, 외부 리소스 0."""
    data = json.dumps(result, ensure_ascii=False)
    data = data.replace("</", "<\\/")          # </script> 조기 종료 방지
    title = _slug(result["meta"])
    return (_HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__DATA__", data))

if __name__ == "__main__":
    main()
