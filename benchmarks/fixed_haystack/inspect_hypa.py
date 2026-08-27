"""HypaV3 동작 인스펙터 — 턴 1~N 전진 루프, 내부 상태 턴별 기록.

스펙: docs/superpowers/specs/2026-08-27-hypa-inspector-design.md
채점 없음, run.py 경로 무접촉. 출력은 dreaming_data/fixed_haystack/ 에만.

요약 캐시 규약: 기본은 캐시-히트만 소비(미스 = 실콜 0으로 열거 후 출력
차단·종료코드 1). --allow-llm 일 때만 미스분을 실제 호출한다.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import pathlib
import sys
from typing import Any, Callable, Dict, List, Optional

from benchmarks.eval import hypa, hypa_main, transport
from benchmarks.fixed_haystack import assemble, haystack
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
        "misses": cache_only.misses if cache_only else [],
    }


def _slug(meta: Dict[str, Any]) -> str:
    s = meta["settings"]
    base = f"hypa-inspect-{meta['hsid']}-{meta['impl']}"
    tags = []
    if s.get("memory_tokens_ratio") not in (None, 0.39):
        tags.append(f"mtr{s['memory_tokens_ratio']}")
    if s.get("extra_summarization_ratio") not in (None, 0.01):
        tags.append(f"esr{s['extra_summarization_ratio']}")
    return "-".join([base] + tags)


def write_outputs(result: Dict[str, Any], out_dir: pathlib.Path = OUT_DIR) -> bool:
    """미스가 있으면 아무것도 쓰지 않는다 — 미완성 관찰을 남기지 않는다."""
    if result["misses"]:
        return False
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(result["meta"])
    (out_dir / f"{slug}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (out_dir / f"{slug}.html").write_text(render_html(result), encoding="utf-8")
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hs", required=True)
    ap.add_argument("--impl", required=True, choices=["exp", "main"])
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
    write_outputs(result, pathlib.Path(a.out_dir))
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
  <table class="meta" id="meta"></table>
  <svg id="chart" viewBox="0 0 800 260" preserveAspectRatio="none"></svg>
  <div class="legend">● 압축 발동 턴 · <span style="color:#36c">요약 개수</span> ·
    <span style="color:#c33">kept_start_msg</span> ·
    <span style="color:#999">wire_tokens(우축)</span></div>
  <div id="turns"></div>
</div>
<div id="right"><em style="font-size:13px">턴을 클릭하면 상세.</em></div>
<script>
const DATA = __DATA__;
const T = DATA.turns;
function esc(s) { return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;"); }
// meta
document.getElementById("t-hsid").textContent = DATA.meta.hsid;
document.getElementById("t-impl").textContent = DATA.meta.impl;
document.getElementById("meta").innerHTML =
  Object.entries({max_ctx: DATA.meta.max_ctx, summary_model: DATA.meta.summary_model,
                  ...DATA.meta.settings})
    .map(([k,v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
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
  for (const r of T)
    if (r.events.some(e => e.ev === "batch"))
      svg += `<circle cx="${xs(r.turn)}" cy="${y(r.n_summaries, maxSum)}" r="4" fill="#c33"/>`;
  svg += `<text x="${P}" y="14" font-size="10" fill="#666">요약 max ${maxSum} · ksm max ${maxKsm} · wire max ${maxWire}</text>`;
  document.getElementById("chart").innerHTML = svg;
})();
// turn list
const list = document.getElementById("turns");
list.innerHTML = T.map((r, i) => {
  const nb = r.events.filter(e => e.ev === "batch").length;
  return `<div class="turnrow" data-i="${i}">
    <span>t${r.turn}</span>
    <span>요약 ${r.n_summaries}</span>
    <span>ksm ${r.kept_start_msg}</span>
    ${nb ? `<span class="b">압축×${nb}</span>` : ""}
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
function show(r) {
  let h = `<h1>턴 ${r.turn}</h1>`;
  h += `<div><span class="tag">요약 ${r.n_summaries}</span>` +
       `<span class="tag">kept_start ${r.kept_start_msg}</span>` +
       `<span class="tag">wire ${r.wire_tokens ?? "-"}</span></div>`;
  if (r.error) h += `<pre>ERROR: ${esc(r.error)}</pre>`;
  const batches = r.events.filter(e => e.ev === "batch");
  if (batches.length) {
    h += `<h3 style="font-size:13px">압축 배치</h3>` +
      evTable(batches.map(b => ({memos: b.idx.join(" "), tokens: b.tokens})),
              ["memos", "tokens"]);
  }
  const stops = r.events.filter(e => e.ev === "batch_stop");
  if (stops.length) h += `<div class="legend">중단: ${stops.map(s=>s.reason).join(", ")}</div>`;
  const res = r.events.find(e => e.ev === "reserve");
  if (res) h += `<div class="legend">예약: available ${res.available}, current ${res.current}</div>`;
  for (const stage of ["important", "recent"]) {
    const e = r.events.find(x => x.ev === stage);
    if (e && (e.picked.length || e.ev === "recent"))
      h += `<div class="legend">${stage}: [${e.picked}]` +
           (e.reserved !== undefined ? ` (reserved ${e.reserved}, used ${e.used})` : "") + `</div>`;
  }
  const sim = r.events.find(x => x.ev === "similar");
  if (sim) {
    h += `<h3 style="font-size:13px">similar — picked [${sim.picked}] (reserved ${sim.reserved}, used ${sim.used})</h3>`;
    h += `<div class="legend">rrf_rank: [${sim.rrf_rank}]</div>`;
    sim.queries.forEach((q, qi) => {
      const top = (sim.chunk_scores[qi] || []).slice(0, 5);
      h += `<div class="legend">q${qi} (w=${q.weight.toFixed(3)}): ${esc(q.content.slice(0, 90))}</div>`;
      h += evTable(top.map(([id, sc]) => ({chunk: id, cos: sc})), ["chunk", "cos"]);
    });
  }
  if (r.metrics) h += `<pre>${esc(JSON.stringify(r.metrics))}</pre>`;
  h += `<h3 style="font-size:13px">memory_text</h3><pre>${esc(r.memory_text || "(없음)")}</pre>`;
  document.getElementById("right").innerHTML = h;
}
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
