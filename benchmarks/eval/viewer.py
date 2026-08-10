"""벤치 런 결과(v2-*.json) → 읽기용 자립형 HTML.

타임라인(턴별 말풍선 + 토큰/캐시/프로브 배지), 프로브 카드(사실·질문·응답·
이중 채점), 원장 테이블을 한 파일로 만든다. 외부 리소스 없음 — 파일 하나로
어디서든 열린다.

usage:
    python3 -m benchmarks.eval.viewer dreaming_data/eval/v2-pilot-r0-run0.json
    # → 같은 자리에 .html
"""

from __future__ import annotations

import html
import json
import pathlib
import sys
from typing import Dict, List

_BADGE = {"recall": "#2563eb", "relation": "#7c3aed", "false": "#dc2626",
          "update": "#d97706", "recent": "#059669"}

_CSS = """
:root{--bg:#f6f6f4;--card:#fff;--ink:#1f2937;--sub:#6b7280;--line:#e5e7eb;
      --user:#eef2ff;--asst:#f8fafc;--ok:#059669;--bad:#dc2626}
@media(prefers-color-scheme:dark){
:root{--bg:#111418;--card:#1a1f26;--ink:#e5e7eb;--sub:#9ca3af;--line:#2a3038;
      --user:#1e2436;--asst:#161b22}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.65 -apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:24px 16px 80px}
h1{font-size:20px;margin:0 0 4px}
h2{font-size:16px;margin:36px 0 12px;border-bottom:1px solid var(--line);
   padding-bottom:6px}
.meta{color:var(--sub);font-size:13px;margin-bottom:16px}
.stats{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0 4px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;
      padding:8px 14px;font-size:13px}
.stat b{display:block;font-size:17px;font-variant-numeric:tabular-nums}
.turn{margin:14px 0}
.tno{color:var(--sub);font-size:12px;margin-bottom:3px;display:flex;
     gap:8px;align-items:center;flex-wrap:wrap}
.badge{color:#fff;border-radius:99px;padding:1px 9px;font-size:11px}
.bub{border:1px solid var(--line);border-radius:10px;padding:10px 14px;
     white-space:pre-wrap;word-break:break-word}
.user .bub{background:var(--user)}
.asst .bub{background:var(--asst);margin-top:6px}
.role{font-weight:600;font-size:12px;color:var(--sub)}
details>summary{cursor:pointer;color:var(--sub);font-size:13px;
                margin-top:4px;user-select:none}
.probe{background:var(--card);border:1px solid var(--line);border-radius:10px;
       padding:14px 16px;margin:14px 0}
.probe .row{margin:6px 0}
.k{color:var(--sub);font-size:12px;margin-right:6px}
.verdict{font-weight:700}
.ok{color:var(--ok)}.bad{color:var(--bad)}.na{color:var(--sub)}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;
      vertical-align:top}
th{color:var(--sub);font-weight:600}
.num{font-variant-numeric:tabular-nums;text-align:right}
.scroll{overflow-x:auto}
a.anchor{color:inherit;text-decoration:none}
"""


def _esc(t) -> str:
    return html.escape(str(t if t is not None else ""))


def _clip(text: str, n: int = 700) -> str:
    """긴 응답은 앞부분만 보이고 나머지는 접는다."""
    text = text or ""
    if len(text) <= n:
        return f'<div class="bub">{_esc(text)}</div>'
    return (f'<div class="bub">{_esc(text[:n])}…</div>'
            f'<details><summary>전체 보기 ({len(text):,}자)</summary>'
            f'<div class="bub">{_esc(text)}</div></details>')


def _verdict(v) -> str:
    if v is True:
        return '<span class="verdict ok">PASS</span>'
    if v is False:
        return '<span class="verdict bad">FAIL</span>'
    return '<span class="verdict na">미판정</span>'


def render(result: Dict) -> str:
    t = result.get("totals", {})
    turns: List[Dict] = result.get("turns", [])
    probes: List[Dict] = result.get("probes", [])
    ledger: List[Dict] = result.get("ledger", [])
    cached_hits = sum(1 for x in turns if x.get("cached", 0) > 0)
    total_cached = sum(x.get("cached", 0) for x in turns)
    total_prompt = sum(x.get("prompt", 0) for x in turns)
    probe_by_turn = {p["turn"]: p for p in probes}

    stats = [
        ("변형", result.get("variant", "?")),
        ("턴", len(turns)),
        ("프로브", f'{t.get("judge_pass", 0)}/{t.get("probes", 0)}'),
        ("오라클", t.get("oracle_pass", 0)),
        ("절단", t.get("truncated", 0)),
        ("리롤", f'{t.get("rerolls", 0)}회'),
        ("잔존 병리", t.get("flawed", 0)),
        ("캐시 히트", f"{cached_hits}/{len(turns)}턴"),
        ("캐시 토큰", f"{total_cached:,}/{total_prompt:,}"),
        ("나레이터", f'${t.get("cost", 0)}'),
    ]
    if t.get("aborted"):
        stats.append(("중단", t["aborted"]))
    if "cost_lucid" in t or "cost_director" in t:
        cost_lucid = t.get("cost_lucid", t.get("cost_director", 0))
        grand = t["cost"] + cost_lucid + t.get("cost_judge", 0)
        stats += [("Lucid", f'${cost_lucid} '
                            f'({t.get("lucid_calls", t.get("director_calls", 0))}콜)'),
                  ("judge", f'${t.get("cost_judge", 0)} '
                            f'({t.get("judge_calls", 0)}콜)'),
                  ("총비용", f"${round(grand, 4)}")]
    out = [f"<style>{_CSS}</style>",
           '<div class="wrap">',
           f'<h1>{_esc(result.get("session", ""))} · '
           f'{_esc(result.get("model", ""))}</h1>',
           f'<div class="meta">run {result.get("run", 0)} · '
           f'원장 {len(ledger)}건</div>',
           '<div class="stats">'
           + "".join(f'<div class="stat">{k}<b>{_esc(v)}</b></div>'
                     for k, v in stats)
           + "</div>"]

    # ── 프로브 카드 ──
    out.append(f"<h2>프로브 {len(probes)}건</h2>")
    if not probes:
        out.append('<div class="meta">발사된 프로브 없음</div>')
    for p in probes:
        color = _BADGE.get(p.get("ptype", ""), "#6b7280")
        out.append(
            '<div class="probe">'
            f'<div class="row"><span class="badge" style="background:{color}">'
            f'{_esc(p.get("ptype"))}</span> '
            f'<a class="anchor" href="#t{p["turn"]}">턴 {p["turn"] + 1}</a>'
            f' · <span class="k">거리</span>{p.get("distance_turns")}턴'
            f' · <span class="k">{"창내" if p.get("in_window") else "창밖"}'
            '</span>'
            + (' · <span class="k">값 생존</span>'
               if p.get("value_in_window") else "")
            + '</div>'
            f'<div class="row"><span class="k">사실</span>{_esc(p.get("fact"))}'
            f' <span class="k">값</span><b>{_esc(p.get("value"))}</b></div>'
            f'<div class="row"><span class="k">질문</span>'
            f'{_esc(p.get("question"))}</div>'
            f'<div class="row"><span class="k">judge</span>'
            f'{_verdict(p.get("judge"))} '
            f'<span class="k">oracle</span>{_verdict(p.get("oracle"))}</div>'
            f'<div class="row"><span class="k">근거</span>'
            f'{_esc(p.get("why"))}</div>'
            + (f'<div class="row"><span class="k">미스 원인</span>'
               f'{_esc(p["miss_cause"])}</div>' if p.get("miss_cause") else "")
            + '</div>')

    # ── 타임라인 ──
    out.append("<h2>타임라인</h2>")
    for x in turns:
        i = x.get("turn", 0)
        chips = [f'<b>턴 {i + 1}</b>',
                 f'{x.get("prompt", 0):,}tk',
                 f'캐시 {x.get("cached", 0):,}',
                 f'{x.get("sec", 0)}s']
        if x.get("finish") not in ("stop", None):
            chips.append(f'<span class="verdict bad">'
                         f'finish={_esc(x["finish"])}</span>')
        if x.get("rerolls"):
            chips.append(f'리롤 {x["rerolls"]}')
        if x.get("flaw"):
            chips.append(f'<span class="verdict bad">{_esc(x["flaw"])}</span>')
        p = probe_by_turn.get(i)
        if p:
            color = _BADGE.get(p.get("ptype", ""), "#6b7280")
            chips.append(f'<span class="badge" style="background:{color}">'
                         f'{_esc(p.get("ptype"))} '
                         f'{"✓" if p.get("judge") else "✗"}</span>')
        out.append(
            f'<div class="turn" id="t{i}">'
            f'<div class="tno">{" · ".join(chips)}</div>'
            f'<div class="user"><span class="role">렌</span>'
            f'{_clip(x.get("user", ""), 400)}</div>'
            f'<div class="asst"><span class="role">캐릭터</span>'
            f'{_clip(x.get("reply", ""))}</div></div>')

    # ── 원장 ──
    out.append(f"<h2>사실 원장 {len(ledger)}건</h2>"
               '<details><summary>펼치기</summary><div class="scroll"><table>'
               "<tr><th>턴</th><th>kind</th><th>값</th><th>서술</th>"
               "<th>출제</th></tr>")
    for f in ledger:
        out.append(f'<tr><td class="num">{f.get("turn", 0) + 1}</td>'
                   f'<td>{_esc(f.get("kind"))}</td>'
                   f'<td><b>{_esc(f.get("value"))}</b></td>'
                   f'<td>{_esc(f.get("text"))}</td>'
                   f'<td>{"●" if f.get("probed") else ""}</td></tr>')
    out.append("</table></div></details></div>")
    return "".join(out)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    for src in argv:
        path = pathlib.Path(src)
        dst = path.with_suffix(".html")
        dst.write_text(render(json.loads(path.read_text())))
        print(f"{dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
