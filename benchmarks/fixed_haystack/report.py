"""고정 건초더미 결과 집계·렌더 (구현 계약 §7).

usage: python3 -m benchmarks.fixed_haystack.report <runs/*.json>

**두 건초더미를 평균 내지 않는다** — haystack_id별로 섹션을 나눈다. A/B는
평균 assistant 길이가 1,929자 vs 1,295자로 대본 자체가 다르고 출처 변형도
달라(dreaming/vanilla) 섞으면 출처 편향이 통계에 녹아든다 (설계 §2.3).

리포트는 **구조적 서술만** 싣는다 (계약 §8-7) — 프로브 질문·응답 원문은
결과 JSON에만 남기고 여기 표에는 턴·유형·거리·통과 여부만 낸다.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from benchmarks.fixed_haystack.haystack import DEFAULT_OUT_DIR

_TYPES = ("recall", "relation", "false", "update", "recent")
_SKIP_LABEL = {"probe_leak": "누출제외", "probe_unprotectable": "값길이제외",
               "context_overflow": "오버플로", "hypa_error": "hypa실패"}


def _rate(passed: int, n: int) -> str:
    return f"{passed}/{n} ({passed / n:.0%})" if n else "-"


def aggregate(results: List[Dict]) -> Dict:
    """{haystack_id: {"meta":…, "variants": {variant: {…}}, "probes": […]}}."""
    by_hs: Dict[str, Dict] = {}
    for r in results:
        hid = r["haystack_id"]
        hs = by_hs.setdefault(hid, {
            "meta": {"origin_variant": r.get("origin_variant", "?"),
                     "narrator_model": r.get("narrator_model", "?"),
                     "provider_ctx_limit":
                         r.get("assembly_meta", {}).get("provider_ctx_limit"),
                     "fixed_tokens":
                         r.get("assembly_meta", {}).get("fixed_tokens"),
                     # §9-2 귀속 사슬의 마지막 칸 — 이 표가 어느 코드·어느
                     # 조립 입력으로 나온 숫자인지 리포트만 보고 알 수 있어야
                     # 한다. 동결 파일은 같은 날 두 번 다시 얼렸다(§10.3·10.4).
                     "provenance": r.get("provenance", {})},
            "runs": defaultdict(list)})
        hs["runs"][r["variant"]].append(r)

    out: Dict[str, Dict] = {}
    for hid, hs in by_hs.items():
        variants: Dict[str, Dict] = {}
        # 프로브 축(턴·유형·거리)은 건초더미 고유 — 아무 런에서나 읽으면 같다
        any_run = next(iter(hs["runs"].values()))[0]
        axis = [{"turn": p["turn"], "ptype": p["ptype"],
                 "distance_turns": p.get("distance_turns", 0)}
                for p in any_run["probes"]]
        for variant, runs in hs["runs"].items():
            probes = [p for r in runs for p in r["probes"]]
            # judge 파싱 실패(None)와 제외 프로브는 분모에서 뺀다
            live = [p for p in probes
                    if not p.get("skipped") and p["judge"] is not None]
            by_type = {t: (sum(1 for p in live
                               if p["ptype"] == t and p["judge"]),
                           sum(1 for p in live if p["ptype"] == t))
                       for t in _TYPES}
            inw = [p for p in live if p.get("in_window")]
            outw = [p for p in live if not p.get("in_window")]
            skips: Dict[str, int] = defaultdict(int)
            for p in probes:
                if p.get("skipped"):
                    skips[p["skipped"]] += 1
            # 기억 퍼널: 추출(저장소)→전달(주입)→활용(judge). 캐릭터가 표현을
            # 바꿔 말하는 건 RP에서 허용이라 judge는 참고 지표고, 추출·전달이
            # 시스템 책임 몫이다. 필드 없는 변형(vanilla 등)은 분모 0으로 비운다.
            stored = [p for p in live if p.get("value_in_store") is not None]
            deliv = [p for p in live if p.get("value_in_knowledge") is not None]
            hit_d = [p for p in deliv if p["value_in_knowledge"]]
            variants[variant] = {
                "runs": len(runs),
                "by_type": by_type,
                "funnel": {
                    "store": (sum(1 for p in stored if p["value_in_store"]),
                              len(stored)),
                    "deliver": (len(hit_d), len(deliv)),
                    "judge_delivered": (sum(1 for p in hit_d if p["judge"]),
                                        len(hit_d)),
                },
                "total": (sum(1 for p in live if p["judge"]), len(live)),
                "window": {"in": (sum(1 for p in inw if p["judge"]), len(inw)),
                           "out": (sum(1 for p in outw if p["judge"]),
                                   len(outw))},
                "oracle": (sum(1 for p in probes if p["oracle"] is True),
                           sum(1 for p in probes
                               if not p.get("skipped")
                               and p["oracle"] is not None)),
                "unparsed": sum(1 for p in probes if not p.get("skipped")
                                and p["judge"] is None),
                "skips": dict(skips),
                "cost": sum(r["totals"].get("cost", 0.0) for r in runs),
                "cost_judge": sum(r["totals"].get("cost_judge", 0.0)
                                  for r in runs),
                "cost_hypa": sum(r["totals"].get("cost_hypa_summary", 0.0)
                                 for r in runs),
                # 프로브별 (통과, 분모, 스킵 수, 스킵 사유) — 반복 간 흔들림을
                # 그대로 드러낸다. 스킵 사유만 찍고 통과 수를 버리면 3반복 중
                # run0만 hypa실패한 셀이 "전부 실패"처럼 보인다(오버플로는
                # 결정론이지만 hypa_error는 아니다).
                "grid": {p["turn"]: (
                    sum(1 for q in probes
                        if q["turn"] == p["turn"] and q["judge"] is True),
                    sum(1 for q in probes
                        if q["turn"] == p["turn"] and not q.get("skipped")
                        and q["judge"] is not None),
                    sum(1 for q in probes
                        if q["turn"] == p["turn"] and q.get("skipped")),
                    next((q["skipped"] for q in probes
                          if q["turn"] == p["turn"] and q.get("skipped")), ""))
                    for p in axis},
            }
        out[hid] = {"meta": hs["meta"], "axis": axis, "variants": variants}
    return out


def _provenance_lines(pv: Dict) -> List[str]:
    """§9-2 귀속 — 이 숫자가 어느 조건에서 나왔는지 한 줄로.

    키가 없으면(옛 결과 JSON) 침묵하지 않고 그 사실을 적는다 — "기록이 없다"와
    "조건이 같다"를 리포트에서 구분해야 한다.
    """
    if not pv:
        return ["> provenance 없음 — 이 결과가 어느 코드·조립 입력에서 나왔는지 "
                "복원할 수 없다 (스펙 §9-2 미충족 결과 JSON).", ""]
    dirty = {True: "+dirty", False: "", "": "(오염 미상)"}.get(
        pv.get("git_dirty"), "?")
    return [f"> 원세션 `{pv.get('source_session', '?')}` · git "
            f"`{(pv.get('git_commit') or '?')[:12]}`{dirty} · 동결 "
            f"{pv.get('frozen_at', '?')} · preset "
            f"`{(pv.get('preset_sha256') or '?')[:12]}` · card "
            f"`{(pv.get('card_sha256') or '?')[:12]}` · run_params 출처 "
            f"`{pv.get('run_params_source', '?')}`", ""]


def render(agg: Dict, ingest_meta: Optional[Dict] = None) -> str:
    ingest_meta = ingest_meta or {}
    lines = ["# 고정 건초더미 벤치 결과", "",
             "동일한 얼린 히스토리를 변형별 조립 알고리즘으로만 다르게 만든 "
             "결과다. **건초더미 간 평균은 내지 않는다** (대본·출처가 다르다).",
             ""]
    for hid, h in sorted(agg.items()):
        m = h["meta"]
        lines += [f"## {hid}",
                  f"출처 변형 `{m['origin_variant']}` · 나레이터 "
                  f"`{m['narrator_model']}` · 고정부 {m['fixed_tokens']}tok · "
                  f"프로바이더 창 {m['provider_ctx_limit']}tok", ""]
        lines += _provenance_lines(m.get("provenance") or {})
        vs = h["variants"]

        lines += ["### 변형 × 프로브 유형 (judge 통과/분모)",
                  "| variant | " + " | ".join(_TYPES) + " | 합계 | 반복 |",
                  "|" + "---|" * (len(_TYPES) + 3)]
        for variant, a in sorted(vs.items()):
            cells = [_rate(*a["by_type"][t]) for t in _TYPES]
            lines.append(f"| {variant} | " + " | ".join(cells)
                         + f" | {_rate(*a['total'])} | {a['runs']} |")
        lines += [""]

        funnel_vs = {v: a for v, a in sorted(vs.items())
                     if a["funnel"]["store"][1] or a["funnel"]["deliver"][1]}
        if funnel_vs:
            lines += ["### 기억 퍼널 — 추출 → 전달 → 활용",
                      "추출=저장소에 값 존재, 전달=주입 블록에 값 포함, "
                      "활용=전달된 프로브의 judge 통과. 캐릭터가 표현을 바꿔 "
                      "말하는 건 RP 허용 폭이라 활용은 참고 지표 — 추출·전달이 "
                      "시스템 책임 몫이다.",
                      "| variant | 추출(저장소) | 전달(주입) | 전달분 활용 |",
                      "|" + "---|" * 4]
            for variant, a in funnel_vs.items():
                f = a["funnel"]
                lines.append(f"| {variant} | {_rate(*f['store'])} | "
                             f"{_rate(*f['deliver'])} | "
                             f"{_rate(*f['judge_delivered'])} |")
            lines += [""]

        lines += ["### 창내(LITM) vs 창밖(eviction)",
                  "창내 실패는 주의(LITM) 실패, 창밖 통과는 회수 성공이다. "
                  "vanilla/dreaming은 정의상 항상 창내(dreaming은 프록시 압축 "
                  "때문에 **상한값**).",
                  "| variant | 창내 | 창밖 |", "|" + "---|" * 3]
        for variant, a in sorted(vs.items()):
            lines.append(f"| {variant} | {_rate(*a['window']['in'])} | "
                         f"{_rate(*a['window']['out'])} |")
        lines += [""]

        lines += ["### 채점 제외·채점기 건강",
                  "값길이제외 = 값이 1글자라 마스킹도 오라클도 못 지키는 "
                  "프로브(lucid의 출제 문턱을 freeze가 소급 적용). 분모에서 "
                  "빠진 몫이 여기 다 보여야 통과율을 정직하게 읽을 수 있다.",
                  "| variant | 누출제외 | 값길이제외 | 오버플로 | hypa실패 "
                  "| judge 미파싱 | 오라클 |", "|" + "---|" * 7]
        for variant, a in sorted(vs.items()):
            s = a["skips"]
            lines.append(
                f"| {variant} | {s.get('probe_leak', 0)} | "
                f"{s.get('probe_unprotectable', 0)} | "
                f"{s.get('context_overflow', 0)} | {s.get('hypa_error', 0)} | "
                f"{a['unparsed']} | {_rate(*a['oracle'])} |")
        lines += [""]

        lines += ["### 비용 (전 반복 합계, USD)",
                  "| variant | 나레이터 | judge | hypa요약 | 계 |",
                  "|" + "---|" * 5]
        for variant, a in sorted(vs.items()):
            # hypa요약은 프로브 루프가 태운 캐시 미스분 — 인제스천 상각분과
            # 별개다(아래 줄). 빼면 유상태 변형이 실제보다 싸 보인다.
            hy = a.get("cost_hypa", 0.0)
            lines.append(f"| {variant} | {a['cost']:.4f} | "
                         f"{a['cost_judge']:.4f} | {hy:.4f} | "
                         f"{a['cost'] + a['cost_judge'] + hy:.4f} |")
        im = ingest_meta.get(hid) or {}
        if im:
            lines.append("")
            lines.append("인제스천 상각 비용 (프로브 루프 밖 — 1회 리플레이):")
            d, hy = im.get("dreaming") or {}, im.get("hypa") or {}
            if d:
                lines.append(
                    f"- dreaming 꿈 {d.get('dream_cycles', 0)}사이클 "
                    f"({d.get('dream_sec', 0)}s, ${d.get('dream_cost', 0)}) · "
                    f"격리 {d.get('quarantine_count', 0)}턴"
                    + ("  ⚠️ 0.0은 미측정이다 (usage 미반환 클라이언트)"
                       if not d.get("dream_cost") else ""))
            if hy:
                lines.append(f"- hypa 요약 {hy.get('summary_calls', 0)}콜 "
                             f"(${hy.get('summary_cost', 0)}) · 절단 "
                             f"{hy.get('summary_truncated', 0)}")
        lines += [""]

        order = sorted(vs)
        lines += ["### 프로브별 통과 (통과/분모, 제외는 사유)",
                  "| 턴 | 유형 | 거리 | " + " | ".join(order) + " |",
                  "|" + "---|" * (len(order) + 3)]
        for p in h["axis"]:
            cells = []
            for variant in order:
                hit, n, n_skip, skipped = vs[variant]["grid"][p["turn"]]
                label = _SKIP_LABEL.get(skipped, skipped)
                if n:
                    # 일부 반복만 스킵된 셀 — 채점된 반복의 결과를 살린다
                    cells.append(f"{hit}/{n}"
                                 + (f" ({label}{n_skip})" if n_skip else ""))
                else:
                    cells.append(label or "미파싱")
            lines.append(f"| T{p['turn']} | {p['ptype']} | "
                         f"{p['distance_turns']} | " + " | ".join(cells) + " |")
        lines += [""]
    return "\n".join(lines)


def load_ingest_meta(paths: List[str]) -> Dict:
    """결과 파일 옆(`{root}/ingest-meta.json`)의 인제스천 비용을 끌어온다.

    없으면 조용히 비운다. 이걸 빼면 유상태 변형이 실제보다 싸 보인다 —
    report2.py:103-104가 `cost_hypa`를 빠뜨리지 말라고 남긴 것과 같은 이유.
    그래서 `run --out`으로 결과를 딴 데 쓴 경우도 haystack_id로 표준 위치를
    되짚는다 — 상대 위치 하나만 보면 섹션이 통째로 사라져 "인제스천 비용
    0"으로 오독된다(침묵이 아니라 오독이 문제다).
    """
    out: Dict[str, Dict] = {}
    for p in paths:
        f = pathlib.Path(p).resolve()
        hid = json.loads(f.read_text(encoding="utf-8"))["haystack_id"]
        for meta in (f.parents[1] / "ingest-meta.json",       # {root}/runs/
                     pathlib.Path(DEFAULT_OUT_DIR) / hid / "ingest-meta.json"):
            if meta.is_file():
                out[hid] = json.loads(meta.read_text(encoding="utf-8"))
                break
    return out


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
               for p in argv]
    print(render(aggregate(results), load_ingest_meta(argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
