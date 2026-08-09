"""v2 결과 집계: mean±std + 커뮤 신뢰 포맷.

usage: python3 -m benchmarks.eval.report2 dreaming_data/eval/v2-*.json
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
from collections import defaultdict
from typing import Dict, List

_TYPES = ("recall", "relation", "false", "update", "recent")


def window_split(probes: List[Dict]):
    """(창내 pass, 창내 n), (창밖 pass, 창밖 n) — LITM/eviction 실패 분리.

    구 JSON은 in_window 부재 = 창밖 취급(구 게이팅이 evict-전용이라 사실과
    일치). judge None(파싱 실패)은 분모에서 뺀다.
    """
    inw = [p for p in probes if p.get("in_window") and p["judge"] is not None]
    out = [p for p in probes
           if not p.get("in_window") and p["judge"] is not None]

    def _p(xs):
        return (sum(1 for p in xs if p["judge"] is True), len(xs))
    return _p(inw), _p(out)


def value_survival(probes: List[Dict]):
    """창밖(evicted) 프로브 중 value_in_window True 비율 — 통과가 진짜
    기억 때문인지 서사 반복 때문인지 가늠하는 오염 지표.

    (오염 의심 pass/n, 창밖 전체 pass/n). 구 JSON은 value_in_window 부재라
    보수적으로 False(생존 증거 없음) 취급한다.
    """
    out = [p for p in probes
           if not p.get("in_window") and p["judge"] is not None]
    survived = [p for p in out if p.get("value_in_window")]

    def _p(xs):
        return (sum(1 for p in xs if p["judge"] is True), len(xs))
    return _p(survived), _p(out)


def aggregate(results: List[Dict]) -> Dict:
    agg: Dict = {}
    by_variant = defaultdict(list)
    for r in results:
        by_variant[r["variant"]].append(r)
    for variant, runs in by_variant.items():
        by_type: Dict = {}
        for ptype in _TYPES:
            rates = []
            for r in runs:
                # judge 파싱 실패(None)는 분모에서 뺀다 — 기본값을 두면
                # 유형별로 반대 방향 편향이 생긴다 (scoring._verdict 주석)
                ps = [p for p in r["probes"]
                      if p["ptype"] == ptype and p["judge"] is not None]
                if ps:
                    rates.append(sum(p["judge"] for p in ps) / len(ps))
            if rates:
                by_type[ptype] = {
                    "mean": statistics.mean(rates),
                    "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                    "runs": len(rates)}
        misses = defaultdict(int)
        dist = defaultdict(lambda: [0, 0])
        judged = oracle_hits = oracle_scored = scored = unparsed = 0
        agree = agree_base = 0
        for r in runs:
            for p in r["probes"]:
                # oracle None = 판정 불가(값이 캐릭터 이름 등) — 분모에서 뺀다
                if p["oracle"] is not None:
                    oracle_hits += bool(p["oracle"])
                    oracle_scored += 1
                scored += 1
                if p["judge"] is None:
                    unparsed += 1
                    continue
                judged += p["judge"]
                if p["oracle"] is not None:
                    agree += (bool(p["judge"]) == bool(p["oracle"]))
                    agree_base += 1
                if p["judge"] is False and p["miss_cause"] != "-":
                    misses[p["miss_cause"]] += 1
                bucket = (p["distance_turns"] // 10) * 10
                dist[bucket][0] += p["judge"]
                dist[bucket][1] += 1
        parsed = scored - unparsed
        all_probes = [p for r in runs for p in r["probes"]]
        inw, outw = window_split(all_probes)
        vs_survived, vs_out = value_survival(all_probes)
        agg[variant] = {
            "window": {"in": inw, "out": outw},
            "value_survival": {"survived": vs_survived, "out": vs_out},
            "by_type": by_type,
            "miss_causes": dict(misses),
            "distance": {k: v[0] / v[1] for k, v in sorted(dist.items())},
            # hypa 변형만 요약 비용(cost_hypa)이 나레이터 cost와 별도로
            # 붙는다 — 빠뜨리면 hypa가 실제보다 싸 보인다.
            "cost_mean": statistics.mean(
                r["totals"]["cost"] + r["totals"].get("cost_hypa", 0)
                for r in runs),
            # 오라클과 judge의 불일치 해소 규칙은 문헌에 없다. 둘 다 내고
            # 불일치율 자체를 채점기 건강 지표로 읽는다 (PoLL·Thakur).
            "judge_rate": judged / parsed if parsed else 0.0,
            "oracle_rate": oracle_hits / oracle_scored if oracle_scored else 0.0,
            "disagree_rate": 1 - agree / agree_base if agree_base else 0.0,
            "unparsed": unparsed, "probes": scored,
            "oracle_na": scored - oracle_scored,
            **({"hypa_truncated": sum(r["totals"].get("hypa_truncated", 0)
                                       for r in runs)}
               if any("hypa_truncated" in r["totals"] for r in runs) else {}),
        }
    return agg


def render(agg: Dict, results: List[Dict]) -> str:
    lines = ["# 디렉터 벤치 v2 결과", ""]
    lines.append("| variant | " + " | ".join(_TYPES) + " | $ (mean) |")
    lines.append("|" + "---|" * (len(_TYPES) + 2))
    for variant, a in sorted(agg.items()):
        cells = []
        for t in _TYPES:
            row = a["by_type"].get(t)
            cells.append(f"{row['mean']:.0%}±{row['std']:.0%}" if row else "-")
        lines.append(f"| {variant} | " + " | ".join(cells)
                     + f" | {a['cost_mean']:.2f} |")
    lines.append("")
    lines.append("## 창내(LITM) vs 창밖(eviction) 통과율")
    lines.append("| variant | 창내 | 창밖 |")
    lines.append("|" + "---|" * 3)
    for variant, a in sorted(agg.items()):
        (ip, inn), (op, on) = a["window"]["in"], a["window"]["out"]
        lines.append(f"| {variant} | {ip}/{inn} | {op}/{on} |")
    lines.append("")
    for variant, a in sorted(agg.items()):
        vp, vn = a["value_survival"]["survived"]
        op, on = a["value_survival"]["out"]
        lines.append(f"{variant} — 창밖 통과 중 값 생존 오염 의심: "
                     f"{vp}/{vn} (창밖 전체 {op}/{on})")
        if "hypa_truncated" in a:
            lines.append(f"{variant} — hypa 요약 절단: {a['hypa_truncated']}")
    lines.append("")
    lines.append("## 채점기 건강 — judge / 오라클 / 불일치")
    lines.append("| variant | judge | 오라클 | 불일치 | 파싱실패 | n |")
    lines.append("|" + "---|" * 6)
    for variant, a in sorted(agg.items()):
        lines.append(f"| {variant} | {a['judge_rate']:.0%} | "
                     f"{a['oracle_rate']:.0%} | {a['disagree_rate']:.0%} | "
                     f"{a['unparsed']} | {a['probes']} |")
    lines.append("")
    for variant, a in sorted(agg.items()):
        lines.append(f"## {variant} — 거리별 통과율(턴 구간): "
                     + ", ".join(f"{k}~{k + 9}: {v:.0%}"
                                 for k, v in a["distance"].items()))
        if a["miss_causes"]:
            lines.append(f"미스 원인: {a['miss_causes']}")
    lines.append("\n## 부록 — 프로브 무편집 원문")
    for r in results:
        for p in r["probes"]:
            mark = {True: "○", False: "×"}.get(p["judge"], "?")
            lines.append(f"- [{r['variant']} run{r['run']} T{p['turn'] + 1} "
                         f"{p['ptype']}] {mark} Q: {p['question']}")
            lines.append(f"  A: {' '.join(p['reply'].split())[:400]}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [json.loads(pathlib.Path(p).read_text()) for p in argv]
    print(render(aggregate(results), results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
