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
                ps = [p for p in r["probes"] if p["ptype"] == ptype]
                if ps:
                    rates.append(sum(p["judge"] for p in ps) / len(ps))
            if rates:
                by_type[ptype] = {
                    "mean": statistics.mean(rates),
                    "std": statistics.stdev(rates) if len(rates) > 1 else 0.0,
                    "runs": len(rates)}
        misses = defaultdict(int)
        dist = defaultdict(lambda: [0, 0])
        for r in runs:
            for p in r["probes"]:
                if not p["judge"] and p["miss_cause"] != "-":
                    misses[p["miss_cause"]] += 1
                bucket = (p["distance_turns"] // 10) * 10
                dist[bucket][0] += p["judge"]
                dist[bucket][1] += 1
        agg[variant] = {
            "by_type": by_type,
            "miss_causes": dict(misses),
            "distance": {k: v[0] / v[1] for k, v in sorted(dist.items())},
            "cost_mean": statistics.mean(r["totals"]["cost"] for r in runs),
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
    for variant, a in sorted(agg.items()):
        lines.append(f"## {variant} — 거리별 통과율(턴 구간): "
                     + ", ".join(f"{k}~{k + 9}: {v:.0%}"
                                 for k, v in a["distance"].items()))
        if a["miss_causes"]:
            lines.append(f"미스 원인: {a['miss_causes']}")
    lines.append("\n## 부록 — 프로브 무편집 원문")
    for r in results:
        for p in r["probes"]:
            mark = "○" if p["judge"] else "×"
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
