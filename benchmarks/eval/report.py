# [v1 하네스] 현행은 run2 계열 — 이 파일은 구 테스트·PR 호환용으로 보존.
"""변형별 결과 JSON → 비교 표 (스펙 §9 병기 지표 + 수동 감사용 원문).

usage: python3 -m benchmarks.eval.report dreaming_data/eval/result-*.json
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

_MARK = {"full": "○", "partial": "△", "miss": "×"}


def render_report(results: List[Dict]) -> str:
    lines = ["| variant | oracle(full/5) | partial | recall | $ | cache% | sec/turn |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(results, key=lambda x: x["variant"]):
        t = r["totals"]
        lines.append(
            f"| {r['variant']} | {t['oracle_full']}/5 | {t['oracle_partial']} "
            f"| {t['recall']} | {t['cost']} | {t['avg_hit_t2']} "
            f"| {t['avg_sec']} |")
    lines.append("")
    for r in sorted(results, key=lambda x: x["variant"]):
        lines.append(f"## {r['variant']} — 프로브 응답 원문 (수동 감사용)")
        for p in r["probes"]:
            lines.append(f"- T{p['turn'] + 1:02d} {p['label']} "
                         f"{_MARK[p['hit']]} ({p['matched']}/{p['total']}): "
                         f"{' '.join(p['reply'].split())[:300]}")
        lines.append("")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [json.loads(pathlib.Path(p).read_text()) for p in argv]
    print(render_report(results))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
