"""Generate benchmark report from evaluation results."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone


def generate_report(results: list[dict], output_dir: str) -> str:
    """Generate JSON + markdown report. Returns path to markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Save raw JSON ──
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Aggregate metrics ──
    by_category = defaultdict(lambda: {"baseline_f1": [], "saga_f1": [], "baseline_judge": [], "saga_judge": []})
    overall = {"baseline_f1": [], "saga_f1": [], "baseline_judge": [], "saga_judge": []}

    for r in results:
        cat = r["category_name"]
        for bucket in (by_category[cat], overall):
            bucket["baseline_f1"].append(r["baseline_f1"])
            bucket["saga_f1"].append(r["saga_f1"])
            if "baseline_judge_score" in r:
                bucket["baseline_judge"].append(r["baseline_judge_score"])
                bucket["saga_judge"].append(r["saga_judge_score"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # ── Markdown report ──
    lines = [
        "# SAGA Memory Benchmark — LOCOMO Results",
        f"\nGenerated: {timestamp}",
        f"Total QA pairs evaluated: {len(results)}",
        "",
        "## Overall",
        "",
        "| Metric | Baseline | SAGA | Delta |",
        "|--------|----------|------|-------|",
    ]

    b_f1, s_f1 = avg(overall["baseline_f1"]), avg(overall["saga_f1"])
    lines.append(f"| F1 Score | {b_f1:.3f} | {s_f1:.3f} | {s_f1 - b_f1:+.3f} |")

    if overall["baseline_judge"]:
        b_j, s_j = avg(overall["baseline_judge"]), avg(overall["saga_judge"])
        lines.append(f"| Judge Score (1-5) | {b_j:.2f} | {s_j:.2f} | {s_j - b_j:+.2f} |")

    lines.extend(["", "## By Category", ""])
    lines.append("| Category | N | Baseline F1 | SAGA F1 | Delta | Baseline Judge | SAGA Judge |")
    lines.append("|----------|---|-------------|---------|-------|----------------|------------|")

    for cat in ["single-hop", "multi-hop", "temporal", "commonsense", "adversarial"]:
        if cat not in by_category:
            continue
        b = by_category[cat]
        n = len(b["baseline_f1"])
        bf1, sf1 = avg(b["baseline_f1"]), avg(b["saga_f1"])
        bj = f"{avg(b['baseline_judge']):.2f}" if b["baseline_judge"] else "—"
        sj = f"{avg(b['saga_judge']):.2f}" if b["saga_judge"] else "—"
        lines.append(f"| {cat} | {n} | {bf1:.3f} | {sf1:.3f} | {sf1 - bf1:+.3f} | {bj} | {sj} |")

    # ── Per-question details (top improvements) ──
    lines.extend(["", "## Top SAGA Improvements", ""])

    improvements = sorted(results, key=lambda r: r["saga_f1"] - r["baseline_f1"], reverse=True)
    for r in improvements[:10]:
        delta = r["saga_f1"] - r["baseline_f1"]
        if delta <= 0:
            break
        lines.append(
            f"- **[{r['category_name']}]** F1: {r['baseline_f1']:.2f}→{r['saga_f1']:.2f} (+{delta:.2f})\n"
            f"  Q: {r['question'][:100]}"
        )

    # ── Failures (SAGA worse) ──
    failures = [r for r in results if r["saga_f1"] < r["baseline_f1"]]
    if failures:
        lines.extend(["", "## SAGA Regressions", ""])
        for r in failures[:5]:
            delta = r["saga_f1"] - r["baseline_f1"]
            lines.append(
                f"- **[{r['category_name']}]** F1: {r['baseline_f1']:.2f}→{r['saga_f1']:.2f} ({delta:.2f})\n"
                f"  Q: {r['question'][:100]}"
            )

    lines.append("")

    md_path = os.path.join(output_dir, f"report_{timestamp}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[report] JSON: {json_path}")
    print(f"[report] Markdown: {md_path}")
    return md_path
