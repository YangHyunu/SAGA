"""Generate report from LongMemEval results."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime


def generate_report(results: list[dict], output_dir: str | None = None) -> str:
    """Generate JSON + Markdown report. Returns markdown path."""
    if not output_dir:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "results")
    os.makedirs(output_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = os.path.join(output_dir, f"longmemeval_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[report] JSON: {json_path}")

    # Compute stats
    has_judge = any("baseline_correct" in r for r in results)

    # Overall
    total = len(results)
    if has_judge:
        baseline_acc = sum(1 for r in results if r.get("baseline_correct")) / total
        saga_acc = sum(1 for r in results if r.get("saga_correct")) / total
    else:
        baseline_acc = saga_acc = 0.0

    # By category
    by_type = defaultdict(list)
    for r in results:
        by_type[r["question_type"]].append(r)

    # Markdown
    md_lines = [
        "# SAGA Memory Benchmark — LongMemEval Results",
        "",
        f"Generated: {ts}",
        f"Total QA instances evaluated: {total}",
        "",
    ]

    if has_judge:
        md_lines += [
            "## Overall Accuracy",
            "",
            "| Metric | Baseline | SAGA | Delta |",
            "|--------|----------|------|-------|",
            f"| Accuracy | {baseline_acc:.3f} | {saga_acc:.3f} | {saga_acc - baseline_acc:+.3f} |",
            "",
        ]

        # By category
        md_lines += [
            "## By Question Type",
            "",
            "| Type | N | Baseline Acc | SAGA Acc | Delta |",
            "|------|---|-------------|---------|-------|",
        ]

        type_order = [
            "single-session-user", "single-session-assistant",
            "single-session-preference", "multi-session",
            "temporal-reasoning", "knowledge-update",
        ]
        for qtype in type_order:
            items = by_type.get(qtype, [])
            if not items:
                continue
            n = len(items)
            b_acc = sum(1 for r in items if r.get("baseline_correct")) / n
            s_acc = sum(1 for r in items if r.get("saga_correct")) / n
            delta = s_acc - b_acc
            md_lines.append(
                f"| {qtype} | {n} | {b_acc:.3f} | {s_acc:.3f} | {delta:+.3f} |"
            )
        md_lines.append("")

    # Top improvements (where baseline wrong, SAGA correct)
    if has_judge:
        improvements = [
            r for r in results
            if not r.get("baseline_correct") and r.get("saga_correct")
        ]
        regressions = [
            r for r in results
            if r.get("baseline_correct") and not r.get("saga_correct")
        ]

        md_lines += [
            f"## SAGA Improvements ({len(improvements)} questions)",
            "",
        ]
        for r in improvements[:10]:
            md_lines.append(
                f"- **[{r['question_type']}]** Q: {r['question'][:80]}"
            )
        md_lines.append("")

        md_lines += [
            f"## SAGA Regressions ({len(regressions)} questions)",
            "",
        ]
        for r in regressions[:10]:
            md_lines.append(
                f"- **[{r['question_type']}]** Q: {r['question'][:80]}"
            )
        md_lines.append("")

    # Write markdown
    md_path = os.path.join(output_dir, f"longmemeval_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[report] Markdown: {md_path}")

    return md_path
