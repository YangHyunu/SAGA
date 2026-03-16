"""Generate MemoryAgentBench report — 3-way comparison."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime


def generate_report(results: list[dict], output_dir: str, modes: list[str]) -> str:
    """Generate JSON + Markdown report. Returns path to markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save raw JSON
    json_path = os.path.join(output_dir, f"memoryagent_results_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[report] JSON: {json_path}")

    # Aggregate metrics
    by_split = defaultdict(list)
    by_source = defaultdict(list)
    overall = []

    for r in results:
        by_split[r["split"]].append(r)
        by_source[r["source"]].append(r)
        overall.append(r)

    def avg_metric(items: list[dict], mode: str, metric: str) -> float:
        key = f"{mode}_{metric}"
        vals = [r[key] for r in items if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    def pct(items: list[dict], mode: str, metric: str) -> float:
        key = f"{mode}_{metric}"
        vals = [r[key] for r in items if key in r]
        return (sum(1 for v in vals if v) / len(vals) * 100) if vals else 0.0

    # Markdown
    lines = [
        "# SAGA Memory Benchmark — MemoryAgentBench Results",
        "",
        f"Generated: {ts}",
        f"Total QA pairs: {len(results)}",
        f"Modes evaluated: {', '.join(modes)}",
        "",
    ]

    # Overall table
    lines += ["## Overall", ""]
    header = "| Metric |"
    sep = "|--------|"
    for mode in modes:
        header += f" {mode} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    # SubEM row
    row = "| SubEM (%) |"
    for mode in modes:
        val = pct(overall, mode, "subem")
        row += f" {val:.1f}% |"
    lines.append(row)

    # Recall row
    row = "| Recall |"
    for mode in modes:
        val = avg_metric(overall, mode, "recall")
        row += f" {val:.3f} |"
    lines.append(row)

    lines.append("")

    # Delta table (if we have baseline + saga_full)
    if "baseline" in modes and "saga_full" in modes:
        b_subem = pct(overall, "baseline", "subem")
        s_nc = pct(overall, "saga_no_curator", "subem") if "saga_no_curator" in modes else None
        s_full = pct(overall, "saga_full", "subem")

        lines += ["## Curator Delta", ""]
        lines.append("| | SubEM (%) | vs Baseline |")
        lines.append("|---|---|---|")
        lines.append(f"| Baseline | {b_subem:.1f}% | — |")
        if s_nc is not None:
            lines.append(f"| SAGA (no Curator) | {s_nc:.1f}% | {s_nc - b_subem:+.1f}%p |")
        lines.append(f"| SAGA (full) | {s_full:.1f}% | {s_full - b_subem:+.1f}%p |")
        if s_nc is not None:
            lines.append(f"| **Curator contribution** | | **{s_full - s_nc:+.1f}%p** |")
        lines.append("")

    # By competency (split)
    lines += ["## By Competency", ""]
    header = "| Competency | N |"
    sep = "|------------|---|"
    for mode in modes:
        header += f" {mode} SubEM |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    split_order = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]
    for split in split_order:
        items = by_split.get(split, [])
        if not items:
            continue
        row = f"| {split} | {len(items)} |"
        for mode in modes:
            val = pct(items, mode, "subem")
            row += f" {val:.1f}% |"
        lines.append(row)

    lines.append("")

    # By source (sub-dataset)
    lines += ["## By Sub-dataset", ""]
    header = "| Source | Split | N |"
    sep = "|--------|-------|---|"
    for mode in modes:
        header += f" {mode} |"
        sep += "------|"
    lines.append(header)
    lines.append(sep)

    for source, items in sorted(by_source.items()):
        split = items[0]["split"] if items else ""
        row = f"| {source} | {split} | {len(items)} |"
        for mode in modes:
            val = pct(items, mode, "subem")
            row += f" {val:.1f}% |"
        lines.append(row)

    lines.append("")

    # Curator impact examples (best improvements)
    if "saga_no_curator" in modes and "saga_full" in modes:
        lines += ["## Curator Impact Examples", ""]

        improvements = [
            r for r in results
            if not r.get("saga_no_curator_subem") and r.get("saga_full_subem")
        ]
        regressions = [
            r for r in results
            if r.get("saga_no_curator_subem") and not r.get("saga_full_subem")
        ]

        lines.append(f"### Curator Wins ({len(improvements)} questions)")
        lines.append("")
        for r in improvements[:10]:
            lines.append(
                f"- **[{r['source']}]** Q: {r['question'][:80]}..."
            )
        lines.append("")

        lines.append(f"### Curator Regressions ({len(regressions)} questions)")
        lines.append("")
        for r in regressions[:5]:
            lines.append(
                f"- **[{r['source']}]** Q: {r['question'][:80]}..."
            )
        lines.append("")

    # Write markdown
    md_path = os.path.join(output_dir, f"memoryagent_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] Markdown: {md_path}")

    return md_path
