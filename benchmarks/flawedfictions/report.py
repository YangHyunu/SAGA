"""Generate FlawedFictions benchmark reports."""

from __future__ import annotations

import json
import os
from datetime import datetime


def _compute_metrics(results: list[dict], mode: str) -> dict:
    """Compute Accuracy, Precision, Recall, F1 for one mode."""
    pred_key = f"{mode}_predicted"
    correct_key = f"{mode}_correct"

    tp = fp = tn = fn = 0
    unparseable = 0

    for r in results:
        gt = r["has_error"]
        pred = r.get(pred_key)

        if pred is None:
            unparseable += 1
            continue

        if gt and pred:
            tp += 1
        elif not gt and pred:
            fp += 1
        elif gt and not pred:
            fn += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "unparseable": unparseable,
        "total": total + unparseable,
    }


def generate_report(
    results: list[dict],
    output_dir: str,
    modes: list[str],
) -> str:
    """Generate Markdown + JSON report. Returns markdown path."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute metrics per mode
    metrics = {}
    for mode in modes:
        metrics[mode] = _compute_metrics(results, mode)

    # --- JSON report ---
    json_path = os.path.join(output_dir, f"flawedfictions_results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[report] JSON: {json_path}")

    # --- Markdown report ---
    md_path = os.path.join(output_dir, f"flawedfictions_report_{ts}.md")
    lines = [
        "# SAGA FlawedFictions — Contradiction Detection Accuracy",
        "",
        f"Generated: {ts}",
        f"Total instances: {len(results)}",
        f"Modes: {', '.join(modes)}",
        "",
        "## Overall Metrics",
        "",
        "| Metric | " + " | ".join(modes) + " |",
        "|--------" + "|------" * len(modes) + "|",
    ]

    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        row = f"| {metric_name.capitalize()} |"
        for mode in modes:
            val = metrics[mode][metric_name]
            row += f" {val:.1%} |"
        lines.append(row)

    # Confusion matrix
    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "| | " + " | ".join(modes) + " |",
        "|---|" + "|---" * len(modes) + "|",
    ])
    for label in ["tp", "fp", "tn", "fn", "unparseable"]:
        row = f"| {label.upper()} |"
        for mode in modes:
            row += f" {metrics[mode][label]} |"
        lines.append(row)

    # Curator contribution
    if "saga_no_curator" in modes and "saga_full" in modes:
        nc = metrics["saga_no_curator"]
        full = metrics["saga_full"]
        lines.extend([
            "",
            "## Curator Contribution",
            "",
            f"| Metric | No-Curator | Full | Delta |",
            f"|--------|-----------|------|-------|",
            f"| Accuracy | {nc['accuracy']:.1%} | {full['accuracy']:.1%} | {full['accuracy']-nc['accuracy']:+.1%}p |",
            f"| F1 | {nc['f1']:.1%} | {full['f1']:.1%} | {full['f1']-nc['f1']:+.1%}p |",
            f"| Precision | {nc['precision']:.1%} | {full['precision']:.1%} | {full['precision']-nc['precision']:+.1%}p |",
            f"| Recall | {nc['recall']:.1%} | {full['recall']:.1%} | {full['recall']-nc['recall']:+.1%}p |",
        ])

    # Per-instance disagreements
    if "saga_no_curator" in modes and "saga_full" in modes:
        disagree = []
        for r in results:
            nc_pred = r.get("saga_no_curator_predicted")
            full_pred = r.get("saga_full_predicted")
            if nc_pred != full_pred:
                disagree.append(r)

        if disagree:
            lines.extend([
                "",
                f"## Curator Disagreements ({len(disagree)} instances)",
                "",
                "| Example | GT | No-Curator | Full |",
                "|---------|----|-----------:|-----:|",
            ])
            for r in disagree[:20]:
                gt = "error" if r["has_error"] else "clean"
                nc = str(r.get("saga_no_curator_predicted", "?"))
                full = str(r.get("saga_full_predicted", "?"))
                lines.append(f"| {r['example_id']} | {gt} | {nc} | {full} |")

    md_content = "\n".join(lines) + "\n"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"[report] Markdown: {md_path}")

    return md_path
