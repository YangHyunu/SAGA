"""CLI entry point for FlawedFictions — Contradiction Detection Benchmark.

Usage:
    python -m benchmarks.flawedfictions.run [OPTIONS]

Examples:
    # Quick test: 2 examples, baseline only
    python -m benchmarks.flawedfictions.run --max-examples 2 --modes baseline

    # Full 3-way comparison
    python -m benchmarks.flawedfictions.run --modes baseline,saga_no_curator,saga_full

    # Long stories
    python -m benchmarks.flawedfictions.run --split flawed_fictions_long --max-examples 10
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("flawedfictions")


async def main(args):
    from saga.config import load_config
    from saga.llm.client import LLMClient
    from benchmarks.flawedfictions.dataset import load_dataset
    from benchmarks.flawedfictions.evaluator import FlawedFictionsEvaluator
    from benchmarks.flawedfictions.report import generate_report

    config = load_config(args.config)
    llm_client = LLMClient(config)

    try:
        # 1. Load dataset
        logger.info(f"Step 1: Loading FlawedFictions (split={args.split}, max={args.max_examples})")
        instances = load_dataset(split=args.split, max_examples=args.max_examples)

        if not instances:
            logger.error("No instances loaded.")
            return

        # 2. Evaluate
        modes = args.modes.split(",")
        logger.info(f"Step 2: Evaluating ({len(instances)} instances, modes={modes})")

        base_dir = os.path.join(os.path.dirname(__file__), "data")
        checkpoint_path = os.path.join(base_dir, "checkpoint.jsonl")

        if args.clear_checkpoint:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
            # Also remove per-instance data directories
            for entry in os.listdir(base_dir) if os.path.exists(base_dir) else []:
                entry_path = os.path.join(base_dir, entry)
                if os.path.isdir(entry_path) and entry != "results":
                    shutil.rmtree(entry_path, ignore_errors=True)
            logger.info("Cleared checkpoint + data directories")

        evaluator = FlawedFictionsEvaluator(
            llm_client=llm_client,
            config=config,
            qa_model=args.qa_model,
            base_dir=base_dir,
            modes=modes,
            curator_interval=args.curator_interval,
        )

        results = await evaluator.evaluate_all(instances, checkpoint_path=checkpoint_path)

        # 3. Report
        logger.info("Step 3: Generating report")
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        report_path = generate_report(results, output_dir, modes)
        logger.info(f"Done! Report: {report_path}")

        _print_summary(results, modes)

    finally:
        await llm_client.close()


def _print_summary(results: list[dict], modes: list[str]):
    """Print quick summary to console."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*60}")
    print(f"  FlawedFictions Summary ({len(results)} instances)")
    print(f"{'='*60}")

    for mode in modes:
        key = f"{mode}_correct"
        vals = [r[key] for r in results if key in r]
        if vals:
            acc = sum(1 for v in vals if v) / len(vals) * 100
            print(f"  {mode:20s}: Accuracy {acc:.1f}%")

    if "saga_no_curator" in modes and "saga_full" in modes:
        nc_key, full_key = "saga_no_curator_correct", "saga_full_correct"
        nc_vals = [r[nc_key] for r in results if nc_key in r]
        full_vals = [r[full_key] for r in results if full_key in r]
        if nc_vals and full_vals:
            nc_acc = sum(1 for v in nc_vals if v) / len(nc_vals) * 100
            full_acc = sum(1 for v in full_vals if v) / len(full_vals) * 100
            print(f"\n  Curator contribution: {full_acc - nc_acc:+.1f}%p")

    print(f"{'='*60}\n")


def cli():
    parser = argparse.ArgumentParser(
        description="SAGA FlawedFictions — Contradiction Detection Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split", type=str, default="flawed_fictions",
        help="Dataset split: flawed_fictions (414, ~730w) or flawed_fictions_long (200, ~2700w)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=None,
        help="Limit number of examples (default: all)",
    )
    parser.add_argument(
        "--modes", type=str, default="baseline,saga_no_curator,saga_full",
        help="Comma-separated modes (default: baseline,saga_no_curator,saga_full)",
    )
    parser.add_argument(
        "--qa-model", type=str, default="gemini-2.5-flash",
        help="Model for detection judgment (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--curator-interval", type=int, default=5,
        help="Run Curator every N paragraphs (default: 5)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to SAGA config.yaml",
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Clear checkpoint and start fresh",
    )
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
