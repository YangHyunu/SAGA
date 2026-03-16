"""CLI entry point for MemoryAgentBench — Curator effectiveness benchmark.

Usage:
    python -m benchmarks.memoryagent.run [OPTIONS]

Examples:
    # Quick test: Conflict Resolution only, 1 sequence
    python -m benchmarks.memoryagent.run --splits Conflict_Resolution --max-seq 1

    # Full benchmark: all splits, 3-way comparison
    python -m benchmarks.memoryagent.run --modes baseline saga_no_curator saga_full

    # Curator-only comparison (skip baseline for speed)
    python -m benchmarks.memoryagent.run --modes saga_no_curator saga_full --splits Conflict_Resolution
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("memoryagent-bench")


async def main(args):
    from saga.config import load_config
    from saga.llm.client import LLMClient
    from benchmarks.memoryagent.dataset import load_dataset
    from benchmarks.memoryagent.evaluator import MemoryAgentEvaluator
    from benchmarks.memoryagent.report import generate_report

    # Config
    config = load_config(args.config)
    llm_client = LLMClient(config)

    try:
        # 1. Load dataset
        splits = args.splits.split(",") if args.splits else None
        logger.info(f"Step 1: Loading MemoryAgentBench dataset (splits={splits or 'all'})")
        sequences = load_dataset(
            splits=splits,
            max_sequences_per_split=args.max_seq,
        )

        if not sequences:
            logger.error("No sequences loaded. Check dataset availability.")
            return

        # 2. Evaluate
        modes = args.modes.split(",")
        logger.info(f"Step 2: Evaluating ({len(sequences)} sequences, modes={modes})")

        base_dir = os.path.join(os.path.dirname(__file__), "data")
        checkpoint_path = os.path.join(base_dir, "checkpoint.jsonl")

        if args.clear_checkpoint and os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info("Cleared checkpoint file")

        evaluator = MemoryAgentEvaluator(
            llm_client=llm_client,
            config=config,
            qa_model=args.qa_model,
            judge_model=args.judge_model,
            base_dir=base_dir,
            modes=modes,
            curator_interval=args.curator_interval,
            chunk_size=args.chunk_size,
        )

        results = await evaluator.evaluate_all(
            sequences,
            checkpoint_path=checkpoint_path,
        )

        # 3. Report
        logger.info("Step 3: Generating report")
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        report_path = generate_report(results, output_dir, modes)
        logger.info(f"Done! Report: {report_path}")

        # Print quick summary
        _print_summary(results, modes)

    finally:
        await llm_client.close()


def _print_summary(results: list[dict], modes: list[str]):
    """Print quick summary to console."""
    if not results:
        print("\nNo results to summarize.")
        return

    print(f"\n{'='*60}")
    print(f"  MemoryAgentBench Summary ({len(results)} QA pairs)")
    print(f"{'='*60}")

    for mode in modes:
        key = f"{mode}_subem"
        vals = [r[key] for r in results if key in r]
        if vals:
            acc = sum(1 for v in vals if v) / len(vals) * 100
            print(f"  {mode:20s}: SubEM {acc:.1f}%")

    # Curator delta
    if "saga_no_curator" in modes and "saga_full" in modes:
        nc_key, full_key = "saga_no_curator_subem", "saga_full_subem"
        nc_vals = [r[nc_key] for r in results if nc_key in r]
        full_vals = [r[full_key] for r in results if full_key in r]
        if nc_vals and full_vals:
            nc_acc = sum(1 for v in nc_vals if v) / len(nc_vals) * 100
            full_acc = sum(1 for v in full_vals if v) / len(full_vals) * 100
            print(f"\n  Curator contribution: {full_acc - nc_acc:+.1f}%p")

    print(f"{'='*60}\n")


def cli():
    parser = argparse.ArgumentParser(
        description="SAGA MemoryAgentBench — Curator Effectiveness Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Splits: Accurate_Retrieval, Test_Time_Learning, "
            "Long_Range_Understanding, Conflict_Resolution\n"
            "Modes: baseline, saga_no_curator, saga_full"
        ),
    )
    parser.add_argument(
        "--splits", type=str, default=None,
        help="Comma-separated splits to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-seq", type=int, default=None,
        help="Max sequences per split (default: all)",
    )
    parser.add_argument(
        "--modes", type=str, default="baseline,saga_no_curator,saga_full",
        help="Comma-separated modes (default: baseline,saga_no_curator,saga_full)",
    )
    parser.add_argument(
        "--qa-model", type=str, default="gemini-2.5-flash",
        help="Model for QA answering (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gemini-2.5-flash",
        help="Model for LLM-as-Judge (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--curator-interval", type=int, default=10,
        help="Run Curator every N chunks (default: 10)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=4096,
        help="Chunk size in tokens for memorization (default: 4096)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to SAGA config.yaml",
    )
    parser.add_argument(
        "--clear-checkpoint", action="store_true",
        help="Clear checkpoint file and start fresh",
    )
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
