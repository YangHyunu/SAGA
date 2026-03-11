"""CLI entry point for SAGA LongMemEval benchmark.

Usage:
    python -m benchmarks.longmemeval.run [--instances N] [--no-judge] [--qa-model MODEL]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmarks.longmemeval.download import load_longmemeval
from benchmarks.longmemeval.evaluator import LongMemEvalEvaluator
from benchmarks.longmemeval.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("longmemeval-bench")


async def main(args):
    from saga.config import load_config
    from saga.llm.client import LLMClient

    config = load_config(args.config)
    llm_client = LLMClient(config)

    try:
        # 1. Load dataset
        logger.info("Step 1: Loading LongMemEval dataset")
        data = load_longmemeval(variant=args.variant)
        logger.info(f"Loaded {len(data)} instances")

        # 2. Evaluate
        logger.info("Step 2: Running QA evaluation (Baseline vs SAGA)")
        evaluator = LongMemEvalEvaluator(
            llm_client=llm_client,
            qa_model=args.qa_model,
            judge_model=args.judge_model,
            openai_api_key=config.api_keys.openai,
            checkpoint_dir=os.path.join(os.path.dirname(__file__), "data"),
        )
        results = await evaluator.evaluate_all(
            data,
            max_instances=args.instances,
            use_judge=not args.no_judge,
            concurrency=args.concurrency,
        )

        # 3. Report
        logger.info("Step 3: Generating report")
        output_dir = os.path.join(os.path.dirname(__file__), "data", "results")
        report_path = generate_report(results, output_dir)
        logger.info(f"Done! Report: {report_path}")

    finally:
        await llm_client.close()


def cli():
    parser = argparse.ArgumentParser(description="SAGA LongMemEval Benchmark")
    parser.add_argument(
        "--instances", "-n", type=int, default=None,
        help="Number of instances to evaluate (default: all 500)",
    )
    parser.add_argument(
        "--variant", type=str, default="s",
        choices=["s", "oracle"],
        help="Dataset variant: 's' (115K tokens) or 'oracle' (evidence only)",
    )
    parser.add_argument(
        "--qa-model", type=str, default="gemini-2.5-flash",
        help="Model for QA answering (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gemini-2.5-flash",
        help="Model for Yes/No judge (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip judge evaluation",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Number of concurrent evaluations (default: 4)",
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

    if args.clear_checkpoint:
        cp_path = os.path.join(os.path.dirname(__file__), "data", "checkpoint.jsonl")
        if os.path.exists(cp_path):
            os.remove(cp_path)
            print("[checkpoint] Cleared")

    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
