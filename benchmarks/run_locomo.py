"""CLI entry point for SAGA LOCOMO benchmark.

Usage:
    python -m benchmarks.run_locomo [--conversations N] [--no-judge] [--qa-model MODEL] [--skip-ingestion]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.download import load_locomo, download_locomo
from benchmarks.adapter import parse_all
from benchmarks.ingestion import LocomoIngestion
from benchmarks.evaluator import LocomoEvaluator
from benchmarks.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("locomo-bench")


async def main(args):
    from saga.config import load_config
    from saga.storage.sqlite_db import SQLiteDB
    from saga.storage.vector_db import VectorDB
    from saga.storage.md_cache import MdCache
    from saga.llm.client import LLMClient

    # ── Config ──
    config = load_config(args.config)

    # ── Use isolated DB/cache for benchmark ──
    bench_dir = os.path.join(os.path.dirname(__file__), "data")
    db_path = os.path.join(bench_dir, "bench_state.db")
    chroma_path = os.path.join(bench_dir, "bench_chroma")
    cache_path = os.path.join(bench_dir, "bench_cache")

    sqlite_db = SQLiteDB(db_path=db_path)
    await sqlite_db.initialize()

    vector_db = VectorDB(db_path=chroma_path)
    vector_db.initialize()

    md_cache = MdCache(cache_dir=cache_path)
    llm_client = LLMClient(config)

    try:
        # ── 1. Download & Parse ──
        logger.info("Step 1: Loading LOCOMO dataset")
        raw_data = load_locomo()
        conversations = parse_all(raw_data)
        logger.info(f"Parsed {len(conversations)} conversations, total {sum(c.total_turns for c in conversations)} turns")

        # Limit conversations
        convs = conversations[:args.conversations]
        logger.info(f"Using {len(convs)} conversations for benchmark")

        # ── 2. Ingestion ──
        session_mapping = {}
        if not args.skip_ingestion:
            logger.info("Step 2: Ingesting through Sub-B pipeline")
            ingestion = LocomoIngestion(sqlite_db, vector_db, md_cache, llm_client, config)
            session_mapping = await ingestion.ingest_conversations(convs, max_conversations=args.conversations)
            logger.info(f"Ingestion complete: {ingestion.stats}")
        else:
            logger.info("Step 2: Skipping ingestion (--skip-ingestion)")
            for conv in convs:
                session_mapping[conv.sample_id] = f"locomo-{conv.sample_id}"

        # ── 3. Evaluation ──
        logger.info("Step 3: Running QA evaluation (Baseline vs SAGA)")
        evaluator = LocomoEvaluator(
            sqlite_db=sqlite_db,
            vector_db=vector_db,
            md_cache=md_cache,
            llm_client=llm_client,
            config=config,
            qa_model=args.qa_model,
            judge_model=args.judge_model,
        )
        results = await evaluator.evaluate_all(
            convs, session_mapping, use_judge=not args.no_judge
        )

        # ── 4. Report ──
        logger.info("Step 4: Generating report")
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        report_path = generate_report(results, output_dir)
        logger.info(f"Done! Report: {report_path}")

    finally:
        await sqlite_db.close()
        await llm_client.close()


def cli():
    parser = argparse.ArgumentParser(description="SAGA LOCOMO Memory Benchmark")
    parser.add_argument(
        "--conversations", "-n", type=int, default=2,
        help="Number of LOCOMO conversations to evaluate (default: 2)",
    )
    parser.add_argument(
        "--qa-model", type=str, default="gemini-2.5-flash-lite",
        help="Model for QA answering (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--judge-model", type=str, default="gemini-2.5-flash-lite",
        help="Model for LLM-as-Judge (default: gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--no-judge", action="store_true",
        help="Skip LLM-as-Judge (F1 only)",
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true",
        help="Skip Sub-B ingestion (reuse existing data)",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to SAGA config.yaml",
    )
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli()
