"""3-way evaluator: Baseline vs SAGA(no-Curator) vs SAGA(full) on MemoryAgentBench."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from functools import partial
from typing import Optional

from benchmarks.memoryagent.dataset import Sequence
from benchmarks.memoryagent.isolation import (
    IsolatedEnv,
    cleanup_env,
    create_isolated_env,
    create_letta_agent,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3


async def _call_with_retry(llm_client, **kwargs):
    """Call LLM with exponential backoff on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await llm_client.call_llm(**kwargs)
        except Exception as e:
            err_str = str(e)
            is_retryable = any(code in err_str for code in ("503", "429", "UNAVAILABLE"))
            if is_retryable and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"[eval] LLM retry {attempt+1}/{MAX_RETRIES} in {delay}s: {err_str[:100]}")
                await asyncio.sleep(delay)
            else:
                raise


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def substring_exact_match(prediction: str, answers: list[str]) -> bool:
    """SubEM: True if any ground-truth answer is a substring of the prediction."""
    pred_lower = prediction.lower().strip()
    return any(ans.lower().strip() in pred_lower for ans in answers if ans)


def recall_score(prediction: str, answers: list[str]) -> float:
    """Recall: fraction of ground-truth answers found in prediction."""
    if not answers:
        return 0.0
    pred_lower = prediction.lower().strip()
    found = sum(1 for ans in answers if ans and ans.lower().strip() in pred_lower)
    return found / len(answers)


# ---------------------------------------------------------------------------
# Memorization (Sub-B ingestion)
# ---------------------------------------------------------------------------

async def _memorize_chunks(
    env: IsolatedEnv,
    chunks: list[str],
    session_id: str,
    llm_client,
    config,
    run_curator: bool = False,
    curator_interval: int = 10,
) -> dict:
    """Feed chunks through Sub-B pipeline. Optionally run Curator every N chunks.

    Returns timing stats.
    """
    from saga.agents.post_turn import PostTurnExtractor
    from saga.agents.extractors import narrative_extract

    extract_fn = partial(narrative_extract, llm_client=llm_client, config=config)
    post_turn = PostTurnExtractor(
        sqlite_db=env.sqlite_db,
        vector_db=env.vector_db,
        md_cache=env.md_cache,
        llm_client=llm_client,
        config=config,
        extract_fn=extract_fn,
    )

    # Optionally set up Curator
    curator_runner = None
    if run_curator:
        from saga.agents.curator import CuratorRunner
        curator_runner = CuratorRunner(
            sqlite_db=env.sqlite_db,
            vector_db=env.vector_db,
            md_cache=env.md_cache,
            llm_client=llm_client,
            config=config,
        )
        curator_runner.initialize()
        curator_runner.lore_defer_delay = 0  # no delay in benchmark mode

    # Create session
    await env.sqlite_db.create_session(session_id=session_id, name=f"bench-{session_id}")
    import os
    os.makedirs(env.md_cache.get_session_dir(session_id), exist_ok=True)

    stats = {"chunks": len(chunks), "errors": 0, "curator_runs": 0, "elapsed_sec": 0.0}
    t_start = time.time()

    for i, chunk in enumerate(chunks):
        turn_number = i + 1
        try:
            await post_turn.extract_and_update(
                session_id=session_id,
                response_text=chunk,
                turn_number=turn_number,
                user_input=f"(chunk {turn_number}/{len(chunks)})",
            )
        except Exception as e:
            logger.error(f"[memorize] Chunk {turn_number} error: {e}")
            stats["errors"] += 1

        # Run Curator periodically
        if curator_runner and turn_number % curator_interval == 0:
            try:
                await curator_runner.run(session_id, turn_number)
                stats["curator_runs"] += 1
            except Exception as e:
                logger.warning(f"[memorize] Curator run failed at chunk {turn_number}: {e}")

        if turn_number % 50 == 0:
            logger.info(f"[memorize] Progress: {turn_number}/{len(chunks)} chunks")

    # Final Curator run
    if curator_runner and len(chunks) % curator_interval != 0:
        try:
            await curator_runner.run(session_id, len(chunks))
            stats["curator_runs"] += 1
        except Exception as e:
            logger.warning(f"[memorize] Final curator run failed: {e}")

    # Wait for deferred lore tasks to finish before cleanup closes DB connections
    if curator_runner and curator_runner._lore_tasks:
        logger.info(f"[memorize] Waiting for {len(curator_runner._lore_tasks)} deferred lore tasks...")
        await asyncio.gather(*curator_runner._lore_tasks, return_exceptions=True)

    stats["elapsed_sec"] = time.time() - t_start
    logger.info(
        f"[memorize] Done: {stats['chunks']} chunks, {stats['errors']} errors, "
        f"{stats['curator_runs']} curator runs, {stats['elapsed_sec']:.1f}s"
    )
    return stats


# ---------------------------------------------------------------------------
# Query answering
# ---------------------------------------------------------------------------

async def _answer_baseline(
    chunks: list[str],
    question: str,
    llm_client,
    qa_model: str,
    max_context_chars: int = 100_000,
) -> str:
    """Baseline: answer from raw context (last N chunks that fit in context)."""
    # Build context from the end (most recent chunks first)
    context_parts = []
    total = 0
    for chunk in reversed(chunks):
        if total + len(chunk) > max_context_chars:
            break
        context_parts.insert(0, chunk)
        total += len(chunk)

    context_str = "\n\n".join(context_parts)

    prompt = (
        "Below is a long document split into sections. "
        "Answer the question based ONLY on the document content. "
        "If the answer is not in the document, say 'I don't know'.\n\n"
        f"--- Document ---\n{context_str}\n\n"
        f"--- Question ---\n{question}\n\n"
        "Answer concisely:"
    )

    response = await _call_with_retry(
        llm_client,
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return response.strip()


async def _answer_saga(
    env: IsolatedEnv,
    session_id: str,
    question: str,
    llm_client,
    config,
    qa_model: str,
) -> str:
    """SAGA: answer using ContextBuilder (Sub-A) memory retrieval."""
    from saga.agents.context_builder import ContextBuilder

    ctx_builder = ContextBuilder(
        sqlite_db=env.sqlite_db,
        vector_db=env.vector_db,
        md_cache=env.md_cache,
        config=config,
    )

    # Build context from memory
    # Use question as the "last user message" for semantic retrieval
    ctx = await ctx_builder.build_context(
        session_id=session_id,
        messages=[{"role": "user", "content": question}],
        token_budget=config.token_budget.dynamic_context_max,
    )

    md_prefix = ctx.get("md_prefix", "")
    dynamic_suffix = ctx.get("dynamic_suffix", "")

    prompt = (
        "You have access to a memory system that has tracked document content.\n\n"
        f"--- Memory Context ---\n{md_prefix}\n{dynamic_suffix}\n\n"
        "Answer the question based on the memory context. "
        "If the answer is not available, say 'I don't know'.\n\n"
        f"--- Question ---\n{question}\n\n"
        "Answer concisely:"
    )

    response = await _call_with_retry(
        llm_client,
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return response.strip()


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class MemoryAgentEvaluator:
    """Evaluate sequences in 3 modes: baseline, saga-no-curator, saga-full."""

    def __init__(
        self,
        llm_client,
        config,
        qa_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        base_dir: str = "benchmarks/memoryagent/data",
        modes: list[str] | None = None,
        curator_interval: int = 10,
        chunk_size: int = 4096,
    ):
        self.llm_client = llm_client
        self.config = config
        self.qa_model = qa_model
        self.judge_model = judge_model
        self.base_dir = base_dir
        self.modes = modes or ["baseline", "saga_no_curator", "saga_full"]
        self.curator_interval = curator_interval
        self.chunk_size = chunk_size

    async def evaluate_sequence(self, seq: Sequence) -> list[dict]:
        """Evaluate one sequence across all modes. Returns per-question results."""
        seq_id = f"{seq.split}_{seq.seq_index}"
        chunks = seq.get_chunks(self.chunk_size)
        logger.info(
            f"[eval] Sequence {seq_id}: {len(chunks)} chunks, "
            f"{seq.num_questions} questions, source={seq.source}"
        )

        results = []
        mode_answers: dict[str, list[str]] = {}
        mode_stats: dict[str, dict] = {}

        for mode in self.modes:
            logger.info(f"[eval] === Mode: {mode} ===")

            if mode == "baseline":
                # No memorization needed — answer from raw chunks
                answers = []
                for i, q in enumerate(seq.questions):
                    ans = await _answer_baseline(
                        chunks, q, self.llm_client, self.qa_model,
                    )
                    answers.append(ans)
                    if (i + 1) % 20 == 0:
                        logger.info(f"[eval] baseline: {i+1}/{seq.num_questions} questions")
                mode_answers[mode] = answers
                mode_stats[mode] = {"chunks": len(chunks), "elapsed_sec": 0}

            else:
                # SAGA modes need isolated env + memorization
                run_curator = mode == "saga_full"
                env = await create_isolated_env(self.base_dir, f"{seq_id}_{mode}")

                try:
                    # Create Letta agent if running full curator
                    if run_curator:
                        await create_letta_agent(env, self.config)

                    session_id = f"bench-{seq_id}"

                    # Memorize chunks through Sub-B
                    stats = await _memorize_chunks(
                        env=env,
                        chunks=chunks,
                        session_id=session_id,
                        llm_client=self.llm_client,
                        config=self.config,
                        run_curator=run_curator,
                        curator_interval=self.curator_interval,
                    )
                    mode_stats[mode] = stats

                    # Answer questions using SAGA retrieval
                    answers = []
                    for i, q in enumerate(seq.questions):
                        ans = await _answer_saga(
                            env, session_id, q,
                            self.llm_client, self.config, self.qa_model,
                        )
                        answers.append(ans)
                        if (i + 1) % 20 == 0:
                            logger.info(f"[eval] {mode}: {i+1}/{seq.num_questions} questions")
                    mode_answers[mode] = answers

                finally:
                    await cleanup_env(env, delete_files=False)

        # Assemble per-question results
        for i, (question, ground_truths) in enumerate(zip(seq.questions, seq.answers)):
            result = {
                "seq_id": seq_id,
                "split": seq.split,
                "source": seq.source,
                "question_index": i,
                "question_type": seq.question_types[i] if seq.question_types and i < len(seq.question_types) else "",
                "question": question,
                "ground_truths": ground_truths,
            }

            for mode in self.modes:
                pred = mode_answers.get(mode, [""])[i] if i < len(mode_answers.get(mode, [])) else ""
                result[f"{mode}_answer"] = pred
                result[f"{mode}_subem"] = substring_exact_match(pred, ground_truths)
                result[f"{mode}_recall"] = recall_score(pred, ground_truths)

            results.append(result)

        return results

    async def evaluate_all(
        self,
        sequences: list[Sequence],
        checkpoint_path: str | None = None,
    ) -> list[dict]:
        """Evaluate all sequences. Supports checkpoint resume."""
        all_results = []
        done_ids: set[str] = set()

        # Load checkpoint if exists
        if checkpoint_path:
            try:
                with open(checkpoint_path, "r") as f:
                    for line in f:
                        r = json.loads(line)
                        all_results.append(r)
                        done_ids.add(f"{r['seq_id']}_{r['question_index']}")
                if done_ids:
                    logger.info(f"[eval] Resumed from checkpoint: {len(done_ids)} questions done")
            except FileNotFoundError:
                pass

        for seq_idx, seq in enumerate(sequences):
            seq_id = f"{seq.split}_{seq.seq_index}"

            # Skip if all questions already done
            all_done = all(
                f"{seq_id}_{qi}" in done_ids
                for qi in range(seq.num_questions)
            )
            if all_done:
                logger.info(f"[eval] Skipping {seq_id} (all {seq.num_questions} questions done)")
                continue

            logger.info(
                f"[eval] Sequence {seq_idx+1}/{len(sequences)}: {seq_id} "
                f"({seq.num_questions} QAs)"
            )

            try:
                results = await self.evaluate_sequence(seq)

                # Filter out already-done questions
                new_results = [
                    r for r in results
                    if f"{r['seq_id']}_{r['question_index']}" not in done_ids
                ]
                all_results.extend(new_results)

                # Save checkpoint
                if checkpoint_path:
                    import os
                    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    with open(checkpoint_path, "a") as f:
                        for r in new_results:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

            except Exception as e:
                logger.error(f"[eval] Sequence {seq_id} failed: {e}", exc_info=True)

        return all_results
