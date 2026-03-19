"""FlawedFictions evaluator — 3-way contradiction detection comparison.

Modes:
- baseline: full story + prompt → LLM direct judgment
- saga_no_curator: Sub-B episode ingestion → ContextBuilder → judgment
- saga_full: Sub-B + Curator (contradiction_log) → judgment
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from functools import partial

from benchmarks.flawedfictions.dataset import FlawedFictionInstance
from benchmarks.memoryagent.isolation import (
    IsolatedEnv,
    cleanup_env,
    create_isolated_env,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3

DETECTION_PROMPT = """\
You are tasked with detecting the presence of continuity errors in a short story. \
A continuity error occurs when an event or detail in the story contradicts or is \
incompatible with previously established information about the story's world or characters.

Here is the story to analyze:

<story>
{story_context}
</story>

Carefully analyze the story for any continuity errors — contradictions between \
earlier and later parts regarding characters, settings, timelines, or established facts.

Answer with ONLY one of these two words:
- "YES" if you found a continuity error
- "NO" if the story is consistent

Your answer:"""

DETECTION_PROMPT_SAGA = """\
You have access to a memory system that has tracked the content of a story across multiple turns.

--- Memory Context ---
{memory_context}

Based on the memory context above, determine whether the story contains any \
continuity errors — contradictions between earlier and later parts regarding \
characters, settings, timelines, or established facts.

Answer with ONLY one of these two words:
- "YES" if you found a continuity error
- "NO" if the story is consistent

Your answer:"""


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


def _parse_decision(response: str) -> bool | None:
    """Parse YES/NO from LLM response. Returns True=has_error, False=clean, None=unparseable."""
    text = response.strip().upper()
    # Check first word
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^A-Z]", "", first_word)
    if first_word == "YES":
        return True
    if first_word == "NO":
        return False
    # Fallback: search anywhere
    if "YES" in text and "NO" not in text:
        return True
    if "NO" in text and "YES" not in text:
        return False
    return None


# ---------------------------------------------------------------------------
# Memorization (Sub-B ingestion)
# ---------------------------------------------------------------------------

async def _memorize_paragraphs(
    env: IsolatedEnv,
    paragraphs: list[str],
    session_id: str,
    llm_client,
    config,
    run_curator: bool = False,
    curator_interval: int = 5,
) -> dict:
    """Feed story paragraphs through Sub-B pipeline."""
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
        curator_runner.lore_defer_delay = 0

    await env.sqlite_db.create_session(session_id=session_id, name=f"ff-{session_id}")
    os.makedirs(env.md_cache.get_session_dir(session_id), exist_ok=True)

    stats = {"paragraphs": len(paragraphs), "errors": 0, "curator_runs": 0, "elapsed_sec": 0.0}
    t_start = time.time()

    for i, para in enumerate(paragraphs):
        turn_number = i + 1
        try:
            await post_turn.extract_and_update(
                session_id=session_id,
                response_text=para,
                turn_number=turn_number,
                user_input=f"(paragraph {turn_number}/{len(paragraphs)})",
            )
        except Exception as e:
            logger.error(f"[memorize] Paragraph {turn_number} error: {e}")
            stats["errors"] += 1

        if curator_runner and turn_number % curator_interval == 0:
            try:
                await curator_runner.run(session_id, turn_number)
                stats["curator_runs"] += 1
            except Exception as e:
                logger.warning(f"[memorize] Curator run failed at paragraph {turn_number}: {e}")

    # Final Curator run
    if curator_runner and len(paragraphs) % curator_interval != 0:
        try:
            await curator_runner.run(session_id, len(paragraphs))
            stats["curator_runs"] += 1
        except Exception as e:
            logger.warning(f"[memorize] Final curator run failed: {e}")

    # Wait for deferred lore tasks
    if curator_runner and curator_runner._lore_tasks:
        logger.info(f"[memorize] Waiting for {len(curator_runner._lore_tasks)} deferred lore tasks...")
        await asyncio.gather(*curator_runner._lore_tasks, return_exceptions=True)

    stats["elapsed_sec"] = time.time() - t_start
    logger.info(
        f"[memorize] Done: {stats['paragraphs']} paragraphs, {stats['errors']} errors, "
        f"{stats['curator_runs']} curator runs, {stats['elapsed_sec']:.1f}s"
    )
    return stats


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

async def _detect_baseline(
    story: str,
    llm_client,
    qa_model: str,
) -> str:
    """Baseline: full story → LLM direct judgment."""
    prompt = DETECTION_PROMPT.format(story_context=story)
    response = await _call_with_retry(
        llm_client,
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    return response.strip()


async def _detect_saga(
    env: IsolatedEnv,
    session_id: str,
    llm_client,
    config,
    qa_model: str,
) -> str:
    """SAGA: detect using ContextBuilder memory retrieval."""
    from saga.agents.context_builder import ContextBuilder

    ctx_builder = ContextBuilder(
        sqlite_db=env.sqlite_db,
        vector_db=env.vector_db,
        md_cache=env.md_cache,
        config=config,
    )

    ctx = await ctx_builder.build_context(
        session_id=session_id,
        messages=[{"role": "user", "content": "Are there any continuity errors or contradictions in this story?"}],
        token_budget=config.token_budget.dynamic_context_max,
    )

    md_prefix = ctx.get("md_prefix", "")
    dynamic_suffix = ctx.get("dynamic_suffix", "")
    memory_context = f"{md_prefix}\n{dynamic_suffix}".strip()

    prompt = DETECTION_PROMPT_SAGA.format(memory_context=memory_context)
    response = await _call_with_retry(
        llm_client,
        model=qa_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=64,
    )
    return response.strip()


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

class FlawedFictionsEvaluator:
    """Evaluate FlawedFictions in 3 modes: baseline, saga_no_curator, saga_full."""

    def __init__(
        self,
        llm_client,
        config,
        qa_model: str = "gemini-2.5-flash",
        base_dir: str = "benchmarks/flawedfictions/data",
        modes: list[str] | None = None,
        curator_interval: int = 5,
    ):
        self.llm_client = llm_client
        self.config = config
        self.qa_model = qa_model
        self.base_dir = base_dir
        self.modes = modes or ["baseline", "saga_no_curator", "saga_full"]
        self.curator_interval = curator_interval

    async def evaluate_instance(self, inst: FlawedFictionInstance, inst_index: int) -> dict:
        """Evaluate a single instance across all modes."""
        result = {
            "example_id": inst.example_id,
            "instance_index": inst_index,
            "has_error": inst.has_error,
            "num_paragraphs": inst.num_paragraphs,
            "story_len": len(inst.story),
        }

        for mode in self.modes:
            try:
                predicted = await self._evaluate_mode(inst, mode)
                decision = _parse_decision(predicted)
                result[f"{mode}_raw"] = predicted
                result[f"{mode}_predicted"] = decision
                result[f"{mode}_correct"] = decision == inst.has_error if decision is not None else False
                logger.info(
                    f"[eval] {inst.example_id} {mode}: "
                    f"predicted={decision}, gt={inst.has_error}, "
                    f"correct={result[f'{mode}_correct']}"
                )
            except Exception as e:
                logger.error(f"[eval] {inst.example_id} {mode} failed: {e}")
                result[f"{mode}_raw"] = f"ERROR: {e}"
                result[f"{mode}_predicted"] = None
                result[f"{mode}_correct"] = False

        return result

    async def _evaluate_mode(self, inst: FlawedFictionInstance, mode: str) -> str:
        """Run one mode for one instance."""
        if mode == "baseline":
            return await _detect_baseline(inst.story, self.llm_client, self.qa_model)

        # SAGA modes need isolated environment
        # Note: Letta agent 생성은 생략 — CuratorRunner가 Letta 연결 가능 시 자체 생성,
        # 불가 시 DirectLLMCuratorAdapter fallback 사용 (벤치마크에서는 fallback이 더 빠름)
        run_curator = mode == "saga_full"
        session_id = f"ff-{inst.example_id}-{mode}"
        env = await create_isolated_env(self.base_dir, f"{inst.example_id}_{mode}")

        try:
            await _memorize_paragraphs(
                env=env,
                paragraphs=inst.paragraphs,
                session_id=session_id,
                llm_client=self.llm_client,
                config=self.config,
                run_curator=run_curator,
                curator_interval=self.curator_interval,
            )

            return await _detect_saga(
                env=env,
                session_id=session_id,
                llm_client=self.llm_client,
                config=self.config,
                qa_model=self.qa_model,
            )
        finally:
            await cleanup_env(env, delete_files=False)

    async def evaluate_all(
        self,
        instances: list[FlawedFictionInstance],
        checkpoint_path: str | None = None,
    ) -> list[dict]:
        """Evaluate all instances with checkpoint support."""
        # Load existing checkpoint
        completed: dict[str, dict] = {}
        if checkpoint_path and os.path.exists(checkpoint_path):
            with open(checkpoint_path, "r") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        completed[r["example_id"]] = r
            logger.info(f"[eval] Loaded {len(completed)} checkpointed results")

        results = list(completed.values())

        for i, inst in enumerate(instances):
            if inst.example_id in completed:
                logger.info(f"[eval] Skipping {inst.example_id} (checkpointed)")
                continue

            logger.info(f"[eval] Instance {i+1}/{len(instances)}: {inst.example_id} ({inst.num_paragraphs} paragraphs)")
            result = await self.evaluate_instance(inst, i)
            results.append(result)

            # Save checkpoint
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                with open(checkpoint_path, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # Log summary
        for mode in self.modes:
            key = f"{mode}_correct"
            vals = [r[key] for r in results if key in r]
            if vals:
                acc = sum(1 for v in vals if v) / len(vals)
                logger.info(f"[eval] {mode}: {sum(1 for v in vals if v)}/{len(vals)} correct ({acc:.1%})")

        return results
