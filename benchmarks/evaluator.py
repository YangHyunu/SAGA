"""Evaluate Baseline vs SAGA-augmented QA on LOCOMO."""

from __future__ import annotations

import asyncio
import json
import logging
import time

from benchmarks.adapter import LocomoConversation, QAPair, turns_to_messages
from benchmarks.metrics import build_judge_prompt, parse_judge_response, token_f1

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3  # seconds


async def _call_with_retry(llm_client, **kwargs):
    """Call LLM with exponential backoff retry on transient errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return await llm_client.call_llm(**kwargs)
        except Exception as e:
            err_str = str(e)
            is_retryable = "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str
            if is_retryable and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(f"[eval] LLM call failed (attempt {attempt + 1}), retrying in {delay}s: {err_str[:100]}")
                await asyncio.sleep(delay)
            else:
                raise

# Maximum turns to include in baseline context (simulates context window limit)
BASELINE_MAX_TURNS = 60


class LocomoEvaluator:
    """Run QA evaluation in two modes: baseline (raw context) and SAGA (memory-augmented)."""

    def __init__(
        self,
        sqlite_db,
        vector_db,
        md_cache,
        llm_client,
        config,
        qa_model: str = "gemini-2.5-flash-lite",
        judge_model: str = "gemini-2.5-flash-lite",
    ):
        from saga.agents.context_builder import ContextBuilder

        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config
        self.qa_model = qa_model
        self.judge_model = judge_model

        self.context_builder = ContextBuilder(
            sqlite_db=sqlite_db,
            vector_db=vector_db,
            md_cache=md_cache,
            config=config,
        )

    # ─────────────────────────────────────────
    # Baseline: raw truncated context
    # ─────────────────────────────────────────

    async def _answer_baseline(
        self, conv: LocomoConversation, qa: QAPair
    ) -> str:
        """Answer QA using raw conversation context (truncated to last N turns)."""
        messages = turns_to_messages(conv.all_turns, conv.speaker_a, conv.speaker_b)

        # Truncate to simulate context window limitation
        truncated = messages[-BASELINE_MAX_TURNS:]

        # Build conversation context string
        context_lines = []
        for msg in truncated:
            speaker = conv.speaker_a if msg["role"] == "user" else conv.speaker_b
            context_lines.append(f"{speaker}: {msg['content']}")
        context_str = "\n".join(context_lines)

        prompt = (
            f"Below is a conversation between {conv.speaker_a} and {conv.speaker_b}.\n"
            f"Answer the question based ONLY on the conversation. "
            f"If the answer is not in the conversation, say 'Unanswerable'.\n\n"
            f"--- Conversation ---\n{context_str}\n\n"
            f"--- Question ---\n{qa.question}\n\n"
            f"Answer concisely:"
        )

        response = await _call_with_retry(
            self.llm_client,
            model=self.qa_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.strip()

    # ─────────────────────────────────────────
    # SAGA: memory-augmented context
    # ─────────────────────────────────────────

    async def _answer_saga(
        self, conv: LocomoConversation, qa: QAPair, session_id: str
    ) -> str:
        """Answer QA using SAGA's ContextBuilder (Sub-A) memory retrieval."""
        messages = turns_to_messages(conv.all_turns, conv.speaker_a, conv.speaker_b)

        # Use ContextBuilder to retrieve relevant context from memory
        ctx = await self.context_builder.build_context(
            session_id=session_id,
            messages=messages[-10:],  # only recent messages (simulates sliding window)
            token_budget=self.config.token_budget.dynamic_context_max,
        )

        md_prefix = ctx.get("md_prefix", "")
        dynamic_suffix = ctx.get("dynamic_suffix", "")

        # Build conversation context: last 10 turns + memory-retrieved context
        recent_lines = []
        for msg in messages[-10:]:
            speaker = conv.speaker_a if msg["role"] == "user" else conv.speaker_b
            recent_lines.append(f"{speaker}: {msg['content']}")
        recent_str = "\n".join(recent_lines)

        prompt = (
            f"You have access to a memory system that has tracked this conversation.\n\n"
            f"--- Memory Context ---\n{md_prefix}\n{dynamic_suffix}\n\n"
            f"--- Recent Conversation ---\n{recent_str}\n\n"
            f"Answer the question based on the memory context and conversation. "
            f"If the answer is not available, say 'Unanswerable'.\n\n"
            f"--- Question ---\n{qa.question}\n\n"
            f"Answer concisely:"
        )

        response = await _call_with_retry(
            self.llm_client,
            model=self.qa_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        return response.strip()

    # ─────────────────────────────────────────
    # LLM-as-Judge
    # ─────────────────────────────────────────

    async def _judge(self, question: str, ground_truth: str, prediction: str) -> dict:
        """Score a prediction using LLM-as-Judge."""
        prompt = build_judge_prompt(question, ground_truth, prediction)
        response = await _call_with_retry(
            self.llm_client,
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        return parse_judge_response(response)

    # ─────────────────────────────────────────
    # Main evaluation loop
    # ─────────────────────────────────────────

    async def _evaluate_single_qa(
        self,
        conv: LocomoConversation,
        session_id: str,
        i: int,
        qa: QAPair,
        use_judge: bool,
    ) -> dict:
        """Evaluate a single QA pair."""
        logger.info(
            f"[eval] {conv.sample_id} QA {i + 1}/{len(conv.qa_pairs)} "
            f"({qa.category_name})"
        )

        # Get answers from both modes in parallel
        baseline_answer, saga_answer = await asyncio.gather(
            self._answer_baseline(conv, qa),
            self._answer_saga(conv, qa, session_id),
        )

        # Compute F1
        baseline_f1 = token_f1(baseline_answer, qa.answer)
        saga_f1 = token_f1(saga_answer, qa.answer)

        result = {
            "sample_id": conv.sample_id,
            "qa_index": i,
            "category": qa.category,
            "category_name": qa.category_name,
            "question": qa.question,
            "ground_truth": qa.answer,
            "evidence_dia_ids": qa.evidence,
            "baseline_answer": baseline_answer,
            "saga_answer": saga_answer,
            "baseline_f1": baseline_f1["f1"],
            "saga_f1": saga_f1["f1"],
        }

        # Optional LLM-as-Judge (parallel)
        if use_judge:
            baseline_judge, saga_judge = await asyncio.gather(
                self._judge(qa.question, qa.answer, baseline_answer),
                self._judge(qa.question, qa.answer, saga_answer),
            )
            result["baseline_judge_score"] = baseline_judge.get("score", 0)
            result["baseline_judge_reason"] = baseline_judge.get("reason", "")
            result["saga_judge_score"] = saga_judge.get("score", 0)
            result["saga_judge_reason"] = saga_judge.get("reason", "")

        return result

    async def evaluate_conversation(
        self,
        conv: LocomoConversation,
        session_id: str,
        use_judge: bool = True,
        concurrency: int = 8,
    ) -> list[dict]:
        """Evaluate all QA pairs for a conversation with concurrency."""
        sem = asyncio.Semaphore(concurrency)

        async def _limited(coro):
            async with sem:
                return await coro

        tasks = [
            _limited(self._evaluate_single_qa(conv, session_id, i, qa, use_judge))
            for i, qa in enumerate(conv.qa_pairs)
        ]
        results = await asyncio.gather(*tasks)
        # Sort by qa_index to maintain order
        return sorted(results, key=lambda r: r["qa_index"])

    async def evaluate_all(
        self,
        conversations: list[LocomoConversation],
        session_mapping: dict[str, str],
        use_judge: bool = True,
    ) -> list[dict]:
        """Evaluate all conversations. Returns flat list of all QA results."""
        all_results = []
        for conv in conversations:
            session_id = session_mapping.get(conv.sample_id)
            if not session_id:
                logger.warning(f"[eval] No session for {conv.sample_id}, skipping")
                continue
            results = await self.evaluate_conversation(conv, session_id, use_judge)
            all_results.extend(results)
        return all_results
