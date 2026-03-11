"""Evaluate Baseline vs SAGA-augmented QA on LongMemEval.

Approach (lightweight — no Sub-B ingestion):
  1. For each question, embed all haystack sessions into ChromaDB
  2. Baseline: last N sessions as context → answer
  3. SAGA: ChromaDB search for top-k relevant sessions → answer
  4. Judge both answers (Yes/No accuracy)

Features:
  - Checkpoint resume: results saved per-question to jsonl
  - Retry with exponential backoff on transient errors
  - Parallel baseline/SAGA answering
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import hashlib

import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 3
BASELINE_SESSIONS = 10  # last N sessions for baseline
SAGA_TOP_K = 10  # top-k retrieved sessions for SAGA


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


def _sessions_to_context(sessions: list[list[dict]], dates: list[str] | None = None) -> str:
    """Convert sessions to a context string."""
    lines = []
    for i, session in enumerate(sessions):
        date_str = f" ({dates[i]})" if dates and i < len(dates) else ""
        lines.append(f"--- Session {i + 1}{date_str} ---")
        for turn in session:
            role = turn["role"].capitalize()
            lines.append(f"{role}: {turn['content']}")
        lines.append("")
    return "\n".join(lines)


def _session_to_text(session: list[dict]) -> str:
    """Convert a single session to searchable text."""
    return "\n".join(f"{t['role']}: {t['content']}" for t in session)


class LongMemEvalEvaluator:
    """Run QA evaluation: baseline (truncated) vs SAGA (retrieved)."""

    def __init__(
        self,
        llm_client,
        qa_model: str = "gemini-2.5-flash",
        judge_model: str = "gemini-2.5-flash",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
        checkpoint_dir: str | None = None,
    ):
        self.llm_client = llm_client
        self.qa_model = qa_model
        self.judge_model = judge_model
        self.checkpoint_dir = checkpoint_dir or os.path.join(
            os.path.dirname(__file__), "data"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.jsonl")

        # ChromaDB with OpenAI embeddings
        self._chroma_client = chromadb.Client()
        self._embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model_name=embedding_model,
        )

    def _load_checkpoint(self) -> dict[str, dict]:
        """Load completed results from checkpoint file."""
        completed = {}
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        result = json.loads(line)
                        completed[result["question_id"]] = result
            logger.info(f"[checkpoint] Loaded {len(completed)} completed results")
        return completed

    def _save_checkpoint(self, result: dict):
        """Append a single result to checkpoint file."""
        with open(self.checkpoint_path, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    def _embed_sessions(self, question_id: str, sessions: list[list[dict]]) -> chromadb.Collection:
        """Embed all haystack sessions into a temporary ChromaDB collection."""
        col_name = f"lme_{hashlib.md5(question_id.encode()).hexdigest()[:12]}"

        # Delete if exists (re-run safety)
        try:
            self._chroma_client.delete_collection(col_name)
        except Exception:
            pass

        collection = self._chroma_client.create_collection(
            name=col_name,
            embedding_function=self._embed_fn,
        )

        docs = []
        ids = []
        max_chars = 25000  # ~8000 tokens, text-embedding-3-small limit is 8192
        for i, session in enumerate(sessions):
            text = _session_to_text(session)[:max_chars]
            if text.strip():
                docs.append(text)
                ids.append(f"sess_{i}")

        if docs:
            # Batch embed (ChromaDB handles batching internally)
            collection.add(documents=docs, ids=ids)

        return collection

    async def _answer_baseline(self, instance: dict) -> str:
        """Answer using last N sessions (truncated context)."""
        sessions = instance["haystack_sessions"]
        dates = instance.get("haystack_dates", [])

        # Take last N sessions
        truncated_sessions = sessions[-BASELINE_SESSIONS:]
        truncated_dates = dates[-BASELINE_SESSIONS:] if dates else None

        context = _sessions_to_context(truncated_sessions, truncated_dates)

        prompt = (
            f"Below is a conversation history between a user and an assistant.\n"
            f"Answer the question based ONLY on the conversation history. "
            f"If the answer is not in the conversation, say 'I don't have enough information'.\n\n"
            f"{context}\n"
            f"Question: {instance['question']}\n\n"
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

    async def _answer_saga(self, instance: dict, collection: chromadb.Collection) -> str:
        """Answer using ChromaDB-retrieved relevant sessions."""
        sessions = instance["haystack_sessions"]
        dates = instance.get("haystack_dates", [])

        # Search for relevant sessions
        results = collection.query(
            query_texts=[instance["question"]],
            n_results=min(SAGA_TOP_K, len(sessions)),
        )

        # Reconstruct retrieved sessions in original order
        retrieved_indices = sorted(
            int(id_.split("_")[1]) for id_ in results["ids"][0]
        )
        retrieved_sessions = [sessions[i] for i in retrieved_indices]
        retrieved_dates = [dates[i] for i in retrieved_indices if i < len(dates)] if dates else None

        context = _sessions_to_context(retrieved_sessions, retrieved_dates)

        prompt = (
            f"Below is a conversation history between a user and an assistant.\n"
            f"These are the most relevant conversation sessions retrieved from memory.\n"
            f"Answer the question based ONLY on the conversation history. "
            f"If the answer is not in the conversation, say 'I don't have enough information'.\n\n"
            f"{context}\n"
            f"Question: {instance['question']}\n\n"
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

    async def _judge(self, question_type: str, question: str, answer: str, prediction: str) -> bool:
        """Judge if prediction is correct (Yes/No). Returns True if correct."""
        abstention = question_type.endswith("_abs") or "abstention" in question_type

        if not abstention:
            prompt = (
                f"I will give you a question, a correct answer, and a response from a model. "
                f"Please answer yes if the response contains the correct answer. Otherwise, answer no. "
                f"If the response is equivalent to the correct answer or contains all the intermediate "
                f"steps to get the correct answer, you should also answer yes. "
                f"If the response only contains a subset of the information required by the answer, answer no.\n\n"
                f"Question: {question}\n\n"
                f"Correct Answer: {answer}\n\n"
                f"Model Response: {prediction}\n\n"
                f"Is the model response correct? Answer yes or no only."
            )
        else:
            prompt = (
                f"I will give you an unanswerable question, an explanation, and a response from a model. "
                f"Please answer yes if the model correctly identifies the question as unanswerable. "
                f"The model could say that the information is incomplete, or some other information is "
                f"given but the asked information is not.\n\n"
                f"Question: {question}\n\n"
                f"Explanation: {answer}\n\n"
                f"Model Response: {prediction}\n\n"
                f"Does the model correctly identify the question as unanswerable? Answer yes or no only."
            )

        response = await _call_with_retry(
            self.llm_client,
            model=self.judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )

        return response.strip().lower().startswith("yes")

    async def _evaluate_single(self, instance: dict, use_judge: bool = True) -> dict:
        """Evaluate a single QA instance."""
        qid = instance["question_id"]
        qtype = instance["question_type"]

        # Embed sessions into ChromaDB
        collection = self._embed_sessions(qid, instance["haystack_sessions"])

        try:
            # Answer in parallel
            baseline_answer, saga_answer = await asyncio.gather(
                self._answer_baseline(instance),
                self._answer_saga(instance, collection),
            )

            result = {
                "question_id": qid,
                "question_type": qtype,
                "question": instance["question"],
                "ground_truth": instance["answer"],
                "baseline_answer": baseline_answer,
                "saga_answer": saga_answer,
                "num_sessions": len(instance["haystack_sessions"]),
            }

            # Judge in parallel
            if use_judge:
                baseline_correct, saga_correct = await asyncio.gather(
                    self._judge(qtype, instance["question"], str(instance["answer"]), baseline_answer),
                    self._judge(qtype, instance["question"], str(instance["answer"]), saga_answer),
                )
                result["baseline_correct"] = baseline_correct
                result["saga_correct"] = saga_correct

            return result

        finally:
            # Cleanup ChromaDB collection
            try:
                col_name = f"lme_{hashlib.md5(qid.encode()).hexdigest()[:12]}"
                self._chroma_client.delete_collection(col_name)
            except Exception:
                pass

    async def evaluate_all(
        self,
        data: list[dict],
        max_instances: int | None = None,
        use_judge: bool = True,
        concurrency: int = 4,
    ) -> list[dict]:
        """Evaluate all instances with checkpoint resume and concurrency."""
        instances = data[:max_instances] if max_instances else data
        completed = self._load_checkpoint()
        results = list(completed.values())

        remaining = [inst for inst in instances if inst["question_id"] not in completed]
        logger.info(
            f"[eval] {len(completed)} already done, {len(remaining)} remaining "
            f"(total {len(instances)})"
        )

        sem = asyncio.Semaphore(concurrency)
        errors = 0

        async def _process(inst: dict, idx: int):
            nonlocal errors
            async with sem:
                qid = inst["question_id"]
                try:
                    logger.info(
                        f"[eval] {idx + 1 + len(completed)}/{len(instances)} "
                        f"QA {qid} ({inst['question_type']})"
                    )
                    result = await self._evaluate_single(inst, use_judge)
                    self._save_checkpoint(result)
                    return result
                except Exception as e:
                    logger.error(f"[eval] {qid} failed: {e}")
                    errors += 1
                    return None

        tasks = [_process(inst, i) for i, inst in enumerate(remaining)]
        new_results = await asyncio.gather(*tasks)

        for r in new_results:
            if r is not None:
                results.append(r)

        logger.info(
            f"[eval] Done: {len(results)} completed, {errors} errors"
        )
        return results
