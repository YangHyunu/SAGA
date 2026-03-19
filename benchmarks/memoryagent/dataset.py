"""Load MemoryAgentBench dataset from HuggingFace."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Sequence:
    """A single benchmark sequence: one long context + multiple QA pairs."""

    context: str
    questions: list[str]
    answers: list[list[str]]  # multiple valid answers per question
    split: str  # Accurate_Retrieval, Test_Time_Learning, etc.
    source: str  # sub-dataset name
    seq_index: int  # row index within the split
    question_types: list[str] = field(default_factory=list)
    chunk_size: int = 4096  # tokens per chunk for memorization

    @property
    def num_questions(self) -> int:
        return len(self.questions)

    def get_chunks(self, chunk_size: int | None = None) -> list[str]:
        """Split context into fixed-size character chunks for memorization.

        Uses character-based splitting (approx 4 chars/token) since exact
        tokenization depends on the model.
        """
        cs = (chunk_size or self.chunk_size) * 4  # approx chars
        text = self.context
        chunks = []
        for i in range(0, len(text), cs):
            chunk = text[i : i + cs]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks


def load_dataset(
    splits: list[str] | None = None,
    max_sequences_per_split: int | None = None,
) -> list[Sequence]:
    """Load MemoryAgentBench from HuggingFace.

    Args:
        splits: Which splits to load. Default: all four.
        max_sequences_per_split: Limit sequences per split for faster testing.

    Returns:
        List of Sequence objects ready for evaluation.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError(
            "datasets package required: pip install datasets"
        )

    all_splits = splits or [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]

    logger.info(f"Loading MemoryAgentBench from HuggingFace (splits={all_splits})")

    sequences: list[Sequence] = []

    for split_name in all_splits:
        try:
            split_data = hf_load("ai-hyz/MemoryAgentBench", split=split_name)
        except Exception as e:
            logger.warning(f"Split '{split_name}' load failed: {e}, skipping")
            continue
        n = min(len(split_data), max_sequences_per_split) if max_sequences_per_split else len(split_data)

        for idx in range(n):
            row = split_data[idx]
            meta = row.get("metadata", {}) or {}

            seq = Sequence(
                context=row["context"],
                questions=row["questions"],
                answers=row["answers"],
                split=split_name,
                source=meta.get("source", "unknown"),
                seq_index=idx,
                question_types=meta.get("question_types", []),
            )
            sequences.append(seq)

        logger.info(f"  {split_name}: loaded {n} sequences")

    logger.info(f"Total: {len(sequences)} sequences, {sum(s.num_questions for s in sequences)} QA pairs")
    return sequences
