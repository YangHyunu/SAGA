"""Load FlawedFictions dataset from HuggingFace (kahuja/flawed-fictions)."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FlawedFictionInstance:
    """A single story with ground-truth contradiction label."""

    story: str  # plain text (HTML tags removed)
    has_error: bool  # True if story contains a continuity error
    error_explanation: str  # human explanation (empty if no error)
    error_lines: str  # lines introducing the error
    contradicted_lines: str  # earlier lines contradicted
    example_id: str
    paragraphs: list[str] = field(default_factory=list)  # split by <p> tags

    @property
    def num_paragraphs(self) -> int:
        return len(self.paragraphs)


def _html_to_paragraphs(raw_html: str) -> tuple[str, list[str]]:
    """Convert HTML story to plain text + paragraph list.

    Stories use <p>...</p> tags. We split on them and strip HTML.
    """
    # Extract paragraph contents
    parts = re.findall(r"<p>(.*?)</p>", raw_html, re.DOTALL)
    if not parts:
        # Fallback: split on double newlines
        text = re.sub(r"<[^>]+>", "", raw_html)
        text = html.unescape(text).strip()
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        return text, parts

    paragraphs = []
    for p in parts:
        clean = re.sub(r"<[^>]+>", "", p)
        clean = html.unescape(clean).strip()
        if clean:
            paragraphs.append(clean)

    plain_text = "\n\n".join(paragraphs)
    return plain_text, paragraphs


def load_dataset(
    split: str = "flawed_fictions",
    max_examples: int | None = None,
) -> list[FlawedFictionInstance]:
    """Load FlawedFictions from HuggingFace.

    Args:
        split: Dataset split (flawed_fictions or flawed_fictions_long).
        max_examples: Limit examples for quick testing.

    Returns:
        List of FlawedFictionInstance objects.
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError:
        raise ImportError("datasets package required: pip install datasets")

    logger.info(f"Loading FlawedFictions from HuggingFace (split={split})")
    data = hf_load("kahuja/flawed-fictions", split=split)

    n = min(len(data), max_examples) if max_examples else len(data)
    instances: list[FlawedFictionInstance] = []

    for idx in range(n):
        row = data[idx]
        plain_text, paragraphs = _html_to_paragraphs(row["story"])

        inst = FlawedFictionInstance(
            story=plain_text,
            has_error=row["cont_error"] == 1.0,
            error_explanation=row.get("cont_error_expl", "") or "",
            error_lines=row.get("cont_error_lines", "") or "",
            contradicted_lines=row.get("contradicted_lines", "") or "",
            example_id=row["example_id"],
            paragraphs=paragraphs,
        )
        instances.append(inst)

    pos = sum(1 for i in instances if i.has_error)
    neg = len(instances) - pos
    logger.info(f"Loaded {len(instances)} instances ({pos} with errors, {neg} clean)")
    return instances
