"""Evaluation metrics for LOCOMO benchmark."""

from __future__ import annotations

import re
import string
from collections import Counter


# ──────────────────────────────────────────────
# SQuAD-style F1 (from LOCOMO paper methodology)
# ──────────────────────────────────────────────

def normalize_answer(s: str) -> str:
    """Lower text, remove punctuation, articles, and extra whitespace."""
    s = str(s).lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> dict:
    """Compute token-level precision, recall, F1 between prediction and ground_truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not gold_tokens and not pred_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not gold_tokens or not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match after normalization."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ──────────────────────────────────────────────
# LLM-as-Judge
# ──────────────────────────────────────────────

JUDGE_PROMPT = """\
You are an expert evaluator for a conversational memory benchmark.

Given:
- **Question**: {question}
- **Ground Truth Answer**: {ground_truth}
- **Model's Answer**: {prediction}

Rate the model's answer on a scale of 1-5:
1 = Completely wrong or irrelevant
2 = Partially relevant but mostly wrong
3 = Partially correct, missing key details
4 = Mostly correct with minor omissions
5 = Fully correct and complete

For "Unanswerable" ground truths: 5 if model correctly abstains, 1 if model fabricates.

Respond with ONLY a JSON object:
{{"score": <1-5>, "reason": "<brief explanation>"}}"""


def build_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    """Build the LLM-as-Judge prompt."""
    return JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction,
    )


def parse_judge_response(response: str) -> dict:
    """Parse judge LLM response into score + reason."""
    import json

    # Try direct JSON parse
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"\{[^}]+\}", response)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: extract score number from various formats
    score_match = re.search(r'(?:score|rating)["\s]*[:=]\s*(\d)', response, re.IGNORECASE)
    if score_match:
        return {"score": int(score_match.group(1)), "reason": response.strip()}

    # Last resort: find any standalone digit 1-5
    digit_match = re.search(r'\b([1-5])\b', response)
    if digit_match:
        return {"score": int(digit_match.group(1)), "reason": response.strip()}

    return {"score": 0, "reason": f"Failed to parse: {response[:200]}"}
