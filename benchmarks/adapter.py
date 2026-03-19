"""Adapt LOCOMO dataset format to SAGA message format."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Turn:
    """Single dialogue turn."""
    dia_id: str  # e.g. "D1:3" (session:turn)
    speaker: str
    text: str
    session_idx: int  # which session this turn belongs to (1-based)


@dataclass
class QAPair:
    """Single QA evaluation pair."""
    question: str
    answer: str
    evidence: list[str]  # dia_ids like ["D1:3", "D5:2"]
    category: int  # 1=single-hop, 2=multi-hop, 3=temporal, 4=commonsense, 5=adversarial

    @property
    def category_name(self) -> str:
        names = {1: "single-hop", 2: "multi-hop", 3: "temporal", 4: "commonsense", 5: "adversarial"}
        return names.get(self.category, "unknown")


@dataclass
class LocomoConversation:
    """Parsed LOCOMO conversation."""
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[list[Turn]] = field(default_factory=list)
    session_dates: list[str] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)
    all_turns: list[Turn] = field(default_factory=list)

    @property
    def total_turns(self) -> int:
        return len(self.all_turns)

    @property
    def total_sessions(self) -> int:
        return len(self.sessions)


def parse_conversation(raw: dict) -> LocomoConversation:
    """Parse a raw LOCOMO JSON object into structured LocomoConversation.

    LOCOMO format:
    {
        "sample_id": "locomo_0",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1": [...], "session_1_date_time": "...",
            ...
        },
        "qa": [{"question": ..., "answer": ..., "evidence": ["D1:3"], "category": 1}]
    }
    """
    conv_data = raw.get("conversation", raw)  # fallback for flat format

    result = LocomoConversation(
        sample_id=raw.get("sample_id", "unknown"),
        speaker_a=conv_data.get("speaker_a", "A"),
        speaker_b=conv_data.get("speaker_b", "B"),
    )

    # Extract sessions (session_1, session_2, ...) — sorted numerically
    session_keys = sorted(
        [k for k in conv_data.keys() if re.match(r"^session_\d+$", k)],
        key=lambda k: int(k.split("_")[1]),
    )

    for skey in session_keys:
        session_idx = int(skey.split("_")[1])
        turns = []
        for t in conv_data[skey]:
            turn = Turn(
                dia_id=t["dia_id"],
                speaker=t["speaker"],
                text=t["text"],
                session_idx=session_idx,
            )
            turns.append(turn)
            result.all_turns.append(turn)
        result.sessions.append(turns)

        # Session date
        date_key = f"{skey}_date_time"
        result.session_dates.append(conv_data.get(date_key, ""))

    # Extract QA pairs
    for qa in raw.get("qa", []):
        # Adversarial (category 5) uses "adversarial_answer" instead of "answer"
        answer = qa.get("answer") or qa.get("adversarial_answer", "Unanswerable")
        result.qa_pairs.append(QAPair(
            question=qa["question"],
            answer=answer,
            evidence=qa.get("evidence", []),
            category=qa.get("category", 0),
        ))

    return result


def turns_to_messages(turns: list[Turn], speaker_a: str, speaker_b: str) -> list[dict]:
    """Convert turns into OpenAI-compatible message list.

    Convention: speaker_a = user, speaker_b = assistant.
    """
    messages = []
    for turn in turns:
        role = "user" if turn.speaker == speaker_a else "assistant"
        messages.append({"role": role, "content": turn.text})
    return messages


def parse_all(raw_data: list[dict]) -> list[LocomoConversation]:
    """Parse all LOCOMO conversations."""
    return [parse_conversation(r) for r in raw_data]
