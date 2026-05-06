"""System Message Stabilizer — keeps system prompt stable across turns for cache efficiency.

RisuAI dynamically inserts/removes Lorebook entries into the system message each turn.
Since Anthropic's prompt caching is prefix-based, any change to the system message
invalidates all cache breakpoints (BP1/BP2/BP3).

This module:
1. Stores the first system message as "canonical" per session (in world_state KV)
2. On subsequent turns, extracts the delta (new Lorebook entries) from the system message
3. Keeps the canonical system message intact (for cache stability)
4. Moves the delta to the dynamic context area (outside cache breakpoints)
"""

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from langsmith import traceable

logger = logging.getLogger(__name__)

InjectKind = Literal["append", "shrink", "replace"]


@dataclass
class InjectMatch:
    """Classification result for a (removed, added) paragraph pair."""
    r_para: str
    a_para: str
    kind: InjectKind
    appended: str = ""  # only for kind="append"
    overlap: float = 0.0  # only for kind="replace" (for logging)


def _classify_inject(r_para: str, a_para: str) -> InjectMatch | None:
    """Classify a (removed, added) paragraph pair as one of three inject patterns.

    - append: r_para is a strict prefix of a_para → injected suffix becomes delta
    - shrink: a_para is a substring of r_para → paragraph was shortened (excluded)
    - replace: >50% word overlap with no substring relation → canonical needs update
    - None: paragraphs are unrelated; caller treats a_para as a pure addition
    """
    if a_para.startswith(r_para):
        return InjectMatch(r_para, a_para, "append", appended=a_para[len(r_para):].strip())
    if a_para in r_para:
        return InjectMatch(r_para, a_para, "shrink")
    r_words = set(r_para.lower().split())
    a_words = set(a_para.lower().split())
    if r_words and a_words:
        overlap = len(r_words & a_words) / max(len(r_words), len(a_words))
        if overlap > 0.5:
            return InjectMatch(r_para, a_para, "replace", overlap=overlap)
    return None


class SystemStabilizer:
    """Stabilize system messages for prompt caching efficiency."""

    def __init__(self, sqlite_db, config):
        self.db = sqlite_db
        self.config = config

    @traceable(name="pipeline.stabilizer")
    async def stabilize(
        self, session_id: str, messages: list[dict]
    ) -> tuple[list[dict], str]:
        """Stabilize system message and extract lorebook delta.

        Args:
            session_id: Current session ID.
            messages: List of message dicts (role/content).

        Returns:
            (stabilized_messages, lorebook_delta) where:
            - stabilized_messages: messages with canonical system prompt restored
            - lorebook_delta: text of new/changed lorebook entries to inject as dynamic context
        """
        if not self.config.prompt_caching.stabilize_system:
            return messages, ""

        # Find system message
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        if system_idx is None:
            return messages, ""

        current_system = messages[system_idx]["content"]

        # Get stored canonical
        canonical = await self.db.get_world_state_value(
            session_id, "canonical_system_prompt"
        )
        canonical_hash = await self.db.get_world_state_value(
            session_id, "canonical_system_hash"
        )

        current_hash = hashlib.md5(current_system.encode()).hexdigest()

        # First turn: store as canonical, no delta
        if canonical is None:
            await self._save_canonical(session_id, current_system, current_hash)
            logger.info(
                f"[Stabilizer] Canonical saved for session {session_id} "
                f"({len(current_system)} chars, hash={current_hash[:8]})"
            )
            return messages, ""

        # Same system message: no delta needed
        if current_hash == canonical_hash:
            logger.debug("[Stabilizer] System message unchanged, cache stable")
            return messages, ""

        # Extract delta: paragraphs in current but not in canonical
        stable_text, delta_text, canonical_needs_update = self._extract_delta(
            canonical, current_system
        )

        # inject_replace: update canonical so next turns see the corrected content
        if canonical_needs_update:
            await self._save_canonical(session_id, current_system, current_hash)
            logger.info(
                f"[Stabilizer] inject_replace: canonical updated for session {session_id} "
                f"({len(current_system)} chars, hash={current_hash[:8]})"
            )
            # Use current system as-is (already has replaced content);
            # if there are also pure additions, they appear in delta as dynamic context
            if delta_text:
                stabilized = list(messages)
                stabilized[system_idx] = dict(messages[system_idx])
                stabilized[system_idx]["content"] = current_system
                logger.info(
                    f"[Stabilizer] inject_replace + pure adds: "
                    f"{len(delta_text)} chars delta alongside updated canonical"
                )
                return stabilized, delta_text
            return messages, ""

        if delta_text:
            # Replace system message with canonical (stable) version
            stabilized = list(messages)
            stabilized[system_idx] = dict(messages[system_idx])
            stabilized[system_idx]["content"] = canonical
            logger.info(
                f"[Stabilizer] Delta extracted: {len(delta_text)} chars "
                f"({delta_text.count(chr(10)) + 1} lines) moved to dynamic context"
            )
            return stabilized, delta_text

        # No meaningful delta found — use current as-is
        logger.debug("[Stabilizer] No delta detected despite hash mismatch (whitespace?)")
        return messages, ""

    def _extract_delta(self, canonical: str, current: str) -> tuple[str, str, bool]:
        """Extract paragraph-level delta between canonical and current system message.

        Delegates per-pair classification to :func:`_classify_inject` and pair
        matching to :meth:`_match_inject_pairs`. Builds final delta text from
        pure-add paragraphs plus appended content extracted from inject_append.

        Returns:
            (stable_text, delta_text, canonical_needs_update) where:
            - stable_text: canonical (always unchanged here; caller decides what to send)
            - delta_text: content for dynamic context injection
            - canonical_needs_update: True when inject_replace detected; caller must persist
        """
        canonical_paras = self._split_paragraphs(canonical)
        current_paras = self._split_paragraphs(current)

        canonical_counts = Counter(canonical_paras)
        current_counts = Counter(current_paras)

        added = set(p for p in current_paras if current_counts[p] > canonical_counts[p])
        removed = set(p for p in canonical_paras if canonical_counts[p] > current_counts[p])

        if not added:
            return canonical, "", False

        if not removed:
            delta_parts = [p for p in current_paras if p in added]
            return canonical, "\n\n".join(delta_parts), False

        matches = self._match_inject_pairs(removed, added)
        for m in matches:
            if m.kind == "append":
                logger.info(f"[Stabilizer] inject_append: +{len(m.appended)} chars → delta")
            elif m.kind == "shrink":
                logger.info("[Stabilizer] inject_shrink: paragraph shortened, excluded")
            elif m.kind == "replace":
                logger.info(
                    f"[Stabilizer] inject_replace: {m.overlap:.0%} overlap "
                    f"→ canonical update required"
                )

        matched_added = {m.a_para for m in matches}
        canonical_needs_update = any(m.kind == "replace" for m in matches)
        inject_delta_parts = [m.appended for m in matches if m.kind == "append" and m.appended]

        pure_added = added - matched_added
        if matched_added:
            logger.info(
                f"[Stabilizer] {len(matched_added)} modified para(s): "
                f"{len(inject_delta_parts)} inject delta(s), "
                f"{len(pure_added)} pure addition(s)"
            )

        delta_parts = [p for p in current_paras if p in pure_added] + inject_delta_parts
        return canonical, "\n\n".join(delta_parts), canonical_needs_update

    def _match_inject_pairs(self, removed: set[str], added: set[str]) -> list[InjectMatch]:
        """For each removed paragraph, find at most one added paragraph that
        classifies as a non-trivial inject pattern. Each added paragraph is
        consumed at most once.
        """
        matches: list[InjectMatch] = []
        used_added: set[str] = set()
        for r in removed:
            for a in added:
                if a in used_added:
                    continue
                m = _classify_inject(r, a)
                if m is not None:
                    matches.append(m)
                    used_added.add(a)
                    break
        return matches

    @staticmethod
    def _split_paragraphs(text: str) -> list[str]:
        """Split text into normalized paragraphs (double-newline separated).

        Each paragraph is stripped and empty paragraphs are removed.
        """
        raw_paras = re.split(r'\n\s*\n', text)
        return [p.strip() for p in raw_paras if p.strip()]

    async def _save_canonical(self, session_id: str, text: str, text_hash: str):
        """Persist canonical system prompt to world_state KV."""
        await self.db.upsert_world_state(
            session_id, "canonical_system_prompt", text
        )
        await self.db.upsert_world_state(
            session_id, "canonical_system_hash", text_hash
        )
