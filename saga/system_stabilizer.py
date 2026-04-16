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

from langsmith import traceable

logger = logging.getLogger(__name__)


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

        Handles inject-modified paragraphs:
        - inject_append (r_para is prefix of a_para): extracts the appended portion as delta.
          The canonical paragraph is stable; only the injected addition goes to dynamic context.
        - inject_replace (>50% word overlap, no substring): signals caller to update canonical.
          The replaced paragraph is excluded from delta; caller persists current as new canonical.
        - inject_shrink (a_para is prefix of r_para): rare, excluded from delta.
        - pure add: genuinely new paragraphs included in delta.

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

        # Paragraphs with more occurrences in current than canonical
        added = set(p for p in current_paras if current_counts[p] > canonical_counts[p])
        # Paragraphs with more occurrences in canonical than current
        removed = set(p for p in canonical_paras if canonical_counts[p] > current_counts[p])

        if not added:
            return canonical, "", False

        if not removed:
            # No removals — all added are genuinely new
            delta_parts = [p for p in current_paras if p in added]
            return canonical, "\n\n".join(delta_parts), False

        # Detect modified paragraphs (inject_at/inject_lore/inject_replace)
        matched_added: set[str] = set()
        inject_delta_parts: list[str] = []  # appended content extracted from inject_append
        canonical_needs_update = False

        for r_para in removed:
            for a_para in added:
                if a_para in matched_added:
                    continue
                # inject_append: r_para is a strict prefix of a_para
                if a_para.startswith(r_para):
                    appended = a_para[len(r_para):].strip()
                    if appended:
                        inject_delta_parts.append(appended)
                    matched_added.add(a_para)
                    logger.info(
                        f"[Stabilizer] inject_append: +{len(appended)} chars → delta"
                    )
                    break
                # inject_shrink: a_para is substring of r_para (content removed from para)
                if a_para in r_para:
                    matched_added.add(a_para)
                    logger.info("[Stabilizer] inject_shrink: paragraph shortened, excluded")
                    break
                # inject_replace: >50% word overlap, no substring relation
                r_words = set(r_para.lower().split())
                a_words = set(a_para.lower().split())
                if r_words and a_words:
                    overlap = len(r_words & a_words) / max(len(r_words), len(a_words))
                    if overlap > 0.5:
                        matched_added.add(a_para)
                        canonical_needs_update = True
                        logger.info(
                            f"[Stabilizer] inject_replace: {overlap:.0%} overlap "
                            f"→ canonical update required"
                        )
                        break

        pure_added = added - matched_added
        if matched_added:
            logger.info(
                f"[Stabilizer] {len(matched_added)} modified para(s): "
                f"{len(inject_delta_parts)} inject delta(s), "
                f"{len(pure_added)} pure addition(s)"
            )

        delta_parts = [p for p in current_paras if p in pure_added]
        all_delta = delta_parts + inject_delta_parts
        delta_text = "\n\n".join(all_delta)

        return canonical, delta_text, canonical_needs_update

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
