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

logger = logging.getLogger(__name__)


class SystemStabilizer:
    """Stabilize system messages for prompt caching efficiency."""

    def __init__(self, sqlite_db, config):
        self.db = sqlite_db
        self.config = config

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

        # Check if this is a completely different character (major change)
        threshold = self.config.prompt_caching.canonical_similarity_threshold
        if self._should_update_canonical(canonical, current_system, threshold):
            await self._save_canonical(session_id, current_system, current_hash)
            logger.info(
                f"[Stabilizer] Canonical replaced (similarity < {threshold}) "
                f"for session {session_id}"
            )
            return messages, ""

        # Extract delta: paragraphs in current but not in canonical
        stable_text, delta_text = self._extract_delta(canonical, current_system)

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

    def _extract_delta(self, canonical: str, current: str) -> tuple[str, str]:
        """Extract paragraph-level delta between canonical and current system message.

        Handles inject-modified paragraphs: if a removed paragraph and an added
        paragraph are related (substring or >50% word overlap), the added one is
        treated as a modification rather than new content, avoiding duplication.

        Returns:
            (stable_text, delta_text) where delta_text contains paragraphs
            present in current but absent from canonical.
        """
        canonical_paras = self._split_paragraphs(canonical)
        current_paras = self._split_paragraphs(current)

        canonical_set = set(canonical_paras)
        current_set = set(current_paras)

        # Paragraphs added in current (not in canonical)
        added = current_set - canonical_set
        # Paragraphs removed from current (in canonical but not current)
        removed = canonical_set - current_set

        if not added:
            return canonical, ""

        if not removed:
            # No removals — all added are genuinely new
            delta_parts = [p for p in current_paras if p in added]
            return canonical, "\n\n".join(delta_parts)

        # Detect modified paragraphs (inject_at/inject_lore/inject_replace)
        matched_added: set[str] = set()
        for r_para in removed:
            for a_para in added:
                if a_para in matched_added:
                    continue
                # Substring check: append or prepend
                if r_para in a_para or a_para in r_para:
                    matched_added.add(a_para)
                    logger.info(
                        f"[Stabilizer] inject detected: substring match "
                        f"({len(r_para)}\u2192{len(a_para)} chars)"
                    )
                    break
                # Word overlap check: replace
                r_words = set(r_para.lower().split())
                a_words = set(a_para.lower().split())
                if r_words and a_words:
                    overlap = len(r_words & a_words) / max(len(r_words), len(a_words))
                    if overlap > 0.5:
                        matched_added.add(a_para)
                        logger.info(
                            f"[Stabilizer] inject detected: word overlap {overlap:.0%} "
                            f"({len(r_para)}\u2192{len(a_para)} chars)"
                        )
                        break

        pure_added = added - matched_added
        if matched_added:
            logger.info(
                f"[Stabilizer] {len(matched_added)} modified para(s) excluded from delta, "
                f"{len(pure_added)} new para(s) in delta"
            )

        delta_parts = [p for p in current_paras if p in pure_added]
        delta_text = "\n\n".join(delta_parts)

        return canonical, delta_text

    def _should_update_canonical(
        self, canonical: str, current: str, threshold: float
    ) -> bool:
        """Determine if current system is a completely different character.

        Uses Jaccard similarity on paragraph sets. If similarity < threshold,
        the character has fundamentally changed and canonical should be replaced.
        """
        canonical_paras = set(self._split_paragraphs(canonical))
        current_paras = set(self._split_paragraphs(current))

        if not canonical_paras and not current_paras:
            return False

        intersection = canonical_paras & current_paras
        union = canonical_paras | current_paras

        if not union:
            return False

        similarity = len(intersection) / len(union)
        logger.debug(
            f"[Stabilizer] Jaccard similarity: {similarity:.2f} "
            f"(threshold={threshold}, canon={len(canonical_paras)} cur={len(current_paras)})"
        )
        return similarity < threshold

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
