"""Proactive message compression — immutable summary chunks for cache-stable prefixes.

When the message token count exceeds a threshold, old turns are replaced with
compact summary chunks.  Each chunk is *immutable* once created: its text never
changes, so the Anthropic prefix cache can keep hitting.

Pipeline position:  SystemStabilizer → **MessageCompressor** → WindowRecovery
"""

import json
import logging
from datetime import datetime

from saga.storage.sqlite_db import SQLiteDB
from saga.utils.tokens import count_tokens, count_messages_tokens

logger = logging.getLogger(__name__)

_KEY_COMPRESSED_CHUNKS = "compressed_chunks"
_KEY_COMPRESSED_THROUGH = "compressed_through_turn"

# Fixed user-message text for chunk pairs (must be deterministic for cache stability)
_CHUNK_USER_PREFIX = "[SAGA: 이전 대화 요약"


class MessageCompressor:
    """Replace old messages with immutable summary-chunk pairs."""

    def __init__(self, sqlite_db: SQLiteDB, config):
        self.sqlite_db = sqlite_db
        self.config = config

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def compress(
        self, session_id: str, messages: list[dict]
    ) -> tuple[list[dict], int]:
        """Compress messages if token count exceeds threshold.

        Returns:
            (compressed_messages, index_shift)
            index_shift = number of original messages removed from the front
                          (used by caller to fix multimodal part indices).
        """
        total_tokens = count_messages_tokens(messages)
        threshold = int(
            self.config.token_budget.total_context_max
            * self.config.prompt_caching.compress_threshold_ratio
        )

        # Load existing chunks (always — even below threshold we must restore them)
        chunks = await self._load_chunks(session_id)

        # Separate system and non-system messages
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        if not non_system:
            return messages, 0

        # If we already have chunks, figure out which non-system messages
        # are already covered by chunks (they came from RisuAI which doesn't
        # know about our compression — we need to match by turn counting).
        compressed_through = await self._get_compressed_through(session_id)

        if not chunks and total_tokens <= threshold:
            # No chunks, below threshold — nothing to do
            return messages, 0

        if chunks:
            # Rebuild with existing chunks first, then check if MORE compression needed
            rebuilt = self._rebuild_with_chunks(system_msgs, non_system, chunks, compressed_through)
            rebuilt_tokens = count_messages_tokens(rebuilt)
            if rebuilt_tokens <= threshold:
                return rebuilt, 0
            # Rebuilt still exceeds threshold — need more compression
            total_tokens = rebuilt_tokens

        # --- Compression needed ---
        # Calculate how many additional turns to compress
        target_tokens = int(self.config.token_budget.total_context_max * 0.50)  # aim for 50% of max context to give ~15-20 turns of headroom
        turns_to_compress = self._calculate_turns_to_compress(
            non_system, total_tokens, target_tokens, compressed_through
        )

        if turns_to_compress < self.config.prompt_caching.min_compress_turns:
            # Not enough turns to justify compression
            if chunks:
                return self._rebuild_with_chunks(system_msgs, non_system, chunks, compressed_through), 0
            return messages, 0

        # Build summary from turn_logs
        from_turn = compressed_through + 1 if compressed_through else 1
        to_turn = from_turn + turns_to_compress - 1
        summary_text = await self._build_summary_from_turn_logs(session_id, from_turn, to_turn)

        if not summary_text:
            logger.warning(f"[Compressor] No turn_log summaries for turns {from_turn}-{to_turn}, skipping")
            if chunks:
                return self._rebuild_with_chunks(system_msgs, non_system, chunks, compressed_through), 0
            return messages, 0

        # Create new immutable chunk
        new_chunk = {
            "id": f"chunk_{len(chunks) + 1:03d}",
            "from_turn": from_turn,
            "to_turn": to_turn,
            "summary_text": summary_text,
            "token_count": count_tokens(summary_text),
            "created_at": datetime.utcnow().isoformat(),
        }
        chunks.append(new_chunk)
        await self._save_chunks(session_id, chunks)
        await self.sqlite_db.upsert_world_state(session_id, _KEY_COMPRESSED_THROUGH, str(to_turn))

        logger.info(
            f"[Compressor] Created chunk {new_chunk['id']}: turns {from_turn}-{to_turn} "
            f"({new_chunk['token_count']} tokens, {turns_to_compress} turns compressed)"
        )

        # Rebuild message array
        index_shift = turns_to_compress * 2  # each turn = user + assistant
        result = self._rebuild_with_chunks(system_msgs, non_system, chunks, to_turn)

        new_tokens = count_messages_tokens(result)
        logger.info(
            f"[Compressor] Tokens: {total_tokens} → {new_tokens} "
            f"(saved {total_tokens - new_tokens}, {len(chunks)} chunks total)"
        )

        return result, index_shift

    # ------------------------------------------------------------------ #
    # Chunk persistence
    # ------------------------------------------------------------------ #

    async def _load_chunks(self, session_id: str) -> list[dict]:
        raw = await self.sqlite_db.get_world_state_value(session_id, _KEY_COMPRESSED_CHUNKS)
        if not raw:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return []

    async def _save_chunks(self, session_id: str, chunks: list[dict]):
        await self.sqlite_db.upsert_world_state(
            session_id, _KEY_COMPRESSED_CHUNKS, json.dumps(chunks, ensure_ascii=False)
        )

    async def _get_compressed_through(self, session_id: str) -> int:
        val = await self.sqlite_db.get_world_state_value(session_id, _KEY_COMPRESSED_THROUGH)
        return int(val) if val else 0

    # ------------------------------------------------------------------ #
    # Message array reconstruction
    # ------------------------------------------------------------------ #

    def _rebuild_with_chunks(
        self,
        system_msgs: list[dict],
        non_system: list[dict],
        chunks: list[dict],
        compressed_through: int,
    ) -> list[dict]:
        """Rebuild message array: [system] + [chunk pairs] + [remaining messages]."""
        result = list(system_msgs)

        # Add chunk pairs (user + assistant for each)
        for i, chunk in enumerate(chunks):
            chunk_num = i + 1
            result.append({
                "role": "user",
                "content": f"{_CHUNK_USER_PREFIX} {chunk_num}: Turn {chunk['from_turn']}-{chunk['to_turn']}]",
            })
            result.append({
                "role": "assistant",
                "content": chunk["summary_text"],
            })

        # Add remaining non-system messages (skip those covered by chunks)
        # Each turn = 1 user + 1 assistant message pair
        # compressed_through tells us the last turn that's been compressed
        # We need to skip the first (compressed_through * 2) non-system messages
        # BUT RisuAI may have already removed some from the front (sliding window)
        # So we calculate based on what's actually in non_system
        msgs_to_skip = self._count_msgs_to_skip(non_system, compressed_through)
        remaining = non_system[msgs_to_skip:]

        result.extend(remaining)
        return result

    def _count_msgs_to_skip(self, non_system: list[dict], compressed_through: int) -> int:
        """Count how many front messages to skip because they're covered by chunks.

        The tricky part: RisuAI may have already removed some messages from the
        front (its own sliding window).  So we can't just skip compressed_through*2.
        We skip the minimum of (compressed_through * 2) and (len(non_system) - 2),
        ensuring at least the last user+assistant pair remains.
        """
        if compressed_through <= 0:
            return 0
        max_skip = compressed_through * 2
        # Never skip more messages than exist (minus at least 2 for last turn)
        safe_skip = min(max_skip, max(0, len(non_system) - 2))
        return safe_skip

    # ------------------------------------------------------------------ #
    # Compression calculation
    # ------------------------------------------------------------------ #

    def _calculate_turns_to_compress(
        self,
        non_system: list[dict],
        current_tokens: int,
        target_tokens: int,
        compressed_through: int,
    ) -> int:
        """Calculate how many turns to compress to get below target_tokens."""
        tokens_to_free = current_tokens - target_tokens
        if tokens_to_free <= 0:
            return 0

        # Count tokens from the front of non_system, turn by turn
        # Skip messages already covered by compression
        start_idx = self._count_msgs_to_skip(non_system, compressed_through)
        freed = 0
        turns = 0
        idx = start_idx

        while idx + 1 < len(non_system) and freed < tokens_to_free:
            # Each turn = user + assistant
            user_msg = non_system[idx]
            asst_msg = non_system[idx + 1] if idx + 1 < len(non_system) else None

            if user_msg.get("role") != "user":
                idx += 1
                continue
            if asst_msg is None or asst_msg.get("role") != "assistant":
                idx += 1
                continue

            user_tokens = count_tokens(user_msg.get("content", "")) + 4
            asst_tokens = count_tokens(asst_msg.get("content", "")) + 4
            freed += user_tokens + asst_tokens
            turns += 1
            idx += 2

        # Don't compress all messages — keep at least 5 turns of real conversation
        min_remaining_turns = 5
        max_compressible = max(0, (len(non_system) - start_idx) // 2 - min_remaining_turns)
        turns = min(turns, max_compressible)

        return turns

    # ------------------------------------------------------------------ #
    # Summary generation (from existing turn_log data)
    # ------------------------------------------------------------------ #

    async def _build_summary_from_turn_logs(
        self, session_id: str, from_turn: int, to_turn: int
    ) -> str:
        """Build summary text from turn_log entries. No LLM call needed."""
        logs = await self.sqlite_db.get_turn_logs(session_id, from_turn, to_turn)
        if not logs:
            return ""

        lines = []
        for log in logs:
            turn = log.get("turn_number", "?")
            state = log.get("state_changes")
            if isinstance(state, str):
                try:
                    state = json.loads(state)
                except (json.JSONDecodeError, TypeError):
                    state = {}

            summary = ""
            if isinstance(state, dict):
                summary = state.get("summary", "")

            if not summary:
                # Fallback: use first 200 chars of assistant output
                output = log.get("assistant_output", "")
                summary = output[:200] + "..." if len(output) > 200 else output

            if summary:
                lines.append(f"Turn {turn}: {summary}")

        return "\n".join(lines) if lines else ""
