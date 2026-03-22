"""Sliding Window Cache Recovery — 슬라이딩 윈도우 이동 시 캐시 BP 복구.

RisuAI가 max_token 초과로 앞쪽 메시지를 잘라내면 캐시 breakpoint가 깨진다.
이 모듈은:
1. 윈도우 이동을 감지 (첫 non-system 메시지 hash 비교)
2. 잘려나간 턴의 에피소드 요약을 조회 (ChromaDB/turn_log)
3. 요약 블록을 생성하여 messages에 주입 → 새 BP2 구성
"""

import hashlib
import json
import logging

from langsmith import traceable

from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# world_state KV keys
_KEY_FIRST_MSG_HASH = "window_first_msg_hash"
_KEY_FIRST_MSG_TURN = "window_first_msg_turn_estimate"
_KEY_SUMMARY_BLOCK = "window_summary_block"
_KEY_SUMMARY_THROUGH_TURN = "window_summary_through_turn"


class WindowRecovery:
    """Detects sliding window shifts and generates summary blocks for cache recovery."""

    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, config):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.config = config

    @staticmethod
    def _hash_message(msg: dict) -> str:
        """Hash a message's content for comparison."""
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        return hashlib.md5(content[:500].encode()).hexdigest()[:12]

    @staticmethod
    def _get_first_non_system(messages: list[dict]) -> dict | None:
        """Get the first non-system message."""
        for msg in messages:
            if msg.get("role") != "system":
                return msg
        return None

    @traceable(name="pipeline.window_detect")
    async def detect_shift(self, session_id: str, messages: list[dict]) -> dict:
        """Detect if a sliding window shift occurred.

        Returns:
            {
                "shifted": bool,
                "previous_hash": str | None,
                "current_hash": str,
                "estimated_lost_turns": int,  # 0 if no shift
            }
        """
        first_msg = self._get_first_non_system(messages)
        if not first_msg:
            return {"shifted": False, "previous_hash": None, "current_hash": "", "estimated_lost_turns": 0}

        current_hash = self._hash_message(first_msg)

        # Load previous hash from DB
        previous_hash = await self.sqlite_db.get_world_state_value(session_id, _KEY_FIRST_MSG_HASH)

        # Save current hash for next turn
        await self.sqlite_db.upsert_world_state(session_id, _KEY_FIRST_MSG_HASH, current_hash)

        if previous_hash is None:
            # First turn — no shift possible
            return {"shifted": False, "previous_hash": None, "current_hash": current_hash, "estimated_lost_turns": 0}

        if previous_hash == current_hash:
            return {"shifted": False, "previous_hash": previous_hash, "current_hash": current_hash, "estimated_lost_turns": 0}

        # Shift detected! Estimate lost turns
        previous_turn_str = await self.sqlite_db.get_world_state_value(session_id, _KEY_FIRST_MSG_TURN)
        previous_first_turn = int(previous_turn_str) if previous_turn_str else 1

        # Estimate current first turn by counting non-system messages and comparing with total turns
        current_turn_count = await self.sqlite_db.get_turn_count(session_id)
        non_system_count = sum(1 for m in messages if m.get("role") != "system")
        # Each turn ≈ 2 messages (user + assistant), rough estimate
        estimated_current_first_turn = max(1, current_turn_count - (non_system_count // 2))
        estimated_lost = max(0, estimated_current_first_turn - previous_first_turn)

        # Update stored first turn estimate
        await self.sqlite_db.upsert_world_state(
            session_id, _KEY_FIRST_MSG_TURN, str(estimated_current_first_turn)
        )

        logger.info(
            f"[WindowRecovery] Shift detected! session={session_id} "
            f"prev_hash={previous_hash} curr_hash={current_hash} "
            f"estimated_lost_turns={estimated_lost} (turn {previous_first_turn}→{estimated_current_first_turn})"
        )

        return {
            "shifted": True,
            "previous_hash": previous_hash,
            "current_hash": current_hash,
            "estimated_lost_turns": estimated_lost,
            "lost_turn_range": (previous_first_turn, estimated_current_first_turn - 1),
        }

    @traceable(name="pipeline.window_summarize")
    async def build_summary_block(self, session_id: str, shift_info: dict) -> str | None:
        """Build a summary block from lost turns' episodes.

        Merges with existing summary block if one exists (cumulative).

        Returns:
            Summary block string, or None if no episodes found.
        """
        if not shift_info.get("shifted"):
            return await self._get_existing_summary(session_id)

        lost_range = shift_info.get("lost_turn_range", (0, 0))
        from_turn, to_turn = lost_range

        if from_turn > to_turn:
            return await self._get_existing_summary(session_id)

        # Skip turns already covered by MessageCompressor immutable chunks
        compressed_through = await self.sqlite_db.get_world_state_value(
            session_id, "compressed_through_turn"
        )
        if compressed_through:
            compressed_through = int(compressed_through)
            if to_turn <= compressed_through:
                # Entire lost range is already compressed — skip
                return await self._get_existing_summary(session_id)
            if from_turn <= compressed_through:
                from_turn = compressed_through + 1

        # Get episode summaries for lost turns from turn_log
        turn_logs = await self.sqlite_db.get_turn_logs(session_id, from_turn, to_turn)

        new_summaries = []
        for log in turn_logs:
            state_changes = log.get("state_changes") or {}
            # sqlite_db.get_turn_logs returns pre-decoded dicts
            if isinstance(state_changes, str):
                try:
                    state_changes = json.loads(state_changes)
                except (json.JSONDecodeError, TypeError):
                    state_changes = {}

            summary = state_changes.get("summary", "") if isinstance(state_changes, dict) else ""
            if summary:
                new_summaries.append(f"Turn {log.get('turn_number', '?')}: {summary}")

        if not new_summaries:
            # Fallback: try ChromaDB episodes
            new_summaries = self._get_episodes_from_vector(session_id, from_turn, to_turn)

        if not new_summaries:
            logger.warning(f"[WindowRecovery] No summaries found for turns {from_turn}-{to_turn}")
            return await self._get_existing_summary(session_id)

        # Merge with existing summary block
        existing = await self._get_existing_summary(session_id)
        new_block = "\n".join(new_summaries)

        if existing:
            merged = f"{existing}\n{new_block}"
        else:
            merged = new_block

        # Wrap in context block
        summary_block = f"[이전 대화 요약 (Turn {from_turn}~{to_turn})]\n{merged}"

        # Check token budget — if too large, truncate oldest
        max_summary_tokens = int(self.config.token_budget.total_context_max * 0.15)
        block_tokens = count_tokens(summary_block)
        if block_tokens > max_summary_tokens:
            lines = summary_block.split("\n")
            while count_tokens("\n".join(lines)) > max_summary_tokens and len(lines) > 2:
                lines.pop(1)  # Remove oldest (keep header)
            summary_block = "\n".join(lines)
            logger.info(f"[WindowRecovery] Summary block truncated: {block_tokens} → {count_tokens(summary_block)} tokens")

        # Save for next turn
        await self.sqlite_db.upsert_world_state(session_id, _KEY_SUMMARY_BLOCK, summary_block)
        await self.sqlite_db.upsert_world_state(
            session_id, _KEY_SUMMARY_THROUGH_TURN, str(to_turn)
        )

        logger.info(
            f"[WindowRecovery] Summary block built: {len(new_summaries)} episodes, "
            f"{count_tokens(summary_block)} tokens, turns {from_turn}-{to_turn}"
        )

        return summary_block

    def _get_episodes_from_vector(self, session_id: str, from_turn: int, to_turn: int) -> list[str]:
        """Fallback: get episode summaries from ChromaDB by turn range."""
        try:
            result = self.vector_db.get_recent_episodes(session_id, n_results=50)
            summaries = []
            metas = result.get("metadatas", [])
            docs = result.get("documents", [])
            for meta, doc in zip(metas, docs):
                turn = meta.get("turn", 0)
                if from_turn <= turn <= to_turn and doc:
                    summaries.append(f"Turn {turn}: {doc[:300]}")
            summaries.sort(key=lambda s: int(s.split(":")[0].replace("Turn ", "").strip()))
            return summaries
        except Exception as e:
            logger.warning(f"[WindowRecovery] ChromaDB fallback failed: {e}")
            return []

    async def _get_existing_summary(self, session_id: str) -> str | None:
        """Get existing summary block from DB."""
        return await self.sqlite_db.get_world_state_value(session_id, _KEY_SUMMARY_BLOCK)

    @traceable(name="pipeline.window_inject")
    def inject_summary(self, messages: list[dict], summary_block: str | None) -> list[dict]:
        """Inject summary block into messages after system, before first non-system.

        The summary block becomes a stable assistant message with cache_control
        to serve as the new BP2.

        Returns:
            Modified messages list (new list, original not mutated).
        """
        if not summary_block:
            return messages

        messages = list(messages)

        # Find insertion point: after all system messages
        insert_idx = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                insert_idx = i + 1

        # Insert as assistant message (stable content → cacheable)
        summary_msg = {
            "role": "assistant",
            "content": summary_block,
        }

        messages.insert(insert_idx, summary_msg)

        logger.info(
            f"[WindowRecovery] Summary injected at index {insert_idx}, "
            f"{count_tokens(summary_block)} tokens"
        )

        return messages
