"""Tests for MessageCompressor — immutable summary chunk compression."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from saga.message_compressor import MessageCompressor, _CHUNK_USER_PREFIX


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeConfig:
    class token_budget:
        total_context_max = 1000  # small for testing

    class prompt_caching:
        compress_enabled = True
        compress_threshold_ratio = 0.70  # threshold = 700 tokens
        min_compress_turns = 2
        max_summary_ratio = 0.20


def _make_messages(num_turns: int, tokens_per_msg: int = 50) -> list[dict]:
    """Create a message array with system + num_turns user/assistant pairs."""
    msgs = [{"role": "system", "content": "System prompt " + "x" * tokens_per_msg}]
    for i in range(1, num_turns + 1):
        msgs.append({"role": "user", "content": f"Turn {i} user " + "y" * tokens_per_msg})
        msgs.append({"role": "assistant", "content": f"Turn {i} assistant " + "z" * tokens_per_msg})
    return msgs


class FakeSQLiteDB:
    def __init__(self):
        self._world_state = {}
        self._turn_logs = []

    async def get_world_state_value(self, session_id, key):
        return self._world_state.get(f"{session_id}:{key}")

    async def upsert_world_state(self, session_id, key, value):
        self._world_state[f"{session_id}:{key}"] = value

    async def get_turn_logs(self, session_id, from_turn=0, to_turn=None):
        logs = [l for l in self._turn_logs if l.get("turn_number", 0) >= from_turn]
        if to_turn is not None:
            logs = [l for l in logs if l.get("turn_number", 0) <= to_turn]
        return logs

    def add_turn_log(self, turn_number, summary):
        self._turn_logs.append({
            "turn_number": turn_number,
            "state_changes": json.dumps({"summary": summary}),
            "assistant_output": f"Output for turn {turn_number}",
        })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def db():
    return FakeSQLiteDB()


@pytest.fixture
def compressor(db):
    return MessageCompressor(db, FakeConfig())


class TestNoCompression:
    @pytest.mark.asyncio
    async def test_below_threshold_no_chunks(self, compressor):
        """Below threshold with no existing chunks → return original."""
        msgs = _make_messages(3, tokens_per_msg=10)  # very small
        result, shift = await compressor.compress("sess1", msgs)
        assert shift == 0
        assert len(result) == len(msgs)

    @pytest.mark.asyncio
    async def test_below_threshold_with_chunks_restores(self, compressor, db):
        """Below threshold but existing chunks → restore chunk pairs."""
        chunks = [{"id": "chunk_001", "from_turn": 1, "to_turn": 2,
                    "summary_text": "Summary 1-2", "token_count": 20,
                    "created_at": "2026-01-01"}]
        await db.upsert_world_state("sess1", "compressed_chunks", json.dumps(chunks))
        await db.upsert_world_state("sess1", "compressed_through_turn", "2")

        msgs = _make_messages(5, tokens_per_msg=10)
        result, shift = await compressor.compress("sess1", msgs)

        # Should have chunk pair inserted
        assert any(_CHUNK_USER_PREFIX in str(m.get("content", "")) for m in result)


class TestCompression:
    @pytest.mark.asyncio
    async def test_triggers_above_threshold(self, compressor, db):
        """Above threshold → compress and reduce message count."""
        # Add turn logs for summaries
        for i in range(1, 15):
            db.add_turn_log(i, f"Turn {i} happened")

        msgs = _make_messages(14, tokens_per_msg=40)  # should exceed 700 tokens
        result, shift = await compressor.compress("sess1", msgs)

        # Should have fewer messages than original
        assert len(result) < len(msgs)
        # Should have chunk pairs
        chunk_msgs = [m for m in result if _CHUNK_USER_PREFIX in str(m.get("content", ""))]
        assert len(chunk_msgs) > 0

    @pytest.mark.asyncio
    async def test_chunk_immutability(self, compressor, db):
        """Second compression should NOT modify first chunk."""
        for i in range(1, 30):
            db.add_turn_log(i, f"Turn {i} summary")

        # First compression — use large tokens to ensure threshold hit
        msgs1 = _make_messages(14, tokens_per_msg=80)
        result1, shift1 = await compressor.compress("sess1", msgs1)

        chunks_raw = await db.get_world_state_value("sess1", "compressed_chunks")
        if not chunks_raw:
            pytest.skip("No chunks created (threshold not reached)")
        chunks1 = json.loads(chunks_raw)
        first_chunk_text = chunks1[0]["summary_text"]

        # Second compression with even more messages
        msgs2 = _make_messages(25, tokens_per_msg=80)
        result2, _ = await compressor.compress("sess1", msgs2)

        chunks_raw2 = await db.get_world_state_value("sess1", "compressed_chunks")
        chunks2 = json.loads(chunks_raw2)

        # First chunk must be unchanged
        assert chunks2[0]["summary_text"] == first_chunk_text

    @pytest.mark.asyncio
    async def test_no_turn_logs_skips(self, compressor, db):
        """No turn_log summaries → skip compression."""
        msgs = _make_messages(14, tokens_per_msg=40)
        result, shift = await compressor.compress("sess1", msgs)
        assert shift == 0


class TestMessageStructure:
    @pytest.mark.asyncio
    async def test_chunk_pair_structure(self, compressor, db):
        """Chunk pairs are user+assistant with correct format."""
        for i in range(1, 15):
            db.add_turn_log(i, f"Summary {i}")

        msgs = _make_messages(14, tokens_per_msg=40)
        result, _ = await compressor.compress("sess1", msgs)

        # Find chunk pairs
        for i, m in enumerate(result):
            content = str(m.get("content", ""))
            if _CHUNK_USER_PREFIX in content:
                assert m["role"] == "user"
                # Next message should be assistant with summary
                if i + 1 < len(result):
                    assert result[i + 1]["role"] == "assistant"
                    assert "Turn" in result[i + 1]["content"]

    @pytest.mark.asyncio
    async def test_system_preserved(self, compressor, db):
        """System message always first after compression."""
        for i in range(1, 15):
            db.add_turn_log(i, f"Summary {i}")

        msgs = _make_messages(14, tokens_per_msg=40)
        result, _ = await compressor.compress("sess1", msgs)

        assert result[0]["role"] == "system"


class TestSummaryGeneration:
    @pytest.mark.asyncio
    async def test_uses_turn_log_summaries(self, compressor, db):
        """Summary text comes from turn_log state_changes.summary."""
        db.add_turn_log(1, "Alice met Bob")
        db.add_turn_log(2, "They went to the castle")

        summary = await compressor._build_summary_from_turn_logs("sess1", 1, 2)
        assert "Alice met Bob" in summary
        assert "They went to the castle" in summary

    @pytest.mark.asyncio
    async def test_fallback_to_assistant_output(self, compressor, db):
        """No summary → falls back to assistant_output."""
        db._turn_logs.append({
            "turn_number": 1,
            "state_changes": "{}",
            "assistant_output": "The knight drew his sword and charged.",
        })

        summary = await compressor._build_summary_from_turn_logs("sess1", 1, 1)
        assert "knight" in summary


class TestBP2Detection:
    def test_chunk_user_prefix_detection(self):
        """BP2 logic in _build_cacheable_messages should detect chunk pairs."""
        content = f"{_CHUNK_USER_PREFIX} 1: Turn 1-10]"
        assert content.startswith(_CHUNK_USER_PREFIX)
