"""Tests for sliding window cache recovery."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from saga.window_recovery import WindowRecovery


class FakeSQLiteDB:
    """Minimal mock for SQLiteDB world_state KV."""

    def __init__(self):
        self._kv: dict[tuple[str, str], str] = {}
        self._turn_count: dict[str, int] = {}
        self._turn_logs: list[dict] = []

    async def get_world_state_value(self, session_id: str, key: str) -> str | None:
        return self._kv.get((session_id, key))

    async def upsert_world_state(self, session_id: str, key: str, value: str):
        self._kv[(session_id, key)] = value

    async def get_turn_count(self, session_id: str) -> int:
        return self._turn_count.get(session_id, 0)

    async def get_turn_logs(self, session_id: str, from_turn: int = 0, to_turn: int | None = None) -> list[dict]:
        return [
            log for log in self._turn_logs
            if log["session_id"] == session_id
            and log["turn_number"] >= from_turn
            and (to_turn is None or log["turn_number"] <= to_turn)
        ]


class FakeVectorDB:
    """Minimal mock for VectorDB."""

    def get_recent_episodes(self, session_id: str, n_results: int = 20) -> dict:
        return {"ids": [], "documents": [], "metadatas": []}


class FakeConfig:
    """Minimal config."""

    class TokenBudget:
        total_context_max = 180000
        dynamic_context_max = 4000

    token_budget = TokenBudget()


def _make_messages(turns: int, start: int = 1) -> list[dict]:
    """Generate a conversation with N turns (system + user/assistant pairs)."""
    msgs = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(start, start + turns):
        msgs.append({"role": "user", "content": f"User message turn {i}"})
        msgs.append({"role": "assistant", "content": f"Assistant response turn {i}"})
    return msgs


@pytest.fixture
def recovery():
    db = FakeSQLiteDB()
    vdb = FakeVectorDB()
    cfg = FakeConfig()
    return WindowRecovery(db, vdb, cfg), db


@pytest.mark.asyncio
async def test_no_shift_on_first_turn(recovery):
    """First turn should never detect a shift."""
    wr, db = recovery
    msgs = _make_messages(5, start=1)
    result = await wr.detect_shift("sess1", msgs)
    assert result["shifted"] is False
    assert result["previous_hash"] is None


@pytest.mark.asyncio
async def test_no_shift_same_messages(recovery):
    """Same first message → no shift."""
    wr, db = recovery
    msgs = _make_messages(5, start=1)

    # First call
    await wr.detect_shift("sess1", msgs)
    # Second call with same messages
    result = await wr.detect_shift("sess1", msgs)
    assert result["shifted"] is False


@pytest.mark.asyncio
async def test_shift_detected_when_messages_trimmed(recovery):
    """When front messages are trimmed, shift should be detected."""
    wr, db = recovery
    db._turn_count["sess1"] = 10

    # Turn N: full conversation
    msgs_full = _make_messages(10, start=1)
    await wr.detect_shift("sess1", msgs_full)

    # Turn N+1: RisuAI trimmed first 5 turns
    msgs_trimmed = _make_messages(6, start=6)  # starts from turn 6
    msgs_trimmed.insert(0, {"role": "system", "content": "You are a helpful assistant."})

    result = await wr.detect_shift("sess1", msgs_trimmed)
    assert result["shifted"] is True
    assert result["estimated_lost_turns"] > 0


@pytest.mark.asyncio
async def test_summary_block_built_from_turn_logs(recovery):
    """Summary block should be built from turn_log entries."""
    wr, db = recovery
    db._turn_count["sess1"] = 10

    # Add turn logs for turns 1-5
    for i in range(1, 6):
        db._turn_logs.append({
            "session_id": "sess1",
            "turn_number": i,
            "state_changes": {"summary": f"Turn {i}에서 일어난 일"},
        })

    shift_info = {
        "shifted": True,
        "lost_turn_range": (1, 5),
        "estimated_lost_turns": 5,
    }

    block = await wr.build_summary_block("sess1", shift_info)
    assert block is not None
    assert "Turn 1" in block
    assert "Turn 5" in block


@pytest.mark.asyncio
async def test_summary_cumulative_merge(recovery):
    """Subsequent shifts should merge with existing summary."""
    wr, db = recovery
    db._turn_count["sess1"] = 20

    # First shift: turns 1-5
    for i in range(1, 6):
        db._turn_logs.append({
            "session_id": "sess1",
            "turn_number": i,
            "state_changes": {"summary": f"First batch turn {i}"},
        })

    shift1 = {"shifted": True, "lost_turn_range": (1, 5), "estimated_lost_turns": 5}
    block1 = await wr.build_summary_block("sess1", shift1)
    assert "First batch turn 1" in block1

    # Second shift: turns 6-10
    for i in range(6, 11):
        db._turn_logs.append({
            "session_id": "sess1",
            "turn_number": i,
            "state_changes": {"summary": f"Second batch turn {i}"},
        })

    shift2 = {"shifted": True, "lost_turn_range": (6, 10), "estimated_lost_turns": 5}
    block2 = await wr.build_summary_block("sess1", shift2)

    # Should contain both batches
    assert "First batch" in block2
    assert "Second batch turn 10" in block2


@pytest.mark.asyncio
async def test_inject_summary_placement(recovery):
    """Summary should be inserted after system, before first non-system."""
    wr, _ = recovery
    msgs = _make_messages(3, start=1)

    result = wr.inject_summary(msgs, "[이전 대화 요약]\nSome summary")

    # Original not mutated
    assert len(msgs) == 7  # system + 3 turns * 2

    # Result has one extra message
    assert len(result) == 8
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "assistant"
    assert "요약" in result[1]["content"]
    assert result[2]["role"] == "user"


@pytest.mark.asyncio
async def test_inject_none_summary_noop(recovery):
    """None summary should return messages unchanged."""
    wr, _ = recovery
    msgs = _make_messages(3, start=1)
    result = wr.inject_summary(msgs, None)
    assert result == msgs


@pytest.mark.asyncio
async def test_no_shift_overhead_minimal(recovery):
    """No-shift path should be fast (just hash comparison)."""
    import time
    wr, db = recovery
    msgs = _make_messages(5, start=1)
    await wr.detect_shift("sess1", msgs)

    t0 = time.monotonic()
    for _ in range(100):
        await wr.detect_shift("sess1", msgs)
    elapsed = (time.monotonic() - t0) / 100

    # Should be well under 1ms per call (in-memory mock)
    assert elapsed < 0.01


@pytest.mark.asyncio
async def test_inject_with_multiple_system_messages(recovery):
    """Summary should be inserted after ALL system messages."""
    wr, _ = recovery
    msgs = [
        {"role": "system", "content": "System prompt 1"},
        {"role": "system", "content": "System prompt 2"},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    result = wr.inject_summary(msgs, "[요약] Some summary")
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "system"
    assert result[2]["role"] == "assistant"  # injected summary
    assert "요약" in result[2]["content"]
    assert result[3]["role"] == "user"


@pytest.mark.asyncio
async def test_empty_lost_range(recovery):
    """from_turn > to_turn should return existing summary or None."""
    wr, db = recovery
    shift_info = {
        "shifted": True,
        "lost_turn_range": (10, 5),  # invalid range
        "estimated_lost_turns": 0,
    }
    block = await wr.build_summary_block("sess1", shift_info)
    assert block is None


@pytest.mark.asyncio
async def test_state_changes_as_dict(recovery):
    """Production: state_changes is already a dict (pre-decoded by sqlite_db)."""
    wr, db = recovery
    db._turn_count["sess1"] = 5
    db._turn_logs.append({
        "session_id": "sess1",
        "turn_number": 1,
        "state_changes": {"summary": "Dict format summary"},
    })
    shift_info = {"shifted": True, "lost_turn_range": (1, 1), "estimated_lost_turns": 1}
    block = await wr.build_summary_block("sess1", shift_info)
    assert block is not None
    assert "Dict format summary" in block
