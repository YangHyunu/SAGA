"""Unit tests for SystemStabilizer."""

import hashlib
import pytest
import pytest_asyncio

from saga.system_stabilizer import SystemStabilizer


# ──────────────────────────────────────────────────────────────────
# Stub DB (in-memory KV)
# ──────────────────────────────────────────────────────────────────

class StubDB:
    """Minimal in-memory stub matching SQLiteDB's world_state interface."""

    def __init__(self):
        self._kv: dict[tuple[str, str], str] = {}

    async def upsert_world_state(self, session_id: str, key: str, value: str):
        self._kv[(session_id, key)] = value

    async def get_world_state_value(self, session_id: str, key: str) -> str | None:
        return self._kv.get((session_id, key))


class StubConfig:
    """Minimal config stub."""

    class PromptCaching:
        enabled = True
        strategy = "md_prefix"
        stabilize_system = True
        canonical_similarity_threshold = 0.30

    prompt_caching = PromptCaching()


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    return StubDB()


@pytest.fixture
def config():
    return StubConfig()


@pytest.fixture
def stabilizer(db, config):
    return SystemStabilizer(db, config)


# ──────────────────────────────────────────────────────────────────
# Test: paragraph splitting
# ──────────────────────────────────────────────────────────────────

def test_split_paragraphs_basic():
    text = "Hello world\n\nSecond paragraph\n\nThird one"
    result = SystemStabilizer._split_paragraphs(text)
    assert result == ["Hello world", "Second paragraph", "Third one"]


def test_split_paragraphs_empty_lines():
    text = "A\n\n\n\nB\n  \n\nC"
    result = SystemStabilizer._split_paragraphs(text)
    assert result == ["A", "B", "C"]


def test_split_paragraphs_single():
    text = "Just one paragraph"
    result = SystemStabilizer._split_paragraphs(text)
    assert result == ["Just one paragraph"]


def test_split_paragraphs_empty():
    assert SystemStabilizer._split_paragraphs("") == []
    assert SystemStabilizer._split_paragraphs("   ") == []


# ──────────────────────────────────────────────────────────────────
# Test: first turn saves canonical
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_first_turn_saves_canonical(stabilizer, db):
    messages = [
        {"role": "system", "content": "You are a helpful assistant.\n\nBe nice."},
        {"role": "user", "content": "Hello"},
    ]

    result_msgs, delta = await stabilizer.stabilize("s1", messages)

    # No delta on first turn
    assert delta == ""
    # Messages unchanged
    assert result_msgs == messages
    # Canonical saved
    saved = await db.get_world_state_value("s1", "canonical_system_prompt")
    assert saved == "You are a helpful assistant.\n\nBe nice."


# ──────────────────────────────────────────────────────────────────
# Test: same system -> no delta
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_same_system_no_delta(stabilizer, db):
    system_text = "You are Yui.\n\nCharacter info here."
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": "Hi"},
    ]

    # First turn
    await stabilizer.stabilize("s1", messages)

    # Second turn, same system
    result_msgs, delta = await stabilizer.stabilize("s1", messages)
    assert delta == ""
    assert result_msgs[0]["content"] == system_text


# ──────────────────────────────────────────────────────────────────
# Test: lorebook addition -> delta extracted
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_lorebook_delta_extraction(stabilizer, db):
    canonical = "Character: Yui\n\nAge: 18\n\nPersonality: cheerful"
    lorebook = "[Lorebook: Beach]\nThe beach is beautiful and magical."
    modified = canonical + "\n\n" + lorebook

    messages_t1 = [
        {"role": "system", "content": canonical},
        {"role": "user", "content": "Hi"},
    ]
    messages_t2 = [
        {"role": "system", "content": modified},
        {"role": "user", "content": "Tell me about the beach"},
    ]

    # Turn 1: save canonical
    await stabilizer.stabilize("s1", messages_t1)

    # Turn 2: lorebook added
    result_msgs, delta = await stabilizer.stabilize("s1", messages_t2)

    # System message should be restored to canonical
    assert result_msgs[0]["content"] == canonical
    # Delta should contain the lorebook entry
    assert "[Lorebook: Beach]" in delta
    assert "beautiful and magical" in delta


# ──────────────────────────────────────────────────────────────────
# Test: multiple lorebook entries
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multiple_lorebook_delta(stabilizer, db):
    canonical = "Base system prompt.\n\nCharacter setup."
    lb1 = "[Lorebook: Ruins]\nAncient ruins description."
    lb2 = "[Lorebook: Village]\nVillage description."
    modified = canonical + "\n\n" + lb1 + "\n\n" + lb2

    msgs_t1 = [{"role": "system", "content": canonical}, {"role": "user", "content": "Hi"}]
    msgs_t2 = [{"role": "system", "content": modified}, {"role": "user", "content": "Hi"}]

    await stabilizer.stabilize("s1", msgs_t1)
    result_msgs, delta = await stabilizer.stabilize("s1", msgs_t2)

    assert result_msgs[0]["content"] == canonical
    assert "[Lorebook: Ruins]" in delta
    assert "[Lorebook: Village]" in delta


# ──────────────────────────────────────────────────────────────────
# Test: different character -> canonical replaced
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_different_character_replaces_canonical(stabilizer, db):
    system_a = "You are Yui, a young girl.\n\nShe lives in a village.\n\nShe is kind."
    system_b = "You are Hana, a warrior.\n\nShe fights in arenas.\n\nShe is fierce."

    msgs_a = [{"role": "system", "content": system_a}, {"role": "user", "content": "Hi"}]
    msgs_b = [{"role": "system", "content": system_b}, {"role": "user", "content": "Hi"}]

    # Turn 1: Yui
    await stabilizer.stabilize("s1", msgs_a)

    # Turn 2: completely different character (Hana)
    result_msgs, delta = await stabilizer.stabilize("s1", msgs_b)

    # Should replace canonical, not extract delta
    assert delta == ""
    saved = await db.get_world_state_value("s1", "canonical_system_prompt")
    assert saved == system_b


# ──────────────────────────────────────────────────────────────────
# Test: disabled config -> passthrough
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_disabled_passthrough(db):
    class DisabledConfig:
        class PromptCaching:
            enabled = True
            strategy = "md_prefix"
            stabilize_system = False
            canonical_similarity_threshold = 0.30
        prompt_caching = PromptCaching()

    stabilizer = SystemStabilizer(db, DisabledConfig())
    messages = [
        {"role": "system", "content": "Some system prompt"},
        {"role": "user", "content": "Hello"},
    ]

    result_msgs, delta = await stabilizer.stabilize("s1", messages)
    assert result_msgs is messages
    assert delta == ""


# ──────────────────────────────────────────────────────────────────
# Test: no system message -> passthrough
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_system_message(stabilizer):
    messages = [
        {"role": "user", "content": "Hello"},
    ]

    result_msgs, delta = await stabilizer.stabilize("s1", messages)
    assert result_msgs is messages
    assert delta == ""


# ──────────────────────────────────────────────────────────────────
# Test: Jaccard similarity calculation
# ──────────────────────────────────────────────────────────────────

def test_jaccard_exact_match(stabilizer):
    text = "A\n\nB\n\nC"
    assert not stabilizer._should_update_canonical(text, text, 0.30)


def test_jaccard_completely_different(stabilizer):
    a = "X\n\nY\n\nZ"
    b = "A\n\nB\n\nC"
    assert stabilizer._should_update_canonical(a, b, 0.30)


def test_jaccard_partial_overlap(stabilizer):
    a = "Shared base.\n\nParagraph A.\n\nParagraph B."
    b = "Shared base.\n\nParagraph C.\n\nParagraph D."
    # 1 shared out of 5 unique = 0.20 < 0.30 -> should update
    assert stabilizer._should_update_canonical(a, b, 0.30)


def test_jaccard_high_overlap(stabilizer):
    a = "Shared 1.\n\nShared 2.\n\nShared 3.\n\nOnly A."
    b = "Shared 1.\n\nShared 2.\n\nShared 3.\n\nOnly B."
    # 3 shared out of 5 unique = 0.60 > 0.30 -> should NOT update
    assert not stabilizer._should_update_canonical(a, b, 0.30)


# ──────────────────────────────────────────────────────────────────
# Test: session isolation
# ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_session_isolation(stabilizer, db):
    system_s1 = "Session 1 system.\n\nCharacter A."
    system_s2 = "Session 2 system.\n\nCharacter B."

    msgs_s1 = [{"role": "system", "content": system_s1}, {"role": "user", "content": "Hi"}]
    msgs_s2 = [{"role": "system", "content": system_s2}, {"role": "user", "content": "Hi"}]

    await stabilizer.stabilize("s1", msgs_s1)
    await stabilizer.stabilize("s2", msgs_s2)

    saved_s1 = await db.get_world_state_value("s1", "canonical_system_prompt")
    saved_s2 = await db.get_world_state_value("s2", "canonical_system_prompt")

    assert saved_s1 == system_s1
    assert saved_s2 == system_s2
