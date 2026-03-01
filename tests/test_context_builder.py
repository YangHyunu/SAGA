"""Unit tests for saga/agents/context_builder.py.

Covers:
- _get_last_user_message        (message scanning)
- _normalize_chroma_result      (nested vs flat ChromaDB formats)
- _get_relevant_episodes_rrf    (RRF ranking, deduplication, weights)
- _merge_episodes               (deduplication, priority ordering)
- _assemble_dynamic             (token budget enforcement, section order)

All tests use lightweight stubs — no real DB or LLM calls.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saga.agents.context_builder import ContextBuilder, STATE_BLOCK_INSTRUCTION


# ─────────────────────────────────────────────────────────────
# Helpers / stubs
# ─────────────────────────────────────────────────────────────

def _make_chroma_result(ids, docs=None, metas=None, nested=True):
    """Build a ChromaDB-shaped result dict."""
    docs = docs or ["" for _ in ids]
    metas = metas or [{} for _ in ids]
    if nested:
        return {"ids": [ids], "documents": [docs], "metadatas": [metas]}
    return {"ids": ids, "documents": docs, "metadatas": metas}


def _make_builder(vector_results=None, sqlite_lore=None, stable=None, live=None):
    """Build a ContextBuilder with mocked storage dependencies."""
    sqlite_db = MagicMock()
    sqlite_db.get_all_lore = AsyncMock(return_value=sqlite_lore or [])

    vector_db = MagicMock()
    vr = vector_results or {}
    # set up returns for the three RRF sources
    vector_db.get_recent_episodes.return_value = vr.get("recent", _make_chroma_result([]))
    vector_db.search_important_episodes.return_value = vr.get("important", _make_chroma_result([]))
    vector_db.search_episodes.return_value = vr.get("similar", _make_chroma_result([]))
    vector_db.search_lorebook.return_value = vr.get("lorebook", {"documents": [[]], "metadatas": [[]]})

    md_cache = MagicMock()
    md_cache.read_stable = AsyncMock(return_value=stable or "")
    md_cache.read_live = AsyncMock(return_value=live or "")

    config = MagicMock()
    return ContextBuilder(sqlite_db, vector_db, md_cache, config)


# ─────────────────────────────────────────────────────────────
# _get_last_user_message
# ─────────────────────────────────────────────────────────────

class TestGetLastUserMessage:
    def setup_method(self):
        self.builder = _make_builder()

    def test_returns_last_user_content(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First user"},
            {"role": "assistant", "content": "Reply"},
            {"role": "user", "content": "Second user"},
        ]
        assert self.builder._get_last_user_message(messages) == "Second user"

    def test_empty_messages_returns_empty(self):
        assert self.builder._get_last_user_message([]) == ""

    def test_no_user_message_returns_empty(self):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "assistant", "content": "Hello"},
        ]
        assert self.builder._get_last_user_message(messages) == ""

    def test_only_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        assert self.builder._get_last_user_message(messages) == "Hello"

    def test_missing_content_key_returns_empty(self):
        messages = [{"role": "user"}]
        assert self.builder._get_last_user_message(messages) == ""


# ─────────────────────────────────────────────────────────────
# _normalize_chroma_result
# ─────────────────────────────────────────────────────────────

class TestNormalizeChromaResult:
    def setup_method(self):
        self.builder = _make_builder()

    def test_nested_format_flattened(self):
        result = _make_chroma_result(
            ["ep1", "ep2"],
            ["Doc 1", "Doc 2"],
            [{"turn": 1, "importance": 50}, {"turn": 2, "importance": 30}],
            nested=True,
        )
        episodes = self.builder._normalize_chroma_result(result)
        assert len(episodes) == 2
        assert episodes[0]["id"] == "ep1"
        assert episodes[0]["text"] == "Doc 1"
        assert episodes[0]["importance"] == 50
        assert episodes[1]["id"] == "ep2"
        assert episodes[1]["turn"] == 2

    def test_flat_format_handled(self):
        result = _make_chroma_result(
            ["ep1"],
            ["Doc 1"],
            [{"turn": 5, "importance": 70}],
            nested=False,
        )
        episodes = self.builder._normalize_chroma_result(result)
        assert len(episodes) == 1
        assert episodes[0]["id"] == "ep1"
        assert episodes[0]["importance"] == 70

    def test_empty_result_returns_empty_list(self):
        assert self.builder._normalize_chroma_result({}) == []
        assert self.builder._normalize_chroma_result(None) == []

    def test_empty_ids_returns_empty(self):
        result = {"ids": [[]], "documents": [[]], "metadatas": [[]]}
        assert self.builder._normalize_chroma_result(result) == []

    def test_missing_metadata_uses_defaults(self):
        result = {"ids": [["ep1"]], "documents": [["Doc"]], "metadatas": []}
        episodes = self.builder._normalize_chroma_result(result)
        assert episodes[0]["importance"] == 10
        assert episodes[0]["turn"] == 0

    def test_episode_type_defaults_to_episode(self):
        result = _make_chroma_result(["ep1"], ["Doc"])
        episodes = self.builder._normalize_chroma_result(result)
        assert episodes[0]["episode_type"] == "episode"


# ─────────────────────────────────────────────────────────────
# _get_relevant_episodes_rrf
# ─────────────────────────────────────────────────────────────

class TestGetRelevantEpisodesRrf:
    @pytest.mark.asyncio
    async def test_deduplication_across_sources(self):
        """Episode appearing in multiple sources should be counted once."""
        shared = _make_chroma_result(["ep1"], ["shared doc"], [{"turn": 1, "importance": 50}])
        builder = _make_builder(vector_results={
            "recent": shared,
            "important": shared,
            "similar": shared,
        })
        episodes = await builder._get_relevant_episodes_rrf("s1", "query")
        ids = [e["id"] for e in episodes]
        assert ids.count("ep1") == 1

    @pytest.mark.asyncio
    async def test_rrf_score_boosts_episode_in_multiple_sources(self):
        """ep1 in all 3 sources should rank higher than ep2 in only 1."""
        shared = _make_chroma_result(["ep1"], ["shared"], [{"turn": 1, "importance": 10}])
        only_similar = _make_chroma_result(["ep2"], ["unique"], [{"turn": 1, "importance": 10}])
        builder = _make_builder(vector_results={
            "recent": shared,
            "important": shared,
            "similar": _make_chroma_result(
                ["ep1", "ep2"],
                ["shared", "unique"],
                [{"turn": 1}, {"turn": 2}],
            ),
        })
        episodes = await builder._get_relevant_episodes_rrf("s1", "query")
        ids = [e["id"] for e in episodes]
        assert ids.index("ep1") < ids.index("ep2")

    @pytest.mark.asyncio
    async def test_empty_sources_returns_empty(self):
        builder = _make_builder()
        episodes = await builder._get_relevant_episodes_rrf("s1", "query")
        assert episodes == []

    @pytest.mark.asyncio
    async def test_source_field_set_to_first_seen_source(self):
        """First source that introduces an episode sets its 'source' label."""
        recent = _make_chroma_result(["ep1"], ["Doc"], [{"turn": 1}])
        builder = _make_builder(vector_results={"recent": recent})
        episodes = await builder._get_relevant_episodes_rrf("s1", "query")
        assert episodes[0]["source"] == "recent"

    @pytest.mark.asyncio
    async def test_multiple_episodes_from_single_source(self):
        recent = _make_chroma_result(
            ["ep1", "ep2", "ep3"],
            ["D1", "D2", "D3"],
            [{"turn": 3}, {"turn": 2}, {"turn": 1}],
        )
        builder = _make_builder(vector_results={"recent": recent})
        episodes = await builder._get_relevant_episodes_rrf("s1", "query")
        assert len(episodes) == 3


# ─────────────────────────────────────────────────────────────
# _merge_episodes
# ─────────────────────────────────────────────────────────────

class TestMergeEpisodes:
    def setup_method(self):
        self.builder = _make_builder()

    def test_deduplication_by_id(self):
        ep = _make_chroma_result(["ep1"], ["Doc"], [{"turn": 1, "importance": 50}])
        merged = self.builder._merge_episodes(ep, ep, ep)
        assert len(merged) == 1

    def test_priority_recent_over_later_sources(self):
        recent = _make_chroma_result(["ep1"], ["From recent"], [{"turn": 5, "importance": 20}])
        similar = _make_chroma_result(["ep1"], ["From similar"], [{"turn": 5, "importance": 20}])
        merged = self.builder._merge_episodes(recent, {}, similar)
        assert merged[0]["source"] == "recent"

    def test_sorted_by_importance_desc_then_turn_desc(self):
        recent = _make_chroma_result(
            ["low", "high"],
            ["low doc", "high doc"],
            [{"turn": 10, "importance": 10}, {"turn": 1, "importance": 90}],
        )
        merged = self.builder._merge_episodes(recent, {}, {})
        assert merged[0]["id"] == "high"
        assert merged[1]["id"] == "low"

    def test_empty_sources_returns_empty(self):
        assert self.builder._merge_episodes({}, {}, {}) == []

    def test_none_source_skipped(self):
        recent = _make_chroma_result(["ep1"], ["Doc"], [{"turn": 1}])
        merged = self.builder._merge_episodes(recent, None, None)
        assert len(merged) == 1

    def test_flat_chroma_format_also_handled(self):
        flat = _make_chroma_result(["ep1"], ["Doc"], [{"turn": 1}], nested=False)
        merged = self.builder._merge_episodes(flat, {}, {})
        assert len(merged) == 1
        assert merged[0]["id"] == "ep1"


# ─────────────────────────────────────────────────────────────
# _assemble_dynamic
# ─────────────────────────────────────────────────────────────

class TestAssembleDynamic:
    def setup_method(self):
        self.builder = _make_builder()

    def test_always_includes_state_block_instruction(self):
        result = self._assemble()
        assert STATE_BLOCK_INSTRUCTION in result

    def test_live_state_included_when_fits_budget(self):
        result = self._assemble(live_state="Current state: forest", budget=5000)
        assert "Current state: forest" in result

    def test_live_state_omitted_when_budget_exhausted(self):
        """live_state is excluded when token budget is too tight."""
        result = self._assemble(live_state="X" * 10000, budget=1)
        assert "X" * 100 not in result

    def test_episodes_included_with_marker(self):
        episodes = [{"id": "ep1", "turn": 5, "summary": "Hero fought a dragon.", "importance": 80}]
        result = self._assemble(episodes=episodes, budget=5000)
        assert "에피소드 기억" in result
        assert "Hero fought a dragon." in result
        assert "[!]" in result  # importance >= 50

    def test_low_importance_episode_gets_r_marker(self):
        episodes = [{"id": "ep1", "turn": 3, "summary": "Nothing happened.", "importance": 20}]
        result = self._assemble(episodes=episodes, budget=5000)
        assert "[R]" in result

    def test_max_10_episodes_included(self):
        episodes = [
            {"id": f"ep{i}", "turn": i, "summary": f"Event {i}", "importance": 10}
            for i in range(20)
        ]
        result = self._assemble(episodes=episodes, budget=50000)
        # Episodes are limited to [:10] in assembly
        assert result.count("Event ") <= 10

    def test_active_lore_included(self):
        lore = ["### 마법\n마법의 역사", "### 검술\n검술의 기원"]
        result = self._assemble(active_lore=lore, budget=5000)
        assert "활성 로어" in result
        assert "마법의 역사" in result

    def test_sections_in_order_live_episodes_lore_instruction(self):
        result = self._assemble(
            live_state="LIVE",
            episodes=[{"id": "ep1", "turn": 1, "summary": "Battle.", "importance": 60}],
            active_lore=["### Lore\nContent"],
            budget=10000,
        )
        live_pos = result.find("LIVE")
        ep_pos = result.find("에피소드 기억")
        lore_pos = result.find("활성 로어")
        inst_pos = result.find("SAGA State Tracking")
        assert live_pos < ep_pos < lore_pos < inst_pos

    def test_empty_everything_still_has_instruction(self):
        result = self._assemble()
        assert result.strip() != ""
        assert "SAGA State Tracking" in result

    def _assemble(self, live_state="", episodes=None, active_lore=None, budget=1000, stable_prefix=""):
        return self.builder._assemble_dynamic(
            live_state, episodes or [], active_lore or [], budget, stable_prefix
        )
