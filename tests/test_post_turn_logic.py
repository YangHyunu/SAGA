"""Unit tests for PostTurnExtractor static methods + MdCache write_live with scriptstate."""
import os
import pytest
from saga.agents.post_turn import PostTurnExtractor
from saga.storage.md_cache import MdCache


# ============================================================
# _calculate_importance (narrative-based)
# ============================================================

class TestCalculateImportance:
    def test_empty_narrative_base_score(self):
        assert PostTurnExtractor._calculate_importance({}) == 10

    def test_combat_scene(self):
        # combat → +40, total=50
        assert PostTurnExtractor._calculate_importance({"scene_type": "combat"}) == 50

    def test_event_scene(self):
        # event → +35, total=45
        assert PostTurnExtractor._calculate_importance({"scene_type": "event"}) == 45

    def test_exploration_scene(self):
        # exploration → +15, total=25
        assert PostTurnExtractor._calculate_importance({"scene_type": "exploration"}) == 25

    def test_dialogue_no_bonus(self):
        assert PostTurnExtractor._calculate_importance({"scene_type": "dialogue"}) == 10

    def test_key_event_bonus(self):
        # key_event → +30, total=40
        assert PostTurnExtractor._calculate_importance({"key_event": "보스 등장"}) == 40

    def test_key_event_null_no_bonus(self):
        assert PostTurnExtractor._calculate_importance({"key_event": None}) == 10

    def test_npcs_single(self):
        # 1 npc → +10, total=20
        assert PostTurnExtractor._calculate_importance({"npcs_mentioned": ["고블린"]}) == 20

    def test_npcs_capped_at_2(self):
        # min(2, 3) * 10 = 20, total=30
        assert PostTurnExtractor._calculate_importance({"npcs_mentioned": ["A", "B", "C"]}) == 30

    def test_combined_capped_at_100(self):
        narrative = {
            "scene_type": "combat",     # +40
            "key_event": "big boss",    # +30
            "npcs_mentioned": ["A", "B", "C"],  # +20
        }
        # 10 + 40 + 30 + 20 = 100
        assert PostTurnExtractor._calculate_importance(narrative) == 100

    def test_event_with_key_event_and_npcs(self):
        narrative = {
            "scene_type": "event",      # +35
            "key_event": "quest done",  # +30
            "npcs_mentioned": ["NPC"],  # +10
        }
        # 10 + 35 + 30 + 10 = 85
        assert PostTurnExtractor._calculate_importance(narrative) == 85


# ============================================================
# MdCache.write_live with scriptstate
# ============================================================

class TestWriteLiveScriptstate:
    @pytest.fixture(autouse=True)
    def setup_md_cache(self, tmp_path):
        self.md_cache = MdCache(cache_dir=str(tmp_path))
        self.session_id = "test-session"
        os.makedirs(self.md_cache.get_session_dir(self.session_id), exist_ok=True)

    def _read_live(self):
        filepath = os.path.join(
            self.md_cache.get_session_dir(self.session_id), "live_state.md"
        )
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    async def test_scriptstate_overrides_player_context(self):
        await (
            self.md_cache.write_live(
                self.session_id, 5,
                {},
                {"location": "town", "hp": 100, "max_hp": 100, "mood": "neutral"},
                scriptstate={"location": "dungeon", "hp": "85", "max_hp": "100", "mood": "angry"},
            )
        )
        content = self._read_live()
        assert "dungeon" in content
        assert "85" in content
        assert "angry" in content
        assert "town" not in content

    async def test_custom_vars_rendered(self):
        await (
            self.md_cache.write_live(
                self.session_id, 3, {}, {},
                scriptstate={"gold": "500", "quest_stage": "3", "reputation": "friendly"},
            )
        )
        content = self._read_live()
        assert "캐릭터 변수" in content
        assert "gold: 500" in content
        assert "quest_stage: 3" in content
        assert "reputation: friendly" in content

    async def test_handled_keys_not_in_custom_vars(self):
        """Keys like hp, location, mood should NOT duplicate in custom vars section."""
        await (
            self.md_cache.write_live(
                self.session_id, 2, {}, {},
                scriptstate={"hp": "50", "max_hp": "100", "location": "forest", "mood": "calm", "gold": "200"},
            )
        )
        content = self._read_live()
        # gold should be in custom vars
        assert "gold: 200" in content
        # hp should be in status, not duplicated in custom vars
        assert content.count("hp") == 1 or "HP: 50/100" in content

    async def test_none_scriptstate_uses_player_context(self):
        await (
            self.md_cache.write_live(
                self.session_id, 1, {}, {"location": "castle"},
                scriptstate=None,
            )
        )
        content = self._read_live()
        assert "castle" in content
        assert "캐릭터 변수" not in content

    async def test_empty_values_filtered_from_custom_vars(self):
        await (
            self.md_cache.write_live(
                self.session_id, 1, {}, {},
                scriptstate={"gold": "500", "empty_var": "", "zero_var": "0", "null_var": None},
            )
        )
        content = self._read_live()
        assert "gold: 500" in content
        assert "empty_var" not in content
        assert "zero_var" not in content
        assert "null_var" not in content
