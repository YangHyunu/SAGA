"""Unit tests for PostTurnExtractor static methods — narrative-based logic."""
import pytest
from saga.agents.post_turn import PostTurnExtractor


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
