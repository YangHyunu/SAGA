"""Unit tests for PostTurnExtractor static methods — pure logic, no I/O."""
import pytest
from saga.agents.post_turn import PostTurnExtractor


# ============================================================
# _calculate_importance
# ============================================================

class TestCalculateImportance:
    def test_empty_state_base_score(self):
        assert PostTurnExtractor._calculate_importance({}) == 10

    def test_hp_change_small(self):
        # hp_change=5 → abs(5)*3=15, min(30,15)=15, total=10+15=25
        assert PostTurnExtractor._calculate_importance({"hp_change": 5}) == 25

    def test_hp_change_negative(self):
        # hp_change=-5 → abs(-5)*3=15, total=25
        assert PostTurnExtractor._calculate_importance({"hp_change": -5}) == 25

    def test_hp_change_large_capped(self):
        # hp_change=20 → abs(20)*3=60, min(30,60)=30, total=40
        assert PostTurnExtractor._calculate_importance({"hp_change": 20}) == 40

    def test_relationship_changes(self):
        changes = [{"from": "A", "to": "B", "type": "trust", "delta": 10}]
        # 1 change → +10, total=20
        assert PostTurnExtractor._calculate_importance({"relationship_changes": changes}) == 20

    def test_relationship_changes_capped_at_3(self):
        changes = [{"from": "A", "to": "B"} for _ in range(5)]
        # min(3, 5) * 10 = 30, total=40
        assert PostTurnExtractor._calculate_importance({"relationship_changes": changes}) == 40

    def test_event_trigger(self):
        # +35, total=45
        assert PostTurnExtractor._calculate_importance({"event_trigger": "boss_appear"}) == 45

    def test_npc_met_single(self):
        # 1 npc → +10, total=20
        assert PostTurnExtractor._calculate_importance({"npc_met": ["고블린"]}) == 20

    def test_npc_met_capped_at_2(self):
        # min(2, 3) * 10 = 20, total=30
        assert PostTurnExtractor._calculate_importance({"npc_met": ["A", "B", "C"]}) == 30

    def test_npc_separated(self):
        # +15, total=25
        assert PostTurnExtractor._calculate_importance({"npc_separated": ["동료"]}) == 25

    def test_items_gained(self):
        # +15, total=25
        assert PostTurnExtractor._calculate_importance({"items_gained": ["검"]}) == 25

    def test_items_lost(self):
        # +15, total=25
        assert PostTurnExtractor._calculate_importance({"items_lost": ["방패"]}) == 25

    def test_items_transferred(self):
        # +20, total=30
        assert PostTurnExtractor._calculate_importance({"items_transferred": [{"item": "검", "to": "NPC"}]}) == 30

    def test_location_moved(self):
        # +10, total=20
        assert PostTurnExtractor._calculate_importance({"location_moved": True}) == 20

    def test_combined_score_capped_at_100(self):
        state = {
            "hp_change": 20,        # +30
            "relationship_changes": [{"from": "A", "to": "B"}] * 3,  # +30
            "event_trigger": "big",  # +35
            "npc_met": ["A", "B"],   # +20
            "npc_separated": ["C"],  # +15
            "items_gained": ["D"],   # +15
            "items_transferred": [{"item": "E", "to": "F"}],  # +20
            "location_moved": True,  # +10
        }
        # 10 + 30 + 30 + 35 + 20 + 15 + 15 + 20 + 10 = 185 → capped at 100
        assert PostTurnExtractor._calculate_importance(state) == 100

    def test_zero_hp_change_no_bonus(self):
        assert PostTurnExtractor._calculate_importance({"hp_change": 0}) == 10


# ============================================================
# _classify_episode
# ============================================================

class TestClassifyEpisode:
    def test_combat(self):
        assert PostTurnExtractor._classify_episode({"hp_change": -5}) == "combat"

    def test_combat_positive_hp(self):
        assert PostTurnExtractor._classify_episode({"hp_change": 10}) == "combat"

    def test_event(self):
        assert PostTurnExtractor._classify_episode({"event_trigger": "boss"}) == "event"

    def test_relationship(self):
        assert PostTurnExtractor._classify_episode({"relationship_changes": [{"from": "A", "to": "B"}]}) == "relationship"

    def test_encounter(self):
        assert PostTurnExtractor._classify_episode({"npc_met": ["goblin"]}) == "encounter"

    def test_item_gained(self):
        assert PostTurnExtractor._classify_episode({"items_gained": ["sword"]}) == "item"

    def test_item_lost(self):
        assert PostTurnExtractor._classify_episode({"items_lost": ["shield"]}) == "item"

    def test_item_transferred(self):
        assert PostTurnExtractor._classify_episode({"items_transferred": [{"item": "x", "to": "y"}]}) == "item"

    def test_exploration(self):
        assert PostTurnExtractor._classify_episode({"location_moved": True}) == "exploration"

    def test_dialogue_default(self):
        assert PostTurnExtractor._classify_episode({}) == "dialogue"
        assert PostTurnExtractor._classify_episode({"mood": "happy"}) == "dialogue"

    def test_priority_combat_over_event(self):
        """Combat takes priority when both hp_change and event_trigger present."""
        state = {"hp_change": -3, "event_trigger": "boss"}
        assert PostTurnExtractor._classify_episode(state) == "combat"

    def test_priority_event_over_relationship(self):
        state = {"event_trigger": "quest", "relationship_changes": [{"from": "A", "to": "B"}]}
        assert PostTurnExtractor._classify_episode(state) == "event"

    def test_zero_hp_not_combat(self):
        """hp_change=0 should not classify as combat."""
        assert PostTurnExtractor._classify_episode({"hp_change": 0}) == "dialogue"


# ============================================================
# _extract_entities
# ============================================================

class TestExtractEntities:
    def test_location(self):
        entities = PostTurnExtractor._extract_entities({"location": "던전"})
        assert "던전" in entities

    def test_npc_met(self):
        entities = PostTurnExtractor._extract_entities({"npc_met": ["고블린", "엘프"]})
        assert entities == ["고블린", "엘프"]

    def test_npc_separated(self):
        entities = PostTurnExtractor._extract_entities({"npc_separated": ["동료"]})
        assert "동료" in entities

    def test_items_gained_and_lost(self):
        entities = PostTurnExtractor._extract_entities({
            "items_gained": ["검"],
            "items_lost": ["방패"],
        })
        assert "검" in entities
        assert "방패" in entities

    def test_transfers(self):
        entities = PostTurnExtractor._extract_entities({
            "items_transferred": [{"item": "보석", "to": "NPC"}],
        })
        assert "보석" in entities
        assert "NPC" in entities

    def test_relationship_from_to(self):
        entities = PostTurnExtractor._extract_entities({
            "relationship_changes": [{"from": "주인공", "to": "엘프"}],
        })
        assert "주인공" in entities
        assert "엘프" in entities

    def test_deduplication_preserves_order(self):
        entities = PostTurnExtractor._extract_entities({
            "location": "마을",
            "npc_met": ["마을", "고블린"],  # "마을" duplicates location
        })
        assert entities == ["마을", "고블린"]

    def test_non_dict_transfer_skipped(self):
        entities = PostTurnExtractor._extract_entities({
            "items_transferred": ["not_a_dict", {"item": "검", "to": "NPC"}],
        })
        assert "검" in entities
        assert "NPC" in entities

    def test_non_dict_relationship_skipped(self):
        entities = PostTurnExtractor._extract_entities({
            "relationship_changes": ["not_a_dict", {"from": "A", "to": "B"}],
        })
        assert "A" in entities
        assert "B" in entities

    def test_empty_state(self):
        assert PostTurnExtractor._extract_entities({}) == []

    def test_all_sources_combined(self):
        entities = PostTurnExtractor._extract_entities({
            "location": "던전",
            "npc_met": ["고블린"],
            "npc_separated": ["동료"],
            "items_gained": ["검"],
            "items_lost": ["방패"],
            "items_transferred": [{"item": "보석", "to": "상인"}],
            "relationship_changes": [{"from": "주인공", "to": "엘프"}],
        })
        expected = ["던전", "고블린", "동료", "검", "방패", "보석", "상인", "주인공", "엘프"]
        assert entities == expected
