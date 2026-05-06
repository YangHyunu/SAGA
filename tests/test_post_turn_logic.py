"""Unit tests for PostTurnExtractor static methods + MdCache write_live with scriptstate."""
import os
import pytest
from saga.agents.narrative import NarrativeSummary
from saga.agents.post_turn import PostTurnExtractor
from saga.storage.md_cache import MdCache


# ============================================================
# NarrativeSummary.importance
# ============================================================

class TestNarrativeImportance:
    def test_empty_narrative_base_score(self):
        assert NarrativeSummary.empty().importance() == 10

    def test_combat_scene(self):
        # combat → +40, total=50
        assert NarrativeSummary(scene_type="combat").importance() == 50

    def test_event_scene(self):
        # event → +35, total=45
        assert NarrativeSummary(scene_type="event").importance() == 45

    def test_exploration_scene(self):
        # exploration → +15, total=25
        assert NarrativeSummary(scene_type="exploration").importance() == 25

    def test_dialogue_no_bonus(self):
        assert NarrativeSummary(scene_type="dialogue").importance() == 10

    def test_key_event_bonus(self):
        # key_event → +30, total=40
        assert NarrativeSummary(key_event="보스 등장").importance() == 40

    def test_key_event_null_no_bonus(self):
        assert NarrativeSummary(key_event=None).importance() == 10

    def test_npcs_single(self):
        # 1 npc → +10, total=20
        assert NarrativeSummary(npcs_mentioned=["고블린"]).importance() == 20

    def test_npcs_capped_at_2(self):
        # min(2, 3) * 10 = 20, total=30
        assert NarrativeSummary(npcs_mentioned=["A", "B", "C"]).importance() == 30

    def test_combined_capped_at_100(self):
        n = NarrativeSummary(
            scene_type="combat",                # +40
            key_event="big boss",               # +30
            npcs_mentioned=["A", "B", "C"],     # +20
        )
        # 10 + 40 + 30 + 20 = 100
        assert n.importance() == 100

    def test_event_with_key_event_and_npcs(self):
        n = NarrativeSummary(
            scene_type="event",                 # +35
            key_event="quest done",             # +30
            npcs_mentioned=["NPC"],             # +10
        )
        # 10 + 35 + 30 + 10 = 85
        assert n.importance() == 85


# ============================================================
# NarrativeSummary.from_llm_dict (defensive parsing)
# ============================================================

class TestNarrativeFromLLMDict:
    def test_none_returns_empty(self):
        n = NarrativeSummary.from_llm_dict(None)
        assert n.summary == ""
        assert n.npcs_mentioned == []
        assert n.scene_type == "dialogue"
        assert n.key_event is None

    def test_invalid_scene_type_falls_back_to_dialogue(self):
        n = NarrativeSummary.from_llm_dict({"scene_type": "bogus"})
        assert n.scene_type == "dialogue"

    def test_valid_fields_pass_through(self):
        n = NarrativeSummary.from_llm_dict({
            "summary": "전투 종료",
            "npcs_mentioned": ["고블린", "왕"],
            "scene_type": "combat",
            "key_event": "보스 처치",
        })
        assert n.summary == "전투 종료"
        assert n.npcs_mentioned == ["고블린", "왕"]
        assert n.scene_type == "combat"
        assert n.key_event == "보스 처치"

    def test_to_dict_roundtrip(self):
        original = {"summary": "x", "npcs_mentioned": ["A"], "scene_type": "event", "key_event": "k"}
        n = NarrativeSummary.from_llm_dict(original)
        assert n.to_dict() == original


# ============================================================
# _extract_alias (parenthetical alias extraction)
# ============================================================

class TestExtractAlias:
    """Layer 0: extract base name and alias from parenthetical notation."""

    @pytest.mark.parametrize("raw,expected_base,expected_alias", [
        ("루비아(Rubia)", "루비아", "Rubia"),
        ("최은지(崔恩智)", "최은지", "崔恩智"),
        ("Pink(핑크)", "Pink", "핑크"),
        ("존슨(Johnson)", "존슨", "Johnson"),
        ("루비아", "루비아", None),
        ("Johnson", "Johnson", None),
        ("테스트()", "테스트", None),
    ])
    def test_extract_alias(self, raw, expected_base, expected_alias):
        base, alias = PostTurnExtractor._extract_alias(raw)
        assert base == expected_base
        assert alias == expected_alias


# ============================================================
# _is_valid_npc_name (NPC filtering)
# ============================================================

class TestIsValidNpcName:
    """Filter unnamed extras, keep real NPC names."""

    # --- Should be rejected ---
    @pytest.mark.parametrize("name", [
        "마을 여인", "병사 1", "거리 상인", "경비", "숲 노인",
        "동굴 병사", "술집 농부", "시장 여자", "성 기사",
        "NPC #3", "병사A", "A",  # too short / numbered
        "", "   ",  # empty / whitespace
        "a" * 31,  # too long
        # Unnamed prefix patterns
        "이름 없는 병사", "이름없는 기사", "무명의 상인", "무명 농부",
        "정체불명의 여인", "Unknown guard", "unnamed soldier",
        # English generic NPC types
        "Guard", "Soldier", "Merchant", "Villager", "Bystander",
        "guard 1", "soldier3", "Peasant", "Servant",
    ])
    def test_rejects_extras(self, name):
        assert not PostTurnExtractor._is_valid_npc_name(name)

    # --- Should be accepted ---
    @pytest.mark.parametrize("name", [
        "존슨 대장", "엘리자베스", "Johnson", "김소연",
        "The Black Knight", "아리아", "세라핀 공작",
    ])
    def test_accepts_real_names(self, name):
        assert PostTurnExtractor._is_valid_npc_name(name)


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
