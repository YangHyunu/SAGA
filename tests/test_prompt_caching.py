"""Unit tests for _build_cacheable_messages() in saga/server.py.

Tests the 3-breakpoint prompt caching strategy:
  BP1: system prompt
  BP2: mid-conversation assistant
  BP3: last assistant
  Dynamic context: prepended to last user message (NOT system)
"""
import copy
import pytest
import saga.server as server_module


@pytest.fixture(autouse=True)
def _patch_server_config(mock_config, monkeypatch):
    """Patch the global config in saga.server for all tests in this module."""
    monkeypatch.setattr(server_module, "config", mock_config)


def _make_conversation(n_turns: int) -> list[dict]:
    """Build a conversation with system + n user/assistant pairs."""
    msgs = [{"role": "system", "content": "You are a helpful RP narrator."}]
    for i in range(n_turns):
        msgs.append({"role": "user", "content": f"User message {i+1}"})
        msgs.append({"role": "assistant", "content": f"Assistant response {i+1}"})
    msgs.append({"role": "user", "content": "Latest user message"})
    return msgs


# ============================================================
# 3-BP structure with full conversation
# ============================================================

class TestThreeBreakpointStructure:
    def test_bp1_on_system(self):
        """BP1: system message gets cache_control."""
        msgs = _make_conversation(3)
        result = server_module._build_cacheable_messages(msgs, "", "")
        system_msg = result[0]
        assert system_msg["cache_control"] == {"type": "ephemeral"}

    def test_bp2_on_mid_assistant(self):
        """BP2: middle assistant gets cache_control when >=2 assistants."""
        msgs = _make_conversation(4)  # 4 assistants, mid = index 2
        result = server_module._build_cacheable_messages(msgs, "", "")
        assistant_indices = [i for i, m in enumerate(result) if m.get("role") == "assistant"]
        mid_idx = assistant_indices[len(assistant_indices) // 2]
        assert result[mid_idx].get("cache_control") == {"type": "ephemeral"}

    def test_bp3_on_last_assistant(self):
        """BP3: last assistant gets cache_control when >=2 assistants."""
        msgs = _make_conversation(4)
        result = server_module._build_cacheable_messages(msgs, "", "")
        assistant_indices = [i for i, m in enumerate(result) if m.get("role") == "assistant"]
        last_idx = assistant_indices[-1]
        assert result[last_idx].get("cache_control") == {"type": "ephemeral"}

    def test_exactly_3_breakpoints_with_long_convo(self):
        """With >=2 assistants, exactly 3 messages should have cache_control."""
        msgs = _make_conversation(6)
        result = server_module._build_cacheable_messages(msgs, "", "")
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 3  # BP1 + BP2 + BP3


# ============================================================
# Short conversations
# ============================================================

class TestShortConversation:
    def test_one_assistant_gets_bp2_only(self):
        """With exactly 1 assistant, system + that assistant get cache_control."""
        msgs = _make_conversation(1)
        result = server_module._build_cacheable_messages(msgs, "", "")
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 2  # BP1 (system) + BP2 (only assistant)

    def test_no_assistants_only_bp1(self):
        """With no assistants, only system gets cache_control."""
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
        ]
        result = server_module._build_cacheable_messages(msgs, "", "")
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 1
        assert cached[0]["role"] == "system"


# ============================================================
# Dynamic context placement (THE critical invariant)
# ============================================================

class TestDynamicContextPlacement:
    def test_dynamic_context_prepended_to_last_user(self):
        """Dynamic context should be in the last user message, not system."""
        msgs = _make_conversation(2)
        result = server_module._build_cacheable_messages(msgs, "stable prefix", "dynamic stuff")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert "[--- SAGA Context Cache ---]" in last_user["content"]
        assert "[--- SAGA Dynamic ---]" in last_user["content"]
        assert "stable prefix" in last_user["content"]
        assert "dynamic stuff" in last_user["content"]

    def test_dynamic_context_never_in_system(self):
        """Dynamic context must NEVER appear in system messages (cache invalidation)."""
        msgs = _make_conversation(3)
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        system_msgs = [m for m in result if m.get("role") == "system"]
        for sys_msg in system_msgs:
            assert "SAGA Context Cache" not in sys_msg["content"]
            assert "SAGA Dynamic" not in sys_msg["content"]
            assert "prefix" not in sys_msg["content"]

    def test_only_md_prefix(self):
        """When only md_prefix exists, SAGA Context Cache header present."""
        msgs = _make_conversation(1)
        result = server_module._build_cacheable_messages(msgs, "stable data", "")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert "[--- SAGA Context Cache ---]" in last_user["content"]
        assert "stable data" in last_user["content"]

    def test_only_dynamic_suffix(self):
        """When only dynamic_suffix exists, SAGA Dynamic header present."""
        msgs = _make_conversation(1)
        result = server_module._build_cacheable_messages(msgs, "", "dynamic only")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert "[--- SAGA Dynamic ---]" in last_user["content"]
        assert "dynamic only" in last_user["content"]

    def test_no_context_no_modification(self):
        """When both prefix and suffix are empty, last user message unchanged."""
        msgs = _make_conversation(1)
        original_last_user = msgs[-1]["content"]
        result = server_module._build_cacheable_messages(msgs, "", "")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert last_user["content"] == original_last_user


# ============================================================
# Non-Claude model fallback
# ============================================================

class TestNonClaudeFallback:
    def test_non_claude_context_in_system(self, mock_config_non_claude, monkeypatch):
        """Non-Claude: dynamic context goes into system message."""
        monkeypatch.setattr(server_module, "config", mock_config_non_claude)
        msgs = _make_conversation(2)
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        system_msg = result[0]
        assert "SAGA Dynamic Context" in system_msg["content"]
        assert "prefix" in system_msg["content"]
        assert "suffix" in system_msg["content"]

    def test_non_claude_no_cache_control(self, mock_config_non_claude, monkeypatch):
        """Non-Claude: no messages should have cache_control."""
        monkeypatch.setattr(server_module, "config", mock_config_non_claude)
        msgs = _make_conversation(3)
        result = server_module._build_cacheable_messages(msgs, "p", "s")
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 0


# ============================================================
# Caching disabled
# ============================================================

class TestCachingDisabled:
    def test_disabled_behaves_like_non_claude(self, mock_config_caching_disabled, monkeypatch):
        """Even with Claude model, disabled caching = no cache_control."""
        monkeypatch.setattr(server_module, "config", mock_config_caching_disabled)
        msgs = _make_conversation(3)
        result = server_module._build_cacheable_messages(msgs, "p", "s")
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 0

    def test_disabled_context_in_system(self, mock_config_caching_disabled, monkeypatch):
        """Disabled caching: context goes in system message."""
        monkeypatch.setattr(server_module, "config", mock_config_caching_disabled)
        msgs = _make_conversation(1)
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        system_msg = result[0]
        assert "prefix" in system_msg["content"]


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases:
    def test_no_system_message_non_claude(self, mock_config_non_claude, monkeypatch):
        """No system message + non-Claude: inserts new system message."""
        monkeypatch.setattr(server_module, "config", mock_config_non_claude)
        msgs = [{"role": "user", "content": "Hi"}]
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        assert result[0]["role"] == "system"
        assert "SAGA Dynamic Context" in result[0]["content"]

    def test_no_user_message_with_claude(self):
        """No user message with Claude: dynamic context appended as new user message."""
        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        user_msgs = [m for m in result if m.get("role") == "user"]
        assert len(user_msgs) == 1
        assert "prefix" in user_msgs[0]["content"]

    def test_original_messages_not_mutated(self):
        """Input list and dicts should not be modified in-place."""
        msgs = _make_conversation(2)
        original = copy.deepcopy(msgs)
        server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        # The outer list may have different identity (list() copy), but
        # original dicts should be unchanged
        for orig_msg, current_msg in zip(original, msgs):
            assert orig_msg == current_msg, f"Original message was mutated: {orig_msg} != {current_msg}"
