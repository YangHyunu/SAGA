"""Unit tests for pure functions in saga/server.py.

Covers:
- _is_continuation           (autoContinue detection)
- _prune_pending_responses   (TTL eviction)
- _partial_state_marker      (streaming state block detection)
- _extract_session_id        (priority: header > user > hash)
- _extract_gen_params        (optional param forwarding)
- _build_cacheable_messages  (lorebook_delta placement — new 4-arg signature)

No FastAPI / HTTP calls — all pure-function level.
"""
import time
import pytest
from unittest.mock import MagicMock

import saga.server as server_module
from saga.config import SagaConfig, ApiKeysConfig, ModelsConfig, PromptCachingConfig


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_config(mock_config, monkeypatch):
    """Inject a Claude config so caching branches are reachable."""
    monkeypatch.setattr(server_module, "config", mock_config)


@pytest.fixture(autouse=True)
def _reset_pending_responses():
    """Ensure _pending_responses is clean for each test."""
    server_module._pending_responses.clear()
    yield
    server_module._pending_responses.clear()


def _raw_request(headers: dict | None = None):
    """Build a minimal FastAPI Request stub."""
    req = MagicMock()
    req.headers = headers or {}
    return req


# ─────────────────────────────────────────────────────────────
# _is_continuation
# ─────────────────────────────────────────────────────────────

class TestIsContinuation:
    def test_detects_continue_pattern_in_trailing_system(self):
        messages = [
            {"role": "user", "content": "Tell me a story"},
            {"role": "assistant", "content": "Once upon a time..."},
            {"role": "system", "content": "Continue the last response"},
        ]
        assert server_module._is_continuation(messages) is True

    def test_case_insensitive_detection(self):
        messages = [
            {"role": "user", "content": "Go"},
            {"role": "system", "content": "CONTINUE THE LAST RESPONSE please"},
        ]
        assert server_module._is_continuation(messages) is True

    def test_does_not_detect_when_no_pattern(self):
        messages = [
            {"role": "system", "content": "You are a narrator"},
            {"role": "user", "content": "Hello"},
        ]
        assert server_module._is_continuation(messages) is False

    def test_stops_at_user_boundary(self):
        """Continue pattern in an early system message (before last user) is ignored."""
        messages = [
            {"role": "system", "content": "Continue the last response"},
            {"role": "user", "content": "New user message"},
        ]
        assert server_module._is_continuation(messages) is False

    def test_empty_messages(self):
        assert server_module._is_continuation([]) is False

    def test_only_user_message(self):
        assert server_module._is_continuation([{"role": "user", "content": "Hi"}]) is False

    def test_continue_in_user_message_ignored(self):
        """The pattern must be in a system message, not user."""
        messages = [
            {"role": "user", "content": "Continue the last response"},
        ]
        assert server_module._is_continuation(messages) is False

    def test_multiple_trailing_system_messages_any_match(self):
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "Some other instruction"},
            {"role": "system", "content": "Continue the last response"},
        ]
        assert server_module._is_continuation(messages) is True

    def test_system_after_user_without_continue(self):
        messages = [
            {"role": "system", "content": "Continue the last response"},
            {"role": "user", "content": "New input"},
            {"role": "system", "content": "Unrelated instruction"},
        ]
        # The trailing system after the last user is "Unrelated instruction" — no match
        assert server_module._is_continuation(messages) is False


# ─────────────────────────────────────────────────────────────
# _prune_pending_responses
# ─────────────────────────────────────────────────────────────

class TestPrunePendingResponses:
    def test_removes_stale_entries(self):
        old_ts = time.time() - (server_module._PENDING_TTL_SECONDS + 10)
        server_module._pending_responses["stale-sid"] = {
            "response": "partial text",
            "timestamp": old_ts,
        }
        server_module._prune_pending_responses()
        assert "stale-sid" not in server_module._pending_responses

    def test_preserves_fresh_entries(self):
        server_module._pending_responses["fresh-sid"] = {
            "response": "partial text",
            "timestamp": time.time(),
        }
        server_module._prune_pending_responses()
        assert "fresh-sid" in server_module._pending_responses

    def test_mixed_stale_and_fresh(self):
        old_ts = time.time() - (server_module._PENDING_TTL_SECONDS + 5)
        server_module._pending_responses["stale"] = {"response": "", "timestamp": old_ts}
        server_module._pending_responses["fresh"] = {"response": "", "timestamp": time.time()}
        server_module._prune_pending_responses()
        assert "stale" not in server_module._pending_responses
        assert "fresh" in server_module._pending_responses

    def test_empty_dict_no_error(self):
        server_module._prune_pending_responses()  # should not raise

    def test_exactly_at_ttl_boundary_is_pruned(self):
        """Entry exactly at TTL+1 second is pruned (elapsed > TTL)."""
        ts = time.time() - server_module._PENDING_TTL_SECONDS - 1
        server_module._pending_responses["boundary"] = {"response": "", "timestamp": ts}
        server_module._prune_pending_responses()
        assert "boundary" not in server_module._pending_responses


# ─────────────────────────────────────────────────────────────
# _partial_state_marker
# ─────────────────────────────────────────────────────────────

class TestPartialStateMarker:
    def test_single_backtick(self):
        assert server_module._partial_state_marker("text`") == 1

    def test_double_backtick(self):
        assert server_module._partial_state_marker("text``") == 2

    def test_triple_backtick(self):
        assert server_module._partial_state_marker("text```") == 3

    def test_triple_backtick_s(self):
        assert server_module._partial_state_marker("text```s") == 4

    def test_triple_backtick_state(self):
        assert server_module._partial_state_marker("text```state") == 8

    def test_triple_backtick_state_newline(self):
        assert server_module._partial_state_marker("text```state\n") == 9

    def test_no_partial_marker(self):
        assert server_module._partial_state_marker("hello world") == 0

    def test_empty_string(self):
        assert server_module._partial_state_marker("") == 0

    def test_complete_marker_not_counted(self):
        # A full completed marker in the middle should not match
        assert server_module._partial_state_marker("some text here.") == 0

    def test_triple_backtick_sta(self):
        assert server_module._partial_state_marker("prefix```sta") == 6


# ─────────────────────────────────────────────────────────────
# _extract_session_id
# ─────────────────────────────────────────────────────────────

class TestExtractSessionId:
    def _make_request(self, user=None, messages=None):
        req = MagicMock()
        req.user = user
        req.messages = messages or []
        return req

    def test_header_takes_priority_over_user_field(self):
        request = self._make_request(user="user-field-id")
        raw = _raw_request({"x-saga-session-id": "header-id"})
        result = server_module._extract_session_id(request, raw)
        assert result == "header-id"

    def test_header_takes_priority_over_system_hash(self):
        msg = MagicMock()
        msg.role = "system"
        msg.get_text_content.return_value = "System content"
        request = self._make_request(messages=[msg])
        raw = _raw_request({"x-saga-session-id": "header-id"})
        result = server_module._extract_session_id(request, raw)
        assert result == "header-id"

    def test_user_field_used_when_no_header(self):
        request = self._make_request(user="my-user-id")
        raw = _raw_request()
        result = server_module._extract_session_id(request, raw)
        assert result == "my-user-id"

    def test_user_field_stripped(self):
        request = self._make_request(user="  spaced-id  ")
        raw = _raw_request()
        result = server_module._extract_session_id(request, raw)
        assert result == "spaced-id"

    def test_system_hash_fallback(self):
        msg = MagicMock()
        msg.role = "system"
        msg.get_text_content.return_value = "You are a helpful assistant."
        request = self._make_request(messages=[msg])
        raw = _raw_request()
        result = server_module._extract_session_id(request, raw)
        # Should return an 8-char hex string
        assert result is not None
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_system_hash_stable_across_calls(self):
        """Same system content must produce the same session hash."""
        msg = MagicMock()
        msg.role = "system"
        msg.get_text_content.return_value = "Stable system prompt text here."
        request = self._make_request(messages=[msg])
        raw = _raw_request()
        r1 = server_module._extract_session_id(request, raw)
        r2 = server_module._extract_session_id(request, raw)
        assert r1 == r2

    def test_no_system_message_returns_none(self):
        msg = MagicMock()
        msg.role = "user"
        msg.get_text_content.return_value = "Hello"
        request = self._make_request(messages=[msg])
        raw = _raw_request()
        result = server_module._extract_session_id(request, raw)
        assert result is None

    def test_empty_user_field_falls_through_to_hash(self):
        """user='' (falsy) should fall through to system-hash."""
        msg = MagicMock()
        msg.role = "system"
        msg.get_text_content.return_value = "Some prompt"
        request = self._make_request(user="", messages=[msg])
        raw = _raw_request()
        result = server_module._extract_session_id(request, raw)
        assert result is not None
        assert len(result) == 16

    def test_header_stripped(self):
        request = self._make_request()
        raw = _raw_request({"x-saga-session-id": "  padded-id  "})
        result = server_module._extract_session_id(request, raw)
        assert result == "padded-id"


# ─────────────────────────────────────────────────────────────
# _extract_gen_params
# ─────────────────────────────────────────────────────────────

class TestExtractGenParams:
    def _make_request(self, **kwargs):
        req = MagicMock()
        req.top_p = kwargs.get("top_p")
        req.frequency_penalty = kwargs.get("frequency_penalty")
        req.presence_penalty = kwargs.get("presence_penalty")
        req.stop = kwargs.get("stop")
        return req

    def test_empty_when_all_none(self):
        req = self._make_request()
        assert server_module._extract_gen_params(req) == {}

    def test_top_p_included(self):
        req = self._make_request(top_p=0.9)
        params = server_module._extract_gen_params(req)
        assert params["top_p"] == 0.9

    def test_frequency_penalty_included(self):
        req = self._make_request(frequency_penalty=0.5)
        params = server_module._extract_gen_params(req)
        assert params["frequency_penalty"] == 0.5

    def test_presence_penalty_included(self):
        req = self._make_request(presence_penalty=0.3)
        params = server_module._extract_gen_params(req)
        assert params["presence_penalty"] == 0.3

    def test_stop_string_included(self):
        req = self._make_request(stop="STOP")
        params = server_module._extract_gen_params(req)
        assert params["stop"] == "STOP"

    def test_stop_list_included(self):
        req = self._make_request(stop=["STOP", "END"])
        params = server_module._extract_gen_params(req)
        assert params["stop"] == ["STOP", "END"]

    def test_all_params_included(self):
        req = self._make_request(top_p=0.95, frequency_penalty=0.1, presence_penalty=0.2, stop="END")
        params = server_module._extract_gen_params(req)
        assert set(params.keys()) == {"top_p", "frequency_penalty", "presence_penalty", "stop"}

    def test_zero_values_included(self):
        """Explicit 0.0 values must be passed through, not treated as None."""
        req = self._make_request(top_p=0.0, frequency_penalty=0.0)
        params = server_module._extract_gen_params(req)
        assert "top_p" in params
        assert "frequency_penalty" in params


# ─────────────────────────────────────────────────────────────
# _build_cacheable_messages — lorebook_delta (4-argument form)
# ─────────────────────────────────────────────────────────────

class TestBuildCacheableMessagesLorebookDelta:
    """Tests for the lorebook_delta parameter added in P1."""

    def _conversation(self, n_turns: int) -> list[dict]:
        msgs = [{"role": "system", "content": "You are a narrator."}]
        for i in range(n_turns):
            msgs.append({"role": "user", "content": f"User {i+1}"})
            msgs.append({"role": "assistant", "content": f"Asst {i+1}"})
        msgs.append({"role": "user", "content": "Latest user input"})
        return msgs

    def test_lorebook_delta_in_last_user_message(self):
        """Lorebook delta must appear in the last user message, not system."""
        msgs = self._conversation(2)
        result = server_module._build_cacheable_messages(msgs, "", "", "[Lorebook: Forest]\nDark woods.")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert "Active Lorebook" in last_user["content"]
        assert "[Lorebook: Forest]" in last_user["content"]
        assert "Dark woods." in last_user["content"]

    def test_lorebook_delta_not_in_system(self):
        """Lorebook delta must never appear in system (would break BP1 cache)."""
        msgs = self._conversation(2)
        result = server_module._build_cacheable_messages(msgs, "", "", "[Lorebook: Forest]\nDark woods.")
        system_msgs = [m for m in result if m.get("role") == "system"]
        for sys_msg in system_msgs:
            assert "Lorebook" not in sys_msg["content"]

    def test_lorebook_delta_combined_with_md_prefix(self):
        """Both lorebook_delta and md_prefix must appear in last user, in order."""
        msgs = self._conversation(2)
        result = server_module._build_cacheable_messages(
            msgs, "stable prefix", "", "[Lorebook: Town]\nTown lore."
        )
        last_user = [m for m in result if m.get("role") == "user"][-1]
        ctx = last_user["content"]
        # Both sections present
        assert "stable prefix" in ctx
        assert "[Lorebook: Town]" in ctx
        # Context Cache section appears before Active Lorebook section
        assert ctx.index("SAGA Context Cache") < ctx.index("Active Lorebook")

    def test_lorebook_delta_combined_with_dynamic_suffix(self):
        msgs = self._conversation(2)
        result = server_module._build_cacheable_messages(
            msgs, "", "dynamic content", "[Lorebook: Cave]\nCave lore."
        )
        last_user = [m for m in result if m.get("role") == "user"][-1]
        ctx = last_user["content"]
        assert "dynamic content" in ctx
        assert "[Lorebook: Cave]" in ctx

    def test_empty_lorebook_delta_no_active_lorebook_header(self):
        """Empty lorebook_delta must not produce the Active Lorebook header."""
        msgs = self._conversation(2)
        result = server_module._build_cacheable_messages(msgs, "prefix", "suffix", "")
        last_user = [m for m in result if m.get("role") == "user"][-1]
        assert "Active Lorebook" not in last_user["content"]

    def test_no_lorebook_delta_arg_defaults_to_empty(self):
        """Calling with 3 args (no lorebook_delta) behaves like empty delta."""
        msgs = self._conversation(2)
        result3 = server_module._build_cacheable_messages(msgs, "prefix", "suffix")
        result4 = server_module._build_cacheable_messages(msgs, "prefix", "suffix", "")
        last3 = [m for m in result3 if m.get("role") == "user"][-1]["content"]
        last4 = [m for m in result4 if m.get("role") == "user"][-1]["content"]
        assert last3 == last4

    def test_cache_breakpoints_still_applied_with_lorebook_delta(self):
        """Adding lorebook_delta must not disrupt BP1/BP2/BP3 placement."""
        msgs = self._conversation(4)
        result = server_module._build_cacheable_messages(
            msgs, "", "", "[Lorebook: X]\nExtra lore."
        )
        cached = [m for m in result if m.get("cache_control")]
        assert len(cached) == 3  # BP1 + BP2 + BP3
