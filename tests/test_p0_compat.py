"""Tests for P0 RisuAI compatibility gaps: auth, session ID, gen params, streaming filter."""
import asyncio
import json
import re
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from saga.models import ChatCompletionRequest, ChatMessage
from saga.config import SagaConfig, ServerConfig, ApiKeysConfig, ModelsConfig, PromptCachingConfig


# ============================================================
# Gap 2: Bearer Auth
# ============================================================

class TestBearerAuth:
    """Test _verify_bearer dependency."""

    @pytest.fixture
    def _patch_config(self):
        """Patch the global config in server module."""
        cfg = SagaConfig(server=ServerConfig(api_key="test-secret-key"))
        with patch("saga.server.config", cfg):
            yield cfg

    @pytest.fixture
    def _patch_config_no_auth(self):
        cfg = SagaConfig(server=ServerConfig(api_key=""))
        with patch("saga.server.config", cfg):
            yield cfg

    @pytest.mark.asyncio
    async def test_valid_key_passes(self, _patch_config):
        from saga.server import _verify_bearer
        creds = MagicMock()
        creds.credentials = "test-secret-key"
        # Should not raise
        await _verify_bearer(creds)

    @pytest.mark.asyncio
    async def test_invalid_key_rejects(self, _patch_config):
        from saga.server import _verify_bearer
        from fastapi import HTTPException
        creds = MagicMock()
        creds.credentials = "wrong-key"
        with pytest.raises(HTTPException) as exc_info:
            await _verify_bearer(creds)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_credentials_rejects(self, _patch_config):
        from saga.server import _verify_bearer
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc_info:
            await _verify_bearer(None)
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_api_key_disables_auth(self, _patch_config_no_auth):
        from saga.server import _verify_bearer
        # Should not raise even without credentials
        await _verify_bearer(None)

    def test_server_config_api_key_default(self):
        cfg = ServerConfig()
        assert cfg.api_key == ""

    def test_server_config_api_key_set(self):
        cfg = ServerConfig(api_key="my-key")
        assert cfg.api_key == "my-key"


# ============================================================
# Gap 3: Session ID Priority
# ============================================================

class TestSessionIdExtraction:
    """Test _extract_session_id priority: header > user > system hash."""

    def _make_request(self, messages=None, user=None):
        return ChatCompletionRequest(
            model="test",
            messages=messages or [ChatMessage(role="user", content="hello")],
            user=user,
        )

    def _make_raw_request(self, headers=None):
        mock = MagicMock()
        mock.headers = headers or {}
        return mock

    def test_header_takes_priority(self):
        from saga.server import _extract_session_id
        req = self._make_request(
            messages=[ChatMessage(role="system", content="sys prompt")],
            user="user-field-id",
        )
        raw = self._make_raw_request({"x-saga-session-id": "header-id"})
        assert _extract_session_id(req, raw) == "header-id"

    def test_user_field_second_priority(self):
        from saga.server import _extract_session_id
        req = self._make_request(
            messages=[ChatMessage(role="system", content="sys prompt")],
            user="user-field-id",
        )
        raw = self._make_raw_request({})
        assert _extract_session_id(req, raw) == "user-field-id"

    def test_system_hash_fallback(self):
        from saga.server import _extract_session_id
        req = self._make_request(
            messages=[ChatMessage(role="system", content="my system prompt")],
        )
        raw = self._make_raw_request({})
        result = _extract_session_id(req, raw)
        assert result is not None
        assert len(result) == 16  # SHA256 hex[:16]

    def test_no_system_returns_none(self):
        from saga.server import _extract_session_id
        req = self._make_request()
        raw = self._make_raw_request({})
        assert _extract_session_id(req, raw) is None

    def test_system_hash_stable(self):
        """Same system message should produce the same session ID."""
        from saga.server import _extract_session_id
        req1 = self._make_request(messages=[ChatMessage(role="system", content="identical prompt")])
        req2 = self._make_request(messages=[ChatMessage(role="system", content="identical prompt")])
        raw = self._make_raw_request({})
        assert _extract_session_id(req1, raw) == _extract_session_id(req2, raw)

    def test_user_field_in_model(self):
        """Verify ChatCompletionRequest accepts user field."""
        req = ChatCompletionRequest(model="test", messages=[], user="my-user")
        assert req.user == "my-user"

    def test_user_field_optional(self):
        req = ChatCompletionRequest(model="test", messages=[])
        assert req.user is None

    def test_header_whitespace_stripped(self):
        from saga.server import _extract_session_id
        req = self._make_request()
        raw = self._make_raw_request({"x-saga-session-id": "  my-id  "})
        assert _extract_session_id(req, raw) == "my-id"


# ============================================================
# Gap 4: Generation Parameters
# ============================================================

class TestGenParamsExtraction:
    """Test _extract_gen_params extracts optional generation params."""

    def test_all_params(self):
        from saga.server import _extract_gen_params
        req = ChatCompletionRequest(
            model="test", messages=[],
            top_p=0.9, frequency_penalty=0.5, presence_penalty=0.3, stop=["END"],
        )
        params = _extract_gen_params(req)
        assert params == {"top_p": 0.9, "frequency_penalty": 0.5, "presence_penalty": 0.3, "stop": ["END"]}

    def test_no_params(self):
        from saga.server import _extract_gen_params
        req = ChatCompletionRequest(model="test", messages=[])
        assert _extract_gen_params(req) == {}

    def test_partial_params(self):
        from saga.server import _extract_gen_params
        req = ChatCompletionRequest(model="test", messages=[], top_p=0.95)
        params = _extract_gen_params(req)
        assert params == {"top_p": 0.95}

    def test_stop_string(self):
        from saga.server import _extract_gen_params
        req = ChatCompletionRequest(model="test", messages=[], stop="STOP")
        params = _extract_gen_params(req)
        assert params["stop"] == "STOP"


# ============================================================
# Gap 1: Streaming State Block Filter
# ============================================================

class TestStreamingStateFilter:
    """Test _partial_state_marker and the streaming filter logic."""

    def test_partial_marker_single_backtick(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("some text`") > 0

    def test_partial_marker_double_backtick(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("text``") == 2

    def test_partial_marker_triple_backtick(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("text```") == 3

    def test_partial_marker_with_state_prefix(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("text```st") == 5

    def test_partial_marker_full_without_newline(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("text```state") == 8

    def test_no_partial_marker(self):
        from saga.server import _partial_state_marker
        assert _partial_state_marker("normal text") == 0

    def test_make_sse_chunk(self):
        from saga.server import _make_sse_chunk
        result = _make_sse_chunk("sess-1", "test-model", "hello")
        assert result.startswith("data: ")
        assert result.endswith("\n\n")
        data = json.loads(result[6:])
        assert data["choices"][0]["delta"]["content"] == "hello"
        assert data["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_stream_filters_state_block_single_chunk(self):
        """State block in a single chunk should be filtered out."""
        from saga.server import _stream_response, _make_sse_chunk
        chunks = ["Hello world!\n```state\nlocation: 마을\n```\nGoodbye!"]
        await self._run_stream_filter_test(chunks, should_contain=["Hello world!", "Goodbye!"], should_not_contain=["```state", "location: 마을"])

    @pytest.mark.asyncio
    async def test_stream_filters_state_block_multi_chunk(self):
        """State block split across multiple chunks."""
        chunks = ["Hello!", "\n```sta", "te\nlocation: 마을\nmood: 평온\n``", "`\nAfter block."]
        await self._run_stream_filter_test(chunks, should_contain=["Hello!", "After block."], should_not_contain=["location: 마을", "mood: 평온"])

    @pytest.mark.asyncio
    async def test_stream_no_state_block_passes_through(self):
        """Normal text without state block should pass through unchanged."""
        chunks = ["Hello ", "world ", "no state here."]
        await self._run_stream_filter_test(chunks, should_contain=["Hello", "world", "no state here."], should_not_contain=[])

    async def _run_stream_filter_test(self, chunks, should_contain, should_not_contain):
        """Helper: run _stream_response with mocked LLM and check filtered output."""
        from saga.server import _stream_response

        # Mock all required globals
        mock_request = ChatCompletionRequest(model="test", messages=[ChatMessage(role="user", content="hi")])

        async def mock_stream(**kwargs):
            for c in chunks:
                yield c

        with patch("saga.server.llm_client") as mock_llm, \
             patch("saga.server.sqlite_db") as mock_db, \
             patch("saga.server.post_turn") as mock_pt, \
             patch("saga.server.config") as mock_cfg:

            mock_llm.call_llm_stream = mock_stream
            mock_db.increment_turn = AsyncMock(return_value=1)
            mock_pt.extract_and_update = AsyncMock()
            mock_cfg.curator.enabled = False

            collected = ""
            async for sse_line in _stream_response("test-sess", {}, [], mock_request, "hi"):
                if sse_line.startswith("data: ") and sse_line.strip() not in ("data: [DONE]",):
                    try:
                        data = json.loads(sse_line[6:])
                        content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        collected += content
                    except json.JSONDecodeError:
                        pass

            for text in should_contain:
                assert text in collected, f"Expected '{text}' in output, got: {collected}"
            for text in should_not_contain:
                assert text not in collected, f"Did NOT expect '{text}' in output, got: {collected}"
