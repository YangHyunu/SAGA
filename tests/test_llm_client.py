"""Unit tests for saga/llm/client.py — provider detection, message formatting, HTTP mocking."""
import json
import pytest
import httpx
import respx
from saga.llm.client import LLMClient


# ============================================================
# _detect_provider
# ============================================================

class TestDetectProvider:
    def test_claude_model(self, llm_client):
        assert llm_client._detect_provider("claude-3-sonnet") == "anthropic"

    def test_claude_haiku(self, llm_client):
        assert llm_client._detect_provider("claude-3-haiku") == "anthropic"

    def test_anthropic_keyword(self, llm_client):
        assert llm_client._detect_provider("anthropic/claude-sonnet") == "anthropic"

    def test_gemini_model(self, llm_client):
        assert llm_client._detect_provider("gemini-1.5-pro") == "google"

    def test_google_keyword(self, llm_client):
        assert llm_client._detect_provider("google/gemini-pro") == "google"

    def test_gpt4(self, llm_client):
        assert llm_client._detect_provider("gpt-4") == "openai"

    def test_gpt4o_mini(self, llm_client):
        assert llm_client._detect_provider("gpt-4o-mini") == "openai"

    def test_unknown_model_fallback_openai(self, llm_client):
        assert llm_client._detect_provider("unknown-model-xyz") == "openai"

    def test_case_insensitive_claude(self, llm_client):
        assert llm_client._detect_provider("Claude-3-SONNET") == "anthropic"

    def test_case_insensitive_gemini(self, llm_client):
        assert llm_client._detect_provider("GEMINI-2.0-FLASH") == "google"


# ============================================================
# _call_anthropic
# ============================================================

class TestCallAnthropic:
    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        """Anthropic returns text from content blocks."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "Hello world"}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            })
        )
        result = await llm_client._call_anthropic(
            "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "Hello world"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_messages_hoisted(self, llm_client):
        """System messages should be in body['system'], not in messages array."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        assert "system" in body
        assert body["system"][0]["text"] == "You are helpful"
        # messages should only contain non-system
        for msg in body["messages"]:
            assert msg["role"] != "system"

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_control_on_system(self, llm_client):
        """System message with cache_control should propagate to system array."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        messages = [
            {"role": "system", "content": "System prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_control_on_non_system(self, llm_client):
        """Non-system messages with cache_control should have content as array."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "How are you?"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        # The assistant message with cache_control should have content as array
        asst_msg = body["messages"][1]
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert asst_msg["content"][0]["text"] == "Hello"

    @respx.mock
    @pytest.mark.asyncio
    async def test_non_cached_message_content_is_string(self, llm_client):
        """Non-system messages without cache_control should have content as plain string."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        messages = [{"role": "user", "content": "Hi"}]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        assert isinstance(body["messages"][0]["content"], str)

    @respx.mock
    @pytest.mark.asyncio
    async def test_401_raises_runtime_error(self, llm_client):
        """401 should raise RuntimeError with Korean message."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": "unauthorized"})
        )
        with pytest.raises(RuntimeError, match="인증 실패"):
            await llm_client._call_anthropic(
                "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )

    @respx.mock
    @pytest.mark.asyncio
    async def test_headers_include_caching_beta(self, llm_client):
        """Request headers should include the prompt-caching beta."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        await llm_client._call_anthropic(
            "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        headers = route.calls[0].request.headers
        assert headers["anthropic-beta"] == "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"
        assert headers["x-api-key"] == "sk-ant-test-key"

    @respx.mock
    @pytest.mark.asyncio
    async def test_multi_content_blocks(self, llm_client):
        """Multiple content blocks should be concatenated."""
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "world"},
                ],
                "usage": {},
            })
        )
        result = await llm_client._call_anthropic(
            "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "Hello world"

    @respx.mock
    @pytest.mark.asyncio
    async def test_cache_stats_logging(self, llm_client, caplog):
        """Cache stats should be logged when present."""
        import logging
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_read_input_tokens": 80,
                    "cache_creation_input_tokens": 20,
                },
            })
        )
        with caplog.at_level(logging.INFO):
            await llm_client._call_anthropic(
                "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
        assert any("cache_read=80" in r.message for r in caplog.records)

    @respx.mock
    @pytest.mark.asyncio
    async def test_no_system_key_when_no_system_messages(self, llm_client):
        """Body should not include 'system' key when there are no system messages."""
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}],
                "usage": {},
            })
        )
        messages = [{"role": "user", "content": "Hi"}]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        assert "system" not in body


# ============================================================
# _call_google
# ============================================================

class TestCallGoogle:
    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "Gemini says hi"}]}}]
            })
        )
        result = await llm_client._call_google(
            "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "Gemini says hi"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_messages_become_instruction(self, llm_client):
        route = respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            })
        )
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_google("gemini-1.5-pro", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        assert "systemInstruction" in body
        assert "Be helpful" in body["systemInstruction"]["parts"][0]["text"]
        assert "Be concise" in body["systemInstruction"]["parts"][0]["text"]
        # contents should not include system messages
        for c in body["contents"]:
            assert c["role"] in ("user", "model")

    @respx.mock
    @pytest.mark.asyncio
    async def test_assistant_mapped_to_model_role(self, llm_client):
        route = respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            })
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        await llm_client._call_google("gemini-1.5-pro", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        roles = [c["role"] for c in body["contents"]]
        assert roles == ["user", "model", "user"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_429_retry(self, llm_client):
        """Should retry up to 4 times on 429."""
        call_count = 0

        def side_effect(request):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}]
            })

        respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            side_effect=side_effect
        )
        # Patch sleep to avoid real waits
        import asyncio as _aio
        original_sleep = _aio.sleep
        _aio.sleep = lambda _: original_sleep(0)
        try:
            result = await llm_client._call_google(
                "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
            assert result == "ok"
            assert call_count == 3
        finally:
            _aio.sleep = original_sleep

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, llm_client):
        respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            return_value=httpx.Response(200, json={"candidates": []})
        )
        result = await llm_client._call_google(
            "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == ""


# ============================================================
# _call_openai
# ============================================================

class TestCallOpenAI:
    @respx.mock
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "GPT says hi"}}]
            })
        )
        result = await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "GPT says hi"

    @respx.mock
    @pytest.mark.asyncio
    async def test_system_messages_merged(self, llm_client):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_openai("gpt-4", messages, 0.7, 2048)

        body = json.loads(route.calls[0].request.content)
        system_msgs = [m for m in body["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Be helpful" in system_msgs[0]["content"]
        assert "Be concise" in system_msgs[0]["content"]

    @respx.mock
    @pytest.mark.asyncio
    async def test_uses_max_completion_tokens(self, llm_client):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 4096
        )
        body = json.loads(route.calls[0].request.content)
        assert body["max_completion_tokens"] == 4096
        assert "max_tokens" not in body

    @respx.mock
    @pytest.mark.asyncio
    async def test_empty_choices_returns_empty(self, llm_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        result = await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == ""

    @respx.mock
    @pytest.mark.asyncio
    async def test_500_raises(self, llm_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        with pytest.raises(httpx.HTTPStatusError):
            await llm_client._call_openai(
                "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )

    @respx.mock
    @pytest.mark.asyncio
    async def test_auth_header(self, llm_client):
        route = respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}]
            })
        )
        await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        headers = route.calls[0].request.headers
        assert headers["authorization"] == "Bearer sk-openai-test-key"


# ============================================================
# call_llm dispatch
# ============================================================

class TestCallLlmDispatch:
    @respx.mock
    @pytest.mark.asyncio
    async def test_dispatches_to_anthropic(self, llm_client):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "anthropic"}],
                "usage": {},
            })
        )
        result = await llm_client.call_llm("claude-3-sonnet", [{"role": "user", "content": "Hi"}])
        assert result == "anthropic"

    @respx.mock
    @pytest.mark.asyncio
    async def test_dispatches_to_google(self, llm_client):
        respx.post(url__startswith="https://generativelanguage.googleapis.com").mock(
            return_value=httpx.Response(200, json={
                "candidates": [{"content": {"parts": [{"text": "google"}]}}]
            })
        )
        result = await llm_client.call_llm("gemini-1.5-pro", [{"role": "user", "content": "Hi"}])
        assert result == "google"

    @respx.mock
    @pytest.mark.asyncio
    async def test_dispatches_to_openai(self, llm_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={
                "choices": [{"message": {"content": "openai"}}]
            })
        )
        result = await llm_client.call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result == "openai"
