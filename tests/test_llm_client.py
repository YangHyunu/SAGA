"""Unit tests for saga/llm/client.py — provider detection, message formatting, SDK mocking."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from saga.llm.client import LLMClient


# ============================================================
# Helpers
# ============================================================

def _mock_anthropic_response(text="ok", input_tokens=10, output_tokens=5,
                              cache_read=0, cache_create=0, multi_blocks=None):
    """Create a mock Anthropic messages.create response."""
    if multi_blocks:
        blocks = []
        for t in multi_blocks:
            b = MagicMock()
            b.text = t
            blocks.append(b)
    else:
        block = MagicMock()
        block.text = text
        blocks = [block]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_create

    response = MagicMock()
    response.content = blocks
    response.usage = usage
    return response


def _mock_openai_response(content="ok"):
    """Create a mock OpenAI chat.completions.create response."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _mock_openai_response_empty():
    response = MagicMock()
    response.choices = []
    return response


def _mock_google_response(text="ok"):
    """Create a mock google-genai generate_content response."""
    response = MagicMock()
    response.text = text
    return response


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
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        """Anthropic returns text from content blocks."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response("Hello world")
        )
        result = await llm_client._call_anthropic(
            "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_system_messages_hoisted(self, llm_client):
        """System messages should be in create_kwargs['system'], not in messages."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response()
        )
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        call_kwargs = llm_client._anthropic.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert call_kwargs["system"][0]["text"] == "You are helpful"
        # messages should only contain non-system
        for msg in call_kwargs["messages"]:
            assert msg["role"] != "system"

    @pytest.mark.asyncio
    async def test_cache_control_on_system(self, llm_client):
        """System message with cache_control should propagate to system array."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response()
        )
        messages = [
            {"role": "system", "content": "System prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        call_kwargs = llm_client._anthropic.messages.create.call_args.kwargs
        assert call_kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_cache_control_on_non_system(self, llm_client):
        """Non-system messages with cache_control should have content as array."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response()
        )
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "How are you?"},
        ]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        call_kwargs = llm_client._anthropic.messages.create.call_args.kwargs
        asst_msg = call_kwargs["messages"][1]
        assert isinstance(asst_msg["content"], list)
        assert asst_msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert asst_msg["content"][0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_non_cached_message_content_is_string(self, llm_client):
        """Non-system messages without cache_control should have content as plain string."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response()
        )
        messages = [{"role": "user", "content": "Hi"}]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        call_kwargs = llm_client._anthropic.messages.create.call_args.kwargs
        assert isinstance(call_kwargs["messages"][0]["content"], str)

    @pytest.mark.asyncio
    async def test_401_raises_runtime_error(self, llm_client):
        """AuthenticationError should raise RuntimeError with Korean message."""
        import anthropic as _anthropic
        llm_client._anthropic.messages.create = AsyncMock(
            side_effect=_anthropic.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )
        with pytest.raises(RuntimeError, match="인증 실패"):
            await llm_client._call_anthropic(
                "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )

    @pytest.mark.asyncio
    async def test_multi_content_blocks(self, llm_client):
        """Multiple content blocks should be concatenated."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(multi_blocks=["Hello ", "world"])
        )
        result = await llm_client._call_anthropic(
            "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_cache_stats_logging(self, llm_client, caplog):
        """Cache stats should be logged when present."""
        import logging
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response(
                cache_read=80, cache_create=20, input_tokens=100, output_tokens=50
            )
        )
        with caplog.at_level(logging.INFO):
            await llm_client._call_anthropic(
                "claude-3-sonnet", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
        assert any("cache_read=80" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_no_system_key_when_no_system_messages(self, llm_client):
        """Create kwargs should not include 'system' key when there are no system messages."""
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response()
        )
        messages = [{"role": "user", "content": "Hi"}]
        await llm_client._call_anthropic("claude-3-sonnet", messages, 0.7, 2048)

        call_kwargs = llm_client._anthropic.messages.create.call_args.kwargs
        assert "system" not in call_kwargs


# ============================================================
# _call_google
# ============================================================

class TestCallGoogle:
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        mock_resp = _mock_google_response("Gemini says hi")
        with patch("saga.llm.client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await llm_client._call_google(
                "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
        assert result == "Gemini says hi"

    @pytest.mark.asyncio
    async def test_system_messages_become_instruction(self, llm_client):
        mock_resp = _mock_google_response("ok")
        with patch("saga.llm.client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp) as mock_thread:
            messages = [
                {"role": "system", "content": "Be helpful"},
                {"role": "system", "content": "Be concise"},
                {"role": "user", "content": "Hi"},
            ]
            await llm_client._call_google("gemini-1.5-pro", messages, 0.7, 2048)

            # asyncio.to_thread(fn, model=..., contents=..., config=...)
            call_kwargs = mock_thread.call_args.kwargs
            config_obj = call_kwargs["config"]
            assert "Be helpful" in config_obj.system_instruction
            assert "Be concise" in config_obj.system_instruction
            # contents should not include system messages
            for c in call_kwargs["contents"]:
                assert c["role"] in ("user", "model")

    @pytest.mark.asyncio
    async def test_assistant_mapped_to_model_role(self, llm_client):
        mock_resp = _mock_google_response("ok")
        with patch("saga.llm.client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp) as mock_thread:
            messages = [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": "Bye"},
            ]
            await llm_client._call_google("gemini-1.5-pro", messages, 0.7, 2048)

            call_kwargs = mock_thread.call_args.kwargs
            roles = [c["role"] for c in call_kwargs["contents"]]
            assert roles == ["user", "model", "user"]

    @pytest.mark.asyncio
    async def test_429_retry(self, llm_client):
        """Should retry up to 4 times on 429."""
        call_count = 0
        mock_resp = _mock_google_response("ok")

        async def fake_to_thread(fn, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("429 rate limited")
            return mock_resp

        with patch("saga.llm.client.asyncio.to_thread", side_effect=fake_to_thread), \
             patch("saga.llm.client.asyncio.sleep", new_callable=AsyncMock):
            result = await llm_client._call_google(
                "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
            assert result == "ok"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_empty_text_returns_empty(self, llm_client):
        mock_resp = MagicMock()
        mock_resp.text = ""
        with patch("saga.llm.client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await llm_client._call_google(
                "gemini-1.5-pro", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )
        assert result == ""


# ============================================================
# _call_openai
# ============================================================

class TestCallOpenAI:
    @pytest.mark.asyncio
    async def test_basic_response(self, llm_client):
        llm_client._openai.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("GPT says hi")
        )
        result = await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == "GPT says hi"

    @pytest.mark.asyncio
    async def test_system_messages_merged(self, llm_client):
        llm_client._openai.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "system", "content": "Be concise"},
            {"role": "user", "content": "Hi"},
        ]
        await llm_client._call_openai("gpt-4", messages, 0.7, 2048)

        call_kwargs = llm_client._openai.chat.completions.create.call_args.kwargs
        system_msgs = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert "Be helpful" in system_msgs[0]["content"]
        assert "Be concise" in system_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_uses_max_completion_tokens(self, llm_client):
        llm_client._openai.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response()
        )
        await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 4096
        )
        call_kwargs = llm_client._openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_empty_choices_returns_empty(self, llm_client):
        llm_client._openai.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response_empty()
        )
        result = await llm_client._call_openai(
            "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
        )
        assert result == ""

    @pytest.mark.asyncio
    async def test_500_raises(self, llm_client):
        import openai as _openai
        import httpx
        llm_client._openai.chat.completions.create = AsyncMock(
            side_effect=_openai.InternalServerError(
                message="Internal Server Error",
                response=httpx.Response(500, request=httpx.Request("POST", "https://api.openai.com")),
                body=None,
            )
        )
        with pytest.raises(_openai.InternalServerError):
            await llm_client._call_openai(
                "gpt-4", [{"role": "user", "content": "Hi"}], 0.7, 2048
            )


# ============================================================
# call_llm dispatch
# ============================================================

class TestCallLlmDispatch:
    @pytest.mark.asyncio
    async def test_dispatches_to_anthropic(self, llm_client):
        llm_client._anthropic.messages.create = AsyncMock(
            return_value=_mock_anthropic_response("anthropic")
        )
        result = await llm_client.call_llm("claude-3-sonnet", [{"role": "user", "content": "Hi"}])
        assert result == "anthropic"

    @pytest.mark.asyncio
    async def test_dispatches_to_google(self, llm_client):
        mock_resp = _mock_google_response("google")
        with patch("saga.llm.client.asyncio.to_thread", new_callable=AsyncMock, return_value=mock_resp):
            result = await llm_client.call_llm("gemini-1.5-pro", [{"role": "user", "content": "Hi"}])
        assert result == "google"

    @pytest.mark.asyncio
    async def test_dispatches_to_openai(self, llm_client):
        llm_client._openai.chat.completions.create = AsyncMock(
            return_value=_mock_openai_response("openai")
        )
        result = await llm_client.call_llm("gpt-4", [{"role": "user", "content": "Hi"}])
        assert result == "openai"


# ============================================================
# _prepare_* static methods
# ============================================================

class TestPrepareMessages:
    def test_prepare_anthropic_system_hoisted(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        system_parts, non_system = LLMClient._prepare_anthropic_messages(messages)
        assert len(system_parts) == 1
        assert system_parts[0]["text"] == "sys"
        assert len(non_system) == 1
        assert non_system[0]["role"] == "user"

    def test_prepare_anthropic_cache_control(self):
        messages = [
            {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}},
            {"role": "assistant", "content": "hi", "cache_control": {"type": "ephemeral"}},
        ]
        system_parts, non_system = LLMClient._prepare_anthropic_messages(messages)
        assert system_parts[0]["cache_control"] == {"type": "ephemeral"}
        assert non_system[0]["content"][0]["cache_control"] == {"type": "ephemeral"}

    def test_prepare_openai_merges_system(self):
        messages = [
            {"role": "system", "content": "A"},
            {"role": "system", "content": "B"},
            {"role": "user", "content": "hi"},
        ]
        merged = LLMClient._prepare_openai_messages(messages)
        assert len(merged) == 2  # 1 system + 1 user
        assert "A" in merged[0]["content"]
        assert "B" in merged[0]["content"]

    def test_prepare_google_system_instruction(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        contents, sys_instr = LLMClient._prepare_google_messages(messages)
        assert sys_instr == "sys"
        assert len(contents) == 2
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"  # assistant -> model
