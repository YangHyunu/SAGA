"""LLM API client supporting Anthropic, Google, and OpenAI providers.

Uses official SDKs with optional LangSmith tracing via wrap_* wrappers.
"""
import asyncio
import logging
from typing import AsyncIterator

import anthropic
import openai
from langsmith import traceable

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config):
        self.config = config
        # Sanitize API keys (strip whitespace/tabs that may leak from env vars or YAML)
        if config.api_keys.anthropic:
            config.api_keys.anthropic = config.api_keys.anthropic.strip()
        if config.api_keys.openai:
            config.api_keys.openai = config.api_keys.openai.strip()
        if config.api_keys.google:
            config.api_keys.google = config.api_keys.google.strip()

        # --- Anthropic ---
        ant_client = None
        if config.api_keys.anthropic:
            ant_client = anthropic.AsyncAnthropic(
                api_key=config.api_keys.anthropic,
                default_headers={
                    "anthropic-beta": "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11"
                },
            )

        # --- OpenAI ---
        oai_client = None
        if config.api_keys.openai:
            oai_client = openai.AsyncOpenAI(api_key=config.api_keys.openai)

        # --- Google GenAI ---
        google_client = None
        if config.api_keys.google:
            try:
                from google import genai
                google_client = genai.Client(api_key=config.api_keys.google)
            except ImportError:
                logger.warning("[LLMClient] google-genai not installed, Google provider unavailable")

        # --- LangSmith wrapping (conditional) ---
        if config.langsmith.enabled:
            try:
                from langsmith.wrappers import wrap_anthropic, wrap_openai
                if ant_client:
                    ant_client = wrap_anthropic(ant_client)
                if oai_client:
                    oai_client = wrap_openai(oai_client)
            except ImportError:
                logger.warning("[LangSmith] langsmith.wrappers not available")
            try:
                from langsmith.wrappers import wrap_gemini
                if google_client:
                    google_client = wrap_gemini(google_client)
            except ImportError:
                logger.debug("[LangSmith] wrap_gemini not available")

        self._anthropic = ant_client
        self._openai = oai_client
        self._google = google_client
        self._last_cache_stats = {"cache_read": 0, "cache_create": 0}
        self._last_usage = {"model": "", "input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_create": 0}

    async def close(self):
        if self._anthropic and hasattr(self._anthropic, "close"):
            await self._anthropic.close()
        if self._openai and hasattr(self._openai, "close"):
            await self._openai.close()

    @traceable(name="llm.call", run_type="llm")
    async def call_llm(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 8192, **kwargs) -> str:
        """Call LLM API and return text response. Auto-detects provider from model name."""
        provider = self._detect_provider(model)
        if provider == "anthropic":
            return await self._call_anthropic(model, messages, temperature, max_tokens, **kwargs)
        elif provider == "google":
            return await self._call_google(model, messages, temperature, max_tokens, **kwargs)
        else:
            return await self._call_openai(model, messages, temperature, max_tokens, **kwargs)

    async def call_llm_stream(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 8192, **kwargs) -> AsyncIterator[str]:
        """Streaming LLM call. Yields text chunks."""
        provider = self._detect_provider(model)
        if provider == "anthropic":
            async for chunk in self._stream_anthropic(model, messages, temperature, max_tokens, **kwargs):
                yield chunk
        elif provider == "google":
            async for chunk in self._stream_google(model, messages, temperature, max_tokens, **kwargs):
                yield chunk
        else:
            async for chunk in self._stream_openai(model, messages, temperature, max_tokens, **kwargs):
                yield chunk

    def _detect_provider(self, model: str) -> str:
        model_lower = model.lower()
        if "claude" in model_lower or "anthropic" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower or "google" in model_lower:
            return "google"
        else:
            return "openai"

    # --- Message preparation helpers ---

    @staticmethod
    def _to_text(content) -> str:
        """Extract plain text from content (string or multimodal array)."""
        if isinstance(content, str):
            return content
        return "\n".join(
            p.get("text", "") for p in content if p.get("type") == "text"
        )

    @staticmethod
    def _convert_openai_to_anthropic_parts(content_parts: list[dict]) -> list[dict]:
        """Convert OpenAI multimodal content parts to Anthropic format.

        OpenAI image_url → Anthropic image source (base64 or url).
        """
        result = []
        for part in content_parts:
            ptype = part.get("type")
            if ptype == "text":
                result.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    # data:image/png;base64,<data>
                    header, _, data = url.partition(",")
                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                    result.append({
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": data},
                    })
                else:
                    result.append({
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    })
        return result

    @staticmethod
    def _convert_openai_to_google_parts(content_parts: list[dict]) -> list[dict]:
        """Convert OpenAI multimodal content parts to Gemini format."""
        result = []
        for part in content_parts:
            ptype = part.get("type")
            if ptype == "text":
                result.append({"text": part.get("text", "")})
            elif ptype == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    header, _, data = url.partition(",")
                    media_type = header.split(":")[1].split(";")[0] if ":" in header else "image/png"
                    result.append({"inline_data": {"mime_type": media_type, "data": data}})
        return result

    @staticmethod
    def _prepare_anthropic_messages(messages: list[dict]):
        """Split messages into system content blocks and non-system messages.

        Returns (system_parts, non_system) preserving cache_control.
        Handles multimodal content arrays (image_url → Anthropic image source).
        """
        system_parts = []
        non_system = []
        for msg in messages:
            if msg["role"] == "system":
                text = LLMClient._to_text(msg["content"])
                part = {"type": "text", "text": text}
                if msg.get("cache_control"):
                    part["cache_control"] = msg["cache_control"]
                system_parts.append(part)
            else:
                content = msg["content"]
                if isinstance(content, list):
                    # Multimodal: convert OpenAI format → Anthropic format
                    parts = LLMClient._convert_openai_to_anthropic_parts(content)
                    if msg.get("cache_control") and parts:
                        parts[-1]["cache_control"] = msg["cache_control"]
                    entry = {"role": msg["role"], "content": parts}
                elif msg.get("cache_control"):
                    entry = {
                        "role": msg["role"],
                        "content": [
                            {"type": "text", "text": content, "cache_control": msg["cache_control"]}
                        ],
                    }
                else:
                    entry = {"role": msg["role"], "content": content}
                non_system.append(entry)
        return system_parts, non_system

    @staticmethod
    def _prepare_openai_messages(messages: list[dict]):
        """Merge multiple system messages into one (OpenAI only supports one).

        Multimodal content arrays are passed through as-is (OpenAI native format).
        """
        merged = []
        system_parts = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(LLMClient._to_text(m["content"]))
            else:
                merged.append({"role": m["role"], "content": m["content"]})
        if system_parts:
            merged.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        return merged

    @staticmethod
    def _prepare_google_messages(messages: list[dict]):
        """Convert to Gemini format: contents list + system instruction string.

        Multimodal content arrays are converted to Gemini parts format.
        """
        contents = []
        system_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(LLMClient._to_text(msg["content"]))
            else:
                role = "user" if msg["role"] == "user" else "model"
                content = msg["content"]
                if isinstance(content, list):
                    parts = LLMClient._convert_openai_to_google_parts(content)
                else:
                    parts = [{"text": content}]
                contents.append({"role": role, "parts": parts})
        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return contents, system_instruction

    # --- Anthropic ---

    async def _call_anthropic(self, model, messages, temperature, max_tokens, **kwargs):
        """Call Anthropic Messages API with 3-BP prompt caching support."""
        if not self._anthropic:
            raise RuntimeError("Anthropic API key not configured. Set api_keys.anthropic in config.yaml or $ANTHROPIC_API_KEY")
        system_parts, non_system = self._prepare_anthropic_messages(messages)

        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
        }
        if system_parts:
            create_kwargs["system"] = system_parts
        if kwargs.get("top_p") is not None:
            create_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            create_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        try:
            response = await self._anthropic.messages.create(**create_kwargs)
        except anthropic.AuthenticationError:
            raise RuntimeError(
                "Anthropic API 인증 실패 (401). API 키 확인: config.yaml api_keys.anthropic 또는 $ANTHROPIC_API_KEY"
            )

        # Log prompt caching stats
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self._last_cache_stats = {"cache_read": cache_read, "cache_create": cache_create}
        self._last_usage = {
            "model": model, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read": cache_read, "cache_create": cache_create,
        }
        if cache_read or cache_create:
            logger.info(
                f"[Cache] input={usage.input_tokens} cache_read={cache_read} "
                f"cache_create={cache_create} output={usage.output_tokens}"
            )

        return "".join(block.text for block in response.content if hasattr(block, "text"))

    async def _stream_anthropic(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from Anthropic with 3-BP prompt caching support."""
        if not self._anthropic:
            raise RuntimeError("Anthropic API key not configured. Set api_keys.anthropic in config.yaml or $ANTHROPIC_API_KEY")
        system_parts, non_system = self._prepare_anthropic_messages(messages)

        create_kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
        }
        if system_parts:
            create_kwargs["system"] = system_parts
        if kwargs.get("top_p") is not None:
            create_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            create_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        async with self._anthropic.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            # Get final message for cache stats after stream completes
            response = await stream.get_final_message()

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self._last_cache_stats = {"cache_read": cache_read, "cache_create": cache_create}
        self._last_usage = {
            "model": model, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read": cache_read, "cache_create": cache_create,
        }
        if cache_read or cache_create:
            logger.info(
                f"[Cache] input={usage.input_tokens} cache_read={cache_read} "
                f"cache_create={cache_create} output={usage.output_tokens}"
            )

    # --- Google ---

    async def _call_google(self, model, messages, temperature, max_tokens, **kwargs):
        """Call Google Gemini API via google-genai SDK."""
        if not self._google:
            raise RuntimeError("Google API key not configured. Set api_keys.google in config.yaml or $GOOGLE_API_KEY")
        from google import genai

        contents, system_instruction = self._prepare_google_messages(messages)

        gen_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if kwargs.get("top_p") is not None:
            gen_config["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            gen_config["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            gen_config["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            gen_config["stop_sequences"] = [stop] if isinstance(stop, str) else stop
        if kwargs.get("response_mime_type") is not None:
            gen_config["response_mime_type"] = kwargs["response_mime_type"]
        if system_instruction:
            gen_config["system_instruction"] = system_instruction

        config_obj = genai.types.GenerateContentConfig(**gen_config)

        for attempt in range(4):
            try:
                response = await asyncio.to_thread(
                    self._google.models.generate_content,
                    model=model,
                    contents=contents,
                    config=config_obj,
                )
                # Track usage for Google
                usage_meta = getattr(response, "usage_metadata", None)
                if usage_meta:
                    self._last_usage = {
                        "model": model,
                        "input_tokens": getattr(usage_meta, "prompt_token_count", 0) or 0,
                        "output_tokens": getattr(usage_meta, "candidates_token_count", 0) or 0,
                        "cache_read": 0, "cache_create": 0,
                    }
                return response.text or ""
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 2 ** attempt
                    logger.warning(f"[Google] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                    await asyncio.sleep(wait)
                    continue
                raise
        return ""

    async def _stream_google(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from Google (simplified — Gemini streaming)."""
        # For simplicity, use non-streaming and yield full result
        result = await self._call_google(model, messages, temperature, max_tokens, **kwargs)
        yield result

    # --- OpenAI ---

    async def _call_openai(self, model, messages, temperature, max_tokens, **kwargs):
        """Call OpenAI Chat Completions API."""
        if not self._openai:
            raise RuntimeError("OpenAI API key not configured. Set api_keys.openai in config.yaml or $OPENAI_API_KEY")
        merged = self._prepare_openai_messages(messages)

        create_kwargs = {
            "model": model,
            "messages": merged,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if kwargs.get("top_p") is not None:
            create_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            create_kwargs["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            create_kwargs["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            create_kwargs["stop"] = kwargs["stop"]

        response = await self._openai.chat.completions.create(**create_kwargs)

        # Track usage for OpenAI
        if response.usage:
            self._last_usage = {
                "model": model,
                "input_tokens": response.usage.prompt_tokens or 0,
                "output_tokens": response.usage.completion_tokens or 0,
                "cache_read": getattr(response.usage, "prompt_tokens_details", None) and getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0,
                "cache_create": 0,
            }

        choices = response.choices
        return choices[0].message.content if choices else ""

    async def _stream_openai(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from OpenAI."""
        if not self._openai:
            raise RuntimeError("OpenAI API key not configured. Set api_keys.openai in config.yaml or $OPENAI_API_KEY")
        merged = self._prepare_openai_messages(messages)

        create_kwargs = {
            "model": model,
            "messages": merged,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
        if kwargs.get("top_p") is not None:
            create_kwargs["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            create_kwargs["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            create_kwargs["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            create_kwargs["stop"] = kwargs["stop"]

        stream = await self._openai.chat.completions.create(**create_kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
