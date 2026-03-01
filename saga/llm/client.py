"""LLM API client supporting Anthropic, Google, and OpenAI providers."""
import httpx
import json
import logging
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, config):
        self.config = config
        self._http = httpx.AsyncClient(timeout=60.0)
        # Sanitize API keys (strip whitespace/tabs that may leak from env vars or YAML)
        if config.api_keys.anthropic:
            config.api_keys.anthropic = config.api_keys.anthropic.strip()
        if config.api_keys.openai:
            config.api_keys.openai = config.api_keys.openai.strip()
        if config.api_keys.google:
            config.api_keys.google = config.api_keys.google.strip()

    async def close(self):
        await self._http.aclose()

    async def call_llm(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> str:
        """Call LLM API and return text response. Auto-detects provider from model name."""
        provider = self._detect_provider(model)
        if provider == "anthropic":
            return await self._call_anthropic(model, messages, temperature, max_tokens, **kwargs)
        elif provider == "google":
            return await self._call_google(model, messages, temperature, max_tokens, **kwargs)
        else:
            return await self._call_openai(model, messages, temperature, max_tokens, **kwargs)

    async def call_llm_stream(self, model: str, messages: list[dict], temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> AsyncIterator[str]:
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

    async def _call_anthropic(self, model, messages, temperature, max_tokens, **kwargs):
        """Call Anthropic Messages API with 3-BP prompt caching support."""
        api_key = self.config.api_keys.anthropic
        system_parts = []
        non_system = []

        for msg in messages:
            if msg["role"] == "system":
                part = {"type": "text", "text": msg["content"]}
                if msg.get("cache_control"):
                    part["cache_control"] = msg["cache_control"]
                system_parts.append(part)
            else:
                if msg.get("cache_control"):
                    entry = {
                        "role": msg["role"],
                        "content": [
                            {"type": "text", "text": msg["content"], "cache_control": msg["cache_control"]}
                        ],
                    }
                else:
                    entry = {"role": msg["role"], "content": msg["content"]}
                non_system.append(entry)

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
        }
        if system_parts:
            body["system"] = system_parts
        # Anthropic supports top_p and stop_sequences
        if kwargs.get("top_p") is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        resp = await self._http.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11",
                "content-type": "application/json",
            },
            json=body,
        )
        if resp.status_code == 401:
            raise RuntimeError(f"Anthropic API 인증 실패 (401). API 키 확인: config.yaml api_keys.anthropic 또는 $ANTHROPIC_API_KEY")
        resp.raise_for_status()
        data = resp.json()

        # Log prompt caching stats
        usage = data.get("usage", {})
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_create = usage.get("cache_creation_input_tokens", 0)
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        if cache_read or cache_create:
            logger.info(
                f"[Cache] input={input_tokens} cache_read={cache_read} cache_create={cache_create} output={output_tokens}"
            )

        return "".join(block.get("text", "") for block in data.get("content", []))

    async def _call_google(self, model, messages, temperature, max_tokens, **kwargs):
        """Call Google Gemini API."""
        api_key = self.config.api_keys.google
        # Convert messages to Gemini format (merge all system messages)
        contents = []
        system_parts = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        system_instruction = "\n\n".join(system_parts) if system_parts else None

        gen_config = {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        }
        if kwargs.get("top_p") is not None:
            gen_config["topP"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            gen_config["frequencyPenalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            gen_config["presencePenalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            gen_config["stopSequences"] = [stop] if isinstance(stop, str) else stop

        body = {
            "contents": contents,
            "generationConfig": gen_config,
        }
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        import asyncio as _aio
        for attempt in range(4):
            resp = await self._http.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=body,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning(f"[Google] 429 rate limit, retrying in {wait}s (attempt {attempt+1}/4)")
                await _aio.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts)
            return ""
        resp.raise_for_status()
        return ""

    async def _call_openai(self, model, messages, temperature, max_tokens, **kwargs):
        """Call OpenAI Chat Completions API."""
        api_key = self.config.api_keys.openai
        # Merge multiple system messages into one (OpenAI only supports one)
        merged = []
        system_parts = []
        for m in messages:
            if m["role"] == "system":
                system_parts.append(m["content"])
            else:
                merged.append({"role": m["role"], "content": m["content"]})
        if system_parts:
            merged.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
        body = {
            "model": model,
            "messages": merged,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if kwargs.get("top_p") is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            body["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            body["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            body["stop"] = kwargs["stop"]
        resp = await self._http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body,
        )
        if resp.status_code != 200:
            logger.error(f"[OpenAI] {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        return choices[0]["message"]["content"] if choices else ""

    async def _stream_anthropic(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from Anthropic with 3-BP prompt caching support."""
        api_key = self.config.api_keys.anthropic
        system_parts = []
        non_system = []

        for msg in messages:
            if msg["role"] == "system":
                part = {"type": "text", "text": msg["content"]}
                if msg.get("cache_control"):
                    part["cache_control"] = msg["cache_control"]
                system_parts.append(part)
            else:
                if msg.get("cache_control"):
                    entry = {
                        "role": msg["role"],
                        "content": [
                            {"type": "text", "text": msg["content"], "cache_control": msg["cache_control"]}
                        ],
                    }
                else:
                    entry = {"role": msg["role"], "content": msg["content"]}
                non_system.append(entry)

        body = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": non_system,
            "stream": True,
        }
        if system_parts:
            body["system"] = system_parts
        if kwargs.get("top_p") is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("stop") is not None:
            stop = kwargs["stop"]
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        cache_stats = {}

        async with self._http.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "anthropic-beta": "prompt-caching-2024-07-31,extended-cache-ttl-2025-04-11",
                "content-type": "application/json",
            },
            json=body,
        ) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    event_type = data.get("type", "")

                    if event_type == "message_start":
                        usage = data.get("message", {}).get("usage", {})
                        cache_stats["input"] = usage.get("input_tokens", 0)
                        cache_stats["cache_read"] = usage.get("cache_read_input_tokens", 0)
                        cache_stats["cache_create"] = usage.get("cache_creation_input_tokens", 0)
                    elif event_type == "message_delta":
                        cache_stats["output"] = data.get("usage", {}).get("output_tokens", 0)
                    elif event_type == "content_block_delta":
                        text = data.get("delta", {}).get("text", "")
                        if text:
                            yield text

        # Log cache stats after stream completes
        if cache_stats.get("cache_read") or cache_stats.get("cache_create"):
            logger.info(
                f"[Cache] input={cache_stats.get('input', 0)} "
                f"cache_read={cache_stats.get('cache_read', 0)} "
                f"cache_create={cache_stats.get('cache_create', 0)} "
                f"output={cache_stats.get('output', 0)}"
            )

    async def _stream_openai(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from OpenAI."""
        api_key = self.config.api_keys.openai
        body = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "temperature": temperature, "max_tokens": max_tokens, "stream": True,
        }
        if kwargs.get("top_p") is not None:
            body["top_p"] = kwargs["top_p"]
        if kwargs.get("frequency_penalty") is not None:
            body["frequency_penalty"] = kwargs["frequency_penalty"]
        if kwargs.get("presence_penalty") is not None:
            body["presence_penalty"] = kwargs["presence_penalty"]
        if kwargs.get("stop") is not None:
            body["stop"] = kwargs["stop"]
        async with self._http.stream("POST", "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=body) as resp:
            async for line in resp.aiter_lines():
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        yield delta["content"]

    async def _stream_google(self, model, messages, temperature, max_tokens, **kwargs):
        """Stream from Google (simplified — Gemini streaming)."""
        # For simplicity, use non-streaming and yield full result
        result = await self._call_google(model, messages, temperature, max_tokens, **kwargs)
        yield result
