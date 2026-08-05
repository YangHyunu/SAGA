"""dreaming/llm.py — LLMClient 추상화 (스펙 §8).

Dreamer 전용. 동기 경로는 LLM 0콜 (스펙 §2.2).
OpenAI 호환 chat/completions 하나로 OpenRouter·Gemini(OpenAI compat) 전부 커버.
"""

from __future__ import annotations

from typing import Optional, Protocol

import httpx


class LLMClient(Protocol):
    async def complete(self, system: str, user: str) -> str: ...


class OpenAICompatLLM:
    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        r = await self._client.post("/chat/completions", json={
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        })
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
