"""dreaming/upstream.py — OpenAI 호환 업스트림 (기본 OpenRouter).

cache_control은 OpenRouter 규약대로 content part 안에 넣는다.
업스트림이 OpenAI 호환이므로 스트리밍은 SSE 바이트 passthrough다.
"""

from __future__ import annotations

from typing import AsyncIterator, Dict, List, Optional

import httpx


def to_wire(messages: List[Dict]) -> List[Dict]:
    out = []
    for m in messages:
        cc = m.get("cache_control")
        if cc is None:
            out.append(dict(m))
            continue
        mm = {k: v for k, v in m.items() if k != "cache_control"}
        mm["content"] = [{"type": "text", "text": m["content"],
                          "cache_control": cc}]
        out.append(mm)
    return out


class OpenAIUpstream:
    def __init__(self, base_url: str, api_key: str, timeout: float = 300.0,
                 client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url, timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def complete(self, payload: Dict) -> Dict:
        r = await self._client.post("/chat/completions", json=payload)
        r.raise_for_status()
        return r.json()

    async def stream(self, payload: Dict) -> AsyncIterator[bytes]:
        async with self._client.stream(
                "POST", "/chat/completions", json=payload) as r:
            r.raise_for_status()
            async for chunk in r.aiter_bytes():
                yield chunk
