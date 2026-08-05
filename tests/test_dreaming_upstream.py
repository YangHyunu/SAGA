"""OpenAI 호환 업스트림 — cache_control content part 변환 + passthrough."""
import asyncio
import json

import httpx
import pytest

from dreaming.upstream import OpenAIUpstream, to_wire


def test_to_wire_moves_cache_control_into_content_part():
    msgs = [
        {"role": "system", "content": "봇 정의",
         "cache_control": {"type": "ephemeral", "ttl": "5m"}},
        {"role": "user", "content": "안녕"},
    ]
    wire = to_wire(msgs)
    assert wire[0]["content"] == [{
        "type": "text", "text": "봇 정의",
        "cache_control": {"type": "ephemeral", "ttl": "5m"}}]
    assert "cache_control" not in wire[0]          # 메시지 레벨에선 제거
    assert wire[1] == {"role": "user", "content": "안녕"}   # 무마킹은 그대로
    assert "cache_control" in msgs[0]              # 원본 불변


def _upstream(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                               base_url="http://up")
    return OpenAIUpstream("http://up", "k", client=client)


def test_complete_posts_payload_and_returns_json():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "어서 와."}}]})

    up = _upstream(handler)
    resp = asyncio.run(up.complete({"model": "m", "messages": []}))
    assert seen["model"] == "m"
    assert resp["choices"][0]["message"]["content"] == "어서 와."


def test_complete_raises_on_http_error():
    def handler(request):
        return httpx.Response(429, json={"error": "rate"})

    up = _upstream(handler)
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(up.complete({"model": "m", "messages": []}))


def test_stream_yields_raw_bytes():
    body = b'data: {"choices":[{"delta":{"content":"\xec\x96\xb4\xec\x84\x9c"}}]}\n\ndata: [DONE]\n\n'

    def handler(request):
        return httpx.Response(200, content=body)

    up = _upstream(handler)

    async def collect():
        return b"".join([c async for c in up.stream({"model": "m"})])

    assert asyncio.run(collect()) == body
