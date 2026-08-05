"""LLMClient 추상화 (스펙 §8) — Dreamer 전용. 동기 경로는 LLM 0콜."""
import asyncio
import json

import httpx
import pytest

from dreaming.llm import OpenAICompatLLM


def _client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler),
                             base_url="http://fake")


def test_complete_returns_message_content():
    def handler(request):
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"facts": []}'}}]})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    assert asyncio.run(llm.complete("sys", "usr")) == '{"facts": []}'


def test_sends_model_messages_temperature_zero():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "ok"}}]})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    asyncio.run(llm.complete("sys", "usr"))
    assert seen["model"] == "flash"
    assert seen["temperature"] == 0
    assert seen["messages"] == [{"role": "system", "content": "sys"},
                                {"role": "user", "content": "usr"}]


def test_raises_on_http_error():
    def handler(request):
        return httpx.Response(500, json={"error": "boom"})

    llm = OpenAICompatLLM("http://fake", "k", "flash", client=_client_with(handler))
    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(llm.complete("sys", "usr"))
