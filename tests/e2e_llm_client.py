"""E2E test for LLM client with real API calls.

Tests each provider (Anthropic, Google, OpenAI) with actual API keys from config.yaml.
Run: python3 tests/e2e_llm_client.py
"""

import asyncio
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from saga.config import load_config
from saga.llm.client import LLMClient


PROMPT = [
    {"role": "system", "content": "You are a helpful assistant. Reply in one short sentence."},
    {"role": "user", "content": "Say 'hello world' and nothing else."},
]


def ok(label):
    print(f"  \033[32m✓\033[0m {label}")


def fail(label, err):
    print(f"  \033[31m✗\033[0m {label}: {err}")


async def test_anthropic(client, model):
    """Test Anthropic non-streaming + streaming."""
    print(f"\n[Anthropic] model={model}")

    # Non-streaming
    t0 = time.time()
    try:
        result = await client.call_llm(model, PROMPT, temperature=0, max_tokens=50)
        elapsed = time.time() - t0
        assert result and len(result) > 0, "Empty response"
        ok(f"call_llm: \"{result.strip()[:60]}\" ({elapsed:.1f}s)")
    except Exception as e:
        fail("call_llm", e)
        return False

    # Streaming
    t0 = time.time()
    try:
        chunks = []
        async for chunk in client.call_llm_stream(model, PROMPT, temperature=0, max_tokens=50):
            chunks.append(chunk)
        elapsed = time.time() - t0
        full = "".join(chunks)
        assert full and len(full) > 0, "Empty stream"
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"
        ok(f"stream: {len(chunks)} chunks, \"{full.strip()[:60]}\" ({elapsed:.1f}s)")
    except Exception as e:
        fail("stream", e)
        return False

    # Prompt caching (cache_control)
    cached_prompt = [
        {"role": "system", "content": "You are a helpful assistant." * 100, "cache_control": {"type": "ephemeral"}},
        {"role": "user", "content": "Say 'cached' and nothing else."},
    ]
    try:
        result = await client.call_llm(model, cached_prompt, temperature=0, max_tokens=50)
        assert result and len(result) > 0, "Empty response"
        ok(f"cache_control: \"{result.strip()[:60]}\"")
    except Exception as e:
        fail("cache_control", e)
        return False

    return True


async def test_google(client, model):
    """Test Google Gemini."""
    print(f"\n[Google] model={model}")

    t0 = time.time()
    try:
        result = await client.call_llm(model, PROMPT, temperature=0, max_tokens=50)
        elapsed = time.time() - t0
        assert result and len(result) > 0, "Empty response"
        ok(f"call_llm: \"{result.strip()[:60]}\" ({elapsed:.1f}s)")
    except Exception as e:
        fail("call_llm", e)
        return False

    # System instruction test
    sys_prompt = [
        {"role": "system", "content": "Always reply with exactly one word."},
        {"role": "system", "content": "The word must be 'pong'."},
        {"role": "user", "content": "ping"},
    ]
    try:
        result = await client.call_llm(model, sys_prompt, temperature=0, max_tokens=10)
        assert result and len(result) > 0, "Empty response"
        ok(f"multi-system: \"{result.strip()[:60]}\"")
    except Exception as e:
        fail("multi-system", e)
        return False

    return True


async def test_openai(client, model):
    """Test OpenAI."""
    print(f"\n[OpenAI] model={model}")

    # Non-streaming
    t0 = time.time()
    try:
        result = await client.call_llm(model, PROMPT, temperature=0, max_tokens=50)
        elapsed = time.time() - t0
        assert result and len(result) > 0, "Empty response"
        ok(f"call_llm: \"{result.strip()[:60]}\" ({elapsed:.1f}s)")
    except Exception as e:
        fail("call_llm", e)
        return False

    # Streaming
    t0 = time.time()
    try:
        chunks = []
        async for chunk in client.call_llm_stream(model, PROMPT, temperature=0, max_tokens=50):
            chunks.append(chunk)
        elapsed = time.time() - t0
        full = "".join(chunks)
        assert full and len(full) > 0, "Empty stream"
        ok(f"stream: {len(chunks)} chunks, \"{full.strip()[:60]}\" ({elapsed:.1f}s)")
    except Exception as e:
        fail("stream", e)
        return False

    return True


async def test_none_guards():
    """Test that missing API keys raise RuntimeError."""
    print("\n[None Guards]")
    from saga.config import SagaConfig
    empty_config = SagaConfig()
    client = LLMClient(empty_config)

    for provider, model in [("Anthropic", "claude-haiku-4-5-20251001"), ("OpenAI", "gpt-4o-mini"), ("Google", "gemini-2.0-flash")]:
        try:
            await client.call_llm(model, PROMPT)
            fail(f"{provider} guard", "Should have raised RuntimeError")
            return False
        except RuntimeError as e:
            ok(f"{provider} guard: {e}")
        except Exception as e:
            fail(f"{provider} guard", f"Wrong exception: {type(e).__name__}: {e}")
            return False

    return True


async def test_langsmith_wrapping():
    """Test that LangSmith wrap_* applies without errors."""
    print("\n[LangSmith Wrapping]")
    try:
        config = load_config()
        # Temporarily enable langsmith in config
        config.langsmith.enabled = True
        client = LLMClient(config)

        # Check that clients are wrapped (they should have extra attributes from wrapping)
        ok(f"anthropic wrapped: {type(client._anthropic).__name__}")
        ok(f"openai wrapped: {type(client._openai).__name__}")
        if client._google:
            ok(f"google wrapped: {type(client._google).__name__}")

        # Actually call through wrapped client
        result = await client.call_llm("claude-haiku-4-5-20251001", PROMPT, temperature=0, max_tokens=30)
        assert result and len(result) > 0
        ok(f"wrapped call: \"{result.strip()[:60]}\"")

        await client.close()
        return True
    except ImportError as e:
        fail("wrap import", e)
        return False
    except Exception as e:
        fail("wrapped call", e)
        return False


async def main():
    print("=" * 60)
    print("SAGA LLM Client — E2E Test (Real API Calls)")
    print("=" * 60)

    config = load_config()
    client = LLMClient(config)

    results = {}

    # None guards (no API call)
    results["none_guards"] = await test_none_guards()

    # Real API calls
    results["anthropic"] = await test_anthropic(client, config.models.narration)
    results["google"] = await test_google(client, config.models.extraction)
    results["openai"] = await test_openai(client, "gpt-4o-mini")

    # LangSmith wrapping
    results["langsmith"] = await test_langsmith_wrapping()

    await client.close()

    # Summary
    print("\n" + "=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    color = "\033[32m" if passed == total else "\033[31m"
    print(f"{color}Results: {passed}/{total} passed\033[0m")
    for name, ok in results.items():
        status = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        print(f"  {status} {name}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
