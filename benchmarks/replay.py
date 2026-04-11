#!/usr/bin/env python3
"""SAGA Benchmark Replay — RisuAI 채팅 로그를 SAGA 프록시에 턴별 재전송하여 캐시 절감률 측정.

Usage:
    python benchmarks/replay.py --chat path/to/risuai_export.json --reset --turns 100
    python benchmarks/replay.py --chat path/to/risuai_export.json --dry-run
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

import httpx

# ─── RisuAI Chat Parser ───


def parse_risu_chat(path: str) -> list[dict]:
    """Parse RisuAI chat export JSON → list of {role, content} dicts.

    RisuAI format: {"type": "risuChat", "ver": 2, "data": {"message": [...]}}
    Each message: {"role": "user"|"char", "data": "...", ...}
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if raw.get("type") != "risuChat":
        raise ValueError(f"Not a RisuAI chat export (type={raw.get('type')})")

    risu_msgs = raw["data"]["message"]
    messages = []
    for m in risu_msgs:
        role = m["role"]
        content = m.get("data", "")
        if not content:
            continue
        if role == "char":
            role = "assistant"
        elif role != "user":
            continue  # skip system or other roles
        messages.append({"role": role, "content": content})
    return messages


def load_charx_system(path: str) -> str:
    """Extract system prompt from .charx file (ZIP with card.json).

    Combines description + enabled lorebook entries to build a system prompt
    similar to what RisuAI sends.
    """
    import zipfile
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("card.json"))

    card = data.get("data", {})
    parts = []

    name = card.get("name", "")
    desc = card.get("description", "")
    if name:
        parts.append(f"[Character: {name}]")
    if desc:
        parts.append(desc)

    # Lorebook entries
    entries = card.get("character_book", {}).get("entries", [])
    lore_parts = []
    for e in entries:
        if e.get("enabled", True):
            content = e.get("content", "")
            if content:
                entry_name = e.get("name", "")
                if entry_name:
                    lore_parts.append(f"[{entry_name}]\n{content}")
                else:
                    lore_parts.append(content)
    if lore_parts:
        parts.append("[Lorebook]\n" + "\n\n".join(lore_parts))

    system_prompt = card.get("system_prompt", "")
    if system_prompt:
        parts.append(system_prompt)

    full = "\n\n".join(parts)
    return full


# ─── Auto-generate user messages ───

AUTO_GEN_SYSTEM = (
    "You are role-playing as a user in an interactive RP session. "
    "Continue the conversation naturally based on the context. "
    "Write a short in-character action or dialogue (1-3 sentences in Korean). "
    "Do NOT break character or add meta-commentary."
)


async def auto_generate_user_message(
    recent_messages: list[dict],
    google_api_key: str = "",
    model: str = "gemini-2.5-flash-lite",
) -> str:
    """Generate a synthetic user message via direct Google API (bypasses SAGA).

    Uses Google Generative Language API directly to avoid polluting SAGA session
    metrics with auto-gen LLM calls.
    """
    # Use last 4 messages as context for generation
    context = recent_messages[-4:] if len(recent_messages) >= 4 else recent_messages

    # Build Gemini-compatible request
    contents = []
    for m in context:
        role = "user" if m["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": m["content"]}]})

    # Prepend system instruction as first user message
    if contents and contents[0]["role"] == "model":
        contents.insert(0, {"role": "user", "parts": [{"text": AUTO_GEN_SYSTEM}]})
    else:
        contents[0]["parts"].insert(0, {"text": AUTO_GEN_SYSTEM + "\n\n"})

    api_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}"
        f":generateContent?key={google_api_key}"
    )

    async with httpx.AsyncClient(timeout=60) as gen_client:
        resp = await gen_client.post(
            api_url,
            json={
                "contents": contents,
                "generationConfig": {"maxOutputTokens": 200, "temperature": 0.8},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


# ─── SSE Stream Consumer ───


async def consume_sse_stream(response: httpx.Response) -> str:
    """Consume SSE stream, return concatenated content."""
    full_text = []
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload.strip() == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                full_text.append(content)
        except json.JSONDecodeError:
            continue
    return "".join(full_text)


# ─── Main Replay Loop ───


async def replay(args):
    """Execute the benchmark replay."""
    # Parse chat
    chat_messages = parse_risu_chat(args.chat)
    total_in_file = len(chat_messages)
    user_turns_in_file = sum(1 for m in chat_messages if m["role"] == "user")

    # Determine turn count
    target_turns = args.turns if args.turns > 0 else user_turns_in_file
    replay_turns = min(user_turns_in_file, target_turns)
    auto_gen_turns = max(0, target_turns - replay_turns)

    print(f"\n{'='*60}")
    print(f"  SAGA Benchmark Replay")
    print(f"{'='*60}")
    print(f"  Chat file:      {args.chat}")
    print(f"  Messages:       {total_in_file} ({user_turns_in_file} turns)")
    print(f"  Target turns:   {target_turns}")
    print(f"  Replay turns:   {replay_turns}")
    print(f"  Auto-gen turns: {auto_gen_turns}")
    print(f"  Curator:        {'ON' if args.curator else 'OFF'}")
    print(f"  Base URL:       {args.base_url}")
    print(f"  Model:          {args.model or '(server default)'}")
    print(f"{'='*60}\n")

    if args.dry_run:
        # Estimate token count
        total_chars = sum(len(m["content"]) for m in chat_messages)
        est_tokens = total_chars // 3  # rough estimate: 3 chars ≈ 1 token (Korean)
        print(f"  [DRY RUN] Estimated total tokens: ~{est_tokens:,}")
        print(f"  [DRY RUN] No HTTP calls made.")
        return

    # HTTP client setup
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    session_id = f"bench-{uuid.uuid4().hex[:8]}"
    headers["X-SAGA-Session-ID"] = session_id

    async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=30)) as client:
        # Health check
        try:
            health = await client.get(f"{args.base_url}/health")
            health.raise_for_status()
        except Exception as e:
            print(f"  [ERROR] SAGA server not reachable: {e}")
            sys.exit(1)

        # Reset if requested
        if args.reset:
            print("  [RESET] Clearing all databases...")
            resp = await client.post(f"{args.base_url}/api/reset-all", headers=headers)
            if resp.status_code == 200:
                print(f"  [RESET] Done: {resp.json()}")
            else:
                print(f"  [RESET] Warning: {resp.status_code} {resp.text}")
            await asyncio.sleep(1)

        # Curator toggle reminder
        if not args.curator:
            print("  [INFO] --curator flag not set. Curator on/off is determined by server config.yaml")
        else:
            print("  [INFO] --curator flag set. Ensure config.yaml has curator.enabled: true")

        # Load system prompt from charx if provided
        system_message = None
        if args.charx:
            system_text = load_charx_system(args.charx)
            system_message = {"role": "system", "content": system_text}
            print(f"  [CHARX] Loaded system prompt: {len(system_text)} chars (~{len(system_text)//3} tokens)")

        # Build message pairs: [(user_msg, assistant_msg), ...]
        pairs = []
        i = 0
        while i < len(chat_messages):
            if chat_messages[i]["role"] == "user":
                user_msg = chat_messages[i]
                assistant_msg = chat_messages[i + 1] if i + 1 < len(chat_messages) and chat_messages[i + 1]["role"] == "assistant" else None
                pairs.append((user_msg, assistant_msg))
                i += 2 if assistant_msg else 1
            else:
                i += 1

        # Accumulated conversation for sending
        accumulated = []
        if system_message:
            accumulated.append(system_message)
        turn_results = []
        t_benchmark_start = time.monotonic()

        # Phase 1: Replay from chat log
        for turn_idx in range(replay_turns):
            if turn_idx >= len(pairs):
                break

            user_msg, assistant_msg = pairs[turn_idx]
            accumulated.append({"role": "user", "content": user_msg["content"]})

            turn_start = time.monotonic()
            response_text = await _send_turn(
                client, args.base_url, headers, accumulated, args.model, turn_idx + 1, "replay",
            )
            turn_elapsed = time.monotonic() - turn_start

            # Use original assistant text (discard LLM response)
            if assistant_msg:
                accumulated.append({"role": "assistant", "content": assistant_msg["content"]})
            elif response_text:
                accumulated.append({"role": "assistant", "content": response_text})

            turn_results.append({
                "turn": turn_idx + 1,
                "type": "replay",
                "elapsed_s": round(turn_elapsed, 2),
                "user_input": user_msg["content"],
                "original_assistant": (assistant_msg["content"] if assistant_msg else ""),
                "saga_response": (response_text or ""),
            })

            # Brief pause to let Sub-B process
            await asyncio.sleep(0.5)

        # Phase 2: Auto-generate remaining turns
        for turn_idx in range(auto_gen_turns):
            turn_num = replay_turns + turn_idx + 1
            print(f"  [Auto-gen] Generating user message for turn {turn_num}...")

            try:
                user_text = await auto_generate_user_message(
                    accumulated, google_api_key=args.google_api_key,
                )
            except Exception as e:
                print(f"  [Auto-gen] Failed to generate user message: {e}")
                break

            accumulated.append({"role": "user", "content": user_text})

            turn_start = time.monotonic()
            response_text = await _send_turn(
                client, args.base_url, headers, accumulated, args.model, turn_num, "auto-gen",
            )
            turn_elapsed = time.monotonic() - turn_start

            if response_text:
                accumulated.append({"role": "assistant", "content": response_text})
            else:
                print(f"  [Auto-gen] No response for turn {turn_num}, stopping.")
                break

            turn_results.append({
                "turn": turn_num,
                "type": "auto-gen",
                "elapsed_s": round(turn_elapsed, 2),
                "user_input": user_text,
                "original_assistant": "",
                "saga_response": (response_text or ""),
            })

            await asyncio.sleep(0.5)

        total_elapsed = time.monotonic() - t_benchmark_start

        # ─── Collect Results ───
        print(f"\n{'='*60}")
        print(f"  Collecting cost data...")
        print(f"{'='*60}\n")

        report = await _collect_report(
            client, args.base_url, headers, session_id, turn_results, total_elapsed,
        )

        _print_summary(report)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"\n  [OUTPUT] Full report saved to: {args.output}")


async def _send_turn(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict,
    messages: list[dict],
    model: str | None,
    turn_num: int,
    turn_type: str,
    max_retries: int = 3,
) -> str:
    """Send one turn to SAGA, consume SSE stream, return response text."""
    body = {
        "model": model or "claude-haiku-4-5-20251001",
        "messages": messages,
        "stream": True,
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        try:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=body,
                timeout=httpx.Timeout(300, connect=30),
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    print(f"  [Turn {turn_num}] HTTP {response.status_code}: {error_body[:200]}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return ""

                text = await consume_sse_stream(response)
                est_input = sum(len(m["content"]) for m in messages) // 3
                print(
                    f"  [{turn_type:>8}] Turn {turn_num:3d} | "
                    f"~{est_input:,} input tok | "
                    f"resp {len(text):,} chars | "
                    f"{'OK' if text else 'EMPTY'}"
                )
                return text

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            print(f"  [Turn {turn_num}] Timeout (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"  [Turn {turn_num}] Failed after {max_retries} attempts")
                return ""
        except Exception as e:
            print(f"  [Turn {turn_num}] Error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                return ""
    return ""


async def _collect_report(
    client: httpx.AsyncClient,
    base_url: str,
    headers: dict,
    session_id: str,
    turn_results: list[dict],
    total_elapsed: float,
) -> dict:
    """Collect cost/cache data from SAGA API and build report."""
    report = {
        "session_id": session_id,
        "total_turns": len(turn_results),
        "total_elapsed_s": round(total_elapsed, 1),
        "turns": turn_results,
    }

    # Session cost summary
    try:
        resp = await client.get(f"{base_url}/api/cost/{session_id}", headers=headers)
        if resp.status_code == 200:
            report["cost_summary"] = resp.json()
    except Exception as e:
        report["cost_summary_error"] = str(e)

    # Global cost summary (includes sub_b, curator breakdown by model)
    try:
        resp = await client.get(f"{base_url}/api/cost", headers=headers)
        if resp.status_code == 200:
            report["cost_global"] = resp.json()
    except Exception as e:
        report["cost_global_error"] = str(e)

    # Per-call cache data
    try:
        resp = await client.get(
            f"{base_url}/api/metrics/cache",
            headers=headers,
            params={"limit": 500},
        )
        if resp.status_code == 200:
            cache_data = resp.json().get("data", [])
            # Filter by this session only (endpoint returns all sessions)
            cache_data = [c for c in cache_data if c.get("session_id") == session_id]
            report["cache_timeline"] = cache_data
    except Exception as e:
        report["cache_timeline_error"] = str(e)

    # Latency data
    try:
        resp = await client.get(
            f"{base_url}/api/metrics/latency",
            headers=headers,
            params={"limit": 500},
        )
        if resp.status_code == 200:
            report["latency"] = resp.json()
    except Exception as e:
        report["latency_error"] = str(e)

    return report


def _print_summary(report: dict):
    """Print human-readable benchmark summary."""
    cost = report.get("cost_summary", {})
    total_turns = report.get("total_turns", 0)
    elapsed = report.get("total_elapsed_s", 0)

    total_cost = cost.get("total_cost_usd", 0)
    total_savings = cost.get("total_savings_usd", 0)
    baseline = cost.get("cost_without_cache_usd", 0)
    savings_pct = cost.get("cache_savings_percent", 0)
    cache_read = cost.get("total_cache_read_tokens", 0)
    total_input = cost.get("total_input_tokens", 0)

    # Sub-B overhead from global cost (by_model breakdown)
    global_cost = report.get("cost_global", {})
    by_model = global_cost.get("by_model", [])
    sub_b_cost = 0
    main_cost = 0
    for m in by_model:
        model_name = m.get("model", "").lower()
        if "flash" in model_name or "gemini" in model_name:
            sub_b_cost += m.get("cost_usd", 0)
        else:
            main_cost += m.get("cost_usd", 0)

    net_savings = total_savings - sub_b_cost
    net_pct = (net_savings / baseline * 100) if baseline > 0 else 0

    # Cache hit rate from timeline
    cache_timeline = report.get("cache_timeline", [])
    if cache_timeline:
        avg_cache_hit = sum(c.get("cache_hit_pct", 0) for c in cache_timeline) / len(cache_timeline)
    else:
        avg_cache_hit = 0

    # Latency
    latency = report.get("latency", {}).get("summary", {})
    ttft_avg = latency.get("ttft", {}).get("avg", 0)
    total_ms_avg = latency.get("total", {}).get("avg", 0)

    print(f"\n{'='*60}")
    print(f"  SAGA Benchmark Results")
    print(f"{'='*60}")
    print(f"  Session:           {report.get('session_id', 'N/A')}")
    print(f"  Total turns:       {total_turns}")
    print(f"  Elapsed:           {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"")
    print(f"  --- Cost ---")
    print(f"  Baseline (no cache): ${baseline:.4f}")
    print(f"  Actual cost:         ${total_cost:.4f}")
    print(f"  Cache savings:       ${total_savings:.4f} ({savings_pct:.1f}%)")
    print(f"  Sub-B overhead:      ${sub_b_cost:.4f}")
    print(f"  Net savings:         ${net_savings:.4f} ({net_pct:.1f}%)")
    print(f"")
    print(f"  --- Cache ---")
    print(f"  Total input tokens:  {total_input:,}")
    print(f"  Cache read tokens:   {cache_read:,}")
    print(f"  Avg cache hit rate:  {avg_cache_hit:.1f}%")
    print(f"")
    print(f"  --- Latency ---")
    print(f"  Avg TTFT:            {ttft_avg:.0f}ms")
    print(f"  Avg total:           {total_ms_avg:.0f}ms")
    print(f"")
    print(f"  --- Model Breakdown ---")
    for m in by_model:
        print(f"    {m['model']:40s} calls={m['calls']:3d}  cost=${m['cost_usd']:.4f}  saved=${m['savings_usd']:.4f}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="SAGA Benchmark Replay — RisuAI 채팅 리플레이로 캐시 절감률 측정",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--chat", required=True, help="RisuAI chat export JSON path")
    parser.add_argument("--charx", default="", help="RisuAI .charx file path (provides system prompt for cache)")
    parser.add_argument("--reset", action="store_true", help="Reset all DBs before replay (calls /api/reset-all)")
    parser.add_argument("--turns", type=int, default=0, help="Target turn count (0 = all turns in file)")
    parser.add_argument("--curator", action="store_true", help="Remind to enable Curator (manual config toggle)")
    parser.add_argument("--base-url", default="http://localhost:8000", help="SAGA server URL")
    parser.add_argument("--api-key", default="", help="Bearer API key (if auth enabled)")
    parser.add_argument("--model", default="", help="Override model for main LLM calls")
    parser.add_argument("--google-api-key", default=os.environ.get("GOOGLE_API_KEY", ""),
                        help="Google API key for auto-gen (default: $GOOGLE_API_KEY)")
    parser.add_argument("--output", default="", help="Output JSON report path")
    parser.add_argument("--dry-run", action="store_true", help="Parse and show plan without making HTTP calls")

    args = parser.parse_args()
    asyncio.run(replay(args))


if __name__ == "__main__":
    main()
