#!/usr/bin/env python3
"""SAGA E2E Integration Test — Full pipeline verification.

Tests the complete SAGA pipeline:
  Sub-A (Context Builder) → LLM → Sub-B (Post-Turn) → Curator → Letta

Usage:
  # With charx file
  python3 tests/e2e_integration.py --charx /path/to/character.charx

  # Default (위지소연 fallback)
  python3 tests/e2e_integration.py

  # Options
  python3 tests/e2e_integration.py \\
    --charx /path/to/char.charx \\
    --saga-url http://localhost:8000 \\
    --api-key saga-test-key-2026 \\
    --turns 10 \\
    --letta-url http://localhost:8283
"""

import argparse
import asyncio
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)

# ---------------------------------------------------------------------------
# charx parser
# ---------------------------------------------------------------------------

def parse_charx(charx_path: str) -> dict:
    """Parse a .charx file (ZIP → card.json) and extract RP data.

    Returns dict with keys: name, system_prompt, lorebook_text, first_mes, scenario_prompts
    """
    with zipfile.ZipFile(charx_path) as z:
        card = json.loads(z.read("card.json"))

    data = card.get("data", {})
    name = data.get("name", "Unknown")

    # Build system prompt: description is the main character/world text
    # system_prompt field is often empty in charx; description carries everything
    description = data.get("description", "")
    system_prompt_field = data.get("system_prompt", "").strip()
    scenario = data.get("scenario", "").strip()
    personality = data.get("personality", "").strip()

    parts = []
    if description:
        parts.append(description)
    if personality:
        parts.append(f"\nPersonality:\n{personality}")
    if scenario:
        parts.append(f"\nScenario:\n{scenario}")
    if system_prompt_field:
        parts.append(f"\n{system_prompt_field}")

    system_prompt = "\n".join(parts)

    # Lorebook: combine enabled entries
    book = data.get("character_book", {})
    entries = book.get("entries", [])
    lorebook_parts = []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        content = entry.get("content", "").strip()
        entry_name = entry.get("name", "").strip()
        if content:
            if entry_name:
                lorebook_parts.append(f"### {entry_name}\n{content}")
            else:
                lorebook_parts.append(content)

    lorebook_text = "\n\n".join(lorebook_parts) if lorebook_parts else ""

    first_mes = data.get("first_mes", "")

    print(f"  Parsed charx: {name}")
    print(f"    System prompt: {len(system_prompt)} chars")
    print(f"    Lorebook: {len(lorebook_parts)} entries, {len(lorebook_text)} chars")
    print(f"    First message: {len(first_mes)} chars")

    return {
        "name": name,
        "system_prompt": system_prompt,
        "lorebook_text": lorebook_text,
        "first_mes": first_mes,
    }


# ---------------------------------------------------------------------------
# Fallback: 위지소연 hardcoded data (from ab_state_instruction.py)
# ---------------------------------------------------------------------------

FALLBACK_SYSTEM_PROMPT = """Name: 위지소연 (魏遲甦蘇)

Identity: Shinn-yeo (神女, Divine Maiden), Guardian of the Mountain Village.

Residence: A secluded, traditional Hanok estate located deep within the mountains of Joseon, overlooking a small, superstitious village.

Age/Birthdate: 31 / 9 February
Race: Human / Korean
Sex: Female

Physical Appearance:
- Height: 여덟 자 (八尺) [226 cm]
- Hair/Eye: Long, straight black hair / Dark grey eyes.

Personality Traits:
- Emotionless: Barely feels any emotions.
- Kuudere: Outwardly cold, stoic, and emotionless.
- Maternal (Suppressed): Beneath her icy exterior lies a deep well of maternal instinct.
- Duty-Bound: She takes her role as the village's spiritual guardian seriously.
- Lonely: A profound sense of isolation permeates her existence.

Communication:
- Tone: Monotone, flat, devoid of inflection.
- Dialogue Style: Terse and direct. Uses the fewest words possible.

Occupation: Shaman / Spiritual Guardian of the Village.

Backstory: Orphaned at a young age, Soyeon was taken in by the village shaman, Wiji Rye. She grew up isolated, her life revolving around rituals, chores, and spiritual training. When Rye passed away, Soyeon inherited her title and responsibilities."""

FALLBACK_LOREBOOK = """### World Overview
Genres/Tags: Korean Mythology, Romance, Sitcom, Love Comedy, Alternate Universe, Slice of Life
Background: Medieval Korea

### Named NPCs
- 당채련 (唐綵蓮): Village herbalist and physician. Sharp-tongued, caring, secretly romantic.
- 원향 (元香): Chaeryeon's niece, village tomboy. Energetic, impulsive, idolizes Soyeon.
- 적예령 (赤睿領): Wealthy widow, benevolent matriarch. Immensely strong, hosts village banquets.
- 적월화 (赤月華): Yeryeong's eldest daughter. Rebellious, cynical, white-haired albino.
- 천백화 (天白華): Yeryeong's second daughter. Timid, gentle, blue-eyed beauty."""

FALLBACK_FIRST_MES = (
    "눈이 시리도록 파란 하늘 아래, 겨울 산맥의 능선이 날카롭게 그어져 있었다.\n\n"
    "그 산자락 깊숙한 곳, 기와지붕 위로 하얀 눈이 소복이 쌓인 위지 가문의 고택이 자리하고 있었다.\n\n"
    "\"…….\"\n\n"
    "위지소연은 대청마루에 앉아 멍하니 마당을 응시했다."
)

# ---------------------------------------------------------------------------
# User simulator — LLM generates contextual user replies after turn 1
# ---------------------------------------------------------------------------

USER_SIM_SYSTEM = """당신은 한국어 텍스트 RP의 숙련된 유저를 시뮬레이션합니다.
상대 캐릭터(assistant)의 마지막 응답을 읽고, 몰입감 있는 유저 행동/대사를 2~4문장으로 생성하세요.

행동 스타일:
- 한국어로 작성. 행동은 (괄호) 안에, 대사는 직접 작성
- 대화를 앞으로 진전시키세요: 새로운 장소 탐색, NPC와 상호작용, 감정 표현 등
- 매 턴 다른 종류의 행동: 탐색, 대화, 전투, 아이템 사용, 감정 토로, 과거 회상 등

극적 요소 (3~4턴마다 하나씩 섞어주세요):
- 위험한 선택: 함정에 손을 넣는다, 어두운 길을 택한다, 적에게 도발한다
- 감정 폭발: 갑자기 분노, 슬픔, 공포에 사로잡힌다
- 과거 회상: "(문득 고향에서의 기억이 스쳐지나간다)" 같은 내면 묘사
- NPC에게 예상치 못한 행동: 안아준다, 뺨을 때린다, 무릎을 꿇는다
- 주변 환경 활용: 물건을 집어든다, 벽을 두드린다, 냄새를 맡는다
- 유머: 긴장된 상황에서 부적절한 농담, 헛기침

진행 패턴:
- 초반(1~5턴): 조심스러운 탐색과 관찰
- 중반(6~15턴): 적극적 상호작용, 관계 형성, 갈등 발생
- 후반(16턴~): 대담한 결정, 클라이맥스 행동, 승부수

절대 금지:
- 상대 캐릭터의 행동/대사를 대신 쓰지 마세요
- 설명이나 메타 코멘트 없이 유저 행동/대사만 출력"""

# First-turn opener: used only when no first_mes exists
OPENER_SOYEON = "마을 끝에 있는 커다란 저택이 보인다. 문을 두드려 본다."
OPENER_GENERIC = "(주위를 둘러보며) 여기가 어디죠? 처음 보는 곳이네요."
OPENER_DUNGEON = "어둠 속에서 눈을 뜬다. 차가운 돌바닥 위에 누워 있다. 주위를 둘러본다."

# ---------------------------------------------------------------------------
# Scenario: 던전 보스 (Dungeon Boss)
# ---------------------------------------------------------------------------

DUNGEON_SYSTEM_PROMPT = """Name: 아크리쉬 (Akrish), The Undying Flame

Identity: Ancient Dragon Lord, Final Boss of the Obsidian Spire dungeon.

Setting: A vast underground dungeon complex called the Obsidian Spire (흑요석 첨탑), located beneath a cursed mountain range. The dungeon has 7 floors filled with traps, monsters, and dark magic.

Appearance:
- Height: 3m in humanoid form, 30m in dragon form
- Features: Crimson scales visible on neck and arms, molten gold eyes, silver-white hair
- Always wears obsidian armor inscribed with ancient runes

Personality:
- Prideful: Views mortals as entertainment, not threats
- Cunning: Prefers mind games and traps over direct combat
- Honorable: Respects those who show true courage
- Ancient: Speaks in archaic, formal Korean mixed with dragon tongue

Communication:
- Tone: Deep, resonant, commanding
- Dialogue: Formal, archaic Korean (하오체/하게체). Occasionally uses dragon language phrases.

Backstory: Akrish was once a guardian dragon who protected the land. Betrayed by the humans he protected, he retreated underground and built the Obsidian Spire as both fortress and tomb. He has waited 1000 years for a worthy challenger.

Abilities: Fire manipulation, illusion magic, shapeshifting, time distortion within the dungeon.

NPCs in the dungeon:
- 세라핀 (Seraphine): A cursed elf priestess trapped on Floor 3. Can heal and provide lore.
- 가르텐 (Garten): A dwarven blacksmith on Floor 5. Can upgrade weapons.
- 미라주 (Mirage): Akrish's shadow clone that tests adventurers on Floor 6.

Rules:
- The user is an adventurer who has entered the dungeon
- Track HP (starting 100), inventory, floor progress
- Combat uses dice-like mechanics described narratively
- Each floor has unique challenges and guardians
- Akrish watches through magical mirrors and occasionally taunts the adventurer"""

DUNGEON_LOREBOOK = """### Obsidian Spire (흑요석 첨탑)
7-floor dungeon beneath the Cursed Mountains. Each floor has a theme:
- Floor 1: Hall of Echoes (음향의 전당) — Sound-based traps and bat swarms
- Floor 2: Poison Gardens (독의 정원) — Toxic flora, venomous creatures
- Floor 3: Frozen Cathedral (얼어붙은 성당) — Ice magic, Seraphine's prison
- Floor 4: Labyrinth of Mirrors (거울의 미궁) — Illusion puzzles
- Floor 5: Forge of Souls (영혼의 대장간) — Fire elementals, Garten's workshop
- Floor 6: Shadow Realm (그림자 영역) — Mirage's domain, darkness combat
- Floor 7: Throne of Flames (불꽃의 왕좌) — Akrish's lair, final battle

### Combat System
- HP starts at 100, damage ranges from 5-30 per hit
- Healing items: 하급 포션(+20HP), 중급 포션(+50HP), 세라핀의 축복(full heal, once)
- Critical hits on natural 20 (narratively described)
- Death at 0 HP triggers respawn at last checkpoint

### Dragon Language
- "Krath vol'thun" = "Welcome, mortal"
- "Zhar'keth" = "Burn"
- "Vol'thar ish'ka" = "You have proven worthy"
- "Ash'gul mori" = "Death awaits" """

DUNGEON_FIRST_MES = (
    "흑요석 첨탑의 입구가 당신 앞에 우뚝 솟아 있었다.\n\n"
    "거대한 흑요석 문 위에 고대 용언으로 새겨진 글자가 붉게 빛나고 있었다.\n\n"
    "\"Krath vol'thun... 어서 오라, 필멸자여.\"\n\n"
    "어디선가 울려오는 깊은 목소리가 던전 전체를 진동시켰다. "
    "입구 양쪽의 화염 석상이 저절로 불을 밝혔다."
)


def _read_config_api_key(provider: str) -> str:
    """Read API key from config.yaml as fallback when env var is not set."""
    try:
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        key = cfg.get("api_keys", {}).get(provider, "")
        # Skip env var references like "${ANTHROPIC_API_KEY}"
        if key and not key.startswith("${"):
            return key
    except Exception:
        pass
    return ""


async def generate_user_input(
    client: httpx.AsyncClient,
    conversation_history: list[dict],
    model: str,
) -> str:
    """Use LLM to generate the next user input based on conversation context.

    Calls LLM API directly (bypasses SAGA) to avoid session contamination.
    The simulator's system prompt would otherwise pollute canonical_system_prompt.
    """
    # Build condensed recent history (last 6 messages = 3 exchanges)
    recent_msgs: list[dict] = []
    for msg in conversation_history[-6:]:
        if msg["role"] == "system":
            continue
        recent_msgs.append({"role": msg["role"], "content": msg["content"][:1000]})

    # Ensure last message is from assistant so LLM generates user reply
    if not recent_msgs or recent_msgs[-1]["role"] != "assistant":
        recent_msgs.append({"role": "assistant", "content": "(대기 중)"})

    recent_msgs.append({
        "role": "user",
        "content": "위 대화에 이어서 유저의 다음 행동/대사를 생성하세요.",
    })

    try:
        if model.startswith("claude"):
            # --- Anthropic Messages API ---
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                # Fallback: read from config.yaml
                api_key = _read_config_api_key("anthropic")
            if not api_key:
                return "(주위를 둘러본다)"
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "system": USER_SIM_SYSTEM,
                    "messages": recent_msgs,
                    "temperature": 0.9,
                    "max_tokens": 200,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            generated = data["content"][0]["text"].strip()
        else:
            # --- OpenAI-compatible API (GPT, Gemini via OpenAI compat, etc.) ---
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                api_key = _read_config_api_key("openai")
            if not api_key:
                return "(주위를 둘러본다)"
            sim_messages = [{"role": "system", "content": USER_SIM_SYSTEM}] + recent_msgs
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": sim_messages,
                    "temperature": 0.9,
                    "max_tokens": 200,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            generated = data["choices"][0]["message"]["content"].strip()

        # Clean up: remove any meta-commentary or quotes
        generated = re.sub(r'^(유저|User|플레이어)\s*[:：]\s*', '', generated)
        return generated if generated else "(주위를 둘러본다)"
    except Exception as e:
        print(f"    [sim] LLM direct call failed: {e}")
        return f"(주위를 살펴보며) 그래서, 다음엔 뭘 하면 될까요?"


# ---------------------------------------------------------------------------
# SSE streaming helpers
# ---------------------------------------------------------------------------

async def stream_chat_completion(
    client: httpx.AsyncClient,
    url: str,
    headers: dict,
    body: dict,
) -> dict:
    """Send a streaming chat completion request and collect the full response.

    Returns: {content, latency_ms, chunks, error}
    """
    t_start = time.time()
    content = ""
    chunks = 0
    finish_reason = None

    try:
        async with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=180,
        ) as resp:
            if resp.status_code != 200:
                body_text = ""
                async for chunk in resp.aiter_text():
                    body_text += chunk
                return {
                    "content": "",
                    "latency_ms": (time.time() - t_start) * 1000,
                    "chunks": 0,
                    "error": f"HTTP {resp.status_code}: {body_text[:300]}",
                }

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(payload)
                    delta = chunk_data.get("choices", [{}])[0].get("delta", {})
                    delta_content = delta.get("content", "")
                    if delta_content:
                        content += delta_content
                        chunks += 1
                    fr = chunk_data.get("choices", [{}])[0].get("finish_reason")
                    if fr:
                        finish_reason = fr
                except json.JSONDecodeError:
                    pass

    except Exception as e:
        return {
            "content": content,
            "latency_ms": (time.time() - t_start) * 1000,
            "chunks": chunks,
            "error": str(e),
        }

    return {
        "content": content,
        "latency_ms": (time.time() - t_start) * 1000,
        "chunks": chunks,
        "finish_reason": finish_reason,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Phase 1: Connectivity checks
# ---------------------------------------------------------------------------

async def phase1_connectivity(
    saga_url: str, api_key: str, letta_url: str
) -> list[dict]:
    """Basic connectivity and auth verification."""
    results = []

    async with httpx.AsyncClient(timeout=10) as client:
        # 1a. Health check
        try:
            r = await client.get(
                f"{saga_url}/api/status",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            ok = r.status_code == 200
            detail = r.json() if ok else r.text[:200]
            results.append({
                "check": "SAGA health (/api/status)",
                "pass": ok,
                "detail": detail,
            })
        except Exception as e:
            results.append({
                "check": "SAGA health (/api/status)",
                "pass": False,
                "detail": str(e),
            })

        # 1b. Auth verification (wrong key should → 401)
        try:
            r = await client.get(
                f"{saga_url}/api/status",
                headers={"Authorization": "Bearer WRONG-KEY-12345"},
            )
            ok = r.status_code == 401
            results.append({
                "check": "Auth rejection (bad key → 401)",
                "pass": ok,
                "detail": f"Got {r.status_code}" + ("" if ok else f" (expected 401)"),
            })
        except Exception as e:
            results.append({
                "check": "Auth rejection (bad key → 401)",
                "pass": False,
                "detail": str(e),
            })

        # 1c. Letta health
        try:
            r = await client.get(f"{letta_url}/v1/health/")
            ok = r.status_code == 200
            results.append({
                "check": "Letta health (/v1/health/)",
                "pass": ok,
                "detail": r.text[:200] if ok else f"HTTP {r.status_code}",
            })
        except Exception as e:
            results.append({
                "check": "Letta health (/v1/health/)",
                "pass": False,
                "detail": f"Letta unreachable: {e}",
            })

    return results


# ---------------------------------------------------------------------------
# Phase 2: Multi-turn RP simulation (SSE streaming)
# ---------------------------------------------------------------------------

async def phase2_multi_turn(
    saga_url: str,
    api_key: str,
    char_data: dict,
    opener: str,
    num_turns: int,
    model: str,
    sim_model: str,
) -> dict:
    """Run multi-turn RP via SSE streaming with dynamic user simulation.

    Turn 1 uses a hardcoded opener. Turns 2+ use an LLM to generate
    contextual user inputs based on the assistant's previous response.
    """
    system_content = char_data["system_prompt"]
    if char_data["lorebook_text"]:
        system_content += "\n\n" + char_data["lorebook_text"]

    messages: list[dict] = [
        {"role": "system", "content": system_content},
    ]
    if char_data.get("first_mes"):
        messages.append({"role": "assistant", "content": char_data["first_mes"]})

    headers = {"Authorization": f"Bearer {api_key}"}
    turns: list[dict] = []

    async with httpx.AsyncClient() as client:
        for i in range(num_turns):
            # --- Generate user input ---
            if i == 0:
                prompt = opener
            else:
                print(f"    [sim] Generating user input for turn {i+1}...")
                prompt = await generate_user_input(
                    client, messages, sim_model,
                )

            messages.append({"role": "user", "content": prompt})

            # --- Send to SAGA (SSE streaming) ---
            body = {
                "model": model,
                "messages": messages,
                "stream": True,
                "temperature": 0.7,
                "max_tokens": 8192,
            }

            result = await stream_chat_completion(client, saga_url, headers, body)
            assistant_text = result["content"]

            turn_info = {
                "turn": i + 1,
                "user_input": prompt,
                "assistant_output": assistant_text,
                "response_length": len(assistant_text),
                "latency_ms": result["latency_ms"],
                "chunks": result["chunks"],
                "error": result.get("error"),
                "generated_input": i > 0,  # True if LLM-generated
            }
            turns.append(turn_info)

            if result.get("error"):
                print(f"  Turn {i+1}/{num_turns}: ERROR — {result['error'][:100]}")
                messages.append({"role": "assistant", "content": assistant_text or "(error)"})
            else:
                messages.append({"role": "assistant", "content": assistant_text})
                input_preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                print(
                    f"  Turn {i+1}/{num_turns}: "
                    f"User=[{input_preview}] → {len(assistant_text)}ch "
                    f"{result['latency_ms']:.0f}ms {result['chunks']}chunks"
                )

            # Wait for Sub-B async processing to complete
            if i < num_turns - 1:
                await asyncio.sleep(2)

    # Extra wait after last turn for Sub-B + Curator
    print("  Waiting 8s for Sub-B/Curator async processing...")
    await asyncio.sleep(8)

    return {
        "turns": turns,
        "total_turns": len(turns),
        "successful_turns": sum(1 for t in turns if not t.get("error")),
    }


# ---------------------------------------------------------------------------
# Phase 3: Pipeline verification
# ---------------------------------------------------------------------------

def _find_session_id(db_path: str) -> str | None:
    """Find the most recently updated session ID from SQLite."""
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def verify_live_state(session_id: str, cache_dir: str = "cache/sessions") -> dict:
    """Verify live_state.md file was generated with real data."""
    path = os.path.join(cache_dir, session_id, "live_state.md")
    checks = {}

    if not os.path.exists(path):
        return {
            "file_exists": False,
            "has_turn_frontmatter": False,
            "has_real_location": False,
            "has_recent_events": False,
            "no_phantom_defaults": True,
            "_path": path,
            "_content": "(file not found)",
        }

    content = open(path, "r", encoding="utf-8").read()
    checks["file_exists"] = True
    checks["has_turn_frontmatter"] = "turn:" in content
    # Real location = 위치: present and not "unknown"
    checks["has_real_location"] = (
        "위치:" in content and "unknown" not in content.lower()
    )
    checks["has_recent_events"] = "최근 이벤트" in content or "Turn" in content
    # No phantom HP defaults being rendered unnecessarily
    checks["no_phantom_defaults"] = "HP: 100/100" not in content
    checks["_path"] = path
    checks["_content_preview"] = content[:500]
    return checks


def verify_sqlite(session_id: str, expected_turns: int, db_path: str = "db/state.db") -> dict:
    """Verify SQLite state after RP simulation."""
    if not os.path.exists(db_path):
        return {"db_exists": False}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    checks = {"db_exists": True}

    try:
        # Session exists with expected turn count
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        checks["session_exists"] = row is not None
        if row:
            checks["turn_count"] = dict(row).get("turn_count", 0)
            checks["turn_count_ok"] = checks["turn_count"] >= expected_turns

        # Player character exists
        player = conn.execute(
            "SELECT * FROM characters WHERE session_id = ? AND is_player = 1",
            (session_id,),
        ).fetchone()
        checks["player_exists"] = player is not None
        if player:
            p = dict(player)
            checks["player_name"] = p.get("name", "")
            checks["player_location"] = p.get("location", "unknown")

        # Characters count
        char_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM characters WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checks["character_count"] = char_count["cnt"] if char_count else 0

        # Turn logs
        log_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM turn_log WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checks["turn_log_count"] = log_count["cnt"] if log_count else 0
        checks["turn_log_ok"] = checks["turn_log_count"] >= expected_turns

        # Events
        event_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checks["event_count"] = event_count["cnt"] if event_count else 0
        checks["has_events"] = checks["event_count"] >= 1

        # Relationships
        rel_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM relationships WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checks["relationship_count"] = rel_count["cnt"] if rel_count else 0

        # Locations
        loc_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM locations WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        checks["location_count"] = loc_count["cnt"] if loc_count else 0

    finally:
        conn.close()

    return checks


def verify_chromadb(session_id: str, db_path: str = "db/chroma") -> dict:
    """Verify ChromaDB episodes collection."""
    checks = {}
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        episodes = client.get_or_create_collection("episodes")

        result = episodes.get(where={"session_id": session_id})
        episode_count = len(result.get("ids", []))
        checks["episodes_accessible"] = True
        checks["episode_count"] = episode_count
        checks["episodes_ok"] = episode_count >= 3  # at least a few episodes

        if result.get("metadatas"):
            turns_covered = sorted(set(
                m.get("turn", 0) for m in result["metadatas"]
            ))
            checks["turns_with_episodes"] = turns_covered

        if result.get("documents"):
            checks["sample_episode"] = result["documents"][0][:200] if result["documents"] else ""

    except ImportError:
        checks["episodes_accessible"] = False
        checks["error"] = "chromadb not installed"
    except Exception as e:
        checks["episodes_accessible"] = False
        checks["error"] = str(e)

    return checks


async def verify_letta(session_id: str, letta_url: str) -> dict:
    """Verify Letta curator agent creation and memory blocks."""
    checks = {}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            # List agents
            r = await client.get(f"{letta_url}/v1/agents/")
            if r.status_code != 200:
                checks["letta_accessible"] = False
                checks["error"] = f"HTTP {r.status_code}"
                return checks

            agents = r.json()
            checks["letta_accessible"] = True
            checks["total_agents"] = len(agents)

            # Find curator agent for this session
            curator_agents = [
                a for a in agents
                if f"saga_curator_{session_id}" in a.get("name", "")
            ]
            checks["curator_agent_exists"] = len(curator_agents) > 0

            if curator_agents:
                agent = curator_agents[0]
                agent_id = agent.get("id", "")
                checks["curator_agent_name"] = agent.get("name", "")

                # Check memory blocks
                try:
                    mem_r = await client.get(
                        f"{letta_url}/v1/agents/{agent_id}/memory/"
                    )
                    if mem_r.status_code == 200:
                        memory = mem_r.json()
                        blocks = memory.get("memory", {}).get("blocks", [])
                        if not blocks:
                            blocks = memory.get("blocks", [])
                        checks["memory_blocks"] = len(blocks)
                        block_names = [b.get("label", b.get("name", "?")) for b in blocks]
                        checks["block_labels"] = block_names

                        # Check if narrative_summary has content
                        for block in blocks:
                            label = block.get("label", block.get("name", ""))
                            if "narrative" in label.lower():
                                value = block.get("value", "")
                                checks["narrative_summary_populated"] = bool(value.strip())
                                checks["narrative_summary_preview"] = value[:200]
                                break
                except Exception as e:
                    checks["memory_error"] = str(e)

    except Exception as e:
        checks["letta_accessible"] = False
        checks["error"] = str(e)

    return checks


def verify_prompt_caching(session_id: str, db_path: str = "db/state.db") -> dict:
    """Check turn logs for cache hit evidence."""
    checks = {}

    if not os.path.exists(db_path):
        checks["turn_logs_accessible"] = False
        return checks

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute(
            "SELECT turn_number, state_changes FROM turn_log WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ).fetchall()

        checks["turn_logs_accessible"] = True
        checks["total_logs"] = len(rows)

        # Look for cache metrics in state_changes JSON
        cache_hits = 0
        for row in rows:
            try:
                sc = json.loads(row["state_changes"] or "{}")
                if sc.get("cache_read_tokens", 0) > 0:
                    cache_hits += 1
            except (json.JSONDecodeError, TypeError):
                pass

        checks["turns_with_cache_hits"] = cache_hits

        # Also check log files for cache evidence
        log_dir = Path("logs/turns")
        if log_dir.exists():
            log_files = list(log_dir.glob(f"{session_id}*.json"))
            checks["turn_log_files"] = len(log_files)
    finally:
        conn.close()

    return checks


# ---------------------------------------------------------------------------
# Detail dump: full conversation + DB + .md contents
# ---------------------------------------------------------------------------

def collect_db_dump(session_id: str, db_path: str = "db/state.db") -> dict:
    """Dump all DB data for a session in human-readable form."""
    if not os.path.exists(db_path):
        return {"error": "db not found"}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    dump = {}

    try:
        # Session info
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        dump["session"] = dict(row) if row else None

        # Characters
        rows = conn.execute(
            "SELECT name, is_player, hp, max_hp, location, mood, updated_at FROM characters WHERE session_id = ? ORDER BY is_player DESC, name",
            (session_id,),
        ).fetchall()
        dump["characters"] = [dict(r) for r in rows]

        # Relationships
        rows = conn.execute(
            "SELECT from_name, to_name, rel_type, strength, updated_at FROM relationships WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        dump["relationships"] = [dict(r) for r in rows]

        # Events
        rows = conn.execute(
            "SELECT name, event_type, description, turn, importance FROM events WHERE session_id = ? ORDER BY turn",
            (session_id,),
        ).fetchall()
        dump["events"] = [dict(r) for r in rows]

        # Locations
        rows = conn.execute(
            "SELECT name, description, first_visit_turn FROM locations WHERE session_id = ? ORDER BY first_visit_turn",
            (session_id,),
        ).fetchall()
        dump["locations"] = [dict(r) for r in rows]

        # Turn logs (state_changes summary)
        rows = conn.execute(
            "SELECT turn_number, user_input, assistant_output, state_changes FROM turn_log WHERE session_id = ? ORDER BY turn_number",
            (session_id,),
        ).fetchall()
        turn_logs = []
        for r in rows:
            entry = {
                "turn": r["turn_number"],
                "user_input": (r["user_input"] or "")[:100],
                "assistant_output_length": len(r["assistant_output"] or ""),
            }
            try:
                sc = json.loads(r["state_changes"] or "{}")
                # Only keep non-empty/non-default fields
                entry["state_changes"] = {
                    k: v for k, v in sc.items()
                    if v and v != 0 and v != [] and v != "" and v is not None
                }
            except (json.JSONDecodeError, TypeError):
                entry["state_changes"] = {}
            turn_logs.append(entry)
        dump["turn_logs"] = turn_logs

        # World state KV
        rows = conn.execute(
            "SELECT key, value FROM world_state WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        dump["world_state"] = {r["key"]: r["value"] for r in rows}

    finally:
        conn.close()

    return dump


def collect_chromadb_episodes(session_id: str, db_path: str = "db/chroma") -> list[dict]:
    """Dump all episode summaries for a session."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=db_path)
        episodes = client.get_or_create_collection("episodes")
        result = episodes.get(where={"session_id": session_id})

        entries = []
        for i, doc in enumerate(result.get("documents", [])):
            meta = result["metadatas"][i] if i < len(result.get("metadatas", [])) else {}
            entries.append({
                "turn": meta.get("turn", "?"),
                "location": meta.get("location", "?"),
                "importance": meta.get("importance", "?"),
                "summary": doc[:300],
            })
        entries.sort(key=lambda x: x.get("turn", 0))
        return entries
    except Exception as e:
        return [{"error": str(e)}]


def write_detail_report(
    phase2: dict,
    phase3: dict,
    session_id: str | None,
    output_dir: str,
    char_name: str,
) -> str:
    """Write a detailed human-readable .md report with full conversation + DB state."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"e2e_detail_{ts}.md")
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append(f"# SAGA E2E Detail Report — {char_name}")
    lines.append(f"*Generated: {datetime.now().isoformat()}*\n")

    # ── Conversation Log ──
    lines.append("## 1. Conversation Log (턴별 대화)\n")
    for t in phase2.get("turns", []):
        src = "🤖 sim" if t.get("generated_input") else "📝 static"
        lines.append(f"### Turn {t['turn']} ({t['latency_ms']:.0f}ms, {t['response_length']}ch) [{src}]")
        lines.append(f"**User:**\n> {t['user_input']}\n")
        output = t.get("assistant_output", "")
        if output:
            # Truncate very long responses for readability
            preview = output if len(output) <= 1000 else output[:1000] + f"\n\n... ({len(output)}ch total, truncated)"
            lines.append(f"**Assistant:**\n{preview}\n")
        if t.get("error"):
            lines.append(f"**Error:** `{t['error']}`\n")
        lines.append("---\n")

    # ── live_state.md ──
    lines.append("## 2. live_state.md (최종 상태)\n")
    ls = phase3.get("live_state", {})
    content = ls.get("_content_preview", ls.get("_content", "(없음)"))
    if ls.get("file_exists"):
        # Read full content
        ls_path = ls.get("_path", "")
        if ls_path and os.path.exists(ls_path):
            content = open(ls_path, "r", encoding="utf-8").read()
        lines.append(f"```markdown\n{content}\n```\n")
    else:
        lines.append("*(파일 없음)*\n")

    # ── stable_prefix.md ──
    lines.append("## 3. stable_prefix.md\n")
    if session_id:
        stable_path = f"cache/sessions/{session_id}/stable_prefix.md"
        if os.path.exists(stable_path):
            stable_content = open(stable_path, "r", encoding="utf-8").read()
            preview = stable_content if len(stable_content) <= 2000 else stable_content[:2000] + "\n... (truncated)"
            lines.append(f"```markdown\n{preview}\n```\n")
        else:
            lines.append("*(파일 없음 — Curator 미실행 시 정상)*\n")
    else:
        lines.append("*(세션 없음)*\n")

    # ── SQLite DB Dump ──
    lines.append("## 4. SQLite DB 상태\n")
    if session_id:
        db = collect_db_dump(session_id)

        if db.get("session"):
            s = db["session"]
            lines.append(f"**Session**: `{s.get('id')}` | turns={s.get('turn_count')} | updated={s.get('updated_at')}\n")

        if db.get("characters"):
            lines.append("### Characters")
            for c in db["characters"]:
                tag = "🎮 PLAYER" if c.get("is_player") else "👤 NPC"
                lines.append(f"- [{tag}] **{c['name']}** — HP:{c['hp']}/{c['max_hp']} loc:{c['location']} mood:{c['mood']}")
            lines.append("")

        if db.get("relationships"):
            lines.append("### Relationships")
            for r in db["relationships"]:
                lines.append(f"- {r['from_name']} → {r['to_name']}: {r['rel_type']} (strength={r['strength']})")
            lines.append("")

        if db.get("events"):
            lines.append("### Events")
            for e in db["events"]:
                lines.append(f"- T{e['turn']}: [{e['event_type']}] {e['name']} — {e['description'][:100]} (importance={e['importance']})")
            lines.append("")

        if db.get("locations"):
            lines.append("### Locations")
            for loc in db["locations"]:
                lines.append(f"- {loc['name']} (first visit: T{loc['first_visit_turn']}) {loc['description'][:80]}")
            lines.append("")

        if db.get("world_state"):
            lines.append("### World State KV")
            for k, v in db["world_state"].items():
                lines.append(f"- `{k}` = `{v[:200]}`")
            lines.append("")

        if db.get("turn_logs"):
            lines.append("### Turn Logs (state_changes)")
            for tl in db["turn_logs"]:
                sc = tl.get("state_changes", {})
                sc_str = json.dumps(sc, ensure_ascii=False) if sc else "(empty)"
                lines.append(f"- T{tl['turn']}: output={tl['assistant_output_length']}ch | `{sc_str[:200]}`")
            lines.append("")
    else:
        lines.append("*(세션 없음)*\n")

    # ── ChromaDB Episodes ──
    lines.append("## 5. ChromaDB Episodes\n")
    if session_id:
        episodes = collect_chromadb_episodes(session_id)
        if episodes and not episodes[0].get("error"):
            for ep in episodes:
                lines.append(f"- **T{ep['turn']}** [{ep['location']}] imp={ep['importance']}: {ep['summary'][:200]}")
            lines.append("")
        else:
            err = episodes[0].get("error", "unknown") if episodes else "empty"
            lines.append(f"*(접근 실패: {err})*\n")
    else:
        lines.append("*(세션 없음)*\n")

    # ── Letta ──
    lines.append("## 6. Letta Curator\n")
    lt = phase3.get("letta", {})
    if lt.get("curator_agent_exists"):
        lines.append(f"- Agent: `{lt.get('curator_agent_name', '?')}`")
        lines.append(f"- Memory blocks: {lt.get('memory_blocks', 0)} — {lt.get('block_labels', [])}")
        if lt.get("narrative_summary_preview"):
            lines.append(f"- Narrative summary:\n  > {lt['narrative_summary_preview']}")
    else:
        lines.append(f"*(Curator agent 미생성 — {lt.get('error', 'unknown')})*")
    lines.append("")

    full_text = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return path


# ---------------------------------------------------------------------------
# Phase 4: Report generation
# ---------------------------------------------------------------------------

def print_report(
    phase1: list[dict],
    phase2: dict,
    phase3: dict,
    output_dir: str | None = None,
    session_id: str | None = None,
    char_name: str = "",
) -> dict:
    """Print and save the full E2E report."""
    print("\n" + "=" * 70)
    print("  SAGA E2E Integration Test Report")
    print("=" * 70)

    total_checks = 0
    total_pass = 0

    # Phase 1
    print("\n--- Phase 1: Connectivity ---")
    for r in phase1:
        status = "PASS" if r["pass"] else "FAIL"
        total_checks += 1
        if r["pass"]:
            total_pass += 1
        print(f"  [{status}] {r['check']}")
        if not r["pass"]:
            print(f"         Detail: {r.get('detail', '')}")

    # Phase 2
    print("\n--- Phase 2: Multi-turn RP ---")
    p2 = phase2
    turns = p2.get("turns", [])
    success = p2.get("successful_turns", 0)
    total = p2.get("total_turns", 0)
    p2_ok = success == total and total > 0
    total_checks += 1
    if p2_ok:
        total_pass += 1
    print(f"  [{'PASS' if p2_ok else 'FAIL'}] Turns completed: {success}/{total}")

    if turns:
        avg_latency = sum(t["latency_ms"] for t in turns) / len(turns)
        avg_length = sum(t["response_length"] for t in turns) / len(turns)
        print(f"  Avg latency: {avg_latency:.0f}ms | Avg response: {avg_length:.0f} chars")
        print(f"  Turn details:")
        for t in turns:
            err = f" ERROR={t['error'][:50]}" if t.get("error") else ""
            print(
                f"    T{t['turn']:2d}: {t['response_length']:5d}ch "
                f"{t['latency_ms']:7.0f}ms {t['chunks']:3d}chunks{err}"
            )

    # Phase 3
    print("\n--- Phase 3: Pipeline Verification ---")
    p3 = phase3

    # live_state.md
    ls = p3.get("live_state", {})
    checks_map = [
        ("live_state.md exists", ls.get("file_exists", False)),
        ("live_state.md has turn frontmatter", ls.get("has_turn_frontmatter", False)),
        ("live_state.md has real location", ls.get("has_real_location", False)),
        ("live_state.md has recent events", ls.get("has_recent_events", False)),
    ]
    for label, ok in checks_map:
        total_checks += 1
        if ok:
            total_pass += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    # SQLite
    sq = p3.get("sqlite", {})
    sq_checks = [
        ("SQLite DB exists", sq.get("db_exists", False)),
        ("Session exists", sq.get("session_exists", False)),
        (f"Turn count >= expected ({sq.get('turn_count', 0)})", sq.get("turn_count_ok", False)),
        ("Player character exists", sq.get("player_exists", False)),
        (f"Turn logs recorded ({sq.get('turn_log_count', 0)})", sq.get("turn_log_ok", False)),
        (f"Events recorded ({sq.get('event_count', 0)})", sq.get("has_events", False)),
    ]
    for label, ok in sq_checks:
        total_checks += 1
        if ok:
            total_pass += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    if sq.get("player_name"):
        print(f"         Player: {sq['player_name']} @ {sq.get('player_location', '?')}")
    print(f"         Characters: {sq.get('character_count', 0)} | "
          f"Relationships: {sq.get('relationship_count', 0)} | "
          f"Locations: {sq.get('location_count', 0)}")

    # ChromaDB
    ch = p3.get("chromadb", {})
    ch_ok = ch.get("episodes_ok", False)
    total_checks += 1
    if ch_ok:
        total_pass += 1
    print(f"  [{'PASS' if ch_ok else 'FAIL'}] ChromaDB episodes >= 3 (got {ch.get('episode_count', 0)})")
    if ch.get("error"):
        print(f"         Error: {ch['error']}")

    # Letta
    lt = p3.get("letta", {})
    lt_accessible = lt.get("letta_accessible", False)
    lt_agent = lt.get("curator_agent_exists", False)
    total_checks += 1
    if lt_accessible:
        total_pass += 1
    print(f"  [{'PASS' if lt_accessible else 'FAIL'}] Letta accessible")
    total_checks += 1
    if lt_agent:
        total_pass += 1
    print(f"  [{'PASS' if lt_agent else 'FAIL'}] Curator agent exists")
    if lt.get("curator_agent_name"):
        print(f"         Agent: {lt['curator_agent_name']}")
    if lt.get("memory_blocks"):
        print(f"         Memory blocks: {lt.get('memory_blocks', 0)} — {lt.get('block_labels', [])}")
    if lt.get("narrative_summary_populated") is not None:
        ns_ok = lt["narrative_summary_populated"]
        total_checks += 1
        if ns_ok:
            total_pass += 1
        print(f"  [{'PASS' if ns_ok else 'FAIL'}] narrative_summary populated")

    # Prompt caching
    pc = p3.get("prompt_caching", {})
    pc_note = f"(logs with cache hits: {pc.get('turns_with_cache_hits', 0)})"
    print(f"  [INFO] Prompt caching {pc_note}")

    # Summary
    print("\n" + "=" * 70)
    all_pass = total_pass == total_checks
    status_emoji = "ALL PASS" if all_pass else f"{total_pass}/{total_checks} PASS"
    print(f"  Result: {status_emoji}")
    print("=" * 70)

    # Build result object
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_checks": total_checks,
            "passed": total_pass,
            "failed": total_checks - total_pass,
            "all_pass": all_pass,
        },
        "phase1_connectivity": phase1,
        "phase2_multi_turn": phase2,
        "phase3_verification": phase3,
    }

    # Save to JSON
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(output_dir, f"e2e_result_{ts}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n  Results saved to: {out_path}")

        # Write detailed .md report
        detail_path = write_detail_report(phase2, phase3, session_id, output_dir, char_name)
        print(f"  Detail report: {detail_path}")

    return report


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="SAGA E2E Integration Test")
    parser.add_argument("--charx", type=str, help="Path to .charx character file")
    parser.add_argument("--scenario", type=str, choices=["soyeon", "dungeon"], default="soyeon",
                        help="Built-in scenario: soyeon (default) or dungeon")
    parser.add_argument("--saga-url", default="http://localhost:8000", help="SAGA server URL")
    parser.add_argument("--api-key", default="saga-test-key-2026", help="Bearer API key")
    parser.add_argument("--turns", type=int, default=15, help="Number of RP turns")
    parser.add_argument("--letta-url", default="http://localhost:8283", help="Letta server URL")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="LLM model for RP narration")
    parser.add_argument("--sim-model", default="claude-haiku-4-5-20251001", help="LLM model for user simulator")
    parser.add_argument("--reset-db", action="store_true", help="Reset all data before running")
    parser.add_argument("--output-dir", default="tests/e2e_results", help="Output directory for results")
    parser.add_argument("--skip-letta", action="store_true", help="Skip Letta verification")
    args = parser.parse_args()

    print("=" * 70)
    print("  SAGA E2E Integration Test")
    print("=" * 70)

    # --- Load character data ---
    if args.charx:
        print(f"\n[1/4] Loading charx: {args.charx}")
        char_data = parse_charx(args.charx)
        opener = OPENER_GENERIC
    elif args.scenario == "dungeon":
        print("\n[1/4] Using built-in scenario: 던전 보스 (Akrish)")
        char_data = {
            "name": "아크리쉬 (던전보스)",
            "system_prompt": DUNGEON_SYSTEM_PROMPT,
            "lorebook_text": DUNGEON_LOREBOOK,
            "first_mes": DUNGEON_FIRST_MES,
        }
        opener = OPENER_DUNGEON
    else:
        print("\n[1/4] Using fallback character: 위지소연")
        char_data = {
            "name": "위지소연",
            "system_prompt": FALLBACK_SYSTEM_PROMPT,
            "lorebook_text": FALLBACK_LOREBOOK,
            "first_mes": FALLBACK_FIRST_MES,
        }
        opener = OPENER_SOYEON

    print(f"  Character: {char_data['name']}")
    print(f"  Turns: {args.turns}")
    print(f"  Model (narration): {args.model}")
    print(f"  Model (user sim): {args.sim_model}")

    # --- Optional: Reset DB ---
    if args.reset_db:
        print("\n  Resetting all data...")
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                r = await client.post(
                    f"{args.saga_url}/api/reset-all",
                    headers={"Authorization": f"Bearer {args.api_key}"},
                )
                print(f"  Reset: {r.json()}")
            except Exception as e:
                print(f"  Reset failed: {e}")

    # --- Phase 1 ---
    print(f"\n[2/4] Phase 1: Connectivity checks...")
    phase1 = await phase1_connectivity(args.saga_url, args.api_key, args.letta_url)

    # Check if SAGA is reachable before continuing
    saga_ok = any(r["pass"] for r in phase1 if "SAGA health" in r["check"])
    if not saga_ok:
        print("\n  FATAL: SAGA server is not reachable. Aborting.")
        print("  Start the server: uvicorn saga.server:app --host 0.0.0.0 --port 8000")
        print_report(phase1, {"turns": [], "total_turns": 0, "successful_turns": 0}, {},
                     args.output_dir, None, char_data["name"])
        sys.exit(1)

    for r in phase1:
        print(f"  [{'PASS' if r['pass'] else 'FAIL'}] {r['check']}")

    # --- Phase 2 ---
    print(f"\n[3/4] Phase 2: Multi-turn RP simulation ({args.turns} turns, SSE streaming + dynamic user sim)...")
    phase2 = await phase2_multi_turn(
        args.saga_url,
        args.api_key,
        char_data,
        opener,
        args.turns,
        args.model,
        args.sim_model,
    )

    # --- Phase 3 ---
    print(f"\n[4/4] Phase 3: Pipeline verification...")
    session_id = _find_session_id("db/state.db")
    if not session_id:
        print("  WARNING: No session found in SQLite. Some checks will fail.")

    phase3 = {}
    if session_id:
        print(f"  Session ID: {session_id}")
        expected_turns = phase2.get("successful_turns", 0)

        phase3["live_state"] = verify_live_state(session_id)
        phase3["sqlite"] = verify_sqlite(session_id, expected_turns)
        phase3["chromadb"] = verify_chromadb(session_id)
        phase3["prompt_caching"] = verify_prompt_caching(session_id)

        if not args.skip_letta:
            phase3["letta"] = await verify_letta(session_id, args.letta_url)
        else:
            phase3["letta"] = {"letta_accessible": False, "error": "skipped"}
    else:
        phase3["live_state"] = {"file_exists": False}
        phase3["sqlite"] = {"db_exists": False}
        phase3["chromadb"] = {"episodes_accessible": False}
        phase3["letta"] = {"letta_accessible": False}
        phase3["prompt_caching"] = {"turn_logs_accessible": False}

    # --- Phase 4: Report ---
    print_report(phase1, phase2, phase3, args.output_dir, session_id, char_data["name"])


if __name__ == "__main__":
    asyncio.run(main())
