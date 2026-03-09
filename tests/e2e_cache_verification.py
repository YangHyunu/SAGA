#!/usr/bin/env python3
"""E2E Cache Verification — Full multi-turn RP with cache performance tracking.

Runs a real RP session (100+ turns) with LLM-generated user inputs,
tracking per-turn cache stats, response quality, and latency trends.

Tests:
  Phase 1: Connectivity + config check
  Phase 2: Multi-turn RP (default 100 turns) with per-turn cache tracking
  Phase 3: Cache analysis — hit rate, latency trend, cost savings
  Phase 4: TTL verification — 6min idle gap (optional, --ttl-test)

Usage:
  # Basic (100 turns, ~10-15 min):
  python3 tests/e2e_cache_verification.py

  # Custom turns:
  python3 tests/e2e_cache_verification.py --turns 50

  # With TTL test (~7 min extra):
  python3 tests/e2e_cache_verification.py --turns 100 --ttl-test

  # With scenario (soyeon / dungeon / yui):
  python3 tests/e2e_cache_verification.py --scenario soyeon --turns 100
  python3 tests/e2e_cache_verification.py --scenario dungeon --turns 50

  # With charx:
  python3 tests/e2e_cache_verification.py --charx /path/to/char.charx --turns 100

  # All options:
  python3 tests/e2e_cache_verification.py \\
    --saga-url http://localhost:8000 \\
    --api-key saga-test-key-2026 \\
    --model claude-haiku-4-5-20251001 \\
    --sim-model claude-haiku-4-5-20251001 \\
    --turns 100 --ttl-test
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install: pip install httpx")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Character data
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Name: 유이 (Yui)
Identity: Mysterious girl connected to ancient ruins.
Age: 18 / Residence: Coastal village (해안마을)

## Character Profile
- Personality: Cheerful, curious, empathetic, occasionally melancholic. Fiercely loyal.
- Appearance: Long dark hair past shoulders, bright amber eyes that glow faintly in moonlight. Blue pendant necklace — only memento from her past. Slightly tanned skin, light blue dress with leather belt for herb pouches.
- Background: Found as a child (~3 years old) near the ruins on a stormy night ~15 years ago. Raised by village elder Kim Chonjang. No memories before being found. Experiences vivid dreams about glowing chambers deep within the ruins. Pendant emits faint blue light near ruins or during strong emotions.
- Skills: Advanced herbalism, basic water magic (unconscious), excellent swimming, knowledge of local flora/fauna, basic first aid, cooking with wild ingredients
- Fears: Thunderstorms (trigger fragmented memories), being abandoned, deep darkness inside the ruins, losing pendant
- Hobbies: Collecting shells, comparing patterns to ruin inscriptions, swimming at dawn, tending herb garden, humming unknown melodies, stargazing

## World Setting
The village of 해안마을 sits in a natural harbor between steep granite cliffs and a calm crescent-shaped bay. ~200 residents, mostly fishing families.
To the north, ancient ruins of the 별빛 문명 (Starlight Civilization) sprawl across three hilltops, connected by crumbling stone bridges. Massive stone columns with luminous inscriptions, underground chambers that glow at night, sealed doors no one can open.

Important locations:
- 마을 광장 (Village Square): Heart of the village with market, well, and notice board
- 해변 (Beach): Wide sandy beach where villagers fish and children play
- 석진의 약방 (Seokjin's Apothecary): Cramped but aromatic shop filled with herbs
- 고대 유적 입구 (Ruins Entrance): Massive stone archway with glowing inscriptions
- 인어의 동굴 (Mermaid Cave): Sea cave at eastern beach, accessible at low tide
- 촌장의 집 (Elder's House): Largest house, where 유이 grew up
- 등대 (Lighthouse): Abandoned lighthouse on eastern cliff
- 유이의 약초밭 (Yui's Herb Garden): Small garden behind elder's house

## Key NPCs
- 김 촌장 (Elder Kim): Stern but caring, 60s, raised 유이. Former archaeologist. Knows secrets about ruins and 유이's origins but refuses to share. Locked drawers of research notes.
- 미나 (Mina): 유이's best friend, 17. Fisherman's daughter, bold, loud, protective. Terrified of ruins but would follow 유이 anywhere. Has crush on 해수.
- 석진 (Seokjin): Village apothecary, quiet, 50s. Arrived ~10 years ago. Taught 유이 herbalism. Seems to recognize ruin inscriptions. Mysterious scars on hands.
- 해수 (Haesu): Young female traveler, claims to research folklore. Actually from scholarly order studying ancient magic. Carries journal with ruin inscription sketches from worldwide sites.
- 영준 (Youngjun): Blacksmith's son, 19. Strong, dependable, secretly in love with 유이. Patrols near ruins to keep villagers safe.

## Roleplay Guidelines
- Always stay in character as 유이
- Respond in Korean with natural, age-appropriate dialogue
- Describe actions in third person using asterisks (*action*)
- Include emotional reactions and internal thoughts
- React to environment and weather naturally
- Pendant reacts to magic and strong emotions with faint blue glow
- Can unconsciously manipulate small amounts of water when emotional
- Sometimes hum melodies you don't remember learning (ancient magical songs)

## Speech Patterns
유이 speaks softly, slightly formal tone but becomes energetic when excited. 존댓말 with strangers, 반말 with close friends. Often trails off mid-sentence when lost in thought. Uses onomatopoeia frequently. When nervous, fidgets with pendant. When happy, hums. When sad, stares at sea.

## Story Hooks
- Pendant glows brighter each full moon, pointing toward deepest part of ruins
- 석진 once called 유이 by a different name in his sleep, then denied it
- Strange symbols appearing on beach after storms, matching ruin inscriptions
- Sealed door in ruins has pendant-shaped indentation matching 유이's pendant
- 해수's journal contains a sketch that looks exactly like 유이's pendant
- Village prophecy about "child of the starlight" who will either save or doom the world"""

FIRST_MES = (
    "*햇살이 반짝이는 해변에서 조개를 줍다가 고개를 듭니다*\n\n"
    "아, 안녕하세요! 여기 처음 보는 얼굴이네요. 마을에 새로 오신 분인가요?\n\n"
    "*파란 펜던트가 바람에 살짝 흔들리며*\n\n"
    "저는 유이라고 해요. 이 마을에서 살고 있어요. 혹시... 유적을 보러 오신 건 아니죠?"
)

OPENER_YUI = "(주위를 둘러보며) 안녕, 유이! 이 마을 정말 예쁘다. 여기 살면 좋겠다."

# ---------------------------------------------------------------------------
# Scenario: 위지소연 (Soyeon)
# ---------------------------------------------------------------------------

SOYEON_SYSTEM_PROMPT = """Name: 위지소연 (魏遲甦蘇)

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

SOYEON_LOREBOOK = """### World Overview
Genres/Tags: Korean Mythology, Romance, Sitcom, Love Comedy, Alternate Universe, Slice of Life
Background: Medieval Korea

### Named NPCs
- 당채련 (唐綵蓮): Village herbalist and physician. Sharp-tongued, caring, secretly romantic.
- 원향 (元香): Chaeryeon's niece, village tomboy. Energetic, impulsive, idolizes Soyeon.
- 적예령 (赤睿領): Wealthy widow, benevolent matriarch. Immensely strong, hosts village banquets.
- 적월화 (赤月華): Yeryeong's eldest daughter. Rebellious, cynical, white-haired albino.
- 천백화 (天白華): Yeryeong's second daughter. Timid, gentle, blue-eyed beauty."""

SOYEON_FIRST_MES = (
    "눈이 시리도록 파란 하늘 아래, 겨울 산맥의 능선이 날카롭게 그어져 있었다.\n\n"
    "그 산자락 깊숙한 곳, 기와지붕 위로 하얀 눈이 소복이 쌓인 위지 가문의 고택이 자리하고 있었다.\n\n"
    "\"…….\"\n\n"
    "위지소연은 대청마루에 앉아 멍하니 마당을 응시했다."
)

OPENER_SOYEON = "마을 끝에 있는 커다란 저택이 보인다. 문을 두드려 본다."

# ---------------------------------------------------------------------------
# Scenario: 던전 보스 (Dungeon Boss - Akrish)
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

OPENER_DUNGEON = "어둠 속에서 눈을 뜬다. 차가운 돌바닥 위에 누워 있다. 주위를 둘러본다."

OPENER_GENERIC = "(주위를 둘러보며) 여기가 어디죠? 처음 보는 곳이네요."

USER_SIM_SYSTEM = """당신은 한국어 텍스트 RP의 숙련된 유저를 시뮬레이션합니다.
상대 캐릭터(assistant)의 마지막 응답을 읽고, 몰입감 있는 유저 행동/대사를 2~4문장으로 생성하세요.

행동 스타일:
- 한국어로 작성. 행동은 (괄호) 안에, 대사는 직접 작성
- 대화를 앞으로 진전시키세요: 새로운 장소 탐색, NPC와 상호작용, 감정 표현 등
- 매 턴 다른 종류의 행동: 탐색, 대화, 전투, 아이템 사용, 감정 토로, 과거 회상 등

극적 요소 (3~4턴마다 하나씩 섞어주세요):
- 위험한 선택: 함정에 손을 넣는다, 어두운 길을 택한다
- 감정 폭발: 갑자기 분노, 슬픔, 공포에 사로잡힌다
- 과거 회상: "(문득 고향에서의 기억이 스쳐지나간다)" 같은 내면 묘사
- NPC에게 예상치 못한 행동: 안아준다, 뺨을 때린다, 무릎을 꿇는다
- 주변 환경 활용: 물건을 집어든다, 벽을 두드린다, 냄새를 맡는다

진행 패턴:
- 초반(1~10턴): 조심스러운 탐색과 관찰, 유이와 친해지기
- 중반(11~50턴): 적극적 상호작용, 유적 탐험, NPC 관계, 갈등
- 후반(51턴~): 대담한 결정, 유적 깊숙이, 비밀 해독, 클라이맥스

절대 금지:
- 상대 캐릭터의 행동/대사를 대신 쓰지 마세요
- 설명이나 메타 코멘트 없이 유저 행동/대사만 출력"""


# ---------------------------------------------------------------------------
# charx parser (from e2e_integration.py)
# ---------------------------------------------------------------------------

def parse_charx(charx_path: str) -> dict:
    """Parse a .charx file and extract RP data."""
    with zipfile.ZipFile(charx_path) as z:
        card = json.loads(z.read("card.json"))
    data = card.get("data", {})
    name = data.get("name", "Unknown")
    description = data.get("description", "")
    system_prompt_field = data.get("system_prompt", "").strip()
    scenario = data.get("scenario", "").strip()
    personality = data.get("personality", "").strip()

    parts = [p for p in [description, personality, scenario, system_prompt_field] if p]
    system_prompt = "\n\n".join(parts)

    book = data.get("character_book", {})
    entries = book.get("entries", [])
    lorebook_parts = []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        content = entry.get("content", "").strip()
        entry_name = entry.get("name", "").strip()
        if content:
            lorebook_parts.append(f"### {entry_name}\n{content}" if entry_name else content)

    print(f"  Parsed charx: {name} ({len(system_prompt)} chars, {len(lorebook_parts)} lorebook entries)")
    return {
        "name": name,
        "system_prompt": system_prompt,
        "lorebook_text": "\n\n".join(lorebook_parts),
        "first_mes": data.get("first_mes", ""),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_config_api_key(provider: str) -> str:
    """Read API key from config.yaml."""
    try:
        import yaml
        with open("config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        key = cfg.get("api_keys", {}).get(provider, "")
        if key and not key.startswith("${"):
            return key
    except Exception:
        pass
    return ""


async def generate_user_input(client: httpx.AsyncClient, messages: list[dict], model: str) -> str:
    """Use LLM to generate contextual user input (direct API, bypasses SAGA)."""
    recent = [m for m in messages[-6:] if m["role"] != "system"]
    if not recent or recent[-1]["role"] != "assistant":
        recent.append({"role": "assistant", "content": "(대기 중)"})
    recent.append({"role": "user", "content": "위 대화에 이어서 유저의 다음 행동/대사를 생성하세요."})

    try:
        if model.startswith("claude"):
            api_key = os.environ.get("ANTHROPIC_API_KEY", "") or _read_config_api_key("anthropic")
            if not api_key:
                return "(주위를 둘러본다)"
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": model, "system": USER_SIM_SYSTEM, "messages": recent, "temperature": 0.9, "max_tokens": 200},
                timeout=60,
            )
            resp.raise_for_status()
            generated = resp.json()["content"][0]["text"].strip()
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "") or _read_config_api_key("openai")
            if not api_key:
                return "(주위를 둘러본다)"
            sim_msgs = [{"role": "system", "content": USER_SIM_SYSTEM}] + recent
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": sim_msgs, "temperature": 0.9, "max_tokens": 200},
                timeout=60,
            )
            resp.raise_for_status()
            generated = resp.json()["choices"][0]["message"]["content"].strip()

        generated = re.sub(r'^(유저|User|플레이어)\s*[:：]\s*', '', generated)
        return generated if generated else "(주위를 둘러본다)"
    except Exception as e:
        print(f"    [sim] Error: {e}")
        return "(주위를 살펴보며) 계속 이야기해줘."


# ---------------------------------------------------------------------------
# LLM-as-a-Judge
# ---------------------------------------------------------------------------

JUDGE_SYSTEM = """당신은 텍스트 RP 응답 품질 평가 전문가입니다.
AI 캐릭터의 응답을 평가하여 JSON으로 점수를 매겨주세요.

평가 기준 (각 1~5점):
1. character_consistency: 캐릭터 설정 유지 (말투, 성격, 행동 패턴)
2. narrative_quality: 묘사 품질 (생동감, 오감 묘사, 감정 표현)
3. context_coherence: 이전 대화 맥락 반영 (기억, 연속성, 모순 없음)
4. immersion: 몰입감 (RP 컨벤션 준수, 메타 발언 없음, 자연스러움)
5. creativity: 창의성 (반복 회피, 새로운 전개, 흥미로운 디테일)

반드시 아래 JSON 형식만 출력하세요. 다른 텍스트 없이:
{"character_consistency": N, "narrative_quality": N, "context_coherence": N, "immersion": N, "creativity": N, "comment": "한줄 코멘트"}"""


async def judge_response(
    client: httpx.AsyncClient,
    char_description: str,
    recent_messages: list[dict],
    assistant_response: str,
    model: str,
) -> dict | None:
    """Evaluate response quality using LLM-as-a-Judge. Returns scores or None on failure."""
    # Build judge input: char description + recent context + response to evaluate
    context_msgs = []
    for m in recent_messages[-6:]:
        if m["role"] == "system":
            continue
        context_msgs.append(f"[{m['role']}]: {m['content'][:300]}")
    context_str = "\n".join(context_msgs)

    judge_input = (
        f"## 캐릭터 설정 (요약)\n{char_description[:500]}\n\n"
        f"## 최근 대화 맥락\n{context_str}\n\n"
        f"## 평가 대상 응답\n{assistant_response[:1500]}\n\n"
        f"위 응답을 평가하고 JSON으로만 출력하세요."
    )

    try:
        if model.startswith("claude"):
            api_key = os.environ.get("ANTHROPIC_API_KEY", "") or _read_config_api_key("anthropic")
            if not api_key:
                return None
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={
                    "model": model, "system": JUDGE_SYSTEM,
                    "messages": [{"role": "user", "content": judge_input}],
                    "temperature": 0, "max_tokens": 200,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["content"][0]["text"].strip()
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "") or _read_config_api_key("openai")
            if not api_key:
                return None
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": judge_input},
                    ],
                    "temperature": 0, "max_tokens": 200,
                },
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        # Parse JSON from response (handle markdown code blocks)
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        scores = json.loads(raw)

        # Validate
        required = ["character_consistency", "narrative_quality", "context_coherence", "immersion", "creativity"]
        for key in required:
            if key not in scores or not isinstance(scores[key], (int, float)):
                return None
        scores["avg_score"] = round(sum(scores[key] for key in required) / len(required), 2)
        return scores

    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Phase 1: Connectivity
# ---------------------------------------------------------------------------

async def phase1_connectivity(saga_url: str, api_key: str) -> list[dict]:
    """Check server health and caching config."""
    results = []
    async with httpx.AsyncClient(timeout=10) as client:
        # Health
        try:
            r = await client.get(f"{saga_url}/api/status", headers={"Authorization": f"Bearer {api_key}"})
            ok = r.status_code == 200
            results.append({"check": "SAGA server health", "pass": ok, "detail": r.json() if ok else r.text[:200]})
        except Exception as e:
            results.append({"check": "SAGA server health", "pass": False, "detail": str(e)})

        # Quick non-stream test to verify cache stats in response
        try:
            r = await client.post(
                f"{saga_url}/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": "claude-haiku-4-5-20251001", "messages": [
                    {"role": "system", "content": "Say hi."},
                    {"role": "user", "content": "Hi"},
                ], "max_tokens": 10, "temperature": 0},
                timeout=30,
            )
            data = r.json()
            usage = data.get("usage", {})
            has_cache_fields = "cache_read_input_tokens" in usage or "cache_creation_input_tokens" in usage
            results.append({
                "check": "Cache stats in response",
                "pass": has_cache_fields,
                "detail": f"usage keys: {list(usage.keys())}"
            })
        except Exception as e:
            results.append({"check": "Cache stats in response", "pass": False, "detail": str(e)})

    return results


# ---------------------------------------------------------------------------
# Phase 2: Multi-turn RP with cache tracking
# ---------------------------------------------------------------------------

async def phase2_multi_turn(
    saga_url: str, api_key: str, char_data: dict, opener: str,
    num_turns: int, model: str, sim_model: str,
    judge_every: int = 10,
) -> dict:
    """Run multi-turn RP, tracking cache stats and response quality per turn."""
    system_content = char_data["system_prompt"]
    if char_data.get("lorebook_text"):
        system_content += "\n\n" + char_data["lorebook_text"]

    messages = [{"role": "system", "content": system_content}]
    if char_data.get("first_mes"):
        messages.append({"role": "assistant", "content": char_data["first_mes"]})

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    turns = []
    judge_results = []

    async with httpx.AsyncClient() as client:
        for i in range(num_turns):
            # Generate user input
            if i == 0:
                user_input = opener
            else:
                user_input = await generate_user_input(client, messages, sim_model)

            messages.append({"role": "user", "content": user_input})

            # Send to SAGA (non-streaming for cache stats in response)
            body = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 4096,
                "stream": False,
            }

            t_start = time.time()
            try:
                resp = await client.post(
                    f"{saga_url}/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=180,
                )
                latency = (time.time() - t_start) * 1000

                if resp.status_code != 200:
                    turn_info = {
                        "turn": i + 1, "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
                        "latency_ms": latency, "cache_read": 0, "cache_create": 0,
                    }
                    turns.append(turn_info)
                    messages.append({"role": "assistant", "content": "(error)"})
                    print(f"  Turn {i+1:3d}/{num_turns}: ERROR {resp.status_code}")
                    continue

                data = resp.json()
                usage = data.get("usage", {})
                assistant_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                cache_read = usage.get("cache_read_input_tokens", 0)
                cache_create = usage.get("cache_creation_input_tokens", 0)

                # Sanity check (empty/error detection only — real quality via Judge)
                reasonable_length = len(assistant_text) > 50

                # LLM-as-a-Judge (every N turns + first + last)
                judge_score = None
                is_judge_turn = (
                    (i + 1) % judge_every == 0
                    or i == 0
                    or i == num_turns - 1
                )
                if is_judge_turn and assistant_text and len(assistant_text) > 50:
                    judge_score = await judge_response(
                        client, char_data["system_prompt"][:500],
                        messages, assistant_text, sim_model,
                    )
                    if judge_score:
                        judge_results.append({"turn": i + 1, **judge_score})

                turn_info = {
                    "turn": i + 1,
                    "user_input": user_input[:100],
                    "response_length": len(assistant_text),
                    "latency_ms": round(latency, 1),
                    "cache_read": cache_read,
                    "cache_create": cache_create,
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "reasonable_length": reasonable_length,
                    "judge": judge_score,
                    "error": None,
                }
                turns.append(turn_info)
                messages.append({"role": "assistant", "content": assistant_text})

                # Print progress
                cache_indicator = ""
                if cache_read > 0 and cache_create == 0:
                    cache_indicator = " [FULL HIT]"
                elif cache_read > 0:
                    cache_indicator = f" [PARTIAL cr={cache_read} cc={cache_create}]"
                elif cache_create > 0:
                    cache_indicator = f" [CREATE cc={cache_create}]"

                quality_flag = "" if reasonable_length else " ⚠EMPTY"
                judge_flag = ""
                if judge_score:
                    judge_flag = f" [Judge: {judge_score['avg_score']:.1f}/5]"
                print(
                    f"  Turn {i+1:3d}/{num_turns}: "
                    f"{len(assistant_text):5d}ch {latency:7.0f}ms"
                    f"{cache_indicator}{quality_flag}{judge_flag}"
                )

            except Exception as e:
                latency = (time.time() - t_start) * 1000
                turns.append({"turn": i + 1, "error": str(e), "latency_ms": latency, "cache_read": 0, "cache_create": 0})
                messages.append({"role": "assistant", "content": "(error)"})
                print(f"  Turn {i+1:3d}/{num_turns}: EXCEPTION {str(e)[:80]}")

            # Small delay for Sub-B processing
            if i < num_turns - 1:
                await asyncio.sleep(2)

    return {
        "turns": turns,
        "total_turns": len(turns),
        "successful_turns": sum(1 for t in turns if not t.get("error")),
        "final_messages": messages,
        "judge_results": judge_results,
    }


# ---------------------------------------------------------------------------
# Phase 3: Cache analysis
# ---------------------------------------------------------------------------

def phase3_cache_analysis(turns: list[dict], judge_results: list[dict] | None = None) -> dict:
    """Analyze cache performance and judge scores across all turns."""
    valid = [t for t in turns if not t.get("error")]
    if not valid:
        return {"error": "No valid turns"}

    # Cache stats
    turns_with_read = sum(1 for t in valid if t.get("cache_read", 0) > 0)
    turns_with_create = sum(1 for t in valid if t.get("cache_create", 0) > 0)
    total_cache_read = sum(t.get("cache_read", 0) for t in valid)
    total_cache_create = sum(t.get("cache_create", 0) for t in valid)
    total_prompt = sum(t.get("prompt_tokens", 0) for t in valid)

    # Skip first turn for hit rate (first turn is always a miss)
    subsequent = valid[1:] if len(valid) > 1 else []
    hit_rate = sum(1 for t in subsequent if t.get("cache_read", 0) > 0) / len(subsequent) if subsequent else 0

    # Latency analysis
    latencies = [t["latency_ms"] for t in valid]
    first_5_avg = sum(latencies[:5]) / min(5, len(latencies))
    last_5_avg = sum(latencies[-5:]) / min(5, len(latencies)) if len(latencies) >= 5 else first_5_avg
    overall_avg = sum(latencies) / len(latencies)

    # Sanity check: non-empty responses
    non_empty = sum(1 for t in valid if t.get("reasonable_length", False))

    # Cost savings estimate
    # cache_read tokens cost 10% of normal input
    total_input_equivalent = total_prompt + total_cache_create + total_cache_read
    if total_input_equivalent > 0:
        savings_pct = (total_cache_read * 0.9) / total_input_equivalent * 100
    else:
        savings_pct = 0

    return {
        "total_turns": len(valid),
        "turns_with_cache_read": turns_with_read,
        "turns_with_cache_create": turns_with_create,
        "cache_hit_rate": round(hit_rate * 100, 1),
        "total_cache_read_tokens": total_cache_read,
        "total_cache_create_tokens": total_cache_create,
        "total_prompt_tokens": total_prompt,
        "estimated_savings_pct": round(savings_pct, 1),
        "latency_avg_ms": round(overall_avg, 0),
        "latency_first_5_avg_ms": round(first_5_avg, 0),
        "latency_last_5_avg_ms": round(last_5_avg, 0),
        "latency_trend": "improving" if last_5_avg < first_5_avg * 0.9 else (
            "degrading" if last_5_avg > first_5_avg * 1.5 else "stable"),
        "non_empty_rate": round(non_empty / len(valid) * 100, 1) if valid else 0,
        "judge": _analyze_judge(judge_results or []),
    }


def _analyze_judge(judge_results: list[dict]) -> dict:
    """Summarize LLM judge scores."""
    if not judge_results:
        return {"evaluated": 0}

    dims = ["character_consistency", "narrative_quality", "context_coherence", "immersion", "creativity"]
    avgs = {}
    for dim in dims:
        vals = [j[dim] for j in judge_results if dim in j]
        avgs[dim] = round(sum(vals) / len(vals), 2) if vals else 0

    all_avgs = [j.get("avg_score", 0) for j in judge_results]
    overall = round(sum(all_avgs) / len(all_avgs), 2) if all_avgs else 0

    # Drift detection: compare first half vs second half
    drift = None
    if len(judge_results) >= 4:
        mid = len(judge_results) // 2
        first_half = sum(j.get("avg_score", 0) for j in judge_results[:mid]) / mid
        second_half = sum(j.get("avg_score", 0) for j in judge_results[mid:]) / (len(judge_results) - mid)
        drift = round(second_half - first_half, 2)

    return {
        "evaluated": len(judge_results),
        "overall_avg": overall,
        "per_dimension": avgs,
        "quality_drift": drift,
        "comments": [j.get("comment", "") for j in judge_results if j.get("comment")],
    }


# ---------------------------------------------------------------------------
# Phase 4: TTL verification
# ---------------------------------------------------------------------------

async def phase4_ttl_test(
    saga_url: str, api_key: str, messages: list[dict], model: str, wait_sec: int = 370,
) -> dict:
    """Test extended TTL: warm cache, wait >5min, check if cache survives."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages, "temperature": 0.7, "max_tokens": 100, "stream": False}

    async with httpx.AsyncClient(timeout=180) as client:
        # Warm up
        print("  [TTL] Warming cache...")
        r = await client.post(f"{saga_url}/v1/chat/completions", headers=headers, json=body)
        warm = r.json().get("usage", {})
        warm_create = warm.get("cache_creation_input_tokens", 0)
        warm_read = warm.get("cache_read_input_tokens", 0)
        print(f"  [TTL] Warmup: cache_create={warm_create}, cache_read={warm_read}")

        # Wait
        print(f"\n  [TTL] Waiting {wait_sec}s ({wait_sec/60:.1f}min)...")
        print(f"         (기본 TTL=5min이면 만료, 1h TTL이면 생존)")
        for elapsed in range(0, wait_sec, 30):
            remaining = wait_sec - elapsed
            mins = remaining // 60
            secs = remaining % 60
            print(f"         {mins}분 {secs}초 남음...    ", end="\r", flush=True)
            time.sleep(min(30, remaining))
        print(f"         대기 완료!                       ")

        # Re-request
        print("  [TTL] Post-wait request...")
        r = await client.post(f"{saga_url}/v1/chat/completions", headers=headers, json=body)
        post = r.json().get("usage", {})
        post_read = post.get("cache_read_input_tokens", 0)
        post_create = post.get("cache_creation_input_tokens", 0)

    survived = post_read > 0
    print(f"  [TTL] Result: cache_read={post_read}, cache_create={post_create}")
    print(f"  [TTL] {'PASS — 1h TTL 작동!' if survived else 'FAIL — 캐시 만료됨 (TTL 미적용?)'}")

    return {
        "wait_seconds": wait_sec,
        "warmup_cache_create": warm_create,
        "warmup_cache_read": warm_read,
        "post_wait_cache_read": post_read,
        "post_wait_cache_create": post_create,
        "cache_survived": survived,
    }


# ---------------------------------------------------------------------------
# Phase 5: Pipeline verification (Sub-B + Curator)
# ---------------------------------------------------------------------------

def _find_session_id(db_path: str = "db/state.db") -> str | None:
    """Find most recent session from SQLite."""
    import sqlite3
    if not os.path.exists(db_path):
        return None
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute("SELECT id FROM sessions ORDER BY updated_at DESC LIMIT 1").fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def verify_pipeline(expected_turns: int, db_path: str = "db/state.db", curator_interval: int = 10) -> dict:
    """Verify Sub-B extraction and Curator execution after multi-turn RP."""
    import sqlite3
    checks = {}

    session_id = _find_session_id(db_path)
    checks["session_id"] = session_id

    if not session_id or not os.path.exists(db_path):
        checks["db_exists"] = False
        return checks

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Session turn count
        row = conn.execute("SELECT turn_count FROM sessions WHERE id = ?", (session_id,)).fetchone()
        checks["turn_count"] = row["turn_count"] if row else 0
        checks["turn_count_ok"] = checks["turn_count"] >= expected_turns

        # Sub-B: turn_log populated
        log_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM turn_log WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]
        checks["turn_logs"] = log_count
        checks["turn_logs_ok"] = log_count >= expected_turns

        # Sub-B: characters extracted
        char_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM characters WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]
        checks["characters"] = char_count
        checks["characters_ok"] = char_count >= 1

        # Sub-B: player character
        player = conn.execute(
            "SELECT name, location FROM characters WHERE session_id = ? AND is_player = 1", (session_id,)
        ).fetchone()
        checks["player_exists"] = player is not None
        if player:
            checks["player_name"] = player["name"]
            checks["player_location"] = player["location"]

        # Sub-B: events extracted
        event_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM events WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]
        checks["events"] = event_count

        # Sub-B: relationships
        rel_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM relationships WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]
        checks["relationships"] = rel_count

    finally:
        conn.close()

    # Curator: check if it ran (expected at turns 10, 20, 30...)
    expected_curator_runs = expected_turns // curator_interval
    checks["expected_curator_runs"] = expected_curator_runs

    # Check Letta agent
    try:
        import requests as req
        r = req.get("http://localhost:8283/v1/agents/", timeout=5)
        if r.status_code == 200:
            agents = r.json()
            curator_agents = [a for a in agents if session_id in a.get("name", "")]
            checks["letta_accessible"] = True
            checks["curator_agent_exists"] = len(curator_agents) > 0
            if curator_agents:
                agent_id = curator_agents[0].get("id", "")
                checks["curator_agent_name"] = curator_agents[0].get("name", "")
                # Check memory blocks
                mem_r = req.get(f"http://localhost:8283/v1/agents/{agent_id}/core-memory/blocks", timeout=5)
                if mem_r.status_code == 200:
                    blocks = mem_r.json()
                    checks["memory_blocks"] = len(blocks)
                    for block in blocks:
                        label = block.get("label", "")
                        if "narrative" in label.lower():
                            value = block.get("value", "")
                            checks["narrative_populated"] = bool(value.strip())
                            checks["narrative_preview"] = value[:200]
                            break
        else:
            checks["letta_accessible"] = False
    except Exception:
        checks["letta_accessible"] = False

    # ChromaDB episodes
    try:
        import chromadb
        client = chromadb.PersistentClient(path="db/chroma")
        episodes = client.get_or_create_collection("episodes")
        result = episodes.get(where={"session_id": session_id})
        checks["episodes"] = len(result.get("ids", []))
    except Exception:
        checks["episodes"] = 0

    return checks


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def print_report(phase1, phase2, cache_analysis, ttl_result, pipeline, char_name):
    """Print full report."""
    print("\n" + "=" * 70)
    print("  SAGA E2E Cache Verification Report")
    print("=" * 70)

    total_checks = 0
    total_pass = 0

    # Phase 1
    print("\n--- Phase 1: Connectivity ---")
    for r in phase1:
        ok = r["pass"]
        total_checks += 1
        total_pass += ok
        print(f"  [{PASS if ok else FAIL}] {r['check']}")

    # Phase 2: RP quality
    print("\n--- Phase 2: Multi-turn RP ---")
    p2 = phase2
    success = p2["successful_turns"]
    total = p2["total_turns"]
    rp_ok = success == total and total > 0
    total_checks += 1
    total_pass += rp_ok
    print(f"  [{PASS if rp_ok else FAIL}] RP turns completed: {success}/{total}")

    ne = cache_analysis.get("non_empty_rate", 0)
    ne_ok = ne >= 95
    total_checks += 1
    total_pass += ne_ok
    print(f"  [{PASS if ne_ok else FAIL}] Non-empty responses: {ne}%")

    # Phase 3: Cache performance
    print("\n--- Phase 3: Cache Performance ---")
    ca = cache_analysis

    # Cache hit rate
    hr = ca.get("cache_hit_rate", 0)
    hr_ok = hr >= 80
    total_checks += 1
    total_pass += hr_ok
    print(f"  [{PASS if hr_ok else FAIL}] Cache hit rate (turn 2+): {hr}%")

    # Cache creation on first turn
    turns = phase2.get("turns", [])
    t1_ok = turns[0].get("cache_create", 0) > 0 if turns else False
    total_checks += 1
    total_pass += t1_ok
    print(f"  [{PASS if t1_ok else FAIL}] First turn cache creation: {turns[0].get('cache_create', 0) if turns else 0} tokens")

    # Cost savings
    sv = ca.get("estimated_savings_pct", 0)
    sv_ok = sv > 10
    total_checks += 1
    total_pass += sv_ok
    print(f"  [{PASS if sv_ok else FAIL}] Estimated cost savings: {sv}%")

    # Latency stability
    trend = ca.get("latency_trend", "unknown")
    trend_ok = trend != "degrading"
    total_checks += 1
    total_pass += trend_ok
    print(f"  [{PASS if trend_ok else FAIL}] Latency trend: {trend} (avg={ca.get('latency_avg_ms', 0):.0f}ms)")
    print(f"         First 5 avg: {ca.get('latency_first_5_avg_ms', 0):.0f}ms | Last 5 avg: {ca.get('latency_last_5_avg_ms', 0):.0f}ms")

    # Token totals
    print(f"\n  Token Summary:")
    print(f"    cache_read total  : {ca.get('total_cache_read_tokens', 0):>10,} tokens (90% 할인)")
    print(f"    cache_create total: {ca.get('total_cache_create_tokens', 0):>10,} tokens")
    print(f"    prompt total      : {ca.get('total_prompt_tokens', 0):>10,} tokens")

    # Judge results
    jd = ca.get("judge", {})
    if jd.get("evaluated", 0) > 0:
        print(f"\n--- LLM-as-a-Judge (매 {10}턴 평가) ---")
        overall = jd.get("overall_avg", 0)
        judge_ok = overall >= 3.5
        total_checks += 1
        total_pass += judge_ok
        print(f"  [{PASS if judge_ok else FAIL}] Overall quality: {overall}/5.0 ({jd['evaluated']}턴 평가)")

        dims = jd.get("per_dimension", {})
        for dim, score in dims.items():
            label = {
                "character_consistency": "캐릭터 일관성",
                "narrative_quality": "서사 품질",
                "context_coherence": "맥락 연속성",
                "immersion": "몰입감",
                "creativity": "창의성",
            }.get(dim, dim)
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"    {label:<12}: {bar} {score}/5")

        drift = jd.get("quality_drift")
        if drift is not None:
            drift_ok = drift > -0.5
            total_checks += 1
            total_pass += drift_ok
            direction = "향상" if drift > 0.2 else ("하락" if drift < -0.2 else "안정")
            print(f"  [{PASS if drift_ok else FAIL}] Quality drift: {drift:+.2f} ({direction})")

        comments = jd.get("comments", [])
        if comments:
            print(f"  Judge comments:")
            for c in comments[-3:]:  # last 3
                print(f"    - {c}")

    # Phase 4: TTL
    if ttl_result:
        print("\n--- Phase 4: Extended TTL (1h) ---")
        ttl_ok = ttl_result.get("cache_survived", False)
        total_checks += 1
        total_pass += ttl_ok
        wait = ttl_result.get("wait_seconds", 0)
        print(f"  [{PASS if ttl_ok else FAIL}] Cache survived {wait}s ({wait/60:.1f}min) idle gap")
        if not ttl_ok:
            print(f"         ⚠ 확인: config.yaml prompt_caching.cache_ttl = '1h'")
            print(f"         ⚠ 확인: client.py anthropic-beta 헤더에 extended-cache-ttl-2025-04-11")

    # Phase 5: Pipeline (Sub-B + Curator)
    if pipeline:
        print("\n--- Phase 5: Pipeline (Sub-B + Curator) ---")
        pl = pipeline

        tl_ok = pl.get("turn_logs_ok", False)
        total_checks += 1
        total_pass += tl_ok
        print(f"  [{PASS if tl_ok else FAIL}] Sub-B turn logs: {pl.get('turn_logs', 0)} recorded")

        ch_ok = pl.get("characters_ok", False)
        total_checks += 1
        total_pass += ch_ok
        print(f"  [{PASS if ch_ok else FAIL}] Characters extracted: {pl.get('characters', 0)}")
        if pl.get("player_exists"):
            print(f"         Player: {pl.get('player_name', '?')} @ {pl.get('player_location', '?')}")

        ep = pl.get("episodes", 0)
        ep_ok = ep >= 3
        total_checks += 1
        total_pass += ep_ok
        print(f"  [{PASS if ep_ok else FAIL}] ChromaDB episodes: {ep}")

        if pl.get("events", 0) > 0:
            print(f"         Events: {pl['events']} | Relationships: {pl.get('relationships', 0)}")

        if pl.get("letta_accessible"):
            cur_ok = pl.get("curator_agent_exists", False)
            total_checks += 1
            total_pass += cur_ok
            print(f"  [{PASS if cur_ok else FAIL}] Curator agent created")
            if cur_ok:
                print(f"         Agent: {pl.get('curator_agent_name', '?')}")
                print(f"         Memory blocks: {pl.get('memory_blocks', 0)}")
                if pl.get("narrative_populated") is not None:
                    nar_ok = pl["narrative_populated"]
                    total_checks += 1
                    total_pass += nar_ok
                    print(f"  [{PASS if nar_ok else FAIL}] narrative_summary populated")
                    if pl.get("narrative_preview"):
                        print(f"         Preview: {pl['narrative_preview'][:120]}...")
        else:
            print(f"  [SKIP] Letta 미연결 — Curator 검증 생략")

    # Summary
    print("\n" + "=" * 70)
    all_pass = total_pass == total_checks
    print(f"  Result: {'ALL PASS' if all_pass else f'{total_pass}/{total_checks} PASS'}")
    print("=" * 70)

    return {"total_checks": total_checks, "passed": total_pass, "all_pass": all_pass}


def save_results(phase1, phase2, cache_analysis, ttl_result, summary, output_dir, char_name):
    """Save JSON + markdown report."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON
    report = {
        "timestamp": datetime.now().isoformat(),
        "character": char_name,
        "summary": summary,
        "phase1_connectivity": phase1,
        "phase2_turns": phase2.get("turns", []),
        "phase3_cache_analysis": cache_analysis,
        "phase4_ttl": ttl_result,
    }
    json_path = os.path.join(output_dir, f"cache_e2e_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  JSON: {json_path}")

    # Markdown detail
    md_path = os.path.join(output_dir, f"cache_e2e_{ts}.md")
    lines = [
        f"# SAGA Cache E2E Report — {char_name}",
        f"*{datetime.now().isoformat()}*\n",
        f"## Summary",
        f"- Turns: {cache_analysis.get('total_turns', 0)}",
        f"- Cache hit rate: {cache_analysis.get('cache_hit_rate', 0)}%",
        f"- Cost savings: {cache_analysis.get('estimated_savings_pct', 0)}%",
        f"- Avg latency: {cache_analysis.get('latency_avg_ms', 0):.0f}ms",
        f"- Non-empty: {cache_analysis.get('non_empty_rate', 0)}%",
    ]

    # Judge summary in markdown
    jd = cache_analysis.get("judge", {})
    if jd.get("evaluated", 0) > 0:
        lines.append(f"- Judge overall: {jd.get('overall_avg', 0)}/5.0 ({jd['evaluated']}턴 평가)")
        lines.append(f"- Quality drift: {jd.get('quality_drift', 'N/A')}")
    lines.append("")

    # Per-turn table
    lines.extend([
        "## Per-Turn Stats\n",
        "| Turn | Latency | cache_read | cache_create | prompt | resp_len | judge |",
        "|------|---------|------------|-------------|--------|----------|-------|",
    ])
    for t in phase2.get("turns", []):
        j = t.get("judge")
        j_mark = f"{j['avg_score']:.1f}" if j else "-"
        lines.append(
            f"| {t['turn']:3d} | {t['latency_ms']:7.0f}ms | {t.get('cache_read', 0):>10,} | "
            f"{t.get('cache_create', 0):>10,} | {t.get('prompt_tokens', 0):>6,} | "
            f"{t.get('response_length', 0):>5d}ch | {j_mark} |"
        )
    lines.append("")

    # Judge detail table
    judge_results = phase2.get("judge_results", [])
    if judge_results:
        lines.extend([
            "## Judge Evaluations\n",
            "| Turn | Avg | Character | Narrative | Context | Immersion | Creativity | Comment |",
            "|------|-----|-----------|-----------|---------|-----------|------------|---------|",
        ])
        for j in judge_results:
            lines.append(
                f"| {j.get('turn', '?')} | {j.get('avg_score', 0):.1f} | "
                f"{j.get('character_consistency', 0)} | {j.get('narrative_quality', 0)} | "
                f"{j.get('context_coherence', 0)} | {j.get('immersion', 0)} | "
                f"{j.get('creativity', 0)} | {j.get('comment', '')} |"
            )
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Report: {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="SAGA E2E Cache Verification")
    parser.add_argument("--charx", type=str, help="Path to .charx character file")
    parser.add_argument("--scenario", type=str, choices=["yui", "soyeon", "dungeon"], default="yui",
                        help="Built-in scenario: yui (default), soyeon, or dungeon")
    parser.add_argument("--saga-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="saga-test-key-2026")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Narration model")
    parser.add_argument("--sim-model", default="claude-haiku-4-5-20251001", help="User simulator model")
    parser.add_argument("--turns", type=int, default=100, help="Number of RP turns (default: 100)")
    parser.add_argument("--ttl-test", action="store_true", help="Run 6-min TTL verification")
    parser.add_argument("--ttl-wait", type=int, default=370, help="TTL wait seconds (default: 370)")
    parser.add_argument("--judge-every", type=int, default=10, help="Run LLM judge every N turns (default: 10)")
    parser.add_argument("--output-dir", default="tests/e2e_cache_results", help="Output directory")
    args = parser.parse_args()

    # Load character
    if args.charx:
        print(f"\n[1/4] Loading charx: {args.charx}")
        char_data = parse_charx(args.charx)
        char_name = char_data.get("name", "Custom")
        opener = OPENER_GENERIC
    elif args.scenario == "soyeon":
        print(f"\n[1/4] Using built-in scenario: 위지소연")
        char_data = {
            "name": "위지소연",
            "system_prompt": SOYEON_SYSTEM_PROMPT,
            "lorebook_text": SOYEON_LOREBOOK,
            "first_mes": SOYEON_FIRST_MES,
        }
        char_name = "위지소연"
        opener = OPENER_SOYEON
    elif args.scenario == "dungeon":
        print(f"\n[1/4] Using built-in scenario: 던전 보스 (Akrish)")
        char_data = {
            "name": "아크리쉬 (던전보스)",
            "system_prompt": DUNGEON_SYSTEM_PROMPT,
            "lorebook_text": DUNGEON_LOREBOOK,
            "first_mes": DUNGEON_FIRST_MES,
        }
        char_name = "아크리쉬"
        opener = OPENER_DUNGEON
    else:
        print(f"\n[1/4] Using built-in character: 유이 (Yui)")
        char_data = {
            "name": "유이 (Yui)",
            "system_prompt": SYSTEM_PROMPT,
            "lorebook_text": "",
            "first_mes": FIRST_MES,
        }
        char_name = "유이"
        opener = OPENER_YUI

    print("=" * 70)
    print("  SAGA E2E Cache Verification")
    print("=" * 70)
    print(f"  Server   : {args.saga_url}")
    print(f"  Character: {char_name}" + (f" (--scenario {args.scenario})" if not args.charx else f" (--charx)"))
    print(f"  Model    : {args.model}")
    print(f"  Sim model: {args.sim_model}")
    print(f"  Turns    : {args.turns}")
    print(f"  TTL test : {'YES' if args.ttl_test else 'NO (--ttl-test)'}")

    # Reset
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"{args.saga_url}/api/reset-all", headers={"Authorization": f"Bearer {args.api_key}"})
            print("  Session reset: OK")
    except Exception:
        print("  Session reset: skipped")

    # Phase 1
    print(f"\n[1/4] Phase 1: Connectivity")
    phase1 = await phase1_connectivity(args.saga_url, args.api_key)
    for r in phase1:
        print(f"  [{'OK' if r['pass'] else 'FAIL'}] {r['check']}")

    if not all(r["pass"] for r in phase1):
        print("\n  ERROR: Phase 1 failed. 서버 확인 후 재실행하세요.")
        sys.exit(1)

    # Phase 2
    print(f"\n[2/4] Phase 2: Multi-turn RP ({args.turns} turns)")
    phase2 = await phase2_multi_turn(
        args.saga_url, args.api_key, char_data, opener,
        args.turns, args.model, args.sim_model,
        judge_every=args.judge_every,
    )

    # Phase 3
    print(f"\n[3/4] Phase 3: Cache Analysis")
    cache_analysis = phase3_cache_analysis(phase2.get("turns", []), phase2.get("judge_results", []))

    # Phase 4 (optional)
    ttl_result = None
    if args.ttl_test:
        print(f"\n[4/4] Phase 4: TTL Verification ({args.ttl_wait}s wait)")
        # Use a subset of the conversation for TTL test
        ttl_messages = phase2.get("final_messages", [])
        if len(ttl_messages) > 20:
            # Trim to last 20 messages to keep it manageable
            ttl_messages = [ttl_messages[0]] + ttl_messages[-19:]
        ttl_result = await phase4_ttl_test(args.saga_url, args.api_key, ttl_messages, args.model, args.ttl_wait)
    else:
        print(f"\n[4/5] Phase 4: TTL — skipped (use --ttl-test)")

    # Phase 5: Pipeline verification
    print(f"\n[5/5] Phase 5: Pipeline Verification (Sub-B + Curator)")
    print("  Waiting 10s for async processing to complete...")
    await asyncio.sleep(10)
    pipeline = verify_pipeline(args.turns)

    # Report
    summary = print_report(phase1, phase2, cache_analysis, ttl_result, pipeline, char_name)
    save_results(phase1, phase2, cache_analysis, ttl_result, summary, args.output_dir, char_name)


if __name__ == "__main__":
    asyncio.run(main())
