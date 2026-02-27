#!/usr/bin/env python3
"""SAGA Prompt Caching Benchmark — 50+ turn RP conversation simulation.

Directly calls Anthropic API to measure 3-BP cache effectiveness:
  - cache_read_input_tokens / cache_creation_input_tokens per turn
  - Latency, token counts, cache hit rate, cost savings

Usage:
  python3 tests/bench_prompt_caching.py --turns 5    # quick test
  python3 tests/bench_prompt_caching.py               # full 50-turn benchmark
  python3 tests/bench_prompt_caching.py --no-cache    # baseline without caching
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saga.config import load_config
import saga.server as server_module

# ── Anthropic pricing (Claude Haiku 4.5, 2025) ────────────────────────
# Min cacheable: 4096 tokens for Haiku 4.5
PRICE_INPUT = 1.00 / 1_000_000        # $/token
PRICE_CACHE_READ = 0.10 / 1_000_000   # 10% of base (cache hit)
PRICE_CACHE_CREATE = 1.25 / 1_000_000 # 125% of base (5m cache write)
PRICE_OUTPUT = 5.00 / 1_000_000       # $/token

# ── RP system prompt (realistic SAGA-style) ───────────────────────────
SYSTEM_PROMPT = """You are a master storyteller and narrator for an immersive fantasy RPG.
You control the world of Eldoria — a vast continent with ancient forests, crumbling dungeons,
coastal trading cities, and mountain strongholds. You narrate in vivid, atmospheric prose.

WORLD LORE:
Eldoria was once a unified empire under the Shattered Crown, a divine artifact forged by the
Five Elemental Titans during the Age of Creation. When the last Emperor, Vareth the Unbound,
attempted to harness the Crown's full power to achieve immortality, the artifact shattered into
five fragments, each scattered across the continent. The cataclysm that followed — known as
the Sundering — reshaped the geography of the world, creating the Abyssal Rift in the south,
the Frozen Wastes in the north, and sinking the great capital city of Luminara beneath the
western sea.

Now, 500 years later, the world is divided into fractious kingdoms and city-states. The five
Crown fragments are sought by many factions: the Restoration Order (who wish to reunite the
empire), the Shadow Guild (who wish to destroy the fragments forever), and the Arcane
Conclave (who wish to study and harness their power). Ancient prophecies speak of a "Crown-
bearer" who will reunite the fragments and either save or doom the world.

GEOGRAPHY OF ELDORIA:
1. The Whispering Woods — An ancient primeval forest in central Eldoria, home to fey creatures,
   forest spirits, and hidden elven sanctuaries. The trees here are sentient and communicate
   through rustling leaves. Deep within lies the Heartwood, a massive tree said to predate the
   Sundering. The forest is dangerous at night when shadow wolves and corrupted dryads roam.
2. Thornhaven — A small but resilient village on the eastern edge of the Whispering Woods.
   Population ~200. Led by Elder Bram, a retired adventurer. The village survives on lumber,
   herbs, and trade with passing merchants. A small shrine to the Earth Titan stands in the
   village square. Notable locations: The Rusty Tankard (inn), Bram's Hall, the Herbalist's
   cottage, and the abandoned watchtower.
3. The Sunken Temple — Once a grand cathedral dedicated to the Water Titan, now partially
   submerged in a lake formed during the Sundering. Rumored to contain one of the Crown
   fragments. The temple is protected by ancient magical wards and inhabited by aquatic
   creatures. Multiple adventuring parties have entered; few have returned.
4. The Crossroads Market — A permanent trading post where four major roads meet, two days'
   travel from Thornhaven. Merchants from across Eldoria gather here. The market is neutral
   ground enforced by the Merchant's Guild. Illegal goods, rare artifacts, and information
   can all be found here for the right price.
5. The Iron Mountains — A massive mountain range in the north, home to dwarvish kingdoms.
   The dwarves mine precious metals and forge legendary weapons. Their capital, Deephold,
   is carved entirely within Mount Ironpeak. The mountains also hide ancient dragon lairs
   and forgotten tunnels from the pre-Sundering era.
6. Port Serenade — A bustling coastal city in the southwest, built on the ruins of a
   pre-Sundering settlement. It serves as the primary naval hub for trade and exploration.
   The Arcane Conclave has a major chapter house here, and ships regularly depart to search
   for the sunken capital of Luminara.

WORLD STATE:
- The player is Kael, a wandering swordsman seeking the Shattered Crown.
- Current location: The Whispering Woods, near the village of Thornhaven.
- Time: Late afternoon, overcast sky.
- Companions: Lyra (elven archer, age 127, reserved but loyal), Maren (dwarf healer, age 45, jovial and protective).
- Active quests: Find the Crown fragment in the Sunken Temple, deliver herbs to Elder Bram.
- Reputation: Neutral with most factions, slightly positive with Thornhaven.
- Known Crown fragments: Water (Sunken Temple, unconfirmed), Fire (Iron Mountains, rumor).

CHARACTER PROFILES:
Kael — Age 28, human male. Former soldier from the fallen kingdom of Valcrest. Carries his
father's sword "Dawncleaver." Motivated by a promise to his dying father to find the Crown
and restore Valcrest. Pragmatic but honorable. Fears losing those close to him. Has a scar
across his left eyebrow from a childhood accident.

Lyra Windshade — Age 127, wood elf female. Ranger and archer of exceptional skill. Left her
elven sanctuary after a disagreement with the Elder Council about isolationism. Seeks to
protect the Whispering Woods from encroaching darkness. Quiet and observant, speaks rarely
but with precision. Has a falcon companion named Zephyr who scouts ahead.

Maren Ironforge — Age 45, dwarf female. Combat medic and healer trained at the Temple of the
Earth Titan in Deephold. Left the mountains after her clinic was destroyed in a tunnel collapse.
Warm and boisterous, she uses humor to cope with loss. Expert in herblore, field surgery, and
earth-aspected healing magic. Carries a warhammer "Boneset" that doubles as a surgical tool.

CHARACTER RULES:
- NPCs have distinct personalities, motivations, and speech patterns.
- Lyra speaks in short, precise sentences. Maren uses colloquialisms and dwarvish expressions.
- Combat follows narrative logic — describe actions cinematically with tactical awareness.
- Track inventory changes and location transitions precisely.
- Maintain emotional continuity across scenes — characters remember and react to past events.
- NPCs may disagree with the player or each other based on their personalities.
- Environmental storytelling: weather, time of day, and surroundings affect mood and options.

NARRATION STYLE:
- Second person ("You step into the clearing...")
- Rich sensory details (sounds, smells, textures, temperature)
- Balance action, dialogue, and atmosphere in roughly equal measure
- End each response with a subtle hook or choice point
- Use paragraph breaks for readability
- Combat scenes should be visceral and kinetic
- Quiet moments should be introspective and atmospheric
- Dialogue should reveal character and advance the story

RESPONSE FORMAT:
- Keep responses between 150-250 words
- Always include at least one line of NPC dialogue when companions are present
- When state changes occur, include a [STATE_BLOCK] at the end

FACTIONS AND ORGANIZATIONS:
1. The Restoration Order — A knightly order dedicated to reuniting the Shattered Crown and
   restoring the old Empire. Led by Grand Commander Seryth, a charismatic human paladin.
   They believe only a worthy Crown-bearer can save the world from the growing darkness.
   The Order maintains chapter houses in every major city and has a network of agents.
   Their symbol is a golden crown with five points. They are generally benevolent but
   increasingly desperate as dark forces grow stronger.

2. The Shadow Guild — A criminal syndicate that believes the Shattered Crown is too
   dangerous to reassemble. Led by the mysterious figure known only as "The Veil," the
   Guild operates through cells in major cities. They seek to destroy the Crown fragments
   or hide them forever. Despite their criminal methods, many members genuinely believe
   they are saving the world. Their agents include assassins, spies, and corrupted mages.
   Symbol: a crescent moon bisected by a dagger.

3. The Arcane Conclave — An organization of scholars and mages dedicated to studying the
   Crown's power. Based primarily in Port Serenade, they maintain the largest library in
   Eldoria. Led by Archmage Thalindra, an ancient elven woman. They want to harness the
   Crown's power for the advancement of magical knowledge. Neutral in the conflict between
   the Order and the Guild, but will ally with either for access to Crown fragments.

4. The Wild Court — The ruling body of the fey creatures in the Whispering Woods. Led by
   Queen Aelindra, they are ancient and powerful but largely unconcerned with mortal affairs.
   However, the Sundering damaged the barrier between the mortal world and the Feywild, and
   they may seek the Crown to repair it. They communicate through dreams and nature signs.

5. The Iron Covenant — The governing council of the dwarvish kingdoms in the Iron Mountains.
   Conservative and isolationist, they guard one of the Crown fragments (the Fire fragment)
   but refuse to acknowledge it publicly. Led by High Thane Borik Stoneheart.

MAGIC SYSTEM:
Magic in Eldoria flows from the five elemental sources tied to the original Titans:
- Earth: Healing, defense, nature control. Green energy. Common among dwarves and druids.
- Water: Illusion, divination, mind magic. Blue energy. Common in coastal regions.
- Fire: Destruction, transformation, forge magic. Red energy. Common in the Iron Mountains.
- Air: Speed, lightning, weather control. White energy. Common among elves and highland folk.
- Void: Necromancy, shadow, dimensional magic. Purple energy. Forbidden by most societies.

Each Crown fragment amplifies its associated element. As fragments are gathered, their combined
power creates unpredictable magical surges in the surrounding area. The Sundering scattered
ambient magical energy unevenly, creating "wild zones" where magic behaves erratically.

BESTIARY (Common Threats):
- Dire Wolves: Pack hunters, size of horses. Intelligent enough to set ambushes. Weak to fire.
- Shadow Wraiths: Remnants of souls lost in the Sundering. Incorporeal, drain life energy.
  Can only be harmed by magical weapons or elemental attacks. Most active at night.
- Corrupted Dryads: Once gentle forest spirits twisted by void energy. Control plants
  aggressively. Their corruption spreads slowly through the forest. Can be purified with
  earth magic.
- Stone Golems: Ancient constructs guarding pre-Sundering ruins. Nearly indestructible but
  slow. Vulnerable to water magic which erodes their joints. Will not attack unless provoked
  or their guarded area is breached.
- River Serpents: Massive aquatic predators in freshwater systems. Semi-intelligent,
  territorial. Some can be reasoned with. Guard the approaches to the Sunken Temple.
- Mountain Drakes: Smaller cousins of true dragons. Wingspan of 20 feet. Breathe fire.
  Territorial but can be tamed by skilled handlers. Nest in the upper peaks of the Iron
  Mountains.
- Goblin Tribes: Small, cunning humanoids. Organized into tribes with shamans who wield
  crude void magic. Frequent raiders of trade routes. Some tribes can be negotiated with.

HISTORY OF THE SUNDERING:
The Sundering occurred exactly 500 years ago, on the night of the Grand Conjunction when all
three moons of Eldoria aligned. Emperor Vareth had spent decades gathering the knowledge to
unlock the Crown's full potential. He performed the Ritual of Ascension in the throne room
of Luminara, surrounded by his court of advisors and mages.

The ritual initially seemed successful — Vareth's body glowed with all five elemental energies
simultaneously, something never before witnessed. But the power was too great for any mortal
to contain. The Crown fractured along its five gemstone settings, each fragment rocketing
outward with catastrophic force. The explosion leveled Luminara instantly, killed everyone
within a mile radius, and sent shockwaves across the entire continent.

The aftermath was devastating:
- The western coast collapsed, sinking Luminara beneath the newly formed Sorrow Sea
- The Abyssal Rift tore open in the southern plains, a wound in reality leaking void energy
- The northern glaciers shifted, creating the impassable Frozen Wastes
- The Whispering Woods expanded rapidly as earth energy surged through the root networks
- The Iron Mountains grew taller and more treacherous as fire energy concentrated underground

In the centuries since, civilization has slowly rebuilt. The old Empire fragmented into the
current patchwork of kingdoms, city-states, and wild territories. The five Crown fragments
were scattered across the continent, each warping the environment around them. Some were
found and hidden; others remain lost. Prophecies began appearing within decades of the
Sundering, speaking of a Crown-bearer who would reunite the fragments.

CALENDAR AND TIME SYSTEM:
Eldoria uses the Imperial Calendar, counting years since the Sundering (AS = After Sundering).
The current year is 500 AS. Each year has 12 months of 30 days each, plus a 5-day festival
period at year's end called the Conjunction Days. The three moons (Seluna, Umbra, and Vesper)
create complex tidal and magical patterns. Full alignments of all three moons happen roughly
every 500 years — meaning another Grand Conjunction is imminent.

The day is divided into 24 hours. Most civilized areas use sundials and water clocks.
Mage-crafted timepieces exist but are expensive. The fey in the Whispering Woods experience
time differently — a day in the Feywild can be a week or a moment in the mortal world.

LANGUAGES OF ELDORIA:
- Common: The trade language, descended from Imperial speech. Spoken everywhere.
- Elvish: Musical and complex. Used for magical incantations and poetry.
- Dwarvish: Guttural and precise. Every word has exactly one meaning. Used in engineering.
- Fey: Cannot truly be spoken by mortals. Communicates emotions and images, not words.
- Void Speech: The language of the entities beyond the Abyssal Rift. Hearing it causes
  nausea. Speaking it draws attention from things that should not be noticed.
- Old Imperial: The formal language of the pre-Sundering Empire. Used in legal documents,
  ancient texts, and some magical formulae. Few can read it fluently.

CRAFTING AND ECONOMY:
Eldoria's economy runs on a tiered currency system: copper pieces (cp), silver pieces (sp),
and gold pieces (gp). 100 cp = 10 sp = 1 gp. Most common goods cost copper; weapons and
armor cost silver; magical items and rare materials cost gold. The Merchant's Guild maintains
currency standards across the continent. Bartering is common in rural areas.

Crafting requires materials, tools, and skill. Blacksmiths forge weapons and armor from raw
metals and monster parts. Alchemists brew potions from herbs, minerals, and creature extracts.
Enchanters imbue items with magical properties using elemental crystals. Master crafters in
each discipline are rare and highly sought after. The dwarves of the Iron Mountains are the
finest metalworkers, while the elves excel at enchantment and alchemy.

Material rarities follow a standard classification: Common (wood, iron, leather), Uncommon
(steel, silver, darkwood), Rare (mithril, dragonscale, starstone), Epic (adamantine, phoenix
feather, crown shard dust), and Legendary (materials from other planes or pre-Sundering
artifacts). Higher rarity materials require specialized knowledge and tools to work with.

SOCIAL STRUCTURES AND POLITICS:
Each region of Eldoria has its own governance:
- Thornhaven and similar villages: Elder councils, direct democracy for small decisions
- Port Serenade: Merchant republic with elected council of trade guild representatives
- The Iron Mountains: Clan-based monarchy under the High Thane and Council of Clans
- The Whispering Woods: Fey Court hierarchy based on age, power, and connection to nature
- Free Cities: Various systems including hereditary monarchy, oligarchy, and theocracy

Political tensions are high. The Restoration Order has growing influence in human kingdoms,
pushing for reunification under a new Emperor. The Shadow Guild undermines Order operations
while maintaining its own network of informants and assassins. The Arcane Conclave remains
officially neutral but sells knowledge and magical services to all sides. Border disputes,
trade wars, and occasional skirmishes keep the continent in a state of uneasy tension.

Diplomacy is conducted through formal channels (ambassadors, treaties) and informal ones
(spies, bribes, arranged marriages). The player's reputation with each faction affects what
quests, information, and services are available. Hostile factions may send assassins or
thieves. Allied factions provide safe houses, discounts, and military support.

WEATHER AND ENVIRONMENTAL SYSTEMS:
The Sundering disrupted natural weather patterns across Eldoria. Magical storms (called
"Sunder-storms") occur unpredictably, especially near Crown fragment locations. These storms
can temporarily amplify or suppress magic, cause hallucinations, open brief rifts to other
planes, or awaken dormant creatures. They are preceded by a distinctive purple-green tint
in the sky and a smell like ozone mixed with copper.

Normal weather follows seasonal patterns modified by geography:
- Spring: Mild, rainy. The Whispering Woods bloom. River levels rise.
- Summer: Warm. Trade season. Festival period. Best time for travel.
- Autumn: Cool, misty. Harvest season. Shadow Guild activity increases.
- Winter: Cold, harsh. Mountain passes close. Undead activity increases.

DETAILED INVENTORY TRACKING:
The party inventory system tracks items by category:
- Weapons: Each weapon has a name, type, damage range, and special properties.
- Armor: Provides damage reduction. Can be damaged and repaired.
- Consumables: Potions, food, herbs. Have limited uses and can expire.
- Quest Items: Story-critical items that cannot be sold or discarded.
- Materials: Crafting components, trade goods, and currency.

All transactions must be logged. When items are acquired or lost, update the STATE_BLOCK.

TRAVEL AND NAVIGATION:
Eldoria has a network of roads maintained by the Merchant's Guild, but only main trade routes
are reliably safe. Wilderness travel requires survival skills, navigation ability, and
combat readiness. Travel speed depends on terrain, weather, party condition, and mode of
transport. On foot, a healthy party covers about 25 miles per day on roads, 15 in forest,
10 in mountains. Horses double road speed but are useless in dense forest or underground.
Magical travel (teleportation circles, fey paths, wind-walking) exists but is rare, expensive,
and sometimes unreliable due to Sunder-storms. River travel is common and efficient where
waterways exist, but river serpents and pirates make it dangerous. The party should always
plan routes carefully, considering resupply points, shelter, and known threat zones.

[STATE_BLOCK]
location: whispering_woods
sub_location: eastern_trail
time_of_day: late_afternoon
weather: overcast
party_hp: {kael: 85, lyra: 92, maren: 78}
party_mp: {kael: 0, lyra: 30, maren: 45}
gold: 340
inventory: [dawncleaver, leather_armor, healing_herbs_x3, map_fragment, torch_x5, rations_x7]
active_effects: [forest_blessing]
quest_progress: {sunken_temple: not_started, elder_bram_herbs: in_progress}
reputation: {restoration_order: neutral, shadow_guild: unknown, arcane_conclave: neutral, wild_court: neutral, iron_covenant: neutral}
[/STATE_BLOCK]"""

# ── Pre-defined user inputs (10 unique, cycled for N turns) ───────────
USER_INPUTS = [
    "I draw my sword and cautiously approach the ancient ruins ahead. Lyra, scout the perimeter.",
    "Let's talk to the merchant at the crossroads. I want to sell the wolf pelts and ask about the Sunken Temple.",
    "We set up camp for the night. I take first watch while Maren tends to Lyra's wound from the earlier fight.",
    "I examine the strange rune carved into the stone door. Can Maren read dwarvish inscriptions?",
    "A group of bandits blocks our path. I step forward and offer them a chance to leave peacefully.",
    "We enter the underground passage. I light a torch and move carefully, checking for traps.",
    "I pick up the glowing amulet from the pedestal. What happens when I touch it?",
    "Let's head back to Thornhaven. I need to report what we found to Elder Bram and resupply.",
    "I challenge the arena champion to a duel. Lyra and Maren watch from the stands.",
    "We follow the river downstream toward the coast. I keep an eye out for the temple entrance the map shows.",
]

# ── Simulated dynamic context (grows slightly each turn) ──────────────
MD_PREFIX_BASE = """## World State Summary
Kael's party has been traveling for 3 days through the Whispering Woods.
Morale is moderate. Supplies are running low.

## Recent Events
- Defeated a dire wolf pack (Turn 2)
- Found ancient map fragment (Turn 4)
- Met traveling herbalist (Turn 6)"""

DYNAMIC_SUFFIX_BASE = """## Active Lorebook Entries
- Whispering Woods: Ancient forest, home to fey creatures
- Thornhaven: Small village, Elder Bram is leader
- Sunken Temple: Rumored location of Crown fragment"""


def build_dynamic_context(turn: int) -> tuple[str, str]:
    """Generate growing dynamic context to simulate real usage."""
    md = MD_PREFIX_BASE
    if turn > 5:
        md += f"\n- Discovered hidden cave (Turn {turn - 2})"
    if turn > 15:
        md += f"\n- Alliance with forest spirits (Turn {turn - 5})"
    if turn > 30:
        md += f"\n- Crown fragment located (Turn {turn - 3})"

    dynamic = DYNAMIC_SUFFIX_BASE
    if turn > 10:
        dynamic += "\n- Shattered Crown: Legendary artifact split into 5 pieces"
    if turn > 20:
        dynamic += "\n- Shadow Guild: Criminal organization opposing the party"

    return md, dynamic


def build_anthropic_body(messages: list[dict], model: str, max_tokens: int = 512,
                         auto_cache: bool = False) -> dict:
    """Convert _build_cacheable_messages output to Anthropic API body format.

    Mirrors saga.llm.client.LLMClient._call_anthropic body construction.
    If auto_cache=True, uses top-level cache_control (Anthropic automatic caching)
    instead of per-block explicit breakpoints.
    """
    system_parts = []
    non_system = []

    for msg in messages:
        if msg["role"] == "system":
            part = {"type": "text", "text": msg["content"]}
            if not auto_cache and msg.get("cache_control"):
                part["cache_control"] = msg["cache_control"]
            system_parts.append(part)
        else:
            if not auto_cache and msg.get("cache_control"):
                entry = {
                    "role": msg["role"],
                    "content": [
                        {"type": "text", "text": msg["content"],
                         "cache_control": msg["cache_control"]}
                    ],
                }
            else:
                entry = {"role": msg["role"], "content": msg["content"]}
            non_system.append(entry)

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "messages": non_system,
    }
    if system_parts:
        body["system"] = system_parts
    if auto_cache:
        body["cache_control"] = {"type": "ephemeral"}

    return body


async def run_benchmark(turns: int, no_cache: bool, model: str, auto_cache: bool = False,
                        trace: bool = False):
    """Run the prompt caching benchmark."""
    # Load config
    config_path = os.environ.get("SAGA_CONFIG", "config.yaml")
    cfg = load_config(config_path)

    api_key = cfg.api_keys.anthropic
    if not api_key:
        print("ERROR: No Anthropic API key found in config.yaml")
        sys.exit(1)

    # Monkeypatch server module's config for _build_cacheable_messages
    if no_cache:
        cfg.prompt_caching.enabled = False
    cfg.models.narration = model
    server_module.config = cfg

    if auto_cache:
        cache_label = "ENABLED (automatic top-level)"
    elif no_cache:
        cache_label = "DISABLED (baseline)"
    else:
        cache_label = "ENABLED (3-BP explicit)"

    print(f"=== SAGA Prompt Caching Benchmark ({turns} turns) ===")
    print(f"Model: {model}")
    print(f"Caching: {cache_label}")
    print()

    # Table header
    header = f"{'Turn':>4} | {'Latency':>9} | {'Input':>7} | {'CacheRead':>10} | {'CacheCrte':>10} | {'Output':>7} | {'HitRate':>8}"
    print(header)
    print("-" * len(header))

    # Conversation state
    conversation: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    results = []
    traces = []  # Per-turn trace data (if --trace enabled)

    async with httpx.AsyncClient(timeout=120.0) as http:
        for turn in range(1, turns + 1):
            # Add user message
            user_input = USER_INPUTS[(turn - 1) % len(USER_INPUTS)]
            conversation.append({"role": "user", "content": user_input})

            # Build messages for API call
            md_prefix, dynamic_suffix = build_dynamic_context(turn)
            if auto_cache:
                # Auto-cache: keep prefix stable, append dynamic context to last user msg only
                augmented = list(conversation)
                context_block = ""
                if md_prefix:
                    context_block += f"[--- SAGA Context Cache ---]\n{md_prefix}\n\n"
                if dynamic_suffix:
                    context_block += f"[--- SAGA Dynamic ---]\n{dynamic_suffix}"
                if context_block:
                    augmented[-1] = dict(augmented[-1])
                    augmented[-1]["content"] = context_block + "\n\n" + augmented[-1]["content"]
            else:
                # 3-BP: use SAGA's _build_cacheable_messages with explicit breakpoints
                augmented = server_module._build_cacheable_messages(
                    list(conversation), md_prefix, dynamic_suffix
                )

            # Build API body
            body = build_anthropic_body(augmented, model, max_tokens=300,
                                        auto_cache=auto_cache)

            # Call Anthropic API
            t_start = time.time()
            resp = await http.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json=body,
            )
            latency_ms = (time.time() - t_start) * 1000

            if resp.status_code != 200:
                print(f"  Turn {turn} FAILED: {resp.status_code} {resp.text[:200]}")
                break

            data = resp.json()
            usage = data.get("usage", {})

            input_tokens = usage.get("input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_create = usage.get("cache_creation_input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)

            total_input = input_tokens + cache_read + cache_create
            hit_rate = (cache_read / total_input * 100) if total_input > 0 else 0.0

            # Cost calculations
            cost_cached = (
                input_tokens * PRICE_INPUT
                + cache_read * PRICE_CACHE_READ
                + cache_create * PRICE_CACHE_CREATE
                + output_tokens * PRICE_OUTPUT
            )
            cost_nocache = (
                total_input * PRICE_INPUT
                + output_tokens * PRICE_OUTPUT
            )

            result = {
                "turn": turn,
                "latency_ms": latency_ms,
                "input_tokens": input_tokens,
                "cache_read": cache_read,
                "cache_create": cache_create,
                "output_tokens": output_tokens,
                "hit_rate": hit_rate,
                "cost_cached": cost_cached,
                "cost_nocache": cost_nocache,
            }
            results.append(result)

            print(
                f"{turn:>4} | {latency_ms:>7.0f}ms | {input_tokens:>7} | {cache_read:>10} | {cache_create:>10} | {output_tokens:>7} | {hit_rate:>6.1f}%"
            )

            # Extract assistant response and append to conversation
            assistant_text = "".join(
                block.get("text", "") for block in data.get("content", [])
            )
            conversation.append({"role": "assistant", "content": assistant_text})

            # Trace: record full turn I/O
            if trace:
                # Summarize request messages (truncate content for readability)
                req_summary = []
                for msg in (body.get("system", []) if isinstance(body.get("system"), list) else []):
                    req_summary.append({
                        "type": "system",
                        "text_len": len(msg.get("text", "")),
                        "text_preview": msg.get("text", "")[:200] + "...",
                        "cache_control": msg.get("cache_control"),
                    })
                for msg in body.get("messages", []):
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        text = content[0].get("text", "") if content else ""
                        cc = content[0].get("cache_control") if content else None
                    else:
                        text = content
                        cc = None
                    req_summary.append({
                        "role": msg["role"],
                        "text_len": len(text),
                        "text_preview": text[:200] + ("..." if len(text) > 200 else ""),
                        "cache_control": cc,
                    })
                traces.append({
                    "turn": turn,
                    "user_input": user_input,
                    "dynamic_context": {
                        "md_prefix_len": len(md_prefix),
                        "dynamic_suffix_len": len(dynamic_suffix),
                    },
                    "request_messages": req_summary,
                    "top_level_cache_control": body.get("cache_control"),
                    "response": {
                        "assistant_text": assistant_text,
                        "usage": usage,
                        "model": data.get("model", ""),
                        "stop_reason": data.get("stop_reason", ""),
                    },
                    "metrics": result,
                })

            # Small delay to avoid rate limiting
            if turn < turns:
                await asyncio.sleep(0.5)

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print_summary(results, model, cache_label)

    # Save report (separate files per mode to avoid overwriting)
    if auto_cache:
        suffix = "auto"
    elif no_cache:
        suffix = "nocache"
    else:
        suffix = "3bp"
    report_path = Path(__file__).parent / f"bench_report_{suffix}.txt"
    save_report(results, model, cache_label, report_path)
    print(f"\nReport saved to: {report_path}")

    # Save trace if enabled
    if trace and traces:
        trace_path = Path(__file__).parent / f"bench_trace_{suffix}.json"
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(traces, f, indent=2, ensure_ascii=False)
        print(f"Trace saved to: {trace_path}")


def print_summary(results: list[dict], model: str, cache_label: str):
    """Print aggregated benchmark statistics."""
    if not results:
        print("No results to summarize.")
        return

    n = len(results)
    avg_latency = sum(r["latency_ms"] for r in results) / n
    first_latency = results[0]["latency_ms"]
    last_10 = results[-10:] if n >= 10 else results
    last_10_avg = sum(r["latency_ms"] for r in last_10) / len(last_10)

    # Cache stats (skip first turn for hit rate average since it's always 0)
    steady_results = [r for r in results if r["turn"] >= 5]
    avg_hit_rate = (
        sum(r["hit_rate"] for r in steady_results) / len(steady_results)
        if steady_results else 0.0
    )

    total_input = sum(r["input_tokens"] for r in results)
    total_cache_read = sum(r["cache_read"] for r in results)
    total_cache_create = sum(r["cache_create"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)

    total_cost_cached = sum(r["cost_cached"] for r in results)
    total_cost_nocache = sum(r["cost_nocache"] for r in results)
    savings = (
        (1 - total_cost_cached / total_cost_nocache) * 100
        if total_cost_nocache > 0 else 0.0
    )

    print(f"=== Summary ===")
    print(f"Total turns: {n}")
    print(f"Model: {model}")
    print(f"Caching: {cache_label}")
    print()
    print(f"Avg latency: {avg_latency:.0f}ms (first: {first_latency:.0f}ms, last 10 avg: {last_10_avg:.0f}ms)")
    print(f"Avg cache hit rate: {avg_hit_rate:.1f}% (turns 5+)")
    print()
    print(f"Total input tokens:       {total_input:>10,}")
    print(f"Total cache read tokens:  {total_cache_read:>10,}")
    print(f"Total cache create tokens:{total_cache_create:>10,}")
    print(f"Total output tokens:      {total_output:>10,}")
    print()
    print(f"Total cost (cached):   ${total_cost_cached:.4f}")
    print(f"Total cost (no cache): ${total_cost_nocache:.4f}")
    print(f"Savings: {savings:.1f}%")


def save_report(results: list[dict], model: str, cache_label: str, path: Path):
    """Save full report to file."""
    lines = []
    lines.append(f"=== SAGA Prompt Caching Benchmark ({len(results)} turns) ===")
    lines.append(f"Model: {model}")
    lines.append(f"Caching: {cache_label}")
    lines.append("")

    header = f"{'Turn':>4} | {'Latency':>9} | {'Input':>7} | {'CacheRead':>10} | {'CacheCrte':>10} | {'Output':>7} | {'HitRate':>8} | {'Cost$':>8} | {'NoCacheCost$':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        lines.append(
            f"{r['turn']:>4} | {r['latency_ms']:>7.0f}ms | {r['input_tokens']:>7} | "
            f"{r['cache_read']:>10} | {r['cache_create']:>10} | {r['output_tokens']:>7} | "
            f"{r['hit_rate']:>6.1f}% | ${r['cost_cached']:>.5f} | ${r['cost_nocache']:>.5f}"
        )

    lines.append("")

    # Summary section
    if results:
        n = len(results)
        avg_latency = sum(r["latency_ms"] for r in results) / n
        steady = [r for r in results if r["turn"] >= 5]
        avg_hit = sum(r["hit_rate"] for r in steady) / len(steady) if steady else 0
        total_cached = sum(r["cost_cached"] for r in results)
        total_nocache = sum(r["cost_nocache"] for r in results)
        savings = (1 - total_cached / total_nocache) * 100 if total_nocache > 0 else 0

        lines.append("=== Summary ===")
        lines.append(f"Avg latency: {avg_latency:.0f}ms")
        lines.append(f"Avg cache hit rate (turns 5+): {avg_hit:.1f}%")
        lines.append(f"Total cost (cached):   ${total_cached:.4f}")
        lines.append(f"Total cost (no cache): ${total_nocache:.4f}")
        lines.append(f"Savings: {savings:.1f}%")

    # Also save raw JSON for further analysis
    json_path = path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="SAGA Prompt Caching Benchmark")
    parser.add_argument("--turns", type=int, default=50, help="Number of turns (default: 50)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching for baseline comparison")
    parser.add_argument("--auto-cache", action="store_true", help="Use top-level automatic caching instead of 3-BP")
    parser.add_argument("--trace", action="store_true", help="Save full I/O trace per turn (bench_trace_*.json)")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Anthropic model to use")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.turns, args.no_cache, args.model, args.auto_cache, args.trace))


if __name__ == "__main__":
    main()
