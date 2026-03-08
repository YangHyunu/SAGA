"""Sub-B: Post-turn Extractor — 매 턴 비동기, Flash급, 유저 안 기다림.

state 블록 파싱 → SQLite + ChromaDB 업데이트 → live_state.md 갱신
asyncio.Lock으로 빠른 연속 입력 방지
"""
import asyncio
import logging
import json
import time
from typing import Callable
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.utils.parsers import format_turn_narrative
from saga.llm.client import LLMClient
logger = logging.getLogger(__name__)

# Sub-B concurrency lock
_sub_b_lock = asyncio.Lock()


class PostTurnExtractor:
    # Korean honorific suffixes to strip for matching (longest first)
    _KO_SUFFIXES = ("선배", "후배", "언니", "오빠", "누나", "씨", "님", "양", "군", "형")
    # Common Korean surnames for suffix-match validation
    _KO_SURNAMES = set("김이박최정강조윤장임한오서신권황안송류홍전배노하유남심")

    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache, llm_client: LLMClient, config, extract_fn: Callable = None):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config
        self.extract_fn = extract_fn

    @classmethod
    def _strip_suffix(cls, name: str) -> str:
        """Strip Korean honorific suffixes to get a match key."""
        stripped = name.strip()
        for suffix in cls._KO_SUFFIXES:
            if stripped.endswith(suffix) and len(stripped) > len(suffix):
                stripped = stripped[:-len(suffix)].strip()
                break  # Only strip one suffix
        return stripped

    @classmethod
    def _match_existing_name(cls, raw_name: str, existing_names: list[str]) -> str:
        """Match raw_name against existing character names using suffix matching.

        Returns the existing canonical name if matched, otherwise the raw_name as-is.
        """
        if not raw_name or len(raw_name) < 2:
            return raw_name

        key = cls._strip_suffix(raw_name).lower()
        if len(key) < 2:
            return raw_name

        for existing in existing_names:
            ex_key = cls._strip_suffix(existing).lower()
            if len(ex_key) < 2:
                continue

            # Exact match after stripping
            if key == ex_key:
                return existing

            # Suffix match: shorter is suffix of longer
            shorter, longer = (key, ex_key) if len(key) <= len(ex_key) else (ex_key, key)
            if longer.endswith(shorter):
                prefix = longer[:-len(shorter)]
                # If prefix is 1 char, only allow if it's a Korean surname
                if len(prefix) == 1 and prefix not in cls._KO_SURNAMES:
                    continue
                # If prefix is >1 char, skip (too risky for false positives)
                if len(prefix) > 1:
                    continue
                return existing

        return raw_name

    async def _normalize_state_block_names(self, session_id: str, state_block: dict) -> None:
        """Normalize all NPC names in state_block against existing characters (in-place)."""
        existing_chars = await self.sqlite_db.get_session_characters(session_id)
        existing_names = [c["name"] for c in existing_chars if not c.get("is_player")]

        if not existing_names:
            return

        # npc_met
        if state_block.get("npc_met"):
            state_block["npc_met"] = [
                self._match_existing_name(n, existing_names)
                for n in state_block["npc_met"] if n
            ]

        # npc_separated
        if state_block.get("npc_separated"):
            state_block["npc_separated"] = [
                self._match_existing_name(n, existing_names)
                for n in state_block["npc_separated"] if n
            ]

        # relationship_changes
        for change in (state_block.get("relationship_changes") or []):
            if not isinstance(change, dict):
                continue
            if change.get("from"):
                change["from"] = self._match_existing_name(change["from"], existing_names)
            if change.get("to"):
                change["to"] = self._match_existing_name(change["to"], existing_names)

        # items_transferred
        for transfer in (state_block.get("items_transferred") or []):
            if not isinstance(transfer, dict):
                continue
            if transfer.get("to"):
                transfer["to"] = self._match_existing_name(transfer["to"], existing_names)

    async def extract_and_update(self, session_id: str, response_text: str, turn_number: int, user_input: str = ""):
        """Main entry point. Called as asyncio.create_task after response is sent."""
        logger.info(f"[Sub-B] ▶ Task started for turn {turn_number}, session={session_id}")

        # Wait for previous Sub-B to finish
        await _sub_b_lock.acquire()
        logger.info(f"[Sub-B] Lock acquired for turn {turn_number}")

        t_start = time.monotonic()
        state_block = None
        importance = 0
        try:
            # 1. Extract state block via injected extract_fn
            state_block = await self.extract_fn(response_text, session_id)

            if state_block is None:
                # Both regex and Flash failed → skip this turn's state tracking
                logger.warning(f"[Sub-B] Extraction failed for turn {turn_number}, skipping")
                await self.sqlite_db.insert_turn_log(session_id, turn_number, None, user_input=user_input, assistant_output=response_text)
                return

            # 1.5. Normalize NPC names against existing characters (in-place)
            await self._normalize_state_block_names(session_id, state_block)

            # 2. Update SQLite state tables
            await self._update_state_db(session_id, state_block, turn_number)

            # 3. Record important events
            importance = self._calculate_importance(state_block)
            if importance >= 40:
                await self._record_event(session_id, state_block, turn_number, importance)

            # 4. Update SQLite turn log
            await self._update_sqlite(session_id, state_block, turn_number, user_input, response_text)

            # 5. Record episode to ChromaDB
            self._record_episode(session_id, turn_number, user_input, response_text, state_block)

            # 6. Update live_state.md
            await self._update_md_cache(session_id, turn_number, state_block)

            logger.info(f"[Sub-B] Turn {turn_number} post-processing complete (importance={importance})")

        except Exception as e:
            logger.error(f"[Sub-B] Error processing turn {turn_number}: {e}", exc_info=True)
        finally:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.info(f"[Sub-B] Releasing lock for turn {turn_number}, elapsed={elapsed_ms:.0f}ms")
            _sub_b_lock.release()

    async def _update_state_db(self, session_id: str, state_block: dict, turn_number: int):
        """Update SQLite tables from state block."""
        # Ensure player character exists (idempotent)
        await self.sqlite_db.ensure_player(session_id)

        # Always sync player location if provided
        location = state_block.get("location", "")
        if location:
            await self.sqlite_db.update_character_location(session_id, "", location, turn_number)
            if state_block.get("location_moved"):
                await self.sqlite_db.create_location(session_id, location, turn_number)

        # New NPCs met
        for npc_name in (state_block.get("npc_met") or []):
            await self.sqlite_db.create_character(session_id, npc_name, is_player=False)
            await self.sqlite_db.create_relationship(session_id, "", npc_name, "met", 30)

        # Relationship changes
        for change in (state_block.get("relationship_changes") or []):
            if not isinstance(change, dict):
                logger.debug(f"[Sub-B] Skipping non-dict relationship_change: {change!r}")
                continue
            try:
                await self.sqlite_db.update_relationship(
                    session_id, (change.get("from") or ""), (change.get("to") or ""),
                    (change.get("type") or ""), (change.get("delta") or 0)
                )
            except Exception as e:
                logger.warning(f"[Sub-B] update_relationship failed: {e}")

        # Items gained
        for item in (state_block.get("items_gained") or []):
            current = await self.sqlite_db.get_world_state_value(session_id, "inventory") or "[]"
            inv = json.loads(current)
            if item not in inv:
                inv.append(item)
            await self.sqlite_db.upsert_world_state(session_id, "inventory", json.dumps(inv, ensure_ascii=False))

        # Items lost
        for item in (state_block.get("items_lost") or []):
            current = await self.sqlite_db.get_world_state_value(session_id, "inventory") or "[]"
            inv = json.loads(current)
            if item in inv:
                inv.remove(item)
            await self.sqlite_db.upsert_world_state(session_id, "inventory", json.dumps(inv, ensure_ascii=False))

        # HP
        hp_delta = state_block.get("hp_change") or 0
        if hp_delta != 0:
            await self.sqlite_db.update_character_hp(session_id, hp_delta)

        # Mood
        if state_block.get("mood"):
            await self.sqlite_db.update_character_mood(session_id, state_block["mood"])

    async def _update_sqlite(self, session_id, state_block, turn_number, user_input, response_text):
        """Update SQLite turn log and basic world_state KV (location, mood)."""
        if state_block.get("location"):
            await self.sqlite_db.upsert_world_state(session_id, "player_location", state_block["location"])
        if state_block.get("mood"):
            await self.sqlite_db.upsert_world_state(session_id, "player_mood", state_block["mood"])

        token_count = 0  # Will be filled by caller
        await self.sqlite_db.insert_turn_log(
            session_id, turn_number, state_block,
            user_input=user_input, assistant_output=response_text,
            token_count=token_count
        )

    def _record_episode(self, session_id, turn_number, user_input, response_text, state_block):
        """Record episode summary to ChromaDB with importance scoring and entity tagging."""
        summary = format_turn_narrative(turn_number, user_input, response_text, state_block)
        location = (state_block.get("location") or "unknown")
        importance = self._calculate_importance(state_block)
        entities = self._extract_entities(state_block)
        npcs = list(state_block.get("npc_met") or []) + list(state_block.get("npc_separated") or [])
        # Add relationship targets as NPCs too
        for change in (state_block.get("relationship_changes") or []):
            if not isinstance(change, dict):
                continue
            for key in ("from", "to"):
                name = change.get(key, "")
                if name and name not in npcs:
                    npcs.append(name)

        episode_type = self._classify_episode(state_block)
        self.vector_db.add_episode(
            session_id, turn_number, summary, location,
            episode_type=episode_type,
            importance=importance,
            entities=entities,
            npcs=npcs,
        )
        if importance >= 50:
            logger.info(f"[Sub-B] High-importance episode (turn {turn_number}, score={importance}, type={episode_type})")

    @staticmethod
    def _calculate_importance(state_block: dict) -> int:
        """Calculate episode importance score (0-100) from state changes."""
        score = 10  # base score for any turn

        hp_change = abs(state_block.get("hp_change") or 0)
        if hp_change > 0:
            score += min(30, hp_change * 3)  # up to 30 for combat/damage

        if state_block.get("relationship_changes"):
            score += 10 * min(3, len(state_block["relationship_changes"]))  # up to 30

        if state_block.get("event_trigger"):
            score += 35  # major story events

        if state_block.get("npc_met"):
            score += 10 * min(2, len(state_block["npc_met"]))  # up to 20

        if state_block.get("npc_separated"):
            score += 15

        if state_block.get("items_gained") or state_block.get("items_lost"):
            score += 15

        if state_block.get("items_transferred"):
            score += 20  # inter-character interaction

        if state_block.get("location_moved"):
            score += 10

        return min(score, 100)

    @staticmethod
    def _extract_entities(state_block: dict) -> list[str]:
        """Extract all named entities from state block for tagging."""
        entities = []
        if state_block.get("location"):
            entities.append(state_block["location"])
        for name in (state_block.get("npc_met") or []):
            entities.append(name)
        for name in (state_block.get("npc_separated") or []):
            entities.append(name)
        for item in (state_block.get("items_gained") or []):
            entities.append(item)
        for item in (state_block.get("items_lost") or []):
            entities.append(item)
        for transfer in (state_block.get("items_transferred") or []):
            if not isinstance(transfer, dict):
                continue
            if transfer.get("item"):
                entities.append(transfer["item"])
            if transfer.get("to"):
                entities.append(transfer["to"])
        for change in (state_block.get("relationship_changes") or []):
            if not isinstance(change, dict):
                continue
            for key in ("from", "to"):
                if change.get(key):
                    entities.append(change[key])
        return list(dict.fromkeys(entities))  # dedupe preserving order

    @staticmethod
    def _classify_episode(state_block: dict) -> str:
        """Classify episode type for filtering."""
        if abs(state_block.get("hp_change") or 0) > 0:
            return "combat"
        if state_block.get("event_trigger"):
            return "event"
        if state_block.get("relationship_changes"):
            return "relationship"
        if state_block.get("npc_met"):
            return "encounter"
        if state_block.get("items_gained") or state_block.get("items_lost") or state_block.get("items_transferred"):
            return "item"
        if state_block.get("location_moved"):
            return "exploration"
        return "dialogue"

    async def _record_event(self, session_id: str, state_block: dict, turn_number: int, importance: int):
        """Record high-importance episodes as Event rows in SQLite."""
        event_type = self._classify_episode(state_block)
        event_name = f"turn_{turn_number}_{event_type}"
        description = state_block.get("notes", "") or f"{event_type} at turn {turn_number}"
        entities = self._extract_entities(state_block)
        await self.sqlite_db.create_event(
            session_id, event_name, event_type, description, turn_number, importance, entities
        )

    async def _update_md_cache(self, session_id, turn_number, state_block):
        """Write live_state.md with current game state."""
        player_ctx = await self.sqlite_db.query_player_context(session_id)
        await self.md_cache.write_live(session_id, turn_number, state_block, player_ctx)
