"""Sub-B: Post-turn Extractor — 매 턴 비동기, Flash급, 유저 안 기다림.

state 블록 파싱 → Kuzu + SQLite + ChromaDB 업데이트 → .md 원자적 갱신
asyncio.Event 락으로 빠른 연속 입력 방지
"""
import asyncio
import logging
import json
import re
from datetime import datetime
from mene.storage.sqlite_db import SQLiteDB
from mene.storage.graph_db import GraphDB
from mene.storage.vector_db import VectorDB
from mene.storage.md_cache import MdCache
from mene.utils.parsers import parse_state_block, format_turn_narrative
from mene.llm.client import LLMClient

logger = logging.getLogger(__name__)

# Sub-B concurrency lock
_sub_b_lock = asyncio.Event()
_sub_b_lock.set()  # Initially open


class PostTurnExtractor:
    def __init__(self, sqlite_db: SQLiteDB, graph_db: GraphDB, vector_db: VectorDB, md_cache: MdCache, llm_client: LLMClient, config):
        self.sqlite_db = sqlite_db
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config

    async def extract_and_update(self, session_id: str, response_text: str, turn_number: int, user_input: str = ""):
        """Main entry point. Called as asyncio.create_task after response is sent."""

        # Wait for previous Sub-B to finish
        await _sub_b_lock.wait()
        _sub_b_lock.clear()

        try:
            # 1. Parse state block
            state_block = parse_state_block(response_text)

            if state_block is None:
                # Regex failed → log tail for debugging, then try Flash
                tail = response_text[-500:] if len(response_text) > 500 else response_text
                logger.warning(f"[Sub-B] Regex parse failed for turn {turn_number}, trying Flash extraction. Response tail:\n{tail}")
                state_block = await self._extract_with_flash(response_text)

            if state_block is None:
                # Flash also failed → skip this turn's state tracking
                logger.warning(f"[Sub-B] Flash extraction also failed for turn {turn_number}, skipping")
                await self.sqlite_db.insert_turn_log(session_id, turn_number, None, user_input=user_input, assistant_output=response_text)
                return

            # 2. Update Kuzu graph
            await self._update_graph(session_id, state_block, turn_number)

            # 3. Update SQLite
            await self._update_sqlite(session_id, state_block, turn_number, user_input, response_text)

            # 4. Record episode to ChromaDB
            self._record_episode(session_id, turn_number, user_input, response_text, state_block)

            # 5. Atomic .md cache update
            await self._update_md_cache(session_id, turn_number, state_block)

            logger.info(f"[Sub-B] Turn {turn_number} post-processing complete")

        except Exception as e:
            logger.error(f"[Sub-B] Error processing turn {turn_number}: {e}", exc_info=True)
        finally:
            _sub_b_lock.set()

    async def _extract_with_flash(self, response_text: str) -> dict | None:
        """Use Flash-tier LLM to extract state from unstructured response."""
        try:
            result = await self.llm_client.call_llm(
                model=self.config.models.extraction,
                messages=[
                    {"role": "system", "content": "Extract game state changes from the RP response. Return ONLY valid JSON (no markdown, no explanation) with keys: location, location_moved, hp_change, items_gained, items_lost, items_transferred, npc_met, npc_separated, relationship_changes, mood, event_trigger, notes. Use defaults (false, 0, [], null) for missing values."},
                    {"role": "user", "content": response_text}
                ],
                temperature=0.1,
                max_tokens=1024
            )
            # Try direct parse first
            result = result.strip()
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                pass
            # Try extracting JSON from markdown code block
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)```', result, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1).strip())
            # Try finding first { ... } block
            brace_match = re.search(r'\{.*\}', result, re.DOTALL)
            if brace_match:
                return json.loads(brace_match.group(0))
            logger.warning(f"[Sub-B] Flash returned unparseable: {result[:200]}")
            return None
        except Exception as e:
            logger.error(f"[Sub-B] Flash extraction failed: {e}")
            return None

    async def _update_graph(self, session_id: str, state_block: dict, turn_number: int):
        """Update Kuzu graph from state block."""
        loop = asyncio.get_event_loop()

        # Location move
        if state_block.get("location_moved"):
            new_location = state_block.get("location", "")
            if new_location:
                await loop.run_in_executor(None, self.graph_db.create_location, session_id, new_location)
                await loop.run_in_executor(None, self.graph_db.update_character_location, session_id, "", new_location, turn_number)

        # New NPCs met
        for npc_name in state_block.get("npc_met", []):
            await loop.run_in_executor(None, self.graph_db.create_character, session_id, npc_name, False)
            await loop.run_in_executor(None, self.graph_db.create_relationship, session_id, "", npc_name, "met", 30)

        # NPC separated
        for npc_name in state_block.get("npc_separated", []):
            pass  # Remove LOCATED_AT edge — handled by graph_db

        # Relationship changes
        for change in state_block.get("relationship_changes", []):
            await loop.run_in_executor(
                None, self.graph_db.update_relationship,
                session_id, change.get("from", ""), change.get("to", ""),
                change.get("type", ""), change.get("delta", 0)
            )

        # Items gained
        for item_name in state_block.get("items_gained", []):
            await loop.run_in_executor(None, self.graph_db.create_item, session_id, item_name)
            await loop.run_in_executor(None, self.graph_db.add_ownership, session_id, "", item_name)

        # Items lost
        for item_name in state_block.get("items_lost", []):
            await loop.run_in_executor(None, self.graph_db.remove_ownership, session_id, "", item_name)

        # Items transferred
        for transfer in state_block.get("items_transferred", []):
            await loop.run_in_executor(
                None, self.graph_db.transfer_item,
                session_id, transfer.get("item", ""), "", transfer.get("to", "")
            )

        # HP change
        hp_delta = state_block.get("hp_change", 0)
        if hp_delta != 0:
            await loop.run_in_executor(None, self.graph_db.update_character_hp, session_id, hp_delta)

        # Mood
        if state_block.get("mood"):
            await loop.run_in_executor(None, self.graph_db.update_character_mood, session_id, state_block["mood"])

    async def _update_sqlite(self, session_id, state_block, turn_number, user_input, response_text):
        """Update SQLite tables."""
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
        """Record episode summary to ChromaDB."""
        summary = format_turn_narrative(turn_number, user_input, response_text, state_block)
        location = state_block.get("location", "unknown")
        self.vector_db.add_episode(session_id, turn_number, summary, location)

    async def _update_md_cache(self, session_id, turn_number, state_block):
        """Build fresh .md from DB and write atomically."""
        # Track what changed
        changed = []
        if state_block.get("location_moved"): changed.append("location")
        if state_block.get("hp_change"): changed.append("hp")
        if state_block.get("items_gained") or state_block.get("items_lost") or state_block.get("items_transferred"):
            changed.append("inventory")
        if state_block.get("npc_met") or state_block.get("npc_separated"):
            changed.append("npcs")
        if state_block.get("relationship_changes"):
            changed.append("relationships")

        # Build fresh .md content from DB
        player = self.graph_db.query_player_context(session_id)
        relationships = self.graph_db.get_relationships(session_id, player.get("name", "")) if player else []

        now = datetime.now().isoformat()
        frontmatter = f'---\nupdated_at: "{now}"\nturn: {turn_number}\nsession_id: {session_id}\nchanged: {json.dumps(changed)}\n---\n\n'

        # Format state.md
        items_str = ", ".join(player.get("items", [])) if player else "없음"
        state_md = frontmatter + f"""## 현재 상태
- **플레이어:** {player.get('name', '?') if player else '?'} | HP: {player.get('hp', '?') if player else '?'}/{player.get('max_hp', 100) if player else 100} | 위치: {player.get('location', '?') if player else '?'}
- **인벤토리:** {items_str}
- **분위기:** {state_block.get('mood', 'neutral')}"""

        # Format relations.md
        rel_lines = ["## 관계 요약"]
        for r in relationships:
            rel_lines.append(f"- {r.get('from', '?')} ↔ {r.get('to', '?')}: {r.get('type', '?')}(강도: {r.get('strength', '?')})")
        relations_md = frontmatter + "\n".join(rel_lines)

        # story.md — append new event
        story_md = frontmatter + f"## 서사 흐름\n- Turn {turn_number}: {state_block.get('notes', '진행 중')}"

        # lore.md — keep as-is or empty
        lore_md = frontmatter + "## 활성 로어북\n(동적 로어북은 Sub-A에서 실시간 조회)"

        contents = {
            "state.md": state_md,
            "relations.md": relations_md,
            "story.md": story_md,
            "lore.md": lore_md
        }

        await self.md_cache.write_cache_atomic(session_id, turn_number, contents, changed)
