"""World data loader — CHARACTERS.md / LOREBOOK.md / WORLD.md → SQLite + ChromaDB."""
import os
import logging
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.utils.parsers import parse_characters_md, parse_lorebook_md

logger = logging.getLogger(__name__)


class WorldLoader:
    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache

    async def load_world(self, session_id: str, world_dir: str):
        """Load world data from .md files into SQLite + ChromaDB, then bootstrap .md cache."""
        logger.info(f"[WorldLoader] Loading world from {world_dir} for session {session_id}")

        # Load characters
        chars_path = os.path.join(world_dir, "CHARACTERS.md")
        characters = parse_characters_md(chars_path)
        for char in characters:
            await self.sqlite_db.create_character(
                session_id=session_id,
                name=char["name"],
                is_player=char.get("is_player", False),
                hp=char.get("hp", 100),
                max_hp=char.get("max_hp", 100),
                location=char.get("location", "unknown"),
                mood=char.get("mood", "neutral"),
            )
            # Create location node for character's initial location
            if char.get("location") and char["location"] != "unknown":
                await self.sqlite_db.create_location(session_id, char["location"], 0)
                await self.sqlite_db.update_character_location(session_id, char["name"], char["location"], 0)
            logger.info(f"[WorldLoader] Loaded character: {char['name']} (player={char.get('is_player')})")

        # Load lorebook
        lore_path = os.path.join(world_dir, "LOREBOOK.md")
        entries = parse_lorebook_md(lore_path)
        for entry in entries:
            lore_id = f"{session_id}_{entry['name']}"
            await self.sqlite_db.create_lore(
                session_id=session_id,
                name=entry["name"],
                lore_type=entry.get("type", "lore"),
                keywords=",".join(entry.get("tags", [])),
                content=entry.get("text", ""),
                priority=50,
            )
            self.vector_db.add_lorebook_entry(
                entry_id=lore_id,
                text=entry["text"],
                metadata={
                    "session_id": session_id,
                    "type": entry.get("type", "lore"),
                    "layer": entry.get("layer", "A1"),
                    "tags": ",".join(entry.get("tags", [])),
                    "name": entry["name"],
                    "kuzu_node_id": ""
                }
            )
            logger.info(f"[WorldLoader] Loaded lorebook: {entry['name']} (layer={entry.get('layer')})")

        # Load world description (if exists)
        world_path = os.path.join(world_dir, "WORLD.md")
        if os.path.exists(world_path):
            with open(world_path, 'r', encoding='utf-8') as f:
                world_desc = f.read()
            # Store as a special lorebook entry
            self.vector_db.add_lorebook_entry(
                entry_id=f"{session_id}_WORLD",
                text=world_desc,
                metadata={"session_id": session_id, "type": "world", "layer": "A1", "tags": "world,setting", "name": "WORLD", "kuzu_node_id": ""}
            )

        # Bootstrap .md cache
        await self._bootstrap_md_cache(session_id, characters, entries)
        logger.info(f"[WorldLoader] World loading complete. {len(characters)} characters, {len(entries)} lorebook entries")

    async def _bootstrap_md_cache(self, session_id, characters, entries):
        """Create initial cache files from loaded world data."""
        # Build stable prefix
        content = await self.md_cache.build_stable_content(characters, entries)
        await self.md_cache.write_stable(session_id, content)

        # Build initial live state
        player = next((c for c in characters if c.get("is_player")), None)
        if player:
            player_ctx = {
                "location": player.get("location", "unknown"),
                "hp": player.get("hp", 100),
                "max_hp": player.get("max_hp", 100),
                "mood": player.get("mood", "neutral"),
                "nearby_npcs": [],
                "recent_events": [],
            }
            await self.md_cache.write_live(session_id, 0, {}, player_ctx)
