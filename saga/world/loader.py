"""World data loader — CHARACTERS.md / LOREBOOK.md / WORLD.md → Kuzu + ChromaDB."""
import os
import logging
from saga.storage.graph_db import GraphDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.utils.parsers import parse_characters_md, parse_lorebook_md

logger = logging.getLogger(__name__)


class WorldLoader:
    def __init__(self, graph_db: GraphDB, vector_db: VectorDB, md_cache: MdCache):
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.md_cache = md_cache

    async def load_world(self, session_id: str, world_dir: str):
        """Load world data from .md files into Kuzu + ChromaDB, then bootstrap .md cache."""
        logger.info(f"[WorldLoader] Loading world from {world_dir} for session {session_id}")

        # Load characters
        chars_path = os.path.join(world_dir, "CHARACTERS.md")
        characters = parse_characters_md(chars_path)
        for char in characters:
            self.graph_db.create_character(
                session_id=session_id,
                name=char["name"],
                is_player=char.get("is_player", False),
                hp=char.get("hp", 100),
                max_hp=char.get("max_hp", 100),
                location=char.get("location", "unknown"),
                mood=char.get("mood", "neutral"),
                traits=char.get("traits", []),
                custom=char.get("custom", {})
            )
            # Create location node for character's initial location
            if char.get("location") and char["location"] != "unknown":
                self.graph_db.create_location(session_id, char["location"])
                self.graph_db.update_character_location(session_id, char["name"], char["location"], 0)
            logger.info(f"[WorldLoader] Loaded character: {char['name']} (player={char.get('is_player')})")

        # Load lorebook
        lore_path = os.path.join(world_dir, "LOREBOOK.md")
        entries = parse_lorebook_md(lore_path)
        for entry in entries:
            lore_id = f"{session_id}_{entry['name']}"
            self.graph_db.create_lore(
                session_id=session_id,
                name=entry["name"],
                lore_type=entry.get("type", "lore"),
                layer=entry.get("layer", "A1")
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
                    "kuzu_node_id": lore_id
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
        """Create initial .md cache files from loaded world data."""
        import json
        from datetime import datetime

        now = datetime.now().isoformat()
        fm = f'---\nupdated_at: "{now}"\nturn: 0\nsession_id: {session_id}\nchanged: [init]\n---\n\n'

        # state.md
        player = next((c for c in characters if c.get("is_player")), None)
        if player:
            items_str = "없음"
            state_md = fm + f"""## 현재 상태
- **플레이어:** {player['name']} | HP: {player.get('hp', 100)}/{player.get('max_hp', 100)} | 위치: {player.get('location', 'unknown')}
- **인벤토리:** {items_str}
- **분위기:** {player.get('mood', 'neutral')}"""
        else:
            state_md = fm + "## 현재 상태\n(플레이어 미설정)"

        # relations.md
        relations_md = fm + "## 관계 요약\n(아직 관계가 형성되지 않음)"

        # story.md
        story_md = fm + "## 서사 흐름\n(이야기가 시작되지 않음)\n\n## 미해결 복선\n(없음)\n\n## 핵심 키워드\n(없음)"

        # lore.md
        lore_lines = ["## 활성 로어북"]
        for entry in entries[:10]:
            preview = entry['text'][:100] if entry.get('text') else ''
            lore_lines.append(f"- **{entry['name']}:** {preview}")
        lore_md = fm + "\n".join(lore_lines)

        contents = {"state.md": state_md, "relations.md": relations_md, "story.md": story_md, "lore.md": lore_md}
        await self.md_cache.write_cache_atomic(session_id, 0, contents, ["init"])
