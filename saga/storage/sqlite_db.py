import aiosqlite
import hashlib
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SQLiteDB:
    def __init__(self, db_path: str = "db/state.db"):
        self.db_path = db_path
        self._db = None

    async def initialize(self):
        """Create tables if not exist."""
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT,
                world_config TEXT,
                turn_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS world_state (
                session_id TEXT NOT NULL REFERENCES sessions(id),
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (session_id, key)
            );
            CREATE TABLE IF NOT EXISTS event_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                event_type TEXT NOT NULL,
                trigger_condition TEXT,
                priority INTEGER DEFAULT 0,
                payload TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS turn_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                turn_number INTEGER NOT NULL,
                user_input TEXT,
                assistant_output TEXT,
                state_changes TEXT,
                token_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, turn_number)
            );
            CREATE TABLE IF NOT EXISTS characters (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                name TEXT NOT NULL,
                aliases TEXT DEFAULT '[]',
                is_player BOOLEAN DEFAULT FALSE,
                hp INTEGER DEFAULT 100,
                max_hp INTEGER DEFAULT 100,
                location TEXT DEFAULT 'unknown',
                mood TEXT DEFAULT 'neutral',
                traits TEXT DEFAULT '[]',
                custom TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                from_name TEXT NOT NULL,
                to_name TEXT NOT NULL,
                rel_type TEXT DEFAULT 'met',
                strength INTEGER DEFAULT 30,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, from_name, to_name)
            );
            CREATE TABLE IF NOT EXISTS locations (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                first_visit_turn INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                name TEXT NOT NULL,
                event_type TEXT DEFAULT '',
                description TEXT DEFAULT '',
                turn INTEGER DEFAULT 0,
                importance INTEGER DEFAULT 10,
                entities TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS lore (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                name TEXT NOT NULL,
                lore_type TEXT DEFAULT 'character',
                keywords TEXT DEFAULT '',
                content TEXT DEFAULT '',
                priority INTEGER DEFAULT 50,
                auto_generated BOOLEAN DEFAULT FALSE,
                source_turns TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_characters_session ON characters(session_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_session ON relationships(session_id);
            CREATE INDEX IF NOT EXISTS idx_events_session_importance ON events(session_id, importance);
            CREATE INDEX IF NOT EXISTS idx_lore_session ON lore(session_id);
        """)
        await self._db.commit()

    async def close(self):
        if self._db:
            await self._db.close()

    # ------------------------------------------------------------------ #
    # Session CRUD
    # ------------------------------------------------------------------ #

    async def create_session(
        self,
        session_id: str,
        name: str = "",
        world_config: str = "",
    ) -> dict:
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO sessions (id, name, world_config, turn_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (session_id, name, world_config, now, now),
        )
        await self._db.commit()
        return await self.get_session(session_id)

    async def get_session(self, session_id: str) -> dict | None:
        async with self._db.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def list_sessions(self) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    async def increment_turn(self, session_id: str) -> int:
        """Increment turn_count by 1 and return the new value."""
        await self._db.execute(
            """
            UPDATE sessions
            SET turn_count = turn_count + 1,
                updated_at = ?
            WHERE id = ?
            """,
            (datetime.utcnow().isoformat(), session_id),
        )
        await self._db.commit()
        return await self.get_turn_count(session_id)

    async def get_turn_count(self, session_id: str) -> int:
        async with self._db.execute(
            "SELECT turn_count FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return 0
            return row[0]

    async def reset_session(self, session_id: str):
        """Reset turn_count to 0 and clear world_state / turn_log / event_queue."""
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            "UPDATE sessions SET turn_count = 0, updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        await self._db.execute(
            "DELETE FROM world_state WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM turn_log WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM event_queue WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM characters WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM relationships WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM locations WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM events WHERE session_id = ?", (session_id,)
        )
        await self._db.execute(
            "DELETE FROM lore WHERE session_id = ?", (session_id,)
        )
        await self._db.commit()

    # ------------------------------------------------------------------ #
    # World State KV
    # ------------------------------------------------------------------ #

    async def upsert_world_state(self, session_id: str, key: str, value: str):
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO world_state (session_id, key, value, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (session_id, key, value, now),
        )
        await self._db.commit()
        logger.debug(f"[DB] upsert_world_state: session={session_id} key={key} value={value[:100]}")

    async def get_world_state(self, session_id: str) -> dict:
        async with self._db.execute(
            "SELECT key, value FROM world_state WHERE session_id = ?", (session_id,)
        ) as cursor:
            rows = await cursor.fetchall()
            return {r[0]: r[1] for r in rows}

    async def get_world_state_value(self, session_id: str, key: str) -> str | None:
        async with self._db.execute(
            "SELECT value FROM world_state WHERE session_id = ? AND key = ?",
            (session_id, key),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return row[0]

    # ------------------------------------------------------------------ #
    # Turn Log
    # ------------------------------------------------------------------ #

    async def insert_turn_log(
        self,
        session_id: str,
        turn_number: int,
        state_changes: dict | None,
        user_input: str = "",
        assistant_output: str = "",
        token_count: int = 0,
    ):
        state_changes_json = json.dumps(state_changes or {}, ensure_ascii=False)
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT OR REPLACE INTO turn_log
                (session_id, turn_number, user_input, assistant_output, state_changes, token_count, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                turn_number,
                user_input,
                assistant_output,
                state_changes_json,
                token_count,
                now,
            ),
        )
        await self._db.commit()
        fields = len(state_changes) if state_changes else 0
        logger.debug(f"[DB] insert_turn_log: session={session_id} turn={turn_number} fields={fields}")

    async def get_turn_logs(
        self,
        session_id: str,
        from_turn: int = 0,
        to_turn: int | None = None,
    ) -> list[dict]:
        if to_turn is None:
            async with self._db.execute(
                """
                SELECT * FROM turn_log
                WHERE session_id = ? AND turn_number >= ?
                ORDER BY turn_number ASC
                """,
                (session_id, from_turn),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute(
                """
                SELECT * FROM turn_log
                WHERE session_id = ? AND turn_number >= ? AND turn_number <= ?
                ORDER BY turn_number ASC
                """,
                (session_id, from_turn, to_turn),
            ) as cursor:
                rows = await cursor.fetchall()

        result = []
        for row in rows:
            r = dict(row)
            try:
                r["state_changes"] = json.loads(r.get("state_changes") or "{}")
            except (json.JSONDecodeError, TypeError):
                r["state_changes"] = {}
            result.append(r)
        return result

    # ------------------------------------------------------------------ #
    # Event Queue
    # ------------------------------------------------------------------ #

    async def queue_event(self, session_id: str, event: dict):
        """Insert an event into the queue. Expects keys: event_type, trigger_condition, priority, payload."""
        payload_json = json.dumps(event.get("payload", {}), ensure_ascii=False)
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO event_queue
                (session_id, event_type, trigger_condition, priority, payload, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                session_id,
                event.get("event_type", ""),
                event.get("trigger_condition", ""),
                event.get("priority", 0),
                payload_json,
                now,
            ),
        )
        await self._db.commit()

    # ------------------------------------------------------------------ #
    # Characters
    # ------------------------------------------------------------------ #

    def _char_id(self, session_id: str, name: str) -> str:
        return hashlib.md5(f"{session_id}:{name}".encode()).hexdigest()[:12]

    async def create_character(
        self,
        session_id: str,
        name: str,
        is_player: bool = False,
        hp: int = 100,
        max_hp: int = 100,
        location: str = "unknown",
        mood: str = "neutral",
        aliases: list[str] | None = None,
    ) -> dict:
        char_id = self._char_id(session_id, name)
        now = datetime.utcnow().isoformat()
        aliases_json = json.dumps(aliases or [], ensure_ascii=False)
        await self._db.execute(
            """
            INSERT INTO characters
                (id, session_id, name, aliases, is_player, hp, max_hp, location, mood, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                aliases = excluded.aliases,
                is_player = excluded.is_player,
                hp = excluded.hp,
                max_hp = excluded.max_hp,
                location = excluded.location,
                mood = excluded.mood,
                updated_at = excluded.updated_at
            """,
            (char_id, session_id, name, aliases_json, is_player, hp, max_hp, location, mood, now, now),
        )
        await self._db.commit()
        logger.debug(f"[DB] create_character: session={session_id} name={name!r} aliases={aliases} is_player={is_player}")
        return await self.get_character(session_id, name)

    async def add_character_alias(self, session_id: str, name: str, alias: str) -> None:
        """Add an alias to an existing character."""
        char = await self.get_character(session_id, name)
        if not char:
            return
        existing_aliases = json.loads(char.get("aliases") or "[]")
        alias_lower = alias.strip().lower()
        if alias_lower not in [a.lower() for a in existing_aliases]:
            existing_aliases.append(alias.strip())
            await self._db.execute(
                "UPDATE characters SET aliases = ?, updated_at = ? WHERE id = ?",
                (json.dumps(existing_aliases, ensure_ascii=False), datetime.utcnow().isoformat(), char["id"]),
            )
            await self._db.commit()
            logger.info(f"[DB] Added alias {alias!r} to character {name!r}")

    async def find_character_by_alias(self, session_id: str, alias: str) -> dict | None:
        """Find a character by any of its aliases."""
        chars = await self.get_session_characters(session_id)
        alias_lower = alias.strip().lower()
        for char in chars:
            char_aliases = json.loads(char.get("aliases") or "[]")
            if alias_lower in [a.lower() for a in char_aliases]:
                return char
        return None

    async def merge_characters(self, session_id: str, keep_name: str, remove_name: str) -> bool:
        """Merge remove_name into keep_name: transfer aliases, delete duplicate."""
        keep = await self.get_character(session_id, keep_name)
        remove = await self.get_character(session_id, remove_name)
        if not keep or not remove:
            return False

        # Transfer aliases from remove → keep
        keep_aliases = json.loads(keep.get("aliases") or "[]")
        remove_aliases = json.loads(remove.get("aliases") or "[]")
        # Add remove's name and aliases to keep
        all_new = [remove_name] + remove_aliases
        for a in all_new:
            if a.strip().lower() not in [x.lower() for x in keep_aliases]:
                keep_aliases.append(a.strip())

        now = datetime.utcnow().isoformat()
        await self._db.execute(
            "UPDATE characters SET aliases = ?, updated_at = ? WHERE id = ?",
            (json.dumps(keep_aliases, ensure_ascii=False), now, keep["id"]),
        )
        # Delete the duplicate
        await self._db.execute(
            "DELETE FROM characters WHERE id = ?", (remove["id"],)
        )
        await self._db.commit()
        logger.info(f"[DB] Merged character {remove_name!r} into {keep_name!r}, aliases={keep_aliases}")
        return True

    async def get_character(self, session_id: str, name: str) -> dict | None:
        char_id = self._char_id(session_id, name)
        async with self._db.execute(
            "SELECT * FROM characters WHERE id = ?", (char_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return dict(row)

    async def get_session_characters(self, session_id: str) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM characters WHERE session_id = ? ORDER BY is_player DESC, name ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Relationships
    # ------------------------------------------------------------------ #

    async def get_relationships(
        self, session_id: str, character_name: str | None = None
    ) -> list[dict]:
        if character_name is None:
            async with self._db.execute(
                "SELECT * FROM relationships WHERE session_id = ? ORDER BY updated_at DESC",
                (session_id,),
            ) as cursor:
                rows = await cursor.fetchall()
        else:
            async with self._db.execute(
                """
                SELECT * FROM relationships
                WHERE session_id = ? AND (from_name = ? OR to_name = ?)
                ORDER BY updated_at DESC
                """,
                (session_id, character_name, character_name),
            ) as cursor:
                rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Locations
    # ------------------------------------------------------------------ #

    async def get_locations(self, session_id: str) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM locations WHERE session_id = ? ORDER BY first_visit_turn ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Events
    # ------------------------------------------------------------------ #

    async def get_recent_events(self, session_id: str, limit: int = 5) -> list[dict]:
        async with self._db.execute(
            """
            SELECT * FROM events
            WHERE session_id = ?
            ORDER BY turn DESC, importance DESC
            LIMIT ?
            """,
            (session_id, limit),
        ) as cursor:
            rows = await cursor.fetchall()
        result = []
        for row in rows:
            r = dict(row)
            try:
                r["entities"] = json.loads(r.get("entities") or "[]")
            except (json.JSONDecodeError, TypeError):
                r["entities"] = []
            result.append(r)
        return result

    # ------------------------------------------------------------------ #
    # Lore
    # ------------------------------------------------------------------ #

    def _lore_id(self, session_id: str, name: str) -> str:
        return hashlib.md5(f"{session_id}:lore:{name}".encode()).hexdigest()[:12]

    async def create_lore(
        self,
        session_id: str,
        name: str,
        lore_type: str = "",
        keywords: str = "",
        content: str = "",
        priority: int = 50,
        auto_generated: bool = False,
        source_turns: str = "",
    ):
        lore_id = self._lore_id(session_id, name)
        now = datetime.utcnow().isoformat()
        await self._db.execute(
            """
            INSERT INTO lore
                (id, session_id, name, lore_type, keywords, content, priority, auto_generated, source_turns, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                lore_type = excluded.lore_type,
                keywords = excluded.keywords,
                content = excluded.content,
                priority = excluded.priority,
                auto_generated = excluded.auto_generated,
                source_turns = excluded.source_turns,
                updated_at = excluded.updated_at
            """,
            (lore_id, session_id, name, lore_type, keywords, content, priority, auto_generated, source_turns, now, now),
        )
        await self._db.commit()


    async def get_all_lore(self, session_id: str) -> list[dict]:
        async with self._db.execute(
            "SELECT * FROM lore WHERE session_id = ? ORDER BY priority DESC, name ASC",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Player context helper
    # ------------------------------------------------------------------ #

    async def query_player_context(self, session_id: str) -> dict:
        """Return a summary dict for the player character (replaces graph_db.query_player_context)."""
        # Find player character
        async with self._db.execute(
            "SELECT * FROM characters WHERE session_id = ? AND is_player = TRUE LIMIT 1",
            (session_id,),
        ) as cursor:
            player_row = await cursor.fetchone()

        if player_row is None:
            return {
                "location": "unknown",
                "hp": 100,
                "max_hp": 100,
                "mood": "neutral",
                "nearby_npcs": [],
                "recent_events": [],
                "relationships": [],
                "session_id": session_id,
            }

        player = dict(player_row)
        location = player.get("location", "unknown")

        # NPCs at the same location
        async with self._db.execute(
            """
            SELECT name FROM characters
            WHERE session_id = ? AND is_player = FALSE AND location = ?
            """,
            (session_id, location),
        ) as cursor:
            npc_rows = await cursor.fetchall()
        nearby_npcs = [r[0] for r in npc_rows]

        # Relationships involving the player
        relationships = await self.get_relationships(session_id, player.get("name"))

        # Enrich nearby_npcs with relationship data
        nearby_enriched = []
        for npc_name in nearby_npcs:
            npc_info = {"name": npc_name}
            for rel in relationships:
                if rel.get("to_name") == npc_name or rel.get("from_name") == npc_name:
                    npc_info["rel_type"] = rel.get("rel_type", "unknown")
                    npc_info["strength"] = rel.get("strength", 0)
                    break
            nearby_enriched.append(npc_info)

        # Recent events
        recent_events = await self.get_recent_events(session_id, limit=5)

        return {
            "location": location,
            "hp": player.get("hp", 100),
            "max_hp": player.get("max_hp", 100),
            "mood": player.get("mood", "neutral"),
            "nearby_npcs": nearby_enriched,
            "recent_events": [
                {"turn": e["turn"], "description": e["description"]}
                for e in recent_events
            ],
            "relationships": relationships,
            "session_id": session_id,
        }

    async def get_state_summary(self, session_id: str) -> str:
        """Full state summary for curator (replaces graph_db.get_graph_summary)."""
        lines = []

        # Characters
        chars = await self.get_session_characters(session_id)
        if chars:
            lines.append("## Characters")
            for c in chars:
                tag = "[PLAYER]" if c.get("is_player") else "[NPC]"
                lines.append(
                    f"- {tag} {c.get('name')} | HP:{c.get('hp')}/{c.get('max_hp')} "
                    f"| loc:{c.get('location')} | mood:{c.get('mood')}"
                )

        # Locations
        locs = await self.get_locations(session_id)
        if locs:
            lines.append("\n## Locations")
            for loc in locs:
                lines.append(f"- {loc.get('name')}: {loc.get('description', '')}")

        # Relationships
        rels = await self.get_relationships(session_id)
        if rels:
            lines.append("\n## Relationships")
            for r in rels:
                lines.append(
                    f"- {r.get('from_name')} --[{r.get('rel_type')}({r.get('strength')})]--&gt; {r.get('to_name')}"
                )

        return "\n".join(lines)

    async def detect_contradictions(self, session_id: str) -> list[dict]:
        """Rule-based contradiction detection (replaces graph_db.detect_contradictions)."""
        contradictions = []

        # Characters with HP <= 0
        async with self._db.execute(
            "SELECT name, hp FROM characters WHERE session_id = ? AND hp <= 0",
            (session_id,),
        ) as cursor:
            for row in await cursor.fetchall():
                contradictions.append({
                    "type": "zero_hp",
                    "character": row[0],
                    "hp": row[1],
                    "description": f"Character '{row[0]}' has HP={row[1]}",
                })

        return contradictions

    async def get_entities_without_lore(self, session_id: str) -> list[dict]:
        """Find characters that have no matching lore entries."""
        async with self._db.execute(
            """
            SELECT c.name, 'character' AS entity_type, c.location, c.mood
            FROM characters c
            WHERE c.session_id = ? AND c.is_player = FALSE
            AND NOT EXISTS (
                SELECT 1 FROM lore l
                WHERE l.session_id = c.session_id
                AND (l.name LIKE '%' || c.name || '%' OR l.keywords LIKE '%' || c.name || '%')
            )
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        return [
            {"name": r[0], "entity_type": r[1], "location": r[2], "mood": r[3]}
            for r in rows
        ]
