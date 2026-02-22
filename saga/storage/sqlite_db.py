import aiosqlite
import json
from datetime import datetime


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

    async def get_triggered_events(self, session_id: str) -> list[dict]:
        """Return all pending events for a session, ordered by priority desc."""
        async with self._db.execute(
            """
            SELECT * FROM event_queue
            WHERE session_id = ? AND status = 'pending'
            ORDER BY priority DESC, id ASC
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()

        result = []
        for row in rows:
            r = dict(row)
            try:
                r["payload"] = json.loads(r.get("payload") or "{}")
            except (json.JSONDecodeError, TypeError):
                r["payload"] = {}
            result.append(r)
        return result

    async def mark_event_done(self, event_id: int):
        await self._db.execute(
            "UPDATE event_queue SET status = 'done' WHERE id = ?", (event_id,)
        )
        await self._db.commit()
