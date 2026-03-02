"""Session management — create, get, list, reset sessions."""
import os
import shutil
import uuid
import logging
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.world.loader import WorldLoader

logger = logging.getLogger(__name__)


class SessionManager:
    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache, config):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.config = config
        self.world_loader = WorldLoader(sqlite_db, vector_db, md_cache)

    async def get_or_create_session(self, session_id: str | None = None) -> dict:
        """Get existing session or create new one."""
        if session_id:
            session = await self.sqlite_db.get_session(session_id)
            if session:
                return session

        # Create new session
        new_id = session_id or str(uuid.uuid4())[:8]
        session = await self.sqlite_db.create_session(new_id, name=f"Session {new_id}")

        # Load default world
        world_dir = os.path.join("data", "worlds", self.config.session.default_world)
        if os.path.exists(world_dir):
            await self.world_loader.load_world(new_id, world_dir)

        logger.info(f"[Session] Created new session: {new_id}")
        return session

    async def list_sessions(self) -> list[dict]:
        return await self.sqlite_db.list_sessions()

    async def get_session(self, session_id: str) -> dict | None:
        return await self.sqlite_db.get_session(session_id)

    async def reset_session(self, session_id: str, curator=None):
        """Reset session: clear DB data, vector data, .md cache files, and optionally curator agent."""
        await self.sqlite_db.reset_session(session_id)
        self.vector_db.delete_session_data(session_id)

        # Clear MdCache filesystem directory for this session
        session_dir = self.md_cache.get_session_dir(session_id)
        if os.path.exists(session_dir):
            shutil.rmtree(session_dir)
            logger.info(f"[Session] Cleared cache dir: {session_dir}")

        # Clean up Letta curator agent if provided
        if curator and hasattr(curator, 'letta_adapter') and curator.letta_adapter._initialized:
            agent_name = f"saga_curator_{session_id}"
            try:
                existing = list(curator.letta_adapter.client.agents.list())
                for agent in existing:
                    if getattr(agent, "name", "") == agent_name:
                        curator.letta_adapter.client.agents.delete(agent.id)
                        curator.letta_adapter._agents.pop(session_id, None)
                        logger.info(f"[Session] Deleted Letta agent: {agent_name}")
                        break
            except Exception as e:
                logger.warning(f"[Session] Letta cleanup failed for {session_id}: {e}")

        # Re-bootstrap
        world_dir = os.path.join("data", "worlds", self.config.session.default_world)
        if os.path.exists(world_dir):
            await self.world_loader.load_world(session_id, world_dir)
        logger.info(f"[Session] Reset session: {session_id}")
