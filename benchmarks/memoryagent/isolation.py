"""Per-sequence environment isolation for MemoryAgentBench.

Each benchmark sequence gets completely isolated:
- SQLite DB (fresh)
- ChromaDB directory (fresh)
- MdCache directory (fresh)
- Letta agent (created & deleted per sequence)
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class IsolatedEnv:
    """Isolated storage environment for one benchmark sequence."""

    base_dir: str
    seq_id: str
    sqlite_db: object = None
    vector_db: object = None
    md_cache: object = None
    _letta_agent_id: Optional[str] = None
    _letta_client: object = None

    @property
    def db_path(self) -> str:
        return os.path.join(self.base_dir, self.seq_id, "state.db")

    @property
    def chroma_path(self) -> str:
        return os.path.join(self.base_dir, self.seq_id, "chroma")

    @property
    def cache_path(self) -> str:
        return os.path.join(self.base_dir, self.seq_id, "cache")


async def create_isolated_env(
    base_dir: str,
    seq_id: str,
) -> IsolatedEnv:
    """Create a fresh isolated environment for one benchmark sequence.

    Creates fresh SQLite, ChromaDB, and MdCache instances in a
    per-sequence subdirectory.
    """
    from saga.storage.sqlite_db import SQLiteDB
    from saga.storage.vector_db import VectorDB
    from saga.storage.md_cache import MdCache

    env = IsolatedEnv(base_dir=base_dir, seq_id=seq_id)

    # Create directories
    os.makedirs(os.path.dirname(env.db_path), exist_ok=True)

    # Initialize fresh storage
    env.sqlite_db = SQLiteDB(db_path=env.db_path)
    await env.sqlite_db.initialize()

    env.vector_db = VectorDB(db_path=env.chroma_path)
    env.vector_db.initialize()

    env.md_cache = MdCache(cache_dir=env.cache_path)

    logger.debug(f"[isolation] Created env for {seq_id}: db={env.db_path}")
    return env


async def create_letta_agent(
    env: IsolatedEnv,
    config,
) -> str | None:
    """Create a fresh Letta agent for this sequence. Returns agent_id or None."""
    try:
        from letta_client import Letta
    except ImportError:
        logger.warning("[isolation] letta-client not installed, skipping Letta agent")
        return None

    try:
        client = Letta(base_url=config.curator.letta_base_url, timeout=100.0)
        env._letta_client = client

        agent_name = f"bench_curator_{env.seq_id}"

        # Delete any leftover agent with same name
        try:
            existing = list(client.agents.list())
            for agent in existing:
                if getattr(agent, "name", None) == agent_name:
                    client.agents.delete(agent.id)
                    logger.debug(f"[isolation] Deleted leftover agent: {agent_name}")
        except Exception:
            pass

        # Create fresh agent with memory blocks
        from saga.adapters.curator_adapter import _BLOCK_INITIAL_VALUES, _SYSTEM_PROMPT

        memory_blocks = [
            {"label": label, "value": value}
            for label, value in _BLOCK_INITIAL_VALUES.items()
        ]

        agent = client.agents.create(
            name=agent_name,
            model=config.curator.letta_model,
            embedding=config.curator.letta_embedding,
            memory_blocks=memory_blocks,
            system=_SYSTEM_PROMPT,
            include_base_tools=True,
            message_buffer_autoclear=True,
        )

        env._letta_agent_id = agent.id
        logger.info(f"[isolation] Created Letta agent: {agent_name} (id={agent.id})")
        return agent.id

    except Exception as e:
        logger.warning(f"[isolation] Failed to create Letta agent: {e}")
        return None


async def cleanup_env(env: IsolatedEnv, delete_files: bool = False):
    """Clean up an isolated environment.

    Deletes Letta agent and optionally removes local files.
    """
    # Delete Letta agent
    if env._letta_agent_id and env._letta_client:
        try:
            env._letta_client.agents.delete(env._letta_agent_id)
            logger.debug(f"[isolation] Deleted Letta agent {env._letta_agent_id}")
        except Exception as e:
            logger.warning(f"[isolation] Failed to delete Letta agent: {e}")

    # Close SQLite
    if env.sqlite_db:
        try:
            await env.sqlite_db.close()
        except Exception:
            pass

    # Optionally remove all files
    if delete_files:
        seq_dir = os.path.join(env.base_dir, env.seq_id)
        if os.path.exists(seq_dir):
            shutil.rmtree(seq_dir, ignore_errors=True)
            logger.debug(f"[isolation] Removed {seq_dir}")
