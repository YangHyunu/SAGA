"""Application lifespan — startup/shutdown for SAGA.

NOTE: This module imports from saga.services (cache_warming) which technically
crosses the core→services layer boundary. This is intentional: lifespan owns
the lifecycle of background tasks and must start/stop the warming loop.
"""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI

from saga.config import load_config
from saga.core import dependencies as deps
from saga.services.cache_warming import cache_warming_loop

logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, cleanup on shutdown."""
    from saga.storage.sqlite_db import SQLiteDB
    from saga.storage.vector_db import VectorDB
    from saga.storage.md_cache import MdCache
    from saga.llm.client import LLMClient
    from saga.agents.context_builder import ContextBuilder
    from saga.agents.post_turn import PostTurnExtractor
    from saga.agents.extractors import narrative_extract
    from saga.agents.curator import CuratorRunner
    from saga.session import SessionManager
    from saga.system_stabilizer import SystemStabilizer
    from saga.window_recovery import WindowRecovery
    from saga.cost_tracker import CostTracker
    from saga.message_compressor import MessageCompressor

    # Load config
    config_path = os.environ.get("SAGA_CONFIG", "config.yaml")
    deps.config = load_config(config_path)

    # LangSmith tracing setup
    if deps.config.langsmith.enabled:
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", deps.config.langsmith.project)
        logger.info(f"[LangSmith] Tracing enabled, project={deps.config.langsmith.project}")

    # Ensure directories exist
    os.makedirs("db", exist_ok=True)
    os.makedirs("cache/sessions", exist_ok=True)
    os.makedirs("logs/turns", exist_ok=True)

    # Initialize storage
    deps.sqlite_db = SQLiteDB(db_path="db/state.db")
    await deps.sqlite_db.initialize()

    deps.vector_db = VectorDB(db_path="db/chroma")
    deps.vector_db.initialize()

    deps.md_cache = MdCache(cache_dir=deps.config.md_cache.cache_dir)

    # Initialize LLM client
    deps.llm_client = LLMClient(deps.config)

    # Initialize agents
    deps.context_builder = ContextBuilder(deps.sqlite_db, deps.vector_db, deps.md_cache, deps.config)
    extract_fn = partial(narrative_extract, llm_client=deps.llm_client, config=deps.config)
    deps.post_turn = PostTurnExtractor(
        deps.sqlite_db, deps.vector_db, deps.md_cache, deps.llm_client, deps.config, extract_fn=extract_fn
    )
    deps.curator = CuratorRunner(deps.sqlite_db, deps.vector_db, deps.md_cache, deps.llm_client, deps.config)

    if deps.config.curator.enabled:
        deps.curator.initialize()

    # Initialize window recovery, cost tracker & message compressor
    deps.window_recovery = WindowRecovery(deps.sqlite_db, deps.vector_db, deps.config)
    deps.cost_tracker = CostTracker(deps.sqlite_db)
    await deps.cost_tracker.initialize()

    deps.message_compressor = MessageCompressor(deps.sqlite_db, deps.config)

    # Initialize session manager
    deps.session_mgr = SessionManager(deps.sqlite_db, deps.vector_db, deps.md_cache, deps.config)

    # Initialize system stabilizer
    deps.system_stabilizer = SystemStabilizer(deps.sqlite_db, deps.config)

    logger.info(f"[SAGA] Server initialized. Listening on {deps.config.server.host}:{deps.config.server.port}")

    # Start cache warming background loop
    warming_task = asyncio.create_task(cache_warming_loop())

    yield

    # Cleanup
    warming_task.cancel()
    await deps.sqlite_db.close()
    await deps.llm_client.close()
    logger.info("[SAGA] Server shutdown complete")
