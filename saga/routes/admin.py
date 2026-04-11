"""Admin and utility routes for SAGA."""
import logging

from fastapi import APIRouter, Depends

from saga.core import dependencies as deps
from saga.middleware.auth import verify_bearer
from saga.models import StatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint — no auth required."""
    return {"status": "ok"}


@router.get("/api/cost", dependencies=[Depends(verify_bearer)])
async def get_cost_global():
    """Global cost summary across all sessions."""
    return await deps.cost_tracker.get_global_summary()


@router.get("/api/cost/{session_id}", dependencies=[Depends(verify_bearer)])
async def get_cost_session(session_id: str):
    """Cost summary for a specific session."""
    return await deps.cost_tracker.get_session_summary(session_id)


@router.get("/api/status", dependencies=[Depends(verify_bearer)])
async def get_status():
    sessions = await deps.sqlite_db.list_sessions()
    return StatusResponse(status="running", active_sessions=len(sessions), version="3.0.0")


@router.post("/api/reset-all", dependencies=[Depends(verify_bearer)])
async def reset_all():
    """Full factory reset: SQLite + ChromaDB + MdCache all cleared."""
    import shutil, os

    # 1. SQLite
    sessions = await deps.sqlite_db.list_sessions()
    for s in sessions:
        await deps.sqlite_db.reset_session(s["id"])
        await deps.sqlite_db._db.execute("DELETE FROM sessions WHERE id = ?", (s["id"],))
    await deps.sqlite_db._db.commit()

    # 2. ChromaDB
    try:
        if deps.vector_db.client:
            for col_name in ("lorebook", "episodes"):
                try:
                    deps.vector_db.client.delete_collection(col_name)
                except Exception:
                    pass
            deps.vector_db.lorebook = deps.vector_db.client.get_or_create_collection(
                name="lorebook", metadata={"hnsw:space": "cosine"}
            )
            deps.vector_db.episodes = deps.vector_db.client.get_or_create_collection(
                name="episodes", metadata={"hnsw:space": "cosine"}
            )
    except Exception as e:
        logger.warning(f"[Reset] ChromaDB reset fallback: {e}")
        chroma_path = "db/chroma"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        os.makedirs(chroma_path, exist_ok=True)
        deps.vector_db.initialize()

    # 3. MdCache
    cache_dir = deps.md_cache.cache_dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

    # 4. Letta
    letta_deleted = 0
    if deps.curator and hasattr(deps.curator, 'letta_adapter') and deps.curator.letta_adapter._initialized:
        try:
            existing = list(deps.curator.letta_adapter.client.agents.list())
            for agent in existing:
                name = getattr(agent, "name", "")
                if name.startswith("saga_curator_"):
                    try:
                        deps.curator.letta_adapter.client.agents.delete(agent.id)
                        letta_deleted += 1
                    except Exception:
                        pass
            deps.curator.letta_adapter._agents.clear()
            logger.info(f"[Reset] Letta: deleted {letta_deleted} curator agents")
        except Exception as e:
            logger.warning(f"[Reset] Letta cleanup failed: {e}")

    # 5. ab_metrics
    ab_metrics_dir = "logs/ab_metrics"
    if os.path.exists(ab_metrics_dir):
        shutil.rmtree(ab_metrics_dir)
        logger.info("[Reset] Cleared logs/ab_metrics/")

    return {"status": "reset_all", "sessions_cleared": len(sessions), "letta_agents_deleted": letta_deleted}


@router.get("/api/memory/search", dependencies=[Depends(verify_bearer)])
async def search_memory(q: str, session: str = "", collection: str = "episodes"):
    if not session:
        return {"documents": [], "metadatas": []}
    if collection == "lorebook":
        return deps.vector_db.search_lorebook(session, q, n_results=10)
    else:
        return deps.vector_db.search_episodes(session, q, n_results=10)


@router.get("/api/graph/query", dependencies=[Depends(verify_bearer)])
async def graph_query(q: str = "", session: str = ""):
    """Query state data (replaced Cypher graph queries)."""
    if not session:
        return {"error": "session parameter required"}
    chars = await deps.sqlite_db.get_session_characters(session)
    rels = await deps.sqlite_db.get_relationships(session)
    events = await deps.sqlite_db.get_recent_events(session, limit=10)
    return {"characters": chars, "relationships": rels, "recent_events": events}
