"""Session management routes for SAGA."""
from fastapi import APIRouter, Depends, HTTPException

from saga.core import dependencies as deps
from saga.middleware.auth import verify_bearer

router = APIRouter(dependencies=[Depends(verify_bearer)])


@router.get("/api/sessions")
async def list_sessions():
    sessions = await deps.sqlite_db.list_sessions()
    return sessions


@router.post("/api/sessions")
async def create_session(name: str = "", world: str = ""):
    session = await deps.session_mgr.get_or_create_session()
    return session


@router.get("/api/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    session = await deps.sqlite_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    state = await deps.sqlite_db.get_world_state(session_id)
    return {"session": session, "world_state": state}


@router.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str):
    summary = await deps.sqlite_db.get_state_summary(session_id)
    return {"session_id": session_id, "graph_summary": summary}


@router.get("/api/sessions/{session_id}/cache")
async def get_session_cache(session_id: str):
    stable = await deps.md_cache.read_stable(session_id)
    live = await deps.md_cache.read_live(session_id)
    return {"session_id": session_id, "stable_prefix": bool(stable), "live_state": bool(live)}


@router.get("/api/sessions/{session_id}/turns")
async def get_turn_logs(session_id: str, from_turn: int = 0, to_turn: int | None = None):
    logs = await deps.sqlite_db.get_turn_logs(session_id, from_turn, to_turn)
    return logs


@router.post("/api/sessions/reset-latest")
async def reset_latest_session():
    """Reset the most recently active session (by updated_at)."""
    sessions = await deps.sqlite_db.list_sessions()
    if not sessions:
        raise HTTPException(404, "No active sessions")
    latest = sessions[0]
    await deps.session_mgr.reset_session(latest["id"], curator=deps.curator)
    return {"status": "reset", "session_id": latest["id"], "name": latest.get("name", "")}


@router.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    await deps.session_mgr.reset_session(session_id, curator=deps.curator)
    return {"status": "reset", "session_id": session_id}
