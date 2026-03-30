"""SAGA Proxy Server — FastAPI app factory.

All business logic, routes, and services are in their respective modules.
This file only creates the FastAPI app, registers middleware, and includes routers.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from saga.core.lifespan import lifespan
from saga.routes.chat import router as chat_router
from saga.routes.sessions import router as sessions_router
from saga.routes.admin import router as admin_router

# ── Backward-compatible re-exports for tests ──
# Tests import these symbols from saga.server; re-export from new locations.
from saga.core.dependencies import (  # noqa: F401
    _BoundedDict,
    _CONTINUE_PATTERN,
    _pending_responses,
    _PENDING_TTL_SECONDS,
    _SESSION_ID_RE,
    _SAGA_META_PREFIX,
    _SAGA_STATE_PREFIX,
    _warming_data,
    _warming_lock,
    _background_tasks,
    _MAX_TRACKED_SESSIONS,
)
from saga.services.chat_handler import (  # noqa: F401
    is_continuation as _is_continuation,
    prune_pending_responses as _prune_pending_responses,
    is_anthropic_model as _is_anthropic_model,
    extract_session_id as _extract_session_id,
    extract_scriptstate as _extract_scriptstate,
    extract_gen_params as _extract_gen_params,
    build_cacheable_messages as _build_cacheable_messages,
    handle_chat as _handle_chat,
)
from saga.services.stream import (  # noqa: F401
    stream_response as _stream_response,
    make_sse_chunk as _make_sse_chunk,
    _partial_state_marker,
)
from saga.middleware.auth import verify_bearer as _verify_bearer  # noqa: F401

# Re-export global component refs so `server_module.config` etc. still works.
# Uses __getattr__ to delegate to deps at access time (not import time),
# so values reflect the live state after lifespan initialization.
import saga.core.dependencies as _deps  # noqa: F401

_DEPS_ATTRS = frozenset({
    "config", "sqlite_db", "vector_db", "md_cache", "llm_client",
    "context_builder", "post_turn", "curator", "session_mgr",
    "system_stabilizer", "window_recovery", "cost_tracker", "message_compressor",
})


def __getattr__(name: str):
    if name in _DEPS_ATTRS:
        return getattr(_deps, name)
    raise AttributeError(f"module 'saga.server' has no attribute {name!r}")


# ============================================================
# App Factory
# ============================================================

app = FastAPI(title="SAGA RP Agent Proxy", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "app://.",  # Electron (RisuAI desktop)
    ],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "x-saga-session-id", "x-saga-scriptstate"],
)

# Register routers
app.include_router(chat_router)
app.include_router(sessions_router)
app.include_router(admin_router)
