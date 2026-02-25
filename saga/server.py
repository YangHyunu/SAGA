"""SAGA Proxy Server — FastAPI OpenAI-compatible endpoint."""
import asyncio
import json
import time
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from saga.config import load_config, SagaConfig
from saga.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    Choice, Usage, ChatCompletionChunk, StreamChoice, StreamDelta,
    StatusResponse, SessionInfo
)
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.graph_db import GraphDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.llm.client import LLMClient
from saga.agents.context_builder import ContextBuilder
from saga.agents.post_turn import PostTurnExtractor
from saga.agents.curator import CuratorRunner
from saga.session import SessionManager
from saga.utils.tokens import count_tokens, count_messages_tokens
from saga.utils.parsers import strip_state_block

logger = logging.getLogger(__name__)

# Global instances
config: SagaConfig = None
sqlite_db: SQLiteDB = None
graph_db: GraphDB = None
vector_db: VectorDB = None
md_cache: MdCache = None
llm_client: LLMClient = None
context_builder: ContextBuilder = None
post_turn: PostTurnExtractor = None
curator: CuratorRunner = None
session_mgr: SessionManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, cleanup on shutdown."""
    global config, sqlite_db, graph_db, vector_db, md_cache, llm_client
    global context_builder, post_turn, curator, session_mgr

    # Load config
    config_path = os.environ.get("SAGA_CONFIG", "config.yaml")
    config = load_config(config_path)

    # Ensure directories exist
    os.makedirs("db", exist_ok=True)
    os.makedirs("cache/sessions", exist_ok=True)
    os.makedirs("logs/turns", exist_ok=True)

    # Initialize storage
    sqlite_db = SQLiteDB(db_path="db/state.db")
    await sqlite_db.initialize()

    graph_db = GraphDB(db_path=config.graph.db_path)
    graph_db.initialize()

    vector_db = VectorDB(db_path="db/chroma")
    vector_db.initialize()

    md_cache = MdCache(cache_dir=config.md_cache.cache_dir)

    # Initialize LLM client
    llm_client = LLMClient(config)

    # Initialize agents
    context_builder = ContextBuilder(sqlite_db, graph_db, vector_db, md_cache, config)
    post_turn = PostTurnExtractor(sqlite_db, graph_db, vector_db, md_cache, llm_client, config)
    curator = CuratorRunner(sqlite_db, graph_db, vector_db, md_cache, llm_client, config)

    if config.curator.enabled:
        curator.initialize()

    # Initialize session manager
    session_mgr = SessionManager(sqlite_db, graph_db, vector_db, md_cache, config)

    logger.info(f"[SAGA] Server initialized. Listening on {config.server.host}:{config.server.port}")

    yield

    # Cleanup
    await sqlite_db.close()
    await llm_client.close()
    logger.info("[SAGA] Server shutdown complete")


app = FastAPI(title="SAGA RP Agent Proxy", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Main Proxy Endpoint (OpenAI Compatible)
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    try:
        return await _handle_chat(request)
    except Exception as e:
        logger.error(f"[SAGA] Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=502, content={"error": str(e)})

async def _handle_chat(request: ChatCompletionRequest):
    t_start = time.time()

    # Get or create session (use first system message hash or generate)
    session_id = _extract_session_id(request)
    session = await session_mgr.get_or_create_session(session_id)
    session_id = session["id"]

    # Token counting
    messages_tokens = count_messages_tokens([{"role": m.role, "content": m.content} for m in request.messages])
    remaining_budget = config.token_budget.total_context_max - messages_tokens
    dynamic_budget = min(int(remaining_budget * 0.15), config.token_budget.dynamic_context_max)

    t_ctx_start = time.time()
    # Sub-A: Context Builder
    messages_dicts = [{"role": m.role, "content": m.content} for m in request.messages]
    context_result = await context_builder.build_context(session_id, messages_dicts, dynamic_budget)
    t_ctx_end = time.time()
    logger.info(f"[TIMING] Context Builder: {(t_ctx_end - t_ctx_start)*1000:.0f}ms | tokens_in: {messages_tokens}")

    # Build cacheable messages
    augmented_messages = _build_cacheable_messages(
        messages_dicts,
        context_result["md_prefix"],
        context_result["dynamic_suffix"]
    )

    # Get last user input for post-turn
    last_user_input = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_input = msg.content
            break

    if request.stream:
        return StreamingResponse(
            _stream_response(session_id, session, augmented_messages, request, last_user_input),
            media_type="text/event-stream"
        )

    # Non-streaming
    t_llm_start = time.time()
    llm_response = await llm_client.call_llm(
        model=config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 4096,
    )
    t_llm_end = time.time()
    logger.info(f"[TIMING] LLM call: {(t_llm_end - t_llm_start)*1000:.0f}ms | model: {config.models.narration}")

    # Strip state block for user
    clean_response = strip_state_block(llm_response)

    # Increment turn
    turn_number = await sqlite_db.increment_turn(session_id)
    logger.info(f"[TIMING] Total: {(time.time() - t_start)*1000:.0f}ms | session: {session_id} | turn: {turn_number}")

    # Sub-B: async post-turn processing
    asyncio.create_task(post_turn.extract_and_update(session_id, llm_response, turn_number, last_user_input))

    # Curator check
    if config.curator.enabled and turn_number % config.curator.interval == 0:
        asyncio.create_task(curator.run(session_id, turn_number))

    # Build response
    resp = ChatCompletionResponse(
        id=f"chatcmpl-saga-{session_id}-{turn_number}",
        object="chat.completion",
        created=int(time.time()),
        model=request.model or "saga-proxy",
        choices=[Choice(
            index=0,
            message=ChatMessage(role="assistant", content=clean_response),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=messages_tokens,
            completion_tokens=count_tokens(clean_response),
            total_tokens=messages_tokens + count_tokens(clean_response)
        )
    )
    return resp


async def _stream_response(session_id, session, augmented_messages, request, last_user_input):
    """SSE streaming response generator."""
    full_response = ""

    async for chunk in llm_client.call_llm_stream(
        model=config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 4096,
    ):
        full_response += chunk
        sse_chunk = ChatCompletionChunk(
            id=f"chatcmpl-saga-{session_id}",
            object="chat.completion.chunk",
            created=int(time.time()),
            model=request.model or "saga-proxy",
            choices=[StreamChoice(
                index=0,
                delta=StreamDelta(content=chunk),
                finish_reason=None
            )]
        )
        yield f"data: {sse_chunk.model_dump_json()}\n\n"

    # Final chunk
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

    # Post-stream processing
    turn_number = await sqlite_db.increment_turn(session_id)
    asyncio.create_task(post_turn.extract_and_update(session_id, full_response, turn_number, last_user_input))
    if config.curator.enabled and turn_number % config.curator.interval == 0:
        asyncio.create_task(curator.run(session_id, turn_number))


def _extract_session_id(request: ChatCompletionRequest) -> str | None:
    """Try to extract session ID from request. Use hash of first system message as stable ID."""
    import hashlib
    for msg in request.messages:
        if msg.role == "system":
            return hashlib.md5(msg.content[:200].encode()).hexdigest()[:8]
    return None


def _build_cacheable_messages(original_messages, md_prefix, dynamic_suffix):
    """Build messages with 3-breakpoint prompt caching structure.

    BP1: original system message (cache_control on the message itself)
    BP2: .md cache block inserted as second system message
    BP3: last assistant message in conversation history
    Dynamic: prepended to last user message, outside cache
    """
    messages = list(original_messages)

    # Find system message
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)

    is_claude = "claude" in config.models.narration.lower()

    if system_idx is not None and is_claude and config.prompt_caching.enabled:
        # === 3-BP Anthropic caching ===

        # BP1: original system message (session-invariant)
        messages[system_idx] = dict(messages[system_idx])
        messages[system_idx]["cache_control"] = {"type": "ephemeral"}

        # BP2: .md cache block (~1-turn refresh cycle)
        if md_prefix:
            md_msg = {
                "role": "system",
                "content": f"[--- SAGA Context Cache ---]\n{md_prefix}",
                "cache_control": {"type": "ephemeral"},
            }
            messages.insert(system_idx + 1, md_msg)

        # BP3: last assistant message in conversation history
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is not None:
            messages[last_assistant_idx] = dict(messages[last_assistant_idx])
            messages[last_assistant_idx]["cache_control"] = {"type": "ephemeral"}

        # Dynamic context: prepend to last user message (outside cache)
        if dynamic_suffix:
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    messages[i] = dict(messages[i])
                    messages[i]["content"] = f"[--- SAGA Dynamic ---]\n{dynamic_suffix}\n\n{messages[i]['content']}"
                    break
    else:
        # Non-Claude or caching disabled: legacy single-block approach
        if system_idx is not None:
            messages[system_idx] = dict(messages[system_idx])
            content = messages[system_idx]["content"]
            if md_prefix or dynamic_suffix:
                content += f"\n\n[--- SAGA Dynamic Context ---]\n{md_prefix}\n\n{dynamic_suffix}"
            messages[system_idx]["content"] = content
        else:
            sys_content = f"[--- SAGA Dynamic Context ---]\n{md_prefix}\n\n{dynamic_suffix}"
            messages.insert(0, {"role": "system", "content": sys_content})

    return messages


# ============================================================
# Admin API
# ============================================================

@app.get("/api/status")
async def get_status():
    sessions = await sqlite_db.list_sessions()
    return StatusResponse(status="running", active_sessions=len(sessions), version="3.0.0")

@app.get("/api/sessions")
async def list_sessions():
    sessions = await sqlite_db.list_sessions()
    return sessions

@app.post("/api/sessions")
async def create_session(name: str = "", world: str = ""):
    world_name = world or config.session.default_world
    session = await session_mgr.get_or_create_session()
    return session

@app.get("/api/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    session = await sqlite_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    state = await sqlite_db.get_world_state(session_id)
    return {"session": session, "world_state": state}

@app.get("/api/sessions/{session_id}/graph")
async def get_session_graph(session_id: str):
    summary = graph_db.get_graph_summary(session_id)
    return {"session_id": session_id, "graph_summary": summary}

@app.get("/api/sessions/{session_id}/cache")
async def get_session_cache(session_id: str):
    cache_data = await md_cache.read_cache(session_id)
    turn = md_cache.get_cache_turn(cache_data)
    return {"session_id": session_id, "cache_turn": turn, "files": list(cache_data.keys()), "has_content": any(bool(v) for v in cache_data.values())}

@app.post("/api/sessions/{session_id}/cache/regen")
async def regenerate_cache(session_id: str):
    md_contents = await context_builder._build_md_from_db(session_id)
    turn = await sqlite_db.get_turn_count(session_id)
    await md_cache.write_cache_atomic(session_id, turn, {f"{k}.md": v for k, v in md_contents.items()})
    return {"status": "regenerated", "turn": turn}

@app.get("/api/sessions/{session_id}/turns")
async def get_turn_logs(session_id: str, from_turn: int = 0, to_turn: int | None = None):
    logs = await sqlite_db.get_turn_logs(session_id, from_turn, to_turn)
    return logs

@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    await session_mgr.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}

@app.get("/api/memory/search")
async def search_memory(q: str, session: str = ""):
    if session:
        results = vector_db.search_lorebook(session, q, n_results=10)
    else:
        results = {"documents": [], "metadatas": []}
    return results

@app.get("/api/graph/query")
async def graph_query(cypher: str, session: str = ""):
    try:
        result = graph_db.conn.execute(cypher)
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        return {"results": rows}
    except Exception as e:
        raise HTTPException(400, str(e))
