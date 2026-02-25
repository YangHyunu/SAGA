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
    global config, sqlite_db, vector_db, md_cache, llm_client
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

    vector_db = VectorDB(db_path="db/chroma")
    vector_db.initialize()

    md_cache = MdCache(cache_dir=config.md_cache.cache_dir)

    # Initialize LLM client
    llm_client = LLMClient(config)

    # Initialize agents
    context_builder = ContextBuilder(sqlite_db, vector_db, md_cache, config)
    post_turn = PostTurnExtractor(sqlite_db, vector_db, md_cache, llm_client, config)
    curator = CuratorRunner(sqlite_db, vector_db, md_cache, llm_client, config)

    if config.curator.enabled:
        curator.initialize()

    # Initialize session manager
    session_mgr = SessionManager(sqlite_db, vector_db, md_cache, config)

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

    BP1: system prompt (절대 안 변함)
    BP2: 대화 히스토리 중간 지점 assistant (이전 턴 내용은 안 변함)
    BP3: 대화 히스토리 마지막 assistant (직전 턴까지 안 변함)
    Dynamic: md_prefix + dynamic_suffix → 마지막 user 메시지 직전에 삽입 (캐시 밖)

    핵심: 동적 컨텍스트를 BP 뒤에 배치해야 BP2/BP3 캐시가 유효함.
    [시스템+BP1] → [대화...BP2...BP3] → [동적 컨텍스트] → [유저 입력]
    """
    messages = list(original_messages)
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    is_claude = "claude" in config.models.narration.lower()

    if system_idx is not None and is_claude and config.prompt_caching.enabled:

        # BP1: system prompt (세션 내내 동일)
        messages[system_idx] = dict(messages[system_idx])
        messages[system_idx]["cache_control"] = {"type": "ephemeral"}

        # 대화 히스토리에서 assistant 메시지 위치 찾기
        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]

        if len(assistant_indices) >= 2:
            # BP2: 중간 지점 assistant (긴 대화에서 앞부분 캐시)
            mid_idx = assistant_indices[len(assistant_indices) // 2]
            messages[mid_idx] = dict(messages[mid_idx])
            messages[mid_idx]["cache_control"] = {"type": "ephemeral"}

            # BP3: 마지막 assistant (직전 턴까지 캐시)
            last_idx = assistant_indices[-1]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = {"type": "ephemeral"}

        elif len(assistant_indices) == 1:
            # 대화가 짧으면 BP2만
            last_idx = assistant_indices[0]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = {"type": "ephemeral"}

        # 동적 컨텍스트: 마지막 user 메시지에 prepend로 삽입
        # ⚠️ system role로 삽입하면 _call_anthropic에서 system 배열로 호이스팅되어
        # BP1~BP3 사이 prefix가 매 턴 바뀌므로 캐시가 무효화됨
        # → user 메시지에 prepend하면 모든 BP 뒤에 위치하므로 캐시 유지
        context_block = ""
        if md_prefix:
            context_block += f"[--- SAGA Context Cache ---]\n{md_prefix}\n\n"
        if dynamic_suffix:
            context_block += f"[--- SAGA Dynamic ---]\n{dynamic_suffix}"

        if context_block:
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                messages[last_user_idx] = dict(messages[last_user_idx])
                messages[last_user_idx]["content"] = context_block + "\n\n" + messages[last_user_idx]["content"]
            else:
                messages.append({
                    "role": "user",
                    "content": context_block,
                })

    else:
        # Non-Claude or caching disabled
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
    summary = await sqlite_db.get_state_summary(session_id)
    return {"session_id": session_id, "graph_summary": summary}

@app.get("/api/sessions/{session_id}/cache")
async def get_session_cache(session_id: str):
    stable = await md_cache.read_stable(session_id)
    live = await md_cache.read_live(session_id)
    return {"session_id": session_id, "stable_prefix": bool(stable), "live_state": bool(live)}

@app.get("/api/sessions/{session_id}/turns")
async def get_turn_logs(session_id: str, from_turn: int = 0, to_turn: int | None = None):
    logs = await sqlite_db.get_turn_logs(session_id, from_turn, to_turn)
    return logs

@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    await session_mgr.reset_session(session_id)
    return {"status": "reset", "session_id": session_id}

@app.post("/api/reset-all")
async def reset_all():
    """Full factory reset: SQLite + ChromaDB + MdCache all cleared."""
    import shutil, os
    # 1. SQLite: drop all data
    sessions = await sqlite_db.list_sessions()
    for s in sessions:
        await sqlite_db.reset_session(s["id"])
        await sqlite_db._db.execute("DELETE FROM sessions WHERE id = ?", (s["id"],))
    await sqlite_db._db.commit()

    # 2. ChromaDB: delete and recreate collections
    try:
        vector_db.delete_all_data()
    except Exception:
        # Fallback: wipe directory
        chroma_path = "db/chroma"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
            os.makedirs(chroma_path)
        vector_db.initialize()

    # 3. MdCache: clear all session dirs
    cache_dir = md_cache.cache_dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

    return {"status": "reset_all", "sessions_cleared": len(sessions)}

@app.get("/api/memory/search")
async def search_memory(q: str, session: str = "", collection: str = "episodes"):
    if not session:
        return {"documents": [], "metadatas": []}
    if collection == "lorebook":
        return vector_db.search_lorebook(session, q, n_results=10)
    else:
        return vector_db.search_episodes(session, q, n_results=10)

@app.get("/api/graph/query")
async def graph_query(q: str = "", session: str = ""):
    """Query state data (replaced Cypher graph queries)."""
    if not session:
        return {"error": "session parameter required"}
    chars = await sqlite_db.get_session_characters(session)
    rels = await sqlite_db.get_relationships(session)
    events = await sqlite_db.get_recent_events(session, limit=10)
    return {"characters": chars, "relationships": rels, "recent_events": events}
