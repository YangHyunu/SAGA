"""SAGA Proxy Server — FastAPI OpenAI-compatible endpoint."""
import asyncio
import hashlib
import hmac
import json
import re
import time
import logging
import os
from collections import OrderedDict

# Logging setup: structured format + file handler
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)

# File handler for persistent logs
os.makedirs("logs", exist_ok=True)
_file_handler = logging.FileHandler("logs/saga.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
logging.getLogger().addHandler(_file_handler)

for _name in ("saga", "saga.llm.client", "saga.system_stabilizer"):
    logging.getLogger(_name).setLevel(logging.INFO)
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.background import BackgroundTask
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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
from saga.system_stabilizer import SystemStabilizer
from saga.observability import init_tracing, tracer as otel_tracer

logger = logging.getLogger(__name__)

# ============================================================
# Bounded dict for DoS protection (max sessions in memory)
# ============================================================

_MAX_TRACKED_SESSIONS = 500


class _BoundedDict(OrderedDict):
    """LRU dict with a hard size cap to prevent memory exhaustion."""

    def __init__(self, maxsize: int):
        super().__init__()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self._maxsize:
            oldest = next(iter(self))
            del self[oldest]


# ============================================================
# autoContinue detection & pending response buffer
# ============================================================

_CONTINUE_PATTERN = re.compile(r"Continue the last response", re.IGNORECASE)

# Buffer for autoContinue partial responses (bounded)
# key: session_id -> {"response": str, "timestamp": float}
_pending_responses: _BoundedDict = _BoundedDict(_MAX_TRACKED_SESSIONS)

_PENDING_TTL_SECONDS = 60


def _is_continuation(messages: list[dict]) -> bool:
    """Detect RisuAI autoContinue requests.

    RisuAI inserts '[Continue the last response]' as a system message in
    postEverything when autoContinueChat fires.  We scan from the end,
    stopping at the first user message so we only look at the trailing
    system messages of the current turn.
    """
    for msg in reversed(messages):
        if msg["role"] == "system" and _CONTINUE_PATTERN.search(msg.get("content", "")):
            return True
        if msg["role"] == "user":
            break
    return False


def _prune_pending_responses() -> None:
    """Remove stale pending response entries older than TTL."""
    now = time.time()
    stale = [sid for sid, v in _pending_responses.items()
             if now - v["timestamp"] > _PENDING_TTL_SECONDS]
    for sid in stale:
        logger.info(f"[Trace] autoContinue: pruned stale pending for session={sid}")
        del _pending_responses[sid]


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
system_stabilizer: SystemStabilizer = None

# Keep strong references to fire-and-forget tasks so GC doesn't collect them
# (asyncio event loop only holds weak references to tasks)
_background_tasks: set = set()

# Cache warming: track last request per session (bounded)
# session_id -> {timestamp, messages, model, count}
_warming_data: _BoundedDict = _BoundedDict(_MAX_TRACKED_SESSIONS)
_warming_lock = asyncio.Lock()


async def _cache_warming_loop():
    """Background task: send lightweight keepalive to maintain Anthropic cache.

    Runs every 30 seconds, fires a max_tokens=1 ping for any Anthropic session
    that has been idle longer than cache_warming.interval seconds and has not
    yet exhausted its max_warmings quota.  Sessions idle for 10+ minutes or
    at max_warmings are evicted from tracking.
    """
    while True:
        await asyncio.sleep(30)
        if not config or not config.cache_warming.enabled:
            continue
        now = time.time()
        expired = []
        async with _warming_lock:
            for sid, data in list(_warming_data.items()):
                elapsed = now - data["timestamp"]
                if elapsed > 600 or data["count"] >= config.cache_warming.max_warmings:
                    expired.append(sid)
                elif elapsed > config.cache_warming.interval:
                    try:
                        await llm_client.call_llm(
                            model=data["model"],
                            messages=data["messages"],
                            max_tokens=1,
                            temperature=0,
                        )
                        data["timestamp"] = now
                        data["count"] += 1
                        logger.info(f"[CacheWarming] session={sid} warming #{data['count']}")
                    except Exception as e:
                        logger.warning(f"[CacheWarming] Failed for {sid}: {e}")
            for sid in expired:
                del _warming_data[sid]
                logger.debug(f"[CacheWarming] Cleaned up session {sid}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, cleanup on shutdown."""
    global config, sqlite_db, vector_db, md_cache, llm_client
    global context_builder, post_turn, curator, session_mgr, system_stabilizer

    # Load config
    config_path = os.environ.get("SAGA_CONFIG", "config.yaml")
    config = load_config(config_path)

    # Initialize OpenTelemetry tracing (Phoenix)
    init_tracing()

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

    # Initialize system stabilizer
    system_stabilizer = SystemStabilizer(sqlite_db, config)

    logger.info(f"[SAGA] Server initialized. Listening on {config.server.host}:{config.server.port}")

    # Start cache warming background loop
    warming_task = asyncio.create_task(_cache_warming_loop())

    yield

    # Cleanup
    warming_task.cancel()
    await sqlite_db.close()
    await llm_client.close()
    logger.info("[SAGA] Server shutdown complete")


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks to log unhandled exceptions."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"[SAGA] Background task failed: {exc}", exc_info=exc)


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
    allow_credentials=False,  # SAGA uses Bearer tokens, not cookies
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "x-saga-session-id"],
)


# ============================================================
# Bearer Auth
# ============================================================

_bearer_scheme = HTTPBearer(auto_error=False)


async def _verify_bearer(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """Verify Bearer token if server.api_key is configured."""
    api_key = config.server.api_key if config else ""
    if not api_key:
        return  # Auth disabled
    if credentials is None or not hmac.compare_digest(
        credentials.credentials.encode("utf-8"), api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ============================================================
# Main Proxy Endpoint (OpenAI Compatible)
# ============================================================

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    _auth: None = Depends(_verify_bearer),
):
    """OpenAI-compatible chat completions endpoint."""
    try:
        return await _handle_chat(request, raw_request)
    except Exception as e:
        logger.error(f"[SAGA] Chat error: {e}", exc_info=True)
        return JSONResponse(status_code=502, content={"error": "upstream_error", "message": "LLM backend request failed"})

async def _handle_chat(request: ChatCompletionRequest, raw_request: Request):
    t_start = time.time()
    mode = "stream" if request.stream else "sync"

    # ── OTel root span for the entire pipeline ──
    span_ctx = otel_tracer.start_as_current_span(
        "saga.pipeline",
        attributes={"saga.mode": mode, "saga.model": request.model or config.models.narration},
    )
    pipeline_span = span_ctx.__enter__()

    # ── Trace: Request ──
    sys_msgs = sum(1 for m in request.messages if m.role == "system")
    usr_msgs = sum(1 for m in request.messages if m.role == "user")
    asst_msgs = sum(1 for m in request.messages if m.role == "assistant")
    gen_params = _extract_gen_params(request)
    logger.info(
        f"[Trace] ━━━ NEW REQUEST ({mode}) ━━━ "
        f"model={request.model} msgs=[sys:{sys_msgs} usr:{usr_msgs} asst:{asst_msgs}] "
        f"temp={request.temperature} max_tokens={request.max_tokens}"
        + (f" gen_params={gen_params}" if gen_params else "")
    )

    # Get or create session (priority: header > user field > system hash)
    session_id = _extract_session_id(request, raw_request)
    session_src = "header" if raw_request.headers.get("x-saga-session-id") else \
                  "user" if request.user else "system-hash" if session_id else "auto"
    session = await session_mgr.get_or_create_session(session_id)
    session_id = session["id"]
    logger.info(f"[Trace] Session: {session_id} (via {session_src})")
    pipeline_span.set_attribute("saga.session_id", session_id)

    # Token counting
    messages_tokens = count_messages_tokens([{"role": m.role, "content": m.get_text_content()} for m in request.messages])
    remaining_budget = config.token_budget.total_context_max - messages_tokens
    dynamic_budget = min(remaining_budget, config.token_budget.dynamic_context_max)
    logger.info(f"[Trace] Tokens: input={messages_tokens} budget_remain={remaining_budget} dynamic_budget={dynamic_budget}")

    # Cache diagnostic
    non_system = [m for m in request.messages if m.role != "system"]
    msg_count = len(non_system)
    first_msg_hash = hashlib.md5(non_system[0].get_text_content()[:100].encode()).hexdigest()[:6] if non_system else "none"
    prefix_content = "".join(m.get_text_content()[:50] for m in non_system[:3])
    prefix_hash = hashlib.md5(prefix_content.encode()).hexdigest()[:8]
    logger.debug(f"[CacheDiag] msgs={msg_count} first_hash={first_msg_hash} prefix_hash={prefix_hash}")

    pipeline_span.set_attribute("saga.tokens.input", messages_tokens)
    pipeline_span.set_attribute("saga.tokens.dynamic_budget", dynamic_budget)

    t_ctx_start = time.time()
    # Sub-A: Context Builder
    messages_dicts = [{"role": m.role, "content": m.get_text_content()} for m in request.messages]

    # autoContinue detection BEFORE stabilize (stabilize replaces system content with canonical)
    _prune_pending_responses()
    is_continuation = _is_continuation(messages_dicts)
    if is_continuation:
        logger.info(f"[Trace] autoContinue detected — deferring turn increment & Sub-B")

    # System Stabilizer: extract lorebook delta before context building
    stabilized_messages, lorebook_delta = await system_stabilizer.stabilize(
        session_id, messages_dicts
    )
    messages_dicts = stabilized_messages

    context_result = await context_builder.build_context(session_id, messages_dicts, dynamic_budget)
    t_ctx_end = time.time()

    md_len = len(context_result["md_prefix"]) if context_result["md_prefix"] else 0
    dyn_len = len(context_result["dynamic_suffix"]) if context_result["dynamic_suffix"] else 0
    ctx_ms = (t_ctx_end - t_ctx_start) * 1000
    logger.info(
        f"[Trace] Sub-A Context: {ctx_ms:.0f}ms | "
        f"md_prefix={md_len}ch dynamic_suffix={dyn_len}ch"
    )
    pipeline_span.set_attribute("saga.sub_a.duration_ms", round(ctx_ms))
    pipeline_span.set_attribute("saga.sub_a.md_prefix_len", md_len)
    pipeline_span.set_attribute("saga.sub_a.dynamic_suffix_len", dyn_len)
    if context_result["dynamic_suffix"]:
        # Show first 200 chars of dynamic context for debugging
        logger.debug(f"[Trace] Dynamic suffix preview: {context_result['dynamic_suffix'][:200]}...")

    # Build cacheable messages
    augmented_messages = _build_cacheable_messages(
        messages_dicts,
        context_result["md_prefix"],
        context_result["dynamic_suffix"],
        lorebook_delta,
    )
    bp_count = sum(1 for m in augmented_messages if isinstance(m, dict) and m.get("cache_control"))
    logger.info(f"[Trace] Cacheable messages: {len(augmented_messages)} msgs, {bp_count} breakpoints")

    # DEBUG: Log actual messages being sent to LLM
    for i, msg in enumerate(augmented_messages):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        content_str = content if isinstance(content, str) else str(content)
        logger.info(f"[LLM-Input] msg[{i}] role={role} len={len(content_str)}ch preview: {content_str[:300]}")

    # Get last user input for post-turn
    last_user_input = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_input = msg.get_text_content()
            break

    if request.stream:
        # Mutable container: generator stores full_response here for BackgroundTask
        stream_ctx = {"full_response": "", "pipeline_span": pipeline_span, "span_ctx": span_ctx}

        async def _post_stream_task():
            """Runs AFTER response is fully sent, outside Starlette's cancel scope."""
            try:
                full_response = stream_ctx["full_response"]
                if not full_response:
                    logger.warning("[Trace] Post-stream: empty full_response (client disconnect?), skipping Sub-B")
                    return

                # autoContinue: accumulate partial responses; defer turn+Sub-B
                if is_continuation:
                    prev = _pending_responses.get(session_id, {}).get("response", "")
                    combined = prev + full_response
                    _pending_responses[session_id] = {"response": combined, "timestamp": time.time()}
                    logger.info(
                        f"[Trace] autoContinue (stream): buffered partial response "
                        f"(cumulative={len(combined)}ch) — skipping turn increment"
                    )
                    return

                # Final (non-continuation) response: combine with any pending partial
                pending_entry = _pending_responses.pop(session_id, None)
                if pending_entry:
                    full_response = pending_entry["response"] + full_response
                    logger.info(
                        f"[Trace] autoContinue (stream): final response — combining "
                        f"pending({len(pending_entry['response'])}ch) + current({len(stream_ctx['full_response'])}ch)"
                    )

                turn_number = await sqlite_db.increment_turn(session_id)
                logger.info(f"[Trace] ━━━ DONE turn={turn_number} (stream) ━━━")

                task_b = asyncio.create_task(post_turn.extract_and_update(session_id, full_response, turn_number, last_user_input))
                _background_tasks.add(task_b)
                task_b.add_done_callback(_background_tasks.discard)
                task_b.add_done_callback(_log_task_exception)

                if config.curator.enabled and turn_number % config.curator.interval == 0:
                    task_c = asyncio.create_task(curator.run(session_id, turn_number))
                    _background_tasks.add(task_c)
                    task_c.add_done_callback(_background_tasks.discard)
                    task_c.add_done_callback(_log_task_exception)

                # Cache warming: record last request for this session (Anthropic only)
                if _is_anthropic_model(request.model):
                    async with _warming_lock:
                        _warming_data[session_id] = {
                            "timestamp": time.time(),
                            "messages": augmented_messages,
                            "model": config.models.narration,
                            "count": 0,
                        }
            except Exception as e:
                logger.error(f"[Trace] Post-stream task failed: {e}", exc_info=True)
            finally:
                # Close pipeline span after all post-stream work
                try:
                    stream_ctx["span_ctx"].__exit__(None, None, None)
                except Exception:
                    pass

        return StreamingResponse(
            _stream_response(session_id, session, augmented_messages, request, stream_ctx),
            media_type="text/event-stream",
            background=BackgroundTask(_post_stream_task),
        )

    # Non-streaming
    t_llm_start = time.time()
    logger.info(f"[Trace] LLM call: model={config.models.narration} temp={request.temperature or 0.7}")
    llm_response = await llm_client.call_llm(
        model=config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 4096,
        **gen_params,
    )
    t_llm_end = time.time()
    logger.info(f"[Trace] LLM done: {(t_llm_end - t_llm_start)*1000:.0f}ms | response={len(llm_response)}ch")

    # Strip state block for user
    clean_response = strip_state_block(llm_response)
    has_state = len(clean_response) < len(llm_response)
    logger.info(f"[Trace] State block: {'found & stripped' if has_state else 'not found'} | clean={len(clean_response)}ch")

    # autoContinue: accumulate partial responses; defer turn+Sub-B until final response
    if is_continuation:
        prev = _pending_responses.get(session_id, {}).get("response", "")
        combined = prev + llm_response
        _pending_responses[session_id] = {"response": combined, "timestamp": time.time()}
        logger.info(
            f"[Trace] autoContinue: buffered partial response "
            f"(cumulative={len(combined)}ch) — skipping turn increment"
        )
        combined_clean = strip_state_block(combined)
        span_ctx.__exit__(None, None, None)
        return ChatCompletionResponse(
            id=f"chatcmpl-saga-{session_id}-cont",
            object="chat.completion",
            created=int(time.time()),
            model=request.model or "saga-proxy",
            choices=[Choice(
                index=0,
                message=ChatMessage(role="assistant", content=combined_clean),
                finish_reason="stop"
            )],
            usage=Usage(
                prompt_tokens=messages_tokens,
                completion_tokens=count_tokens(combined_clean),
                total_tokens=messages_tokens + count_tokens(combined_clean)
            )
        )

    # Final (non-continuation) response: combine with any pending partial
    pending_entry = _pending_responses.pop(session_id, None)
    if pending_entry:
        combined_response = pending_entry["response"] + llm_response
        logger.info(
            f"[Trace] autoContinue: final response — combining "
            f"pending({len(pending_entry['response'])}ch) + current({len(llm_response)}ch)"
        )
        llm_response = combined_response
        clean_response = strip_state_block(llm_response)

    # Increment turn
    turn_number = await sqlite_db.increment_turn(session_id)
    logger.info(f"[Trace] ━━━ DONE turn={turn_number} total={( time.time() - t_start)*1000:.0f}ms ━━━")

    # Sub-B: async post-turn processing
    task_b = asyncio.create_task(post_turn.extract_and_update(session_id, llm_response, turn_number, last_user_input))
    _background_tasks.add(task_b)
    task_b.add_done_callback(_background_tasks.discard)
    task_b.add_done_callback(_log_task_exception)

    # Curator check
    if config.curator.enabled and turn_number % config.curator.interval == 0:
        task_c = asyncio.create_task(curator.run(session_id, turn_number))
        _background_tasks.add(task_c)
        task_c.add_done_callback(_background_tasks.discard)
        task_c.add_done_callback(_log_task_exception)

    # Cache warming: record last request for this session (Anthropic only)
    if _is_anthropic_model(request.model):
        async with _warming_lock:
            _warming_data[session_id] = {
                "timestamp": time.time(),
                "messages": augmented_messages,
                "model": config.models.narration,
                "count": 0,
            }

    # Build response
    pipeline_span.set_attribute("saga.turn_number", turn_number)
    pipeline_span.set_attribute("saga.response_len", len(clean_response))
    pipeline_span.set_attribute("saga.total_ms", round((time.time() - t_start) * 1000))
    span_ctx.__exit__(None, None, None)

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


async def _stream_response(session_id, session, augmented_messages, request, stream_ctx):
    """SSE streaming generator — ONLY yields chunks, no post-processing.

    State blocks (```state ... ```) are stripped from the stream sent to the
    client but preserved in full_response for Sub-B post-turn processing.
    Post-stream work (increment_turn, Sub-B, Curator) runs in BackgroundTask.
    """
    full_response = ""      # Unfiltered — for Sub-B
    buffer = ""             # Accumulates text to detect state block markers
    in_state_block = False  # True while inside a ```state block
    t_stream_start = time.time()
    state_block_filtered = False
    ttft_recorded = False   # Track first token time

    gen_params = _extract_gen_params(request)
    logger.info(f"[Trace] Stream start: model={config.models.narration}")

    async for chunk in llm_client.call_llm_stream(
        model=config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 4096,
        **gen_params,
    ):
        full_response += chunk
        stream_ctx["full_response"] = full_response  # incremental update for BackgroundTask

        # TTFT measurement
        if not ttft_recorded:
            ttft_ms = (time.time() - t_stream_start) * 1000
            ttft_recorded = True
            logger.info(f"[Trace] TTFT: {ttft_ms:.0f}ms")
            p_span = stream_ctx.get("pipeline_span")
            if p_span:
                p_span.add_event("first_token", {"ttft_ms": round(ttft_ms)})
                p_span.set_attribute("saga.ttft_ms", round(ttft_ms))

        buffer += chunk

        # Process buffer for state block markers
        while buffer:
            if in_state_block:
                # Look for closing ``` marker
                close_idx = buffer.find("```")
                if close_idx != -1:
                    # Discard everything up to and including the closing ```
                    buffer = buffer[close_idx + 3:]
                    in_state_block = False
                else:
                    # Haven't found closing marker yet — keep buffering
                    # But don't let buffer grow unbounded; if it's large
                    # and no marker, the block content is just accumulating
                    buffer = buffer[-10:]  # Keep tail for partial marker detection
                    break
            else:
                # Look for opening ```state marker
                # Check for partial marker at end of buffer
                marker_match = re.search(r'`{2,3}\s*state\s*\n', buffer)
                if marker_match:
                    # Emit everything before the marker
                    before = buffer[:marker_match.start()]
                    if before:
                        yield _make_sse_chunk(session_id, request.model, before)
                    buffer = buffer[marker_match.end():]
                    in_state_block = True
                    if not state_block_filtered:
                        state_block_filtered = True
                        logger.info("[Trace] Stream: state block detected, filtering...")
                else:
                    # Check if buffer ends with a partial potential marker
                    # (e.g., "`", "``", "```", "```s", "```st", etc.)
                    partial = _partial_state_marker(buffer)
                    if partial > 0:
                        # Emit safe prefix, keep potential marker in buffer
                        safe = buffer[:-partial]
                        if safe:
                            yield _make_sse_chunk(session_id, request.model, safe)
                        buffer = buffer[-partial:]
                        break
                    else:
                        # No marker possibility — emit all
                        if buffer:
                            yield _make_sse_chunk(session_id, request.model, buffer)
                        buffer = ""
                        break

    # Flush remaining buffer (if not inside a state block)
    if buffer and not in_state_block:
        yield _make_sse_chunk(session_id, request.model, buffer)

    # Final chunk
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

    # Store full_response for BackgroundTask (post-stream processing runs there)
    stream_total_ms = (time.time() - t_stream_start) * 1000
    logger.info(
        f"[Trace] Stream done: {stream_total_ms:.0f}ms | "
        f"response={len(full_response)}ch state_filtered={state_block_filtered}"
    )
    p_span = stream_ctx.get("pipeline_span")
    if p_span:
        p_span.set_attribute("saga.stream.total_ms", round(stream_total_ms))
        p_span.set_attribute("saga.response_len", len(full_response))
        p_span.set_attribute("saga.state_block_filtered", state_block_filtered)
    stream_ctx["full_response"] = full_response


def _make_sse_chunk(session_id: str, model: str | None, content: str) -> str:
    """Build a single SSE data line for a streaming chunk."""
    chunk = ChatCompletionChunk(
        id=f"chatcmpl-saga-{session_id}",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model or "saga-proxy",
        choices=[StreamChoice(
            index=0,
            delta=StreamDelta(content=content),
            finish_reason=None,
        )]
    )
    return f"data: {chunk.model_dump_json()}\n\n"


def _partial_state_marker(text: str) -> int:
    """Return the length of a potential partial ```state\\n marker at the end of text.

    Returns 0 if the tail cannot be a prefix of the marker pattern.
    """
    marker_prefixes = ["`", "``", "```", "```s", "```st", "```sta", "```stat", "```state", "```state\r", "```state\n"]
    for p in reversed(marker_prefixes):
        if text.endswith(p):
            return len(p)
    return 0


_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')


def _is_anthropic_model(model: str | None) -> bool:
    """Check if the model (or config narration model) is Anthropic/Claude."""
    return "claude" in (model or "").lower() or "claude" in config.models.narration.lower()


def _extract_session_id(request: ChatCompletionRequest, raw_request: Request) -> str | None:
    """Extract session ID with priority: header > user field > system hash.

    1. X-SAGA-Session-ID header (explicit, e.g. from future RisuAI plugin)
    2. request.user field (OpenAI spec, configurable in RisuAI)
    3. System message hash fallback (legacy, backward-compatible)
    """
    # Priority 1: Custom header
    header_id = raw_request.headers.get("x-saga-session-id", "").strip()
    if header_id:
        if not _SESSION_ID_RE.match(header_id):
            raise HTTPException(400, "Invalid session ID format")
        return header_id

    # Priority 2: user field
    if request.user:
        user = request.user.strip()
        if not _SESSION_ID_RE.match(user):
            raise HTTPException(400, "Invalid user/session ID format")
        return user

    # Priority 3: System message hash (stable fallback)
    # Use first paragraph only — Lorebook entries appended later would cause
    # different hashes each turn, breaking session continuity.
    # NOTE: first-user-message composite hash was tried but breaks when
    # RisuAI truncates old messages (context window sliding).
    for msg in request.messages:
        if msg.role == "system":
            first_para = msg.get_text_content().split('\n\n')[0][:300]
            return hashlib.sha256(first_para.encode()).hexdigest()[:16]
    return None


def _extract_gen_params(request: ChatCompletionRequest) -> dict:
    """Extract optional generation parameters from the request for LLM forwarding."""
    params = {}
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        params["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        params["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        params["stop"] = request.stop
    return params


def _build_cacheable_messages(original_messages, md_prefix, dynamic_suffix, lorebook_delta=""):
    """Build messages with 3-breakpoint prompt caching structure.

    BP1: system prompt (절대 안 변함 — SystemStabilizer가 보장)
    BP2: 대화 히스토리 중간 지점 assistant (이전 턴 내용은 안 변함)
    BP3: 대화 히스토리 마지막 assistant (직전 턴까지 안 변함)
    Dynamic: md_prefix + lorebook_delta + dynamic_suffix → 마지막 user 메시지 직전에 삽입 (캐시 밖)

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
        if lorebook_delta:
            context_block += f"[--- Active Lorebook ---]\n{lorebook_delta}\n\n"
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

@app.get("/api/status", dependencies=[Depends(_verify_bearer)])
async def get_status():
    sessions = await sqlite_db.list_sessions()
    return StatusResponse(status="running", active_sessions=len(sessions), version="3.0.0")

@app.get("/api/sessions", dependencies=[Depends(_verify_bearer)])
async def list_sessions():
    sessions = await sqlite_db.list_sessions()
    return sessions

@app.post("/api/sessions", dependencies=[Depends(_verify_bearer)])
async def create_session(name: str = "", world: str = ""):
    world_name = world or config.session.default_world
    session = await session_mgr.get_or_create_session()
    return session

@app.get("/api/sessions/{session_id}/state", dependencies=[Depends(_verify_bearer)])
async def get_session_state(session_id: str):
    session = await sqlite_db.get_session(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    state = await sqlite_db.get_world_state(session_id)
    return {"session": session, "world_state": state}

@app.get("/api/sessions/{session_id}/graph", dependencies=[Depends(_verify_bearer)])
async def get_session_graph(session_id: str):
    summary = await sqlite_db.get_state_summary(session_id)
    return {"session_id": session_id, "graph_summary": summary}

@app.get("/api/sessions/{session_id}/cache", dependencies=[Depends(_verify_bearer)])
async def get_session_cache(session_id: str):
    stable = await md_cache.read_stable(session_id)
    live = await md_cache.read_live(session_id)
    return {"session_id": session_id, "stable_prefix": bool(stable), "live_state": bool(live)}

@app.get("/api/sessions/{session_id}/turns", dependencies=[Depends(_verify_bearer)])
async def get_turn_logs(session_id: str, from_turn: int = 0, to_turn: int | None = None):
    logs = await sqlite_db.get_turn_logs(session_id, from_turn, to_turn)
    return logs

@app.post("/api/sessions/reset-latest", dependencies=[Depends(_verify_bearer)])
async def reset_latest_session():
    """Reset the most recently active session (by updated_at)."""
    sessions = await sqlite_db.list_sessions()
    if not sessions:
        raise HTTPException(404, "No active sessions")
    latest = sessions[0]  # list_sessions returns ORDER BY updated_at DESC
    await session_mgr.reset_session(latest["id"], curator=curator)
    return {"status": "reset", "session_id": latest["id"], "name": latest.get("name", "")}

@app.post("/api/sessions/{session_id}/reset", dependencies=[Depends(_verify_bearer)])
async def reset_session(session_id: str):
    await session_mgr.reset_session(session_id, curator=curator)
    return {"status": "reset", "session_id": session_id}

@app.post("/api/reset-all", dependencies=[Depends(_verify_bearer)])
async def reset_all():
    """Full factory reset: SQLite + ChromaDB + MdCache all cleared."""
    import shutil, os
    # 1. SQLite: drop all data
    sessions = await sqlite_db.list_sessions()
    for s in sessions:
        await sqlite_db.reset_session(s["id"])
        await sqlite_db._db.execute("DELETE FROM sessions WHERE id = ?", (s["id"],))
    await sqlite_db._db.commit()

    # 2. ChromaDB: delete collections via API (avoids readonly after dir wipe)
    try:
        if vector_db.client:
            for col_name in ("lorebook", "episodes"):
                try:
                    vector_db.client.delete_collection(col_name)
                except Exception:
                    pass
            # Recreate collections
            vector_db.lorebook = vector_db.client.get_or_create_collection(
                name="lorebook", metadata={"hnsw:space": "cosine"}
            )
            vector_db.episodes = vector_db.client.get_or_create_collection(
                name="episodes", metadata={"hnsw:space": "cosine"}
            )
    except Exception as e:
        logger.warning(f"[Reset] ChromaDB reset fallback: {e}")
        chroma_path = "db/chroma"
        if os.path.exists(chroma_path):
            shutil.rmtree(chroma_path)
        os.makedirs(chroma_path, exist_ok=True)
        vector_db.initialize()

    # 3. MdCache: clear all session dirs
    cache_dir = md_cache.cache_dir
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir)

    # 4. Letta: delete all saga_curator_* agents
    letta_deleted = 0
    if curator and hasattr(curator, 'letta_adapter') and curator.letta_adapter._initialized:
        try:
            existing = list(curator.letta_adapter.client.agents.list())
            for agent in existing:
                name = getattr(agent, "name", "")
                if name.startswith("saga_curator_"):
                    try:
                        curator.letta_adapter.client.agents.delete(agent.id)
                        letta_deleted += 1
                    except Exception:
                        pass
            curator.letta_adapter._agents.clear()
            logger.info(f"[Reset] Letta: deleted {letta_deleted} curator agents")
        except Exception as e:
            logger.warning(f"[Reset] Letta cleanup failed: {e}")

    # 5. ab_metrics: clear metrics logs
    ab_metrics_dir = "logs/ab_metrics"
    if os.path.exists(ab_metrics_dir):
        shutil.rmtree(ab_metrics_dir)
        logger.info("[Reset] Cleared logs/ab_metrics/")

    return {"status": "reset_all", "sessions_cleared": len(sessions), "letta_agents_deleted": letta_deleted}

@app.get("/api/memory/search", dependencies=[Depends(_verify_bearer)])
async def search_memory(q: str, session: str = "", collection: str = "episodes"):
    if not session:
        return {"documents": [], "metadatas": []}
    if collection == "lorebook":
        return vector_db.search_lorebook(session, q, n_results=10)
    else:
        return vector_db.search_episodes(session, q, n_results=10)

@app.get("/api/graph/query", dependencies=[Depends(_verify_bearer)])
async def graph_query(q: str = "", session: str = ""):
    """Query state data (replaced Cypher graph queries)."""
    if not session:
        return {"error": "session parameter required"}
    chars = await sqlite_db.get_session_characters(session)
    rels = await sqlite_db.get_relationships(session)
    events = await sqlite_db.get_recent_events(session, limit=10)
    return {"characters": chars, "relationships": rels, "recent_events": events}
