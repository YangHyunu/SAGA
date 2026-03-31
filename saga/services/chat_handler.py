"""Core chat handling logic for SAGA proxy."""
import asyncio
import hashlib
import json
import logging
import time

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from langsmith import traceable
from starlette.background import BackgroundTask

from saga.core import dependencies as deps
from saga.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    Choice, Usage,
)
from saga.utils.tokens import count_tokens, count_messages_tokens
from saga.utils.parsers import strip_state_block
from saga.window_recovery import _KEY_SUMMARY_THROUGH_TURN
from saga.cost_tracker import UsageRecord
from saga.services.stream import stream_response

logger = logging.getLogger(__name__)


# ============================================================
# Helper functions
# ============================================================

def is_continuation(messages: list[dict]) -> bool:
    """Detect RisuAI autoContinue requests.

    RisuAI inserts '[Continue the last response]' as a system message in
    postEverything when autoContinueChat fires.  We scan from the end,
    stopping at the first user message so we only look at the trailing
    system messages of the current turn.
    """
    for msg in reversed(messages):
        if msg["role"] == "system" and deps._CONTINUE_PATTERN.search(msg.get("content", "")):
            return True
        if msg["role"] == "user":
            break
    return False


def prune_pending_responses() -> None:
    """Remove stale pending response entries older than TTL."""
    now = time.time()
    stale = [sid for sid, v in deps._pending_responses.items()
             if now - v["timestamp"] > deps._PENDING_TTL_SECONDS]
    for sid in stale:
        logger.info(f"[Trace] autoContinue: pruned stale pending for session={sid}")
        del deps._pending_responses[sid]


def is_anthropic_model(model: str | None) -> bool:
    """Check if the model (or config narration model) is Anthropic/Claude."""
    return "claude" in (model or "").lower() or "claude" in deps.config.models.narration.lower()


def extract_session_id(request: ChatCompletionRequest, raw_request: Request) -> str | None:
    """Extract session ID with priority: plugin sentinel > header > user field > system hash.

    0. @@SAGA: sentinel in messages[0] (RisuAI plugin injection)
    1. X-SAGA-Session-ID header (explicit)
    2. request.user field (OpenAI spec, configurable in RisuAI)
    3. System message hash fallback (legacy, backward-compatible)
    """
    # Priority 0: SAGA plugin sentinel in messages[0]
    if request.messages and request.messages[0].role == "system":
        text = request.messages[0].get_text_content()
        if text.startswith(deps._SAGA_META_PREFIX):
            lines = text.split("\n")
            meta_line = lines[0]
            meta = dict(
                kv.split("=", 1)
                for kv in meta_line[len(deps._SAGA_META_PREFIX):].split("&")
                if "=" in kv
            )
            for line in lines[1:]:
                if line.startswith(deps._SAGA_STATE_PREFIX):
                    try:
                        raw_state = json.loads(line[len(deps._SAGA_STATE_PREFIX):])
                        if isinstance(raw_state, dict):
                            request._saga_scriptstate = {
                                (k.lstrip("$") if isinstance(k, str) else str(k)): v
                                for k, v in raw_state.items()
                            }
                            logger.info(f"[Session] Plugin scriptstate: {len(request._saga_scriptstate)} vars")
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"[Session] Invalid scriptstate in sentinel: {e}")
                    break
            sid = meta.get("sid", "").strip()
            if sid and deps._SESSION_ID_RE.match(sid):
                request.messages.pop(0)
                logger.info(f"[Session] Plugin sentinel: sid={sid} grp={meta.get('grp', '0')}")
                return sid

    # Priority 1: Custom header
    header_id = raw_request.headers.get("x-saga-session-id", "").strip()
    if header_id:
        if not deps._SESSION_ID_RE.match(header_id):
            raise HTTPException(400, "Invalid session ID format")
        return header_id

    # Priority 2: user field
    if request.user:
        user = request.user.strip()
        if not deps._SESSION_ID_RE.match(user):
            raise HTTPException(400, "Invalid user/session ID format")
        return user

    # Priority 3: System message hash (stable fallback)
    for msg in request.messages:
        if msg.role == "system":
            first_para = msg.get_text_content().split('\n\n')[0][:300]
            sid = hashlib.sha256(first_para.encode()).hexdigest()[:16]
            logger.debug(f"[Session] Hash input (first 150ch): {first_para[:150]!r}")
            logger.debug(f"[Session] Hash result: {sid}")
            return sid
    return None


def extract_scriptstate(raw_request: Request) -> dict | None:
    """Extract scriptstate from x-saga-scriptstate header (JSON-encoded dict).

    RisuAI plugin sends: x-saga-scriptstate: {"$hp":"85","$location":"dungeon"}
    The '$' prefix is stripped — RisuAI internally prefixes scriptstate keys with '$'.
    Returns None if header is absent or invalid.
    """
    raw = raw_request.headers.get("x-saga-scriptstate", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning(f"[Scriptstate] Expected dict, got {type(data).__name__}")
            return None
        cleaned = {}
        for k, v in data.items():
            key = k.lstrip("$") if isinstance(k, str) else str(k)
            cleaned[key] = v
        return cleaned if cleaned else None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Scriptstate] Invalid JSON in header: {e}")
        return None


def extract_gen_params(request: ChatCompletionRequest) -> dict:
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


def build_cacheable_messages(original_messages, md_prefix, dynamic_suffix, lorebook_delta="", window_summary=""):
    """Build messages with 3-breakpoint prompt caching structure.

    BP1: system prompt (절대 안 변함 — SystemStabilizer가 보장)
    BP2: 대화 히스토리 중간 지점 assistant (이전 턴 내용은 안 변함)
    BP3: 대화 히스토리 마지막 assistant (직전 턴까지 안 변함)
    Dynamic: md_prefix + lorebook_delta + dynamic_suffix → 마지막 user 메시지에 prepend (캐시 밖)
    """
    messages = list(original_messages)
    system_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
    is_claude = "claude" in deps.config.models.narration.lower()

    if system_idx is not None and is_claude and deps.config.prompt_caching.enabled:
        cache_ctrl = {"type": "ephemeral"}
        if deps.config.prompt_caching.cache_ttl:
            cache_ctrl["ttl"] = deps.config.prompt_caching.cache_ttl

        # BP1: system prompt
        messages[system_idx] = dict(messages[system_idx])
        messages[system_idx]["cache_control"] = cache_ctrl

        assistant_indices = [
            i for i, m in enumerate(messages) if m.get("role") == "assistant"
        ]

        if len(assistant_indices) >= 2:
            from saga.message_compressor import _CHUNK_USER_PREFIX
            summary_asst_indices = [
                idx for idx in assistant_indices
                if idx > 0 and messages[idx - 1].get("role") == "user"
                and str(messages[idx - 1].get("content", "")).startswith(_CHUNK_USER_PREFIX)
            ]
            if summary_asst_indices:
                mid_idx = summary_asst_indices[-1]
            else:
                mid_idx = assistant_indices[len(assistant_indices) // 2]
            messages[mid_idx] = dict(messages[mid_idx])
            messages[mid_idx]["cache_control"] = cache_ctrl

            last_idx = assistant_indices[-1]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = cache_ctrl

        elif len(assistant_indices) == 1:
            last_idx = assistant_indices[0]
            messages[last_idx] = dict(messages[last_idx])
            messages[last_idx]["cache_control"] = cache_ctrl

        # Dynamic context: prepend to last user message
        context_block = ""
        if window_summary:
            context_block += f"[--- Lost Turn Summary ---]\n{window_summary}\n\n"
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
# Main chat handler
# ============================================================

@traceable(name="saga.handle_chat")
async def handle_chat(request: ChatCompletionRequest, raw_request: Request):
    t_start = time.time()
    mode = "stream" if request.stream else "sync"

    # ── Trace: Request ──
    sys_msgs = sum(1 for m in request.messages if m.role == "system")
    usr_msgs = sum(1 for m in request.messages if m.role == "user")
    asst_msgs = sum(1 for m in request.messages if m.role == "assistant")
    gen_params = extract_gen_params(request)
    logger.info(
        f"[Trace] ━━━ NEW REQUEST ({mode}) ━━━ "
        f"model={request.model} msgs=[sys:{sys_msgs} usr:{usr_msgs} asst:{asst_msgs}] "
        f"temp={request.temperature} max_tokens={request.max_tokens}"
        + (f" gen_params={gen_params}" if gen_params else "")
    )

    # Get or create session
    session_id = extract_session_id(request, raw_request)
    session_src = "header" if raw_request.headers.get("x-saga-session-id") else \
                  "user" if request.user else "system-hash" if session_id else "auto"
    session = await deps.session_mgr.get_or_create_session(session_id)
    session_id = session["id"]
    logger.info(f"[Trace] Session: {session_id} (via {session_src})")

    # ── Scriptstate ──
    scriptstate = getattr(request, '_saga_scriptstate', None) or extract_scriptstate(raw_request)
    if scriptstate:
        await deps.sqlite_db.upsert_world_state(session_id, "scriptstate", json.dumps(scriptstate, ensure_ascii=False))
        logger.info(f"[Trace] Scriptstate: {len(scriptstate)} vars received ({list(scriptstate.keys())[:5]})")

    # Token counting
    messages_tokens = count_messages_tokens([{"role": m.role, "content": m.get_text_content()} for m in request.messages])
    remaining_budget = deps.config.token_budget.total_context_max - messages_tokens
    dynamic_budget = min(remaining_budget, deps.config.token_budget.dynamic_context_max)
    logger.info(f"[Trace] Tokens: input={messages_tokens} budget_remain={remaining_budget} dynamic_budget={dynamic_budget}")

    # Cache diagnostic
    non_system = [m for m in request.messages if m.role != "system"]
    msg_count = len(non_system)
    first_msg_hash = hashlib.md5(non_system[0].get_text_content()[:100].encode()).hexdigest()[:6] if non_system else "none"
    prefix_content = "".join(m.get_text_content()[:50] for m in non_system[:3])
    prefix_hash = hashlib.md5(prefix_content.encode()).hexdigest()[:8]
    logger.debug(f"[CacheDiag] msgs={msg_count} first_hash={first_msg_hash} prefix_hash={prefix_hash}")

    # Extract non-text (image) parts
    _multimodal_parts: dict[int, list] = {}
    for i, m in enumerate(request.messages):
        if m.role == "user" and isinstance(m.content, list):
            non_text = [p for p in m.content if p.get("type") != "text"]
            if non_text:
                _multimodal_parts[i] = non_text

    t_ctx_start = time.time()
    messages_dicts = [{"role": m.role, "content": m.get_text_content()} for m in request.messages]

    # autoContinue detection BEFORE stabilize
    prune_pending_responses()
    _is_continuation = is_continuation(messages_dicts)
    if _is_continuation:
        logger.info(f"[Trace] autoContinue detected — deferring turn increment & Sub-B")

    # System Stabilizer
    stabilized_messages, lorebook_delta = await deps.system_stabilizer.stabilize(
        session_id, messages_dicts
    )
    messages_dicts = stabilized_messages

    # Proactive message compression
    if deps.config.prompt_caching.compress_enabled:
        messages_dicts, compress_shift = await deps.message_compressor.compress(session_id, messages_dicts)
        if compress_shift > 0:
            _multimodal_parts = {
                max(0, k - compress_shift): v
                for k, v in _multimodal_parts.items()
                if k >= compress_shift
            }
            logger.info(f"[Trace] MessageCompressor: {compress_shift} messages replaced with summary chunks")

    # Window Recovery
    window_summary = ""
    shift_info = await deps.window_recovery.detect_shift(session_id, messages_dicts)
    if shift_info["shifted"]:
        summary_block = await deps.window_recovery.build_summary_block(session_id, shift_info)
        if summary_block:
            window_summary = summary_block
        logger.info(f"[Trace] WindowRecovery: shift detected, summary prepared (lost ~{shift_info['estimated_lost_turns']} turns)")
    else:
        existing_summary = await deps.window_recovery._get_existing_summary(session_id)
        last_injected = await deps.sqlite_db.get_world_state_value(session_id, "window_summary_injected_turn")
        summary_through = await deps.sqlite_db.get_world_state_value(session_id, _KEY_SUMMARY_THROUGH_TURN)
        if existing_summary and (last_injected != summary_through):
            window_summary = existing_summary

    context_result = await deps.context_builder.build_context(session_id, messages_dicts, dynamic_budget)
    t_ctx_end = time.time()

    md_len = len(context_result["md_prefix"]) if context_result["md_prefix"] else 0
    dyn_len = len(context_result["dynamic_suffix"]) if context_result["dynamic_suffix"] else 0
    ctx_ms = (t_ctx_end - t_ctx_start) * 1000
    logger.info(
        f"[Trace] Sub-A Context: {ctx_ms:.0f}ms | "
        f"md_prefix={md_len}ch dynamic_suffix={dyn_len}ch"
    )
    if context_result["dynamic_suffix"]:
        logger.debug(f"[Trace] Dynamic suffix preview: {context_result['dynamic_suffix'][:200]}...")

    # Build cacheable messages
    augmented_messages = build_cacheable_messages(
        messages_dicts,
        context_result["md_prefix"],
        context_result["dynamic_suffix"],
        lorebook_delta,
        window_summary,
    )
    if window_summary:
        summary_through = await deps.sqlite_db.get_world_state_value(session_id, _KEY_SUMMARY_THROUGH_TURN)
        if summary_through:
            await deps.sqlite_db.upsert_world_state(session_id, "window_summary_injected_turn", summary_through)

    # Restore multimodal content
    if _multimodal_parts:
        for orig_idx, image_parts in _multimodal_parts.items():
            if orig_idx < len(augmented_messages):
                msg = augmented_messages[orig_idx]
                text = msg["content"] if isinstance(msg["content"], str) else str(msg["content"])
                msg["content"] = [{"type": "text", "text": text}] + image_parts
        logger.info(f"[Trace] Multimodal: restored image parts for {len(_multimodal_parts)} message(s)")

    bp_count = sum(1 for m in augmented_messages if isinstance(m, dict) and m.get("cache_control"))
    logger.info(f"[Trace] Cacheable messages: {len(augmented_messages)} msgs, {bp_count} breakpoints")

    for i, msg in enumerate(augmented_messages):
        role = msg.get("role", "?") if isinstance(msg, dict) else "?"
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        content_str = content if isinstance(content, str) else str(content)
        logger.debug(f"[LLM-Input] msg[{i}] role={role} len={len(content_str)}ch preview: {content_str[:300]}")

    # Get last user input for post-turn
    last_user_input = ""
    for msg in reversed(request.messages):
        if msg.role == "user":
            last_user_input = msg.get_text_content()
            break

    if request.stream:
        stream_ctx = {"full_response": ""}

        async def _post_stream_task():
            try:
                full_response = stream_ctx["full_response"]
                if not full_response:
                    logger.warning("[Trace] Post-stream: empty full_response (client disconnect?), skipping Sub-B")
                    return

                if _is_continuation:
                    prev = deps._pending_responses.get(session_id, {}).get("response", "")
                    combined = prev + full_response
                    deps._pending_responses[session_id] = {"response": combined, "timestamp": time.time()}
                    logger.info(
                        f"[Trace] autoContinue (stream): buffered partial response "
                        f"(cumulative={len(combined)}ch) — skipping turn increment"
                    )
                    return

                pending_entry = deps._pending_responses.pop(session_id, None)
                if pending_entry:
                    full_response = pending_entry["response"] + full_response
                    logger.info(
                        f"[Trace] autoContinue (stream): final response — combining "
                        f"pending({len(pending_entry['response'])}ch) + current({len(stream_ctx['full_response'])}ch)"
                    )

                turn_number = await deps.sqlite_db.increment_turn(session_id)
                logger.info(f"[Trace] ━━━ DONE turn={turn_number} (stream) ━━━")

                task_b = asyncio.create_task(deps.post_turn.extract_and_update(session_id, full_response, turn_number, last_user_input, scriptstate=scriptstate))
                deps._background_tasks.add(task_b)
                task_b.add_done_callback(deps._background_tasks.discard)
                task_b.add_done_callback(deps.log_task_exception)

                if deps.config.curator.enabled and turn_number % deps.config.curator.interval == 0:
                    task_c = asyncio.create_task(deps.curator.run(session_id, turn_number))
                    deps._background_tasks.add(task_c)
                    task_c.add_done_callback(deps._background_tasks.discard)
                    task_c.add_done_callback(deps.log_task_exception)

                if is_anthropic_model(request.model):
                    async with deps._warming_lock:
                        deps._warming_data[session_id] = {
                            "timestamp": time.time(),
                            "messages": augmented_messages,
                            "model": deps.config.models.narration,
                            "count": 0,
                        }
            except Exception as e:
                logger.error(f"[Trace] Post-stream task failed: {e}", exc_info=True)

        return StreamingResponse(
            stream_response(session_id, session, augmented_messages, request, stream_ctx, gen_params=gen_params),
            media_type="text/event-stream",
            background=BackgroundTask(_post_stream_task),
        )

    # Non-streaming
    t_llm_start = time.time()
    logger.info(f"[Trace] LLM call: model={deps.config.models.narration} temp={request.temperature or 0.7}")
    llm_response = await deps.llm_client.call_llm(
        model=deps.config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 8192,
        **gen_params,
    )
    t_llm_end = time.time()
    logger.info(f"[Trace] LLM done: {(t_llm_end - t_llm_start)*1000:.0f}ms | response={len(llm_response)}ch")

    usage = deps.llm_client._last_usage
    total_ms = (t_llm_end - t_llm_start) * 1000
    await deps.cost_tracker.record(UsageRecord(
        model=usage["model"], input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"], cache_read_tokens=usage["cache_read"],
        cache_create_tokens=usage["cache_create"], session_id=session_id, call_type="main",
        total_ms=total_ms,
    ))

    clean_response = strip_state_block(llm_response)
    has_state = len(clean_response) < len(llm_response)
    logger.info(f"[Trace] State block: {'found & stripped' if has_state else 'not found'} | clean={len(clean_response)}ch")

    if _is_continuation:
        prev = deps._pending_responses.get(session_id, {}).get("response", "")
        combined = prev + llm_response
        deps._pending_responses[session_id] = {"response": combined, "timestamp": time.time()}
        logger.info(
            f"[Trace] autoContinue: buffered partial response "
            f"(cumulative={len(combined)}ch) — skipping turn increment"
        )
        combined_clean = strip_state_block(combined)
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

    # Final response
    pending_entry = deps._pending_responses.pop(session_id, None)
    if pending_entry:
        combined_response = pending_entry["response"] + llm_response
        logger.info(
            f"[Trace] autoContinue: final response — combining "
            f"pending({len(pending_entry['response'])}ch) + current({len(llm_response)}ch)"
        )
        llm_response = combined_response
        clean_response = strip_state_block(llm_response)

    turn_number = await deps.sqlite_db.increment_turn(session_id)
    logger.info(f"[Trace] ━━━ DONE turn={turn_number} total={(time.time() - t_start)*1000:.0f}ms ━━━")

    task_b = asyncio.create_task(deps.post_turn.extract_and_update(session_id, llm_response, turn_number, last_user_input, scriptstate=scriptstate))
    deps._background_tasks.add(task_b)
    task_b.add_done_callback(deps._background_tasks.discard)
    task_b.add_done_callback(deps.log_task_exception)

    if deps.config.curator.enabled and turn_number % deps.config.curator.interval == 0:
        task_c = asyncio.create_task(deps.curator.run(session_id, turn_number))
        deps._background_tasks.add(task_c)
        task_c.add_done_callback(deps._background_tasks.discard)
        task_c.add_done_callback(deps.log_task_exception)

    if is_anthropic_model(request.model):
        async with deps._warming_lock:
            deps._warming_data[session_id] = {
                "timestamp": time.time(),
                "messages": augmented_messages,
                "model": deps.config.models.narration,
                "count": 0,
            }

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
            total_tokens=messages_tokens + count_tokens(clean_response),
            cache_read_input_tokens=deps.llm_client._last_cache_stats.get("cache_read", 0),
            cache_creation_input_tokens=deps.llm_client._last_cache_stats.get("cache_create", 0),
        )
    )
    return resp
