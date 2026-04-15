"""Chat completions orchestrator: pipeline glue for SAGA proxy."""
import hashlib
import json
import logging
import time

from fastapi import Request
from fastapi.responses import StreamingResponse
from langsmith import traceable
from starlette.background import BackgroundTask

from saga.core import dependencies as deps
from saga.models import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
    Choice, Usage,
)
from saga.utils.tokens import count_tokens, count_messages_tokens
from saga.utils.parsers import strip_state_block
from saga.cost_tracker import UsageRecord
from saga.services.stream import stream_response
from saga.services.session_extractor import (
    extract_session_id,
    extract_scriptstate,
    extract_gen_params,
)
from saga.services.cache_marker import is_anthropic_model, build_cacheable_messages
from saga.services.post_turn_pipeline import (
    is_continuation,
    prune_pending_responses,
    finalize_turn,
)

logger = logging.getLogger(__name__)


@traceable(name="saga.handle_chat")
async def handle_chat(request: ChatCompletionRequest, raw_request: Request):
    t_start = time.time()
    mode = "stream" if request.stream else "sync"

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

    session_id = extract_session_id(request, raw_request)
    session_src = "header" if raw_request.headers.get("x-saga-session-id") else \
                  "user" if request.user else "system-hash" if session_id else "auto"
    session = await deps.session_mgr.get_or_create_session(session_id)
    session_id = session["id"]
    logger.info(f"[Trace] Session: {session_id} (via {session_src})")

    scriptstate = extract_scriptstate(raw_request)
    if scriptstate:
        await deps.sqlite_db.upsert_world_state(session_id, "scriptstate", json.dumps(scriptstate, ensure_ascii=False))
        logger.info(f"[Trace] Scriptstate: {len(scriptstate)} vars received ({list(scriptstate.keys())[:5]})")

    messages_tokens = count_messages_tokens([{"role": m.role, "content": m.get_text_content()} for m in request.messages])
    remaining_budget = deps.config.token_budget.total_context_max - messages_tokens
    dynamic_budget = min(remaining_budget, deps.config.token_budget.dynamic_context_max)
    logger.info(f"[Trace] Tokens: input={messages_tokens} budget_remain={remaining_budget} dynamic_budget={dynamic_budget}")

    non_system = [m for m in request.messages if m.role != "system"]
    msg_count = len(non_system)
    first_msg_hash = hashlib.md5(non_system[0].get_text_content()[:100].encode()).hexdigest()[:6] if non_system else "none"
    prefix_content = "".join(m.get_text_content()[:50] for m in non_system[:3])
    prefix_hash = hashlib.md5(prefix_content.encode()).hexdigest()[:8]
    logger.debug(f"[CacheDiag] msgs={msg_count} first_hash={first_msg_hash} prefix_hash={prefix_hash}")

    _multimodal_parts: dict[int, list] = {}
    for i, m in enumerate(request.messages):
        if m.role == "user" and isinstance(m.content, list):
            non_text = [p for p in m.content if p.get("type") != "text"]
            if non_text:
                _multimodal_parts[i] = non_text

    t_ctx_start = time.time()
    messages_dicts = [{"role": m.role, "content": m.get_text_content()} for m in request.messages]

    prune_pending_responses()
    _is_continuation = is_continuation(messages_dicts)
    if _is_continuation:
        logger.info(f"[Trace] autoContinue detected — deferring turn increment & Sub-B")

    stabilized_messages, lorebook_delta = await deps.system_stabilizer.stabilize(
        session_id, messages_dicts
    )
    messages_dicts = stabilized_messages

    if deps.config.prompt_caching.compress_enabled:
        messages_dicts, compress_shift = await deps.message_compressor.compress(session_id, messages_dicts)
        if compress_shift > 0:
            _multimodal_parts = {
                max(0, k - compress_shift): v
                for k, v in _multimodal_parts.items()
                if k >= compress_shift
            }
            logger.info(f"[Trace] MessageCompressor: {compress_shift} messages replaced with summary chunks")

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

    augmented_messages = build_cacheable_messages(
        messages_dicts,
        context_result["md_prefix"],
        context_result["dynamic_suffix"],
        lorebook_delta,
    )

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
                await finalize_turn(
                    session_id=session_id,
                    full_response=full_response,
                    is_cont=_is_continuation,
                    last_user_input=last_user_input,
                    scriptstate=scriptstate,
                    augmented_messages=augmented_messages,
                    request_model=request.model,
                    mode="stream",
                )
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

    final_response, turn_number = await finalize_turn(
        session_id=session_id,
        full_response=llm_response,
        is_cont=_is_continuation,
        last_user_input=last_user_input,
        scriptstate=scriptstate,
        augmented_messages=augmented_messages,
        request_model=request.model,
        mode="sync",
    )

    clean_response = strip_state_block(final_response)
    has_state = len(clean_response) < len(final_response)

    if turn_number is None:
        # Buffered continuation: return combined cumulative response, no turn increment
        logger.info(f"[Trace] State block: {'found & stripped' if has_state else 'not found'} | clean={len(clean_response)}ch (buffered)")
        return ChatCompletionResponse(
            id=f"chatcmpl-saga-{session_id}-cont",
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

    logger.info(f"[Trace] State block: {'found & stripped' if has_state else 'not found'} | clean={len(clean_response)}ch | total={(time.time() - t_start)*1000:.0f}ms")

    return ChatCompletionResponse(
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
