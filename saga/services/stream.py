"""SSE streaming response handling for SAGA."""
import json
import logging
import re
import time

from saga.core import dependencies as deps
from saga.cost_tracker import UsageRecord
from saga.models import ChatCompletionChunk, StreamChoice, StreamDelta

logger = logging.getLogger(__name__)


async def stream_response(session_id, session, augmented_messages, request, stream_ctx, gen_params=None):
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

    if gen_params is None:
        gen_params = {}
    logger.info(f"[Trace] Stream start: model={deps.config.models.narration}")

    async for chunk in deps.llm_client.call_llm_stream(
        model=deps.config.models.narration,
        messages=augmented_messages,
        temperature=request.temperature or 0.7,
        max_tokens=request.max_tokens or 8192,
        **gen_params,
    ):
        full_response += chunk
        stream_ctx["full_response"] = full_response  # incremental update for BackgroundTask

        # TTFT measurement
        if not ttft_recorded:
            ttft_ms = (time.time() - t_stream_start) * 1000
            ttft_recorded = True
            logger.info(f"[Trace] TTFT: {ttft_ms:.0f}ms")

        buffer += chunk

        # Process buffer for state block markers
        while buffer:
            if in_state_block:
                close_idx = buffer.find("```")
                if close_idx != -1:
                    buffer = buffer[close_idx + 3:]
                    in_state_block = False
                else:
                    buffer = buffer[-10:]
                    break
            else:
                marker_match = re.search(r'`{2,3}\s*state\s*\n', buffer)
                if marker_match:
                    before = buffer[:marker_match.start()]
                    if before:
                        yield make_sse_chunk(session_id, request.model, before)
                    buffer = buffer[marker_match.end():]
                    in_state_block = True
                    if not state_block_filtered:
                        state_block_filtered = True
                        logger.info("[Trace] Stream: state block detected, filtering...")
                else:
                    partial = _partial_state_marker(buffer)
                    if partial > 0:
                        safe = buffer[:-partial]
                        if safe:
                            yield make_sse_chunk(session_id, request.model, safe)
                        buffer = buffer[-partial:]
                        break
                    else:
                        if buffer:
                            yield make_sse_chunk(session_id, request.model, buffer)
                        buffer = ""
                        break

    # Flush remaining buffer (if not inside a state block)
    if buffer and not in_state_block:
        yield make_sse_chunk(session_id, request.model, buffer)

    # Final chunk
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

    # Record cost for stream call
    usage = deps.llm_client._last_usage
    await deps.cost_tracker.record(UsageRecord(
        model=usage["model"], input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"], cache_read_tokens=usage["cache_read"],
        cache_create_tokens=usage["cache_create"], session_id=session_id, call_type="main_stream",
    ))

    stream_total_ms = (time.time() - t_stream_start) * 1000
    logger.info(
        f"[Trace] Stream done: {stream_total_ms:.0f}ms | "
        f"response={len(full_response)}ch state_filtered={state_block_filtered}"
    )
    stream_ctx["full_response"] = full_response


def make_sse_chunk(session_id: str, model: str | None, content: str) -> str:
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
