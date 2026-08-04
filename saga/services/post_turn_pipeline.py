"""Post-turn pipeline: autoContinue handling, turn finalization, background tasks."""
import asyncio
import logging
import time

from saga.core import dependencies as deps
from saga.services.cache_marker import is_anthropic_model
from saga.utils.parsers import strip_state_block

logger = logging.getLogger(__name__)


def is_continuation(messages: list[dict]) -> bool:
    """Detect RisuAI autoContinue requests.

    RisuAI inserts '[Continue the last response]' as a system message in
    postEverything when autoContinueChat fires. We scan from the end,
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


def combine_with_pending(session_id: str, full_response: str, is_cont: bool) -> tuple[str, bool]:
    """Handle autoContinue buffering.

    Returns (response, buffered):
      - buffered=True  → caller should skip turn increment (more continuation coming)
      - buffered=False → response is final (possibly combined with prior partial)
    """
    if is_cont:
        prev = deps._pending_responses.get(session_id, {}).get("response", "")
        combined = prev + full_response
        deps._pending_responses[session_id] = {"response": combined, "timestamp": time.time()}
        logger.info(
            f"[Trace] autoContinue: buffered partial response "
            f"(cumulative={len(combined)}ch) — skipping turn increment"
        )
        return combined, True

    pending_entry = deps._pending_responses.pop(session_id, None)
    if pending_entry:
        combined = pending_entry["response"] + full_response
        logger.info(
            f"[Trace] autoContinue: final response — combining "
            f"pending({len(pending_entry['response'])}ch) + current({len(full_response)}ch)"
        )
        return combined, False
    return full_response, False


async def spawn_post_turn_tasks(
    session_id: str,
    full_response: str,
    turn_number: int,
    last_user_input: str,
    scriptstate: dict | None,
    augmented_messages: list,
    request_model: str | None,
) -> None:
    """Spawn Sub-B extraction + Curator + cache warming. Does NOT increment turn."""
    task_b = asyncio.create_task(
        deps.post_turn.extract_and_update(
            session_id, full_response, turn_number, last_user_input, scriptstate=scriptstate
        )
    )
    deps._background_tasks.add(task_b)
    task_b.add_done_callback(deps._background_tasks.discard)
    task_b.add_done_callback(deps.log_task_exception)

    if deps.config.curator.enabled and turn_number % deps.config.curator.interval == 0:
        task_c = asyncio.create_task(deps.curator.run(session_id, turn_number))
        deps._background_tasks.add(task_c)
        task_c.add_done_callback(deps._background_tasks.discard)
        task_c.add_done_callback(deps.log_task_exception)

    if is_anthropic_model(request_model):
        async with deps._warming_lock:
            deps._warming_data[session_id] = {
                "timestamp": time.time(),
                "messages": augmented_messages,
                "model": deps.config.models.narration,
                "count": 0,
            }


async def finalize_turn(
    session_id: str,
    full_response: str,
    is_cont: bool,
    last_user_input: str,
    scriptstate: dict | None,
    augmented_messages: list,
    request_model: str | None,
    mode: str = "sync",
    ledger_ctx: dict | None = None,
) -> tuple[str, int | None]:
    """Unified post-turn finalization for both streaming and non-streaming.

    ledger_ctx: {"verdict": ..., "last_user_hash": ...} from the pair ledger.
    A reroll reuses the superseded turn's number instead of incrementing, so
    the regenerated response overwrites the discarded turn's records.

    Returns (final_response, turn_number):
      - turn_number=None → buffered (autoContinue in progress, no turn increment)
      - turn_number=int  → turn completed, tasks spawned
    """
    final_response, buffered = combine_with_pending(session_id, full_response, is_cont)
    if buffered:
        return final_response, None

    reroll_turn = (ledger_ctx or {}).get("verdict", {}).get("reroll_turn_number")
    if reroll_turn is not None:
        turn_number = reroll_turn
        logger.info(f"[Trace] ━━━ DONE turn={turn_number} (reroll, no increment) ━━━")
    else:
        turn_number = await deps.sqlite_db.increment_turn(session_id)
        suffix = f" ({mode})" if mode != "sync" else ""
        logger.info(f"[Trace] ━━━ DONE turn={turn_number}{suffix} ━━━")

    if ledger_ctx and deps.pair_ledger is not None:
        try:
            await deps.pair_ledger.record_turn(
                session_id, ledger_ctx["verdict"], ledger_ctx["last_user_hash"],
                strip_state_block(final_response), turn_number,
            )
        except Exception as e:
            logger.warning(f"[PairLedger] record_turn failed: {e}")

    await spawn_post_turn_tasks(
        session_id, final_response, turn_number,
        last_user_input, scriptstate, augmented_messages, request_model,
    )
    return final_response, turn_number
