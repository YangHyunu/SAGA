"""Global component holders for SAGA.

All components are initialized as None and populated during lifespan startup.
Other modules import from here to access shared state.
"""
from __future__ import annotations

import asyncio
import re
import typing
from collections import OrderedDict

if typing.TYPE_CHECKING:
    from saga.config import SagaConfig
    from saga.storage.sqlite_db import SQLiteDB
    from saga.storage.vector_db import VectorDB
    from saga.storage.md_cache import MdCache
    from saga.llm.client import LLMClient
    from saga.agents.context_builder import ContextBuilder
    from saga.agents.post_turn import PostTurnExtractor
    from saga.agents.curator import CuratorRunner
    from saga.session import SessionManager
    from saga.system_stabilizer import SystemStabilizer
    from saga.cost_tracker import CostTracker
    from saga.message_compressor import MessageCompressor

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

# ============================================================
# Session ID parsing
# ============================================================

_SESSION_ID_RE = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

# ============================================================
# Global component instances (set in lifespan)
# ============================================================

config: SagaConfig | None = None
sqlite_db: SQLiteDB | None = None
vector_db: VectorDB | None = None
md_cache: MdCache | None = None
llm_client: LLMClient | None = None
context_builder: ContextBuilder | None = None
post_turn: PostTurnExtractor | None = None
curator: CuratorRunner | None = None
session_mgr: SessionManager | None = None
system_stabilizer: SystemStabilizer | None = None
cost_tracker: CostTracker | None = None
message_compressor: MessageCompressor | None = None

# Keep strong references to fire-and-forget tasks so GC doesn't collect them
_background_tasks: set = set()


def log_task_exception(task: asyncio.Task) -> None:
    """Callback for fire-and-forget tasks to log unhandled exceptions."""
    import logging
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logging.getLogger("saga").error(f"[SAGA] Background task failed: {exc}", exc_info=exc)

# Cache warming: track last request per session (bounded)
_warming_data: _BoundedDict = _BoundedDict(_MAX_TRACKED_SESSIONS)
_warming_lock = asyncio.Lock()
