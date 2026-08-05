"""dreaming/idle.py — IdleTrigger 추상화 (스펙 §3.2, §8).

유휴 기준 = 캐시 TTL 경과(기본 5m) — 캐시가 이미 죽은 시점이라
꿈(과 Plan 4의 재압축)이 공짜인 창구다. cron 아님, 세션별 타이머.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable, Dict

logger = logging.getLogger(__name__)


class IdleWatcher:
    def __init__(self, idle_seconds: float,
                 on_idle: Callable[[str], Awaitable[None]]) -> None:
        self._idle = idle_seconds
        self._on_idle = on_idle
        self._tasks: Dict[str, asyncio.Task] = {}

    def touch(self, session: str) -> None:
        prev = self._tasks.pop(session, None)
        if prev is not None:
            prev.cancel()
        self._tasks[session] = asyncio.create_task(self._wait(session))

    async def _wait(self, session: str) -> None:
        try:
            await asyncio.sleep(self._idle)
        except asyncio.CancelledError:
            return
        self._tasks.pop(session, None)
        try:
            await self._on_idle(session)
        except Exception:
            logger.exception("[idle] on_idle failed: %s", session)   # fail-open

    def cancel_all(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        self._tasks.clear()
