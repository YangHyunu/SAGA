"""dreaming/idle.py — IdleTrigger 추상화 (스펙 §3.2, §8).

유휴 기준 기본 5m는 Anthropic 캐시 TTL과 동기다. 단, "유휴 = 캐시 죽음
= 재압축 공짜" 등식은 **Anthropic처럼 TTL 만료로 캐시가 소멸하는
프로바이더에서만** 성립한다 (스펙 §6.3 프로바이더 한정 주의). 자동
프리픽스 캐싱(DeepSeek 등)은 유휴와 무관하게 캐시가 수 시간 살아 있어
유휴 재압축이 오히려 살아 있는 캐시를 깨뜨린다 — 그쪽 방어선은
chunks.BOUNDARY_STEP(플랜 바이트 변경 빈도 자체를 낮춤)이다.
cron 아님, 세션별 타이머.
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
