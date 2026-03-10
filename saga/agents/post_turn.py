"""Sub-B: Post-Turn — 매 턴 비동기, Flash 서사 요약, 유저 안 기다림.

프레젠테이션 S8: "scriptstate 스냅샷 저장 + Flash 서사 요약,
ChromaDB episode 임베딩, turn_log 기록"

Flash 서사 요약 → ChromaDB 에피소드 저장 → turn_log 기록
asyncio.Lock으로 빠른 연속 입력 방지
"""
import asyncio
import logging
import time
from typing import Callable
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.llm.client import LLMClient
from saga.utils.parsers import format_turn_narrative
logger = logging.getLogger(__name__)

# Sub-B concurrency lock
_sub_b_lock = asyncio.Lock()


class PostTurnExtractor:
    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache, llm_client: LLMClient, config, extract_fn: Callable = None):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config
        self.extract_fn = extract_fn

    async def extract_and_update(self, session_id: str, response_text: str, turn_number: int, user_input: str = ""):
        """Main entry point. Called as asyncio.create_task after response is sent."""
        logger.info(f"[Sub-B] ▶ Task started for turn {turn_number}, session={session_id}")

        # Wait for previous Sub-B to finish
        await _sub_b_lock.acquire()
        logger.info(f"[Sub-B] Lock acquired for turn {turn_number}")

        t_start = time.monotonic()
        narrative = None
        importance = 0
        try:
            # 1. Flash 서사 요약 추출
            narrative = await self.extract_fn(response_text, session_id)

            if narrative is None:
                logger.warning(f"[Sub-B] Narrative extraction failed for turn {turn_number}, recording minimal")
                narrative = {"summary": "", "npcs_mentioned": [], "scene_type": "dialogue", "key_event": None}

            # 2. ChromaDB 에피소드 저장
            importance = self._calculate_importance(narrative)
            self._record_episode(session_id, turn_number, user_input, response_text, narrative, importance)

            # 3. NPC 레지스트리 업데이트 (Flash의 npcs_mentioned 기반)
            for npc in (narrative.get("npcs_mentioned") or []):
                if npc:
                    await self.sqlite_db.create_character(session_id, npc, is_player=False)

            # 4. turn_log 기록
            await self.sqlite_db.insert_turn_log(
                session_id, turn_number, narrative,
                user_input=user_input, assistant_output=response_text,
            )

            # 5. live_state.md 갱신 (SQLite 캐릭터 상태 기반)
            try:
                player_ctx = await self.sqlite_db.query_player_context(session_id)
                await self.md_cache.write_live(session_id, turn_number, {}, player_ctx)
            except Exception as live_err:
                logger.warning(f"[Sub-B] live_state.md write failed: {live_err}")

            logger.info(f"[Sub-B] Turn {turn_number} post-processing complete (importance={importance})")

        except Exception as e:
            logger.error(f"[Sub-B] Error processing turn {turn_number}: {e}", exc_info=True)
        finally:
            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.info(f"[Sub-B] Releasing lock for turn {turn_number}, elapsed={elapsed_ms:.0f}ms")
            _sub_b_lock.release()

    def _record_episode(self, session_id, turn_number, user_input, response_text, narrative, importance):
        """Record episode summary to ChromaDB."""
        summary = narrative.get("summary", "")
        if not summary:
            summary = format_turn_narrative(turn_number, user_input, response_text, {})

        scene_type = narrative.get("scene_type", "dialogue")
        npcs = narrative.get("npcs_mentioned") or []

        self.vector_db.add_episode(
            session_id, turn_number, summary, "unknown",
            episode_type=scene_type,
            importance=importance,
            entities=npcs,
            npcs=npcs,
        )
        if importance >= 50:
            logger.info(f"[Sub-B] High-importance episode (turn {turn_number}, score={importance}, type={scene_type})")

    @staticmethod
    def _calculate_importance(narrative: dict) -> int:
        """Calculate episode importance from narrative summary."""
        score = 10  # base

        scene_type = narrative.get("scene_type", "dialogue")
        if scene_type == "combat":
            score += 40
        elif scene_type == "event":
            score += 35
        elif scene_type == "exploration":
            score += 15

        if narrative.get("key_event"):
            score += 30

        npcs = narrative.get("npcs_mentioned") or []
        if npcs:
            score += 10 * min(2, len(npcs))

        return min(score, 100)
