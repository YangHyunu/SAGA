"""Sub-B: Post-Turn — 매 턴 비동기, Flash 서사 요약, 유저 안 기다림.

프레젠테이션 S8: "scriptstate 스냅샷 저장 + Flash 서사 요약,
ChromaDB episode 임베딩, turn_log 기록"

Flash 서사 요약 → ChromaDB 에피소드 저장 → turn_log 기록
asyncio.Lock으로 빠른 연속 입력 방지
"""
import asyncio
import logging
import re
import time
from typing import Callable

from langsmith import traceable

try:
    from rapidfuzz import fuzz as _fuzz
except ImportError:
    _fuzz = None
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

    @traceable(name="pipeline.sub_b")
    async def extract_and_update(self, session_id: str, response_text: str, turn_number: int, user_input: str = "", scriptstate: dict | None = None):
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

            # 3. NPC 레지스트리 업데이트 (Flash의 npcs_mentioned 기반, 필터링 + 정규화)
            for npc in (narrative.get("npcs_mentioned") or []):
                if npc and self._is_valid_npc_name(npc):
                    resolved = await self._resolve_npc_name(session_id, npc)
                    await self.sqlite_db.create_character(session_id, resolved, is_player=False)

            # 4. turn_log 기록
            await self.sqlite_db.insert_turn_log(
                session_id, turn_number, narrative,
                user_input=user_input, assistant_output=response_text,
            )

            # 5. live_state.md 갱신 (scriptstate 우선 → SQLite 폴백)
            try:
                player_ctx = await self.sqlite_db.query_player_context(session_id)
                # scriptstate가 없으면 world_state KV에서 마지막 저장분 로드
                ss = scriptstate
                if not ss:
                    raw = await self.sqlite_db.get_world_state_value(session_id, "scriptstate")
                    if raw:
                        import json
                        try:
                            ss = json.loads(raw)
                        except (json.JSONDecodeError, ValueError):
                            ss = None
                await self.md_cache.write_live(session_id, turn_number, {}, player_ctx, scriptstate=ss)
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

    # Patterns that indicate unnamed extras, not real NPC names
    _EXTRA_PATTERN = re.compile(
        r"^(마을|숲|동굴|거리|성|궁|술집|시장)?\s*"
        r"(사람|여인|남자|여자|병사|기사|상인|농부|노인|아이|소녀|소년|시민|주민|행인|경비|하인|종자|시녀|광부|어부|사제|수녀)\s*\d*$"
    )
    _NUMBERED_PATTERN = re.compile(r"^.{1,4}\s*[#\dA-D]+$")  # "병사 1", "NPC #3"

    # Korean particle suffixes for normalization
    _KO_PARTICLE = re.compile(
        r"(이|가|을|를|의|에게|에서|한테|으로|로|와|과|은|는|도|만|까지|부터|보다|처럼|아|야)$"
    )
    # English articles
    _EN_ARTICLE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)
    # Japanese honorific suffixes
    _JP_HONORIFIC = re.compile(r"[-\s]?(san|kun|chan|sama|dono|sensei)$", re.IGNORECASE)
    # Parenthetical annotations: "라쿤(Raccoon)" → "라쿤"
    _PAREN = re.compile(r"\s*\([^)]*\)\s*$")

    _FUZZY_THRESHOLD = 88

    @staticmethod
    def _is_valid_npc_name(name: str) -> bool:
        """Filter out unnamed extras and generic descriptions."""
        name = name.strip()
        if len(name) < 2 or len(name) > 30:
            return False
        if PostTurnExtractor._EXTRA_PATTERN.match(name):
            return False
        if PostTurnExtractor._NUMBERED_PATTERN.match(name):
            return False
        return True

    @staticmethod
    def _normalize_npc_name(name: str) -> str:
        """Normalize NPC name for deduplication (Mem0-style).

        Strips articles, particles, honorifics, parentheticals, then lowercases.
        Used as lookup key; original display name is preserved in DB.
        """
        s = name.strip()
        s = PostTurnExtractor._PAREN.sub("", s)
        s = PostTurnExtractor._EN_ARTICLE.sub("", s)
        s = PostTurnExtractor._JP_HONORIFIC.sub("", s)
        s = s.strip().lower()
        s = PostTurnExtractor._KO_PARTICLE.sub("", s)
        return s.strip()

    async def _resolve_npc_name(self, session_id: str, raw_name: str) -> str:
        """Resolve raw NPC name to canonical name via normalization + fuzzy match.

        Returns the existing canonical name if a match is found, otherwise raw_name.
        """
        normalized = self._normalize_npc_name(raw_name)

        # Get existing characters for this session
        existing = await self.sqlite_db.get_session_characters(session_id)
        if not existing:
            return raw_name

        # Layer 1: exact match on normalized form
        for char in existing:
            if self._normalize_npc_name(char["name"]) == normalized:
                return char["name"]

        # Layer 2: fuzzy match (rapidfuzz)
        if _fuzz is not None:
            best_score, best_name = 0, None
            for char in existing:
                score = _fuzz.token_set_ratio(normalized, self._normalize_npc_name(char["name"]))
                if score > best_score:
                    best_score, best_name = score, char["name"]
            if best_score >= self._FUZZY_THRESHOLD and best_name:
                logger.info(
                    f"[Sub-B] NPC fuzzy match: {raw_name!r} → {best_name!r} "
                    f"(score={best_score:.0f})"
                )
                return best_name

        return raw_name

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
