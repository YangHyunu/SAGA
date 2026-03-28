"""Sub-B: Post-Turn — 매 턴 비동기, Flash 서사 요약, 유저 안 기다림.

Flash 서사 요약 → ChromaDB 에피소드 저장 → turn_log 기록
asyncio.Lock으로 빠른 연속 입력 방지
"""
import asyncio
import json
import logging
import re
import time
from typing import Callable

from langsmith import traceable

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
                    base_name, alias = self._extract_alias(npc)
                    resolved = await self._resolve_npc_name(session_id, base_name)
                    await self.sqlite_db.create_character(session_id, resolved, is_player=False)
                    # alias가 있으면 등록 (괄호에서 추출된 것)
                    if alias:
                        await self.sqlite_db.add_character_alias(session_id, resolved, alias)
                        # base_name 자체도 alias로 등록 (역방향 검색용)
                        if base_name != resolved:
                            await self.sqlite_db.add_character_alias(session_id, resolved, base_name)

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

    # Unnamed extras / generic descriptions
    _EXTRA_PATTERN = re.compile(
        r"^(마을|숲|동굴|거리|성|궁|술집|시장)?\s*"
        r"(사람|여인|남자|여자|병사|기사|상인|농부|노인|아이|소녀|소년|시민|주민|행인|경비|하인|종자|시녀|광부|어부|사제|수녀)\s*\d*$"
    )
    _NUMBERED_PATTERN = re.compile(r"^.{1,4}\s*[#\dA-D]+$")

    # Parenthetical alias: "루비아(Rubia)" → base="루비아", alias="Rubia"
    _PAREN = re.compile(r"\s*\(([^)]*)\)\s*$")

    _NPC_DEDUP_PROMPT = (
        "Given a NEW name and a list of EXISTING character names, "
        "determine if the NEW name refers to the same character as any EXISTING name. "
        "Consider: translations (한국어↔English), alternate spellings, nicknames, "
        "honorifics, and phonetic similarities.\n"
        "If it matches, respond with ONLY the existing name. "
        "If no match, respond with ONLY 'NONE'.\n\n"
        "EXISTING: {existing_names}\n"
        "NEW: {new_name}\n"
        "Answer:"
    )

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
    def _extract_alias(name: str) -> tuple[str, str | None]:
        """Extract base name and alias from parenthetical notation.

        "루비아(Rubia)" → ("루비아", "Rubia")
        "루비아"        → ("루비아", None)
        """
        m = PostTurnExtractor._PAREN.search(name)
        if m:
            alias = m.group(1).strip()
            base = PostTurnExtractor._PAREN.sub("", name).strip()
            return base, alias if alias else None
        return name.strip(), None

    async def _resolve_npc_name(self, session_id: str, raw_name: str) -> str:
        """Resolve NPC name: alias exact match → name exact match → LLM judgment.

        Returns the existing canonical name if a match is found, otherwise raw_name.
        """
        existing = await self.sqlite_db.get_session_characters(session_id)
        if not existing:
            return raw_name

        raw_lower = raw_name.strip().lower()

        # Layer 0: alias exact match (괄호에서 추출된 alias)
        alias_match = await self.sqlite_db.find_character_by_alias(session_id, raw_name)
        if alias_match:
            logger.info(f"[Sub-B] NPC alias match: {raw_name!r} → {alias_match['name']!r}")
            return alias_match["name"]

        # Layer 1: name exact match (case-insensitive)
        for char in existing:
            if char["name"].strip().lower() == raw_lower:
                return char["name"]

        # Layer 2: LLM judgment (한/영, 별명, 번역 등 전부 커버)
        existing_names = [c["name"] for c in existing if not c.get("is_player")]
        if existing_names:
            try:
                match = await self._llm_dedup_check(raw_name, existing_names)
                if match:
                    logger.info(f"[Sub-B] NPC LLM match: {raw_name!r} → {match!r}")
                    # 새 이름을 alias로 등록
                    await self.sqlite_db.add_character_alias(session_id, match, raw_name)
                    return match
            except Exception as e:
                logger.warning(f"[Sub-B] NPC LLM dedup failed: {e}")

        return raw_name

    async def _llm_dedup_check(self, new_name: str, existing_names: list[str]) -> str | None:
        """Ask LLM if new_name matches any existing character. Returns matched name or None."""
        prompt = self._NPC_DEDUP_PROMPT.format(
            existing_names=", ".join(existing_names),
            new_name=new_name,
        )
        response = await self.llm_client.call_llm(
            messages=[{"role": "user", "content": prompt}],
            model=self.config.models.extraction,  # Flash 모델 (저비용)
            max_tokens=50,
            temperature=0,
        )
        answer = response.strip().strip('"').strip("'")
        # 응답이 기존 이름 중 하나와 일치하면 반환
        for name in existing_names:
            if answer.lower() == name.lower():
                return name
        return None

    async def deduplicate_npcs(self, session_id: str) -> list[tuple[str, str]]:
        """Scan existing NPCs and merge duplicates via LLM judgment.

        Returns list of (keep, removed) pairs that were merged.
        """
        chars = await self.sqlite_db.get_session_characters(session_id)
        npcs = [c for c in chars if not c.get("is_player")]
        if len(npcs) < 2:
            return []

        merged = []
        names = [c["name"] for c in npcs]
        skip = set()

        for i, npc in enumerate(npcs):
            if npc["name"] in skip:
                continue
            others = [n for j, n in enumerate(names) if j != i and n not in skip]
            if not others:
                continue
            try:
                match = await self._llm_dedup_check(npc["name"], others)
                if match:
                    # keep the one that appeared first (lower id = earlier)
                    await self.sqlite_db.merge_characters(session_id, match, npc["name"])
                    merged.append((match, npc["name"]))
                    skip.add(npc["name"])
                    logger.info(f"[Sub-B] Dedup merged: {npc['name']!r} → {match!r}")
            except Exception as e:
                logger.warning(f"[Sub-B] Dedup check failed for {npc['name']!r}: {e}")

        return merged

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
