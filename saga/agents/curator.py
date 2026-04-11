"""Letta Curator — N턴마다 비동기 실행. Memory Block 기반 서사 판단 연속성."""
import asyncio
import json
import logging
import time
from langsmith import traceable
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.adapters.curator_adapter import LettaCuratorAdapter, DirectLLMCuratorAdapter
from saga.cost_tracker import UsageRecord
from saga.utils.parsers import parse_llm_json

logger = logging.getLogger(__name__)


def _log_lore_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"[Curator] Deferred lore task failed: {exc}", exc_info=exc)


class CuratorRunner:
    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache, llm_client, config, cost_tracker=None):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config
        self.cost_tracker = cost_tracker

        # Try Letta first, fallback to direct LLM
        self.letta_adapter = LettaCuratorAdapter(config)
        self.fallback_adapter = DirectLLMCuratorAdapter(llm_client, config)
        self._use_letta = False
        self._lore_tasks: set = set()  # GC protection for deferred lore tasks
        self.lore_defer_delay: float = 10.0  # seconds; set to 0 for benchmark mode

    def initialize(self):
        """Connect Letta client only. Agent creation is lazy (per session on first run)."""
        if self.config.curator.enabled:
            try:
                self.letta_adapter.initialize()
                self._use_letta = self.letta_adapter._initialized
                if self._use_letta:
                    logger.info("[Curator] Using Letta Memory Block curator (agents created lazily per session)")
                else:
                    logger.warning("[Curator] Letta client unavailable, using fallback")
            except Exception as e:
                logger.warning(f"[Curator] Letta init failed, using fallback: {e}")
                self._use_letta = False

    @traceable(name="pipeline.curator")
    async def run(self, session_id: str, turn_number: int):
        """Run curator. Called every N turns asynchronously."""
        context = None
        t_start = time.monotonic()
        adapter_name = "letta" if self._use_letta else "fallback"

        try:
            context = await self._gather_context(session_id, turn_number)

            adapter = self.letta_adapter if self._use_letta else self.fallback_adapter
            result = await adapter.run(session_id, context)

            # Record curator cost
            if self.cost_tracker:
                if self._use_letta:
                    # Letta: estimate tokens from prompt length (Letta doesn't expose usage)
                    prompt_text = self.letta_adapter._build_prompt(session_id, context)
                    est_input = len(prompt_text) // 3  # rough: 3 chars ≈ 1 token
                    await self.cost_tracker.record(UsageRecord(
                        model=self.config.curator.letta_model.split("/")[-1],
                        input_tokens=est_input,
                        output_tokens=500,  # estimated curator response
                        session_id=session_id,
                        call_type="curator_letta",
                    ))
                else:
                    usage = self.llm_client._last_usage
                    await self.cost_tracker.record(UsageRecord(
                        model=usage.get("model", self.config.models.curator),
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        cache_read_tokens=usage.get("cache_read", 0),
                        cache_create_tokens=usage.get("cache_create", 0),
                        session_id=session_id,
                        call_type="curator",
                    ))

            await self._apply_results(session_id, turn_number, result)

            # P0-1: Force compress_story if stable_prefix is empty after threshold.
            # Uses _compress_story_md directly to avoid duplicate _apply_results call.
            if (turn_number >= self.config.curator.compress_story_after_turns
                    and not result.get("compress_story")):
                existing = await self.md_cache.read_stable(session_id)
                if not existing.strip():
                    summary = (result.get("compressed_summary") or result.get("narrative_notes") or "").strip()
                    if summary:
                        await self._compress_story_md(session_id, turn_number, summary)
                        logger.info(f"[Curator] Forced compress_story at turn {turn_number} (stable_prefix was empty)")
                    else:
                        logger.warning(f"[Curator] Forced compress skipped at turn {turn_number}: no summary available")

            # P0-2: Sync Letta Memory Block → stable_prefix (overwrites forced compress
            # when Letta has a more authoritative narrative_summary)
            await self._sync_letta_memory(session_id)

            elapsed_ms = (time.monotonic() - t_start) * 1000
            logger.info(f"[Curator] Curation complete for session {session_id} at turn {turn_number} ({elapsed_ms:.0f}ms)")

            # Auto-generate lore: deferred fire-and-forget to avoid LLM API contention
            self._schedule_deferred_lore(session_id, turn_number)

        except Exception as e:
            logger.error(f"[Curator] Error: {e}", exc_info=True)
            # Try fallback if Letta failed and context was gathered
            if self._use_letta and context is not None:
                try:
                    logger.info("[Curator] Retrying with fallback adapter")
                    result = await self.fallback_adapter.run(session_id, context)
                    await self._apply_results(session_id, turn_number, result)
                    self._schedule_deferred_lore(session_id, turn_number)
                except Exception as e2:
                    logger.error(f"[Curator] Fallback also failed: {e2}")

    def _schedule_deferred_lore(self, session_id: str, turn_number: int):
        """Schedule lore generation as a deferred task with delay to avoid LLM API contention."""
        async def _deferred():
            await asyncio.sleep(self.lore_defer_delay)
            logger.info(f"[Curator] Deferred lore generation starting for turn {turn_number}")
            await self._auto_generate_lore(session_id, turn_number)

        task = asyncio.create_task(_deferred())
        self._lore_tasks.add(task)
        task.add_done_callback(self._lore_tasks.discard)
        task.add_done_callback(_log_lore_task_exception)

    async def _gather_context(self, session_id, turn_number):
        graph_summary = await self.sqlite_db.get_state_summary(session_id)

        recent_episodes = self.vector_db.search_episodes(session_id, "최근 일어난 일", n_results=20)
        episodes_text = ""
        if recent_episodes and recent_episodes.get("documents"):
            docs = recent_episodes["documents"][0]
            metas = recent_episodes["metadatas"][0]
            episodes_text = "\n".join(
                f"- [Turn {m.get('turn', '?')}] {doc}" for doc, m in zip(docs, metas)
            )

        from_turn = max(0, turn_number - self.config.curator.interval)
        turn_logs = await self.sqlite_db.get_turn_logs(session_id, from_turn=from_turn, to_turn=turn_number)

        contradictions = await self.sqlite_db.detect_contradictions(session_id)

        return {
            "turn_number": turn_number,
            "graph_summary": graph_summary,
            "episodes_text": episodes_text,
            "turn_logs": turn_logs,
            "contradictions": contradictions,
        }

    async def _apply_results(self, session_id, turn_number, result):
        if result.get("contradictions"):
            for fix in result["contradictions"]:
                logger.info(f"[Curator] Contradiction fix: {fix}")
                # Apply graph fixes as needed

        if result.get("events"):
            for event in result["events"]:
                if not isinstance(event, dict):
                    logger.debug(f"[Curator] Skipping non-dict event: {event!r}")
                    continue
                await self.sqlite_db.queue_event(session_id, event)

        if result.get("compress_story") and result.get("compressed_summary"):
            await self._compress_story_md(session_id, turn_number, result["compressed_summary"])

    async def _auto_generate_lore(self, session_id: str, turn_number: int):
        """Detect entities without lore and auto-generate lore entries."""
        entities = await self.sqlite_db.get_entities_without_lore(session_id)
        if not entities:
            return

        # Limit to 3 per curation cycle to avoid overload
        entities = entities[:3]
        logger.info(f"[Curator] Auto-generating lore for {len(entities)} entities: {[e.get('name') for e in entities]}")

        for entity in entities:
            try:
                await self._generate_single_lore(session_id, turn_number, entity)
            except Exception as e:
                logger.warning(f"[Curator] Lore generation failed for {entity.get('name')}: {e}")

    async def _generate_single_lore(self, session_id: str, turn_number: int, entity: dict):
        """Generate lore for a single entity using episodes and graph context."""
        entity_name = entity.get("name", "")
        entity_type = entity.get("entity_type", "character")

        # Gather episodes mentioning this entity
        episodes = self.vector_db.search_episodes(session_id, entity_name, n_results=10)
        episodes_text = ""
        source_turns = []
        if episodes and episodes.get("documents"):
            docs = episodes["documents"][0] if isinstance(episodes["documents"][0], list) else episodes["documents"]
            metas = episodes["metadatas"][0] if isinstance(episodes["metadatas"][0], list) else episodes["metadatas"]
            ep_lines = []
            for doc, meta in zip(docs, metas):
                turn = meta.get("turn", "?")
                source_turns.append(turn)
                ep_lines.append(f"- [Turn {turn}] {doc[:300]}")
            episodes_text = "\n".join(ep_lines[:8])

        # Get relationships for this entity
        relationships = []
        if entity_type == "character":
            rels = await self.sqlite_db.get_relationships(session_id, entity_name)
            relationships = [
                {"target": r.get("to_name") if r.get("from_name") == entity_name else r.get("from_name"),
                 "rel_type": r.get("rel_type", ""),
                 "strength": r.get("strength", 0)}
                for r in rels
            ]

        rel_text = "\n".join(
            f"- {r.get('target')}: {r.get('rel_type')} (강도 {r.get('strength')})"
            for r in relationships
        ) if relationships else "없음"

        # Build prompt for lore generation
        prompt = (
            f"다음 {entity_type}에 대한 로어(배경 설정)를 생성하세요.\n\n"
            f"이름: {entity_name}\n"
            f"유형: {entity_type}\n"
        )
        if entity.get("location"):
            prompt += f"위치: {entity['location']}\n"
        if entity.get("mood"):
            prompt += f"분위기: {entity['mood']}\n"
        prompt += (
            f"\n[관련 에피소드]\n{episodes_text or '없음'}\n\n"
            f"[관계]\n{rel_text}\n\n"
            "위 정보를 바탕으로 JSON 형식으로 응답하세요:\n"
            '{"content": "로어 텍스트 (2~3문장)", "keywords": "쉼표로 구분된 활성화 키워드", '
            '"lore_type": "character|location|item|event|world", "priority": 50}'
        )

        # Call LLM (use curator model for reliable structured output)
        response = await self.llm_client.call_llm(
            model=self.config.models.curator,
            messages=[
                {"role": "system", "content": "당신은 RP 세계관 구축 전문가입니다. 에피소드와 관계 정보를 분석하여 일관성 있는 로어를 생성합니다. 반드시 JSON으로만 응답하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1024,
            response_mime_type="application/json",
        )

        # Parse response
        lore_data = parse_llm_json(response)

        if not lore_data or not lore_data.get("content") or len(lore_data.get("content", "")) < 20:
            logger.warning(f"[Curator] Could not parse lore for {entity_name} (too short or empty): {response[:200]}")
            return

        content = lore_data["content"]
        keywords = lore_data.get("keywords", entity_name)
        lore_type = lore_data.get("lore_type", entity_type)
        priority = lore_data.get("priority", 50)

        # Store in SQLite
        await self.sqlite_db.create_lore(
            session_id=session_id,
            name=f"lore_{entity_name}",
            lore_type=lore_type,
            keywords=keywords,
            content=content,
            priority=priority,
            auto_generated=True,
            source_turns=json.dumps(source_turns[:5]),
        )

        # Store in ChromaDB for vector search
        lore_id = f"{session_id}_lore_{entity_name}"
        self.vector_db.add_lorebook_entry(
            entry_id=lore_id,
            text=content,
            metadata={
                "session_id": session_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
                "lore_type": lore_type,
                "keywords": keywords,
                "priority": priority,
                "auto_generated": True,
                "source_turns": json.dumps(source_turns[:5]),
            },
        )

        logger.info(f"[Curator] Auto-generated lore for '{entity_name}' ({lore_type}, priority={priority})")

    async def _sync_letta_memory(self, session_id: str):
        """P0-2: Sync Letta narrative_summary Memory Block → stable_prefix.md."""
        if not self._use_letta:
            return
        try:
            blocks = await asyncio.wait_for(
                self.letta_adapter.read_memory_blocks(session_id),
                timeout=5.0,
            )
            narrative = blocks.get("narrative_summary", "")
            if not narrative or len(narrative) < 50:
                return
            chars = await self.sqlite_db.get_session_characters(session_id)
            lore = await self.sqlite_db.get_all_lore(session_id)
            content = await self.md_cache.build_stable_content(
                chars, lore, narrative_summary=narrative
            )
            await self.md_cache.write_stable(session_id, content)
            logger.info(f"[Curator] Synced Letta narrative_summary to stable_prefix ({len(narrative)} chars)")
        except asyncio.TimeoutError:
            logger.warning("[Curator] Letta memory block read timed out (5s)")
        except Exception as e:
            logger.warning(f"[Curator] Memory block sync failed: {e}")

    async def _compress_story_md(self, session_id, turn_number, compressed_summary):
        """Update stable prefix with compressed story summary."""
        chars = await self.sqlite_db.get_session_characters(session_id)
        lore = await self.sqlite_db.get_all_lore(session_id)
        content = await self.md_cache.build_stable_content(
            chars, lore, narrative_summary=compressed_summary
        )
        await self.md_cache.write_stable(session_id, content)
