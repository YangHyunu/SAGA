"""Letta Curator — N턴마다 비동기 실행. Memory Block 기반 서사 판단 연속성."""
import asyncio
import json
import logging
from datetime import datetime
from mene.storage.sqlite_db import SQLiteDB
from mene.storage.graph_db import GraphDB
from mene.storage.vector_db import VectorDB
from mene.storage.md_cache import MdCache
from mene.adapters.curator_adapter import LettaCuratorAdapter, DirectLLMCuratorAdapter

logger = logging.getLogger(__name__)


class CuratorRunner:
    def __init__(self, sqlite_db: SQLiteDB, graph_db: GraphDB, vector_db: VectorDB, md_cache: MdCache, llm_client, config):
        self.sqlite_db = sqlite_db
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.llm_client = llm_client
        self.config = config

        # Try Letta first, fallback to direct LLM
        self.letta_adapter = LettaCuratorAdapter(config)
        self.fallback_adapter = DirectLLMCuratorAdapter(llm_client, config)
        self._use_letta = False

    def initialize(self):
        if self.config.curator.enabled:
            try:
                self.letta_adapter.initialize()
                self._use_letta = True
                logger.info("[Curator] Using Letta Memory Block curator")
            except Exception as e:
                logger.warning(f"[Curator] Letta init failed, using fallback: {e}")
                self._use_letta = False

    async def run(self, session_id: str, turn_number: int):
        """Run curator. Called every N turns asynchronously."""
        context = None
        try:
            context = await self._gather_context(session_id, turn_number)

            adapter = self.letta_adapter if self._use_letta else self.fallback_adapter
            result = await adapter.run(session_id, context)

            await self._apply_results(session_id, turn_number, result)
            logger.info(f"[Curator] Curation complete for session {session_id} at turn {turn_number}")

        except Exception as e:
            logger.error(f"[Curator] Error: {e}", exc_info=True)
            # Try fallback if Letta failed and context was gathered
            if self._use_letta and context is not None:
                try:
                    logger.info("[Curator] Retrying with fallback adapter")
                    result = await self.fallback_adapter.run(session_id, context)
                    await self._apply_results(session_id, turn_number, result)
                except Exception as e2:
                    logger.error(f"[Curator] Fallback also failed: {e2}")

    async def _gather_context(self, session_id, turn_number):
        loop = asyncio.get_event_loop()

        graph_summary = await loop.run_in_executor(None, self.graph_db.get_graph_summary, session_id)

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

        contradictions = self.graph_db.detect_contradictions(session_id)

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
                await self.sqlite_db.queue_event(session_id, event)

        if result.get("compress_story") and result.get("compressed_summary"):
            await self._compress_story_md(session_id, turn_number, result["compressed_summary"])

    async def _compress_story_md(self, session_id, turn_number, compressed_summary):
        now = datetime.now().isoformat()
        frontmatter = f'---\nupdated_at: "{now}"\nturn: {turn_number}\nsession_id: {session_id}\nchanged: [compressed]\n---\n\n'
        content = frontmatter + compressed_summary
        await self.md_cache.write_cache_atomic(session_id, turn_number, {"story.md": content})
