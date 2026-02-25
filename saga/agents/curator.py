"""Letta Curator — N턴마다 비동기 실행. Memory Block 기반 서사 판단 연속성."""
import asyncio
import json
import logging
from datetime import datetime
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.graph_db import GraphDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.adapters.curator_adapter import LettaCuratorAdapter, DirectLLMCuratorAdapter

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

    async def run(self, session_id: str, turn_number: int):
        """Run curator. Called every N turns asynchronously."""
        context = None
        try:
            context = await self._gather_context(session_id, turn_number)

            adapter = self.letta_adapter if self._use_letta else self.fallback_adapter
            result = await adapter.run(session_id, context)

            await self._apply_results(session_id, turn_number, result)

            # Auto-generate lore for entities without lore
            await self._auto_generate_lore(session_id, turn_number)

            logger.info(f"[Curator] Curation complete for session {session_id} at turn {turn_number}")

        except Exception as e:
            logger.error(f"[Curator] Error: {e}", exc_info=True)
            # Try fallback if Letta failed and context was gathered
            if self._use_letta and context is not None:
                try:
                    logger.info("[Curator] Retrying with fallback adapter")
                    result = await self.fallback_adapter.run(session_id, context)
                    await self._apply_results(session_id, turn_number, result)
                    await self._auto_generate_lore(session_id, turn_number)
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

    async def _auto_generate_lore(self, session_id: str, turn_number: int):
        """Detect entities without lore and auto-generate lore entries."""
        loop = asyncio.get_event_loop()

        # Find entities that have no linked Lore nodes
        entities = await loop.run_in_executor(
            None, self.graph_db.get_entities_without_lore, session_id
        )
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
        loop = asyncio.get_event_loop()
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

        # Get graph relationships for this entity
        relationships = []
        if entity_type == "character":
            char_id = self.graph_db._node_id(session_id, entity_name)
            try:
                result = self.graph_db.conn.execute(
                    """
                    MATCH (c:Character {id: $cid})-[e:RELATES_TO]->(other:Character)
                    RETURN other.name AS target, e.rel_type AS rel_type, e.strength AS strength
                    """,
                    {"cid": char_id},
                )
                relationships = self.graph_db._result_to_list(result)
            except RuntimeError:
                pass

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

        # Call LLM
        response = await self.llm_client.call_llm(
            model=self.config.models.curator,
            messages=[
                {"role": "system", "content": "당신은 RP 세계관 구축 전문가입니다. 에피소드와 관계 정보를 분석하여 일관성 있는 로어를 생성합니다. 반드시 JSON으로만 응답하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=512,
        )

        # Parse response
        import re
        lore_data = None
        response = response.strip()
        # Try direct JSON
        try:
            lore_data = json.loads(response)
        except json.JSONDecodeError:
            pass
        # Try code block
        if not lore_data:
            match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', response)
            if match:
                try:
                    lore_data = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
        # Try brace matching
        if not lore_data:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                try:
                    lore_data = json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass

        if not lore_data or not lore_data.get("content"):
            logger.warning(f"[Curator] Could not parse lore for {entity_name}: {response[:200]}")
            return

        content = lore_data["content"]
        keywords = lore_data.get("keywords", entity_name)
        lore_type = lore_data.get("lore_type", entity_type)
        priority = lore_data.get("priority", 50)

        # Store in KuzuDB
        await loop.run_in_executor(
            None, self.graph_db.create_lore,
            session_id, f"lore_{entity_name}",
            lore_type, "core", keywords, content,
            priority, "{}", "{}",
            True, json.dumps(source_turns[:5]),
        )

        # Link to entity
        await loop.run_in_executor(
            None, self.graph_db.link_lore,
            session_id, entity_type, entity_name, f"lore_{entity_name}"
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

    async def _compress_story_md(self, session_id, turn_number, compressed_summary):
        now = datetime.now().isoformat()
        frontmatter = f'---\nupdated_at: "{now}"\nturn: {turn_number}\nsession_id: {session_id}\nchanged: [compressed]\n---\n\n'
        content = frontmatter + compressed_summary
        await self.md_cache.write_cache_atomic(session_id, turn_number, {"story.md": content})
