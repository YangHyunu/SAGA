"""Sub-A: Context Builder — 매 턴 동기 실행, LLM 호출 없음, ~35ms 목표.

2-tier 조회: .md 캐시(~5ms) → DB 동적 보충(~30ms) → 조립
반환: {"md_prefix": str, "dynamic_suffix": str}
"""
import asyncio
import logging
from langsmith import traceable
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.utils.tokens import count_tokens

logger = logging.getLogger(__name__)



class ContextBuilder:
    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB, md_cache: MdCache, config):
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.config = config

    @traceable(name="pipeline.sub_a")
    async def build_context(self, session_id: str, messages: list[dict], token_budget: int) -> dict:
        """Main entry point. Returns {"md_prefix": str, "dynamic_suffix": str}"""
        last_user_msg = self._get_last_user_message(messages)

        # Read stable prefix (cached, rarely changes)
        stable_prefix = await self.md_cache.read_stable(session_id)

        # Read live state (dynamic, every turn)
        live_state = await self.md_cache.read_live(session_id)

        # Query ChromaDB for relevant episodes (RRF-ranked)
        episodes = await self._get_relevant_episodes_rrf(session_id, last_user_msg)

        # Query lore (SQLite-based now)
        active_lore = await self._get_active_lore(session_id, last_user_msg)

        # Assemble dynamic suffix
        dynamic_suffix = self._assemble_dynamic(live_state, episodes, active_lore, token_budget, stable_prefix)

        return {"md_prefix": stable_prefix, "dynamic_suffix": dynamic_suffix}

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def _normalize_chroma_result(self, result: dict) -> list[dict]:
        """Flatten a ChromaDB query/get result into a list of episode dicts."""
        if not result:
            return []
        raw_ids = result.get("ids", [])
        raw_docs = result.get("documents", [])
        raw_metas = result.get("metadatas", [])

        # ChromaDB query returns nested [[...]], get returns flat [...]
        ids = raw_ids[0] if raw_ids and isinstance(raw_ids[0], list) else raw_ids
        docs = raw_docs[0] if raw_docs and isinstance(raw_docs[0], list) else raw_docs
        metas = raw_metas[0] if raw_metas and isinstance(raw_metas[0], list) else raw_metas

        episodes = []
        for i, ep_id in enumerate(ids):
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            episodes.append({
                "id": ep_id,
                "text": doc,
                "summary": doc,
                "turn": meta.get("turn", 0),
                "importance": meta.get("importance", 10),
                "episode_type": meta.get("episode_type", "episode"),
                "location": meta.get("location", ""),
                "npcs": meta.get("npcs", ""),
            })
        return episodes

    async def _get_relevant_episodes_rrf(self, session_id: str, query: str, top_n: int = 10) -> list[dict]:
        """Select episodes using Reciprocal Rank Fusion across 3 sources.

        Sources:
          recent   — get_recent_episodes (turn-ordered, no semantic filter)
          important — search_important_episodes (importance >= 40)
          similar   — search_episodes (semantic similarity to current query)

        RRF formula: score += weight / (k + rank + 1)
        """
        k = 60  # standard RRF constant

        # P2: VectorDB 호출 병렬화 (return_exceptions=True for partial failure resilience)
        results = await asyncio.gather(
            asyncio.to_thread(self.vector_db.get_recent_episodes, session_id, 10),
            asyncio.to_thread(self.vector_db.search_important_episodes, session_id, 40, 10),
            asyncio.to_thread(self.vector_db.search_episodes, session_id, query, 15),
            return_exceptions=True,
        )
        recent_raw, important_raw, similar_raw = [
            r if not isinstance(r, Exception) else {} for r in results
        ]
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.warning(f"[ContextBuilder] VectorDB query {i} failed: {r}")

        sources = [
            ("recent", 1.2, self._normalize_chroma_result(recent_raw)),
            ("important", 1.0, self._normalize_chroma_result(important_raw)),
            ("similar", 0.8, self._normalize_chroma_result(similar_raw)),
        ]

        scores: dict[str, float] = {}
        episode_cache: dict[str, dict] = {}

        for source_name, weight, episodes in sources:
            for rank, ep in enumerate(episodes):
                eid = ep["id"]
                scores[eid] = scores.get(eid, 0.0) + weight / (k + rank + 1)
                if eid not in episode_cache:
                    episode_cache[eid] = {**ep, "source": source_name}

        ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_n]
        selected = [episode_cache[eid] for eid in ranked_ids]

        logger.debug(
            "RRF episode selection: %d candidates → %d ranked (top_n=%d, session=%s)",
            len(scores),
            len(selected),
            top_n,
            session_id,
        )
        return selected

    async def _get_active_lore(self, session_id: str, query: str) -> list[str]:
        """Get relevant lore from SQLite + vector search."""
        # SQLite lore
        all_lore = await self.sqlite_db.get_all_lore(session_id)

        # Vector search for relevant lorebook entries
        vector_lore = self.vector_db.search_lorebook(session_id, query, n_results=5)

        # Combine and deduplicate
        lore_texts = []
        seen = set()
        for lore in all_lore:
            if lore['name'] not in seen:
                seen.add(lore['name'])
                lore_texts.append(f"### {lore['name']}\n{lore['content']}")

        # Add vector results not already in SQLite lore
        docs = vector_lore.get("documents", [[]])[0] if vector_lore.get("documents") else []
        metas = vector_lore.get("metadatas", [[]])[0] if vector_lore.get("metadatas") else []
        for doc, meta in zip(docs, metas):
            name = meta.get("name", "") if meta else ""
            if name and name not in seen:
                seen.add(name)
                lore_texts.append(f"### {name}\n{doc}")

        return lore_texts

    def _assemble_dynamic(self, live_state: str, episodes: list[dict], active_lore: list[str], token_budget: int, stable_prefix: str) -> str:
        """Assemble dynamic suffix from live state + episodes + lore."""
        parts = []
        remaining = token_budget - count_tokens(stable_prefix)

        # Live state first (most important)
        if live_state:
            live_tokens = count_tokens(live_state)
            if live_tokens <= remaining:
                parts.append(live_state)
                remaining -= live_tokens

        # P1: Episodes — 개별 삽입 (예산 초과 시 가능한 만큼만)
        if episodes:
            episode_header = "[에피소드 기억]\n"
            header_tokens = count_tokens(episode_header)
            if header_tokens <= remaining:
                episode_lines = []
                remaining -= header_tokens
                for ep in episodes[:10]:
                    marker = "[!]" if ep.get("importance", 0) >= 50 else "[R]"
                    line = f"{marker} Turn {ep.get('turn', '?')}: {ep.get('summary', '')[:500]}\n"
                    line_tokens = count_tokens(line)
                    if line_tokens <= remaining:
                        episode_lines.append(line)
                        remaining -= line_tokens
                    else:
                        break
                if episode_lines:
                    parts.append(episode_header + "".join(episode_lines))
                else:
                    remaining += header_tokens  # 헤더만 넣고 내용 없으면 되돌림

        # P1: Active lore — 개별 삽입 (per-entry 400자 cap)
        if active_lore:
            lore_header = "[활성 로어]\n"
            header_tokens = count_tokens(lore_header)
            if header_tokens <= remaining:
                lore_lines = []
                remaining -= header_tokens
                for entry in active_lore[:5]:
                    entry = entry[:800]  # 단일 로어 엔트리가 전체 예산 잠식 방지
                    entry_tokens = count_tokens(entry + "\n")
                    if entry_tokens <= remaining:
                        lore_lines.append(entry)
                        remaining -= entry_tokens
                    else:
                        break
                if lore_lines:
                    parts.append(lore_header + "\n".join(lore_lines))
                else:
                    remaining += header_tokens

        return "\n\n".join(parts)
