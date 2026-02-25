"""Sub-A: Context Builder — 매 턴 동기 실행, LLM 호출 없음, ~35ms 목표.

2-tier 조회: .md 캐시(~5ms) → DB 동적 보충(~30ms) → 조립
반환: {"md_prefix": str, "dynamic_suffix": str}
"""
import asyncio
import logging
from saga.storage.sqlite_db import SQLiteDB
from saga.storage.graph_db import GraphDB
from saga.storage.vector_db import VectorDB
from saga.storage.md_cache import MdCache
from saga.utils.tokens import count_tokens, truncate_to_budget

logger = logging.getLogger(__name__)

# State tracking instruction appended to every turn
STATE_BLOCK_INSTRUCTION = """[--- SAGA State Tracking ---]
응답 마지막에 아래 형식의 상태 블록을 추가해주세요:
```state
location: 현재 위치
location_moved: 위치 변경 여부 (true/false)
hp_change: HP 변화량 (없으면 0)
items_gained: [획득 아이템]
items_lost: [소실 아이템]
items_transferred: [{item: 아이템명, to: 받는캐릭터}]
npc_met: [새로 만난 NPC]
npc_separated: [현재 위치에서 분리된 NPC]
relationship_changes: [{from, to, type, delta}]
mood: 현재 분위기
event_trigger: 트리거된 이벤트 (없으면 null)
notes: 다음 턴에 알아야 할 사항
```
이 블록은 시스템 내부용이며 유저에게는 보이지 않습니다."""


class ContextBuilder:
    def __init__(self, sqlite_db: SQLiteDB, graph_db: GraphDB, vector_db: VectorDB, md_cache: MdCache, config):
        self.sqlite_db = sqlite_db
        self.graph_db = graph_db
        self.vector_db = vector_db
        self.md_cache = md_cache
        self.config = config

    async def build_context(self, session_id: str, messages: list[dict], token_budget: int) -> dict:
        """Main entry point. Returns {"md_prefix": str, "dynamic_suffix": str}"""
        last_user_msg = self._get_last_user_message(messages)

        # Phase 0: .md cache read + freshness check
        md_cache_data = await self.md_cache.read_cache(session_id)
        cache_turn = self.md_cache.get_cache_turn(md_cache_data)
        current_turn = await self.sqlite_db.get_turn_count(session_id)
        cache_is_fresh = self.md_cache.is_fresh(md_cache_data, current_turn)

        if not cache_is_fresh:
            md_cache_data = await self._build_md_from_db(session_id)

        md_prefix = self.md_cache.format_as_prefix(md_cache_data, self.config.token_budget.md_cache_max)

        # Phase 1: Dynamic DB queries (parallel)
        # Run Kuzu graph query and ChromaDB vector search concurrently
        player_ctx, lorebook_results = await self._parallel_db_queries(session_id, last_user_msg)

        # Phase 2: Graph × Vector hybrid
        lorebook_results = self._hybrid_rerank(session_id, lorebook_results)

        # Phase 3: Dynamic lorebook filtering
        active_lorebook = self._filter_lorebook(lorebook_results, player_ctx, token_budget)

        # Phase 4: Assemble dynamic suffix
        dynamic_suffix = await self._assemble_suffix(session_id, md_cache_data, player_ctx, lorebook_results, active_lorebook, md_prefix, token_budget)

        return {"md_prefix": md_prefix, "dynamic_suffix": dynamic_suffix}

    def _get_last_user_message(self, messages: list[dict]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    async def _parallel_db_queries(self, session_id, last_user_msg):
        """Run Kuzu and ChromaDB queries in parallel, including 3-stage episode retrieval."""
        loop = asyncio.get_event_loop()
        kuzu_future = loop.run_in_executor(None, self.graph_db.query_player_context, session_id)
        lorebook_future = loop.run_in_executor(None, self.vector_db.search_lorebook, session_id, last_user_msg, 10)
        # 3-stage episode retrieval: Recent + Important + Similar (all parallel)
        recent_future = loop.run_in_executor(None, self.vector_db.get_recent_episodes, session_id, 5)
        important_future = loop.run_in_executor(None, self.vector_db.search_important_episodes, session_id, 40, 5)
        similar_future = loop.run_in_executor(None, self.vector_db.search_episodes, session_id, last_user_msg, 5)

        player_ctx, lorebook_results, recent_eps, important_eps, similar_eps = await asyncio.gather(
            kuzu_future, lorebook_future, recent_future, important_future, similar_future
        )

        # Merge episodes: Recent → Important → Similar (deduplicated)
        merged_episodes = self._merge_episodes(recent_eps, important_eps, similar_eps)
        player_ctx["episodes"] = merged_episodes

        return player_ctx, lorebook_results

    def _merge_episodes(self, recent, important, similar) -> list[dict]:
        """Merge 3-stage episode results, deduplicating by ID. Priority: Recent > Important > Similar."""
        seen_ids = set()
        merged = []

        for source, label in [(recent, "recent"), (important, "important"), (similar, "similar")]:
            if not source:
                continue
            # Handle both nested [[...]] and flat [...] formats from ChromaDB
            raw_ids = source.get("ids", [])
            raw_docs = source.get("documents", [])
            raw_metas = source.get("metadatas", [])

            ids = raw_ids[0] if raw_ids and isinstance(raw_ids[0], list) else raw_ids
            docs = raw_docs[0] if raw_docs and isinstance(raw_docs[0], list) else raw_docs
            metas = raw_metas[0] if raw_metas and isinstance(raw_metas[0], list) else raw_metas

            if not ids:
                continue

            for i, ep_id in enumerate(ids):
                if ep_id in seen_ids:
                    continue
                seen_ids.add(ep_id)
                meta = metas[i] if i < len(metas) else {}
                merged.append({
                    "id": ep_id,
                    "text": docs[i] if i < len(docs) else "",
                    "turn": meta.get("turn", 0),
                    "importance": meta.get("importance", 10),
                    "source": label,
                    "episode_type": meta.get("episode_type", "episode"),
                    "location": meta.get("location", ""),
                    "npcs": meta.get("npcs", ""),
                })

        # Sort: important first, then by turn descending
        merged.sort(key=lambda x: (-x["importance"], -x["turn"]))
        return merged

    def _hybrid_rerank(self, session_id, lorebook_results):
        """Graph × Vector hybrid: expand Kuzu nodes found in lorebook results."""
        if not lorebook_results or not lorebook_results.get("metadatas"):
            return lorebook_results

        metadatas = lorebook_results.get("metadatas", [[]])[0]
        kuzu_node_ids = [m.get("kuzu_node_id") for m in metadatas if m.get("kuzu_node_id")]

        if not kuzu_node_ids:
            return lorebook_results

        expanded = self.graph_db.query_graph_expansion(session_id, kuzu_node_ids, max_hop=2)
        if expanded:
            expanded_names = [item.get("name", "") for item in expanded[:5]]
            expanded_query = " ".join(expanded_names)
            expanded_lorebook = self.vector_db.search_lorebook(session_id, expanded_query, n_results=5)
            lorebook_results = self._merge_and_dedupe(lorebook_results, expanded_lorebook)

        return lorebook_results

    def _merge_and_dedupe(self, results1, results2):
        """Merge two ChromaDB query results, deduplicating by ID."""
        if not results2 or not results2.get("ids"):
            return results1

        existing_ids = set(results1.get("ids", [[]])[0])
        merged = {
            "ids": [list(results1.get("ids", [[]])[0])],
            "documents": [list(results1.get("documents", [[]])[0])],
            "metadatas": [list(results1.get("metadatas", [[]])[0])],
            "distances": [list(results1.get("distances", [[]])[0])] if results1.get("distances") else [[]]
        }

        for i, doc_id in enumerate(results2.get("ids", [[]])[0]):
            if doc_id not in existing_ids:
                merged["ids"][0].append(doc_id)
                merged["documents"][0].append(results2["documents"][0][i])
                merged["metadatas"][0].append(results2["metadatas"][0][i])
                if results2.get("distances"):
                    merged["distances"][0].append(results2["distances"][0][i])

        return merged

    def _filter_lorebook(self, lorebook_results, player_ctx, token_budget):
        """Graph-enhanced lorebook filtering: entity mentions → graph KNOWS → lore retrieval."""
        entries = []

        # 1. Graph-based lore: find entities in player context → query linked Lore nodes
        graph_lore = self._get_graph_lore(player_ctx)
        for lore in graph_lore:
            content = lore.get("content", "")
            if content:
                entries.append(content)

        # 2. Vector-based lore (existing ChromaDB results)
        if lorebook_results and lorebook_results.get("documents"):
            docs = lorebook_results.get("documents", [[]])[0]
            for doc in docs:
                if doc and doc not in entries:  # dedupe
                    entries.append(doc)

        return entries

    def _get_graph_lore(self, player_ctx) -> list[dict]:
        """Retrieve lore entries linked to entities in the current context via graph edges."""
        if not player_ctx:
            return []

        lore_entries = []
        seen_names = set()

        # Lore for nearby NPCs
        for npc in player_ctx.get("nearby_npcs", []):
            name = npc.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                lore = self.graph_db.query_lore_for_entity(
                    player_ctx.get("session_id", ""), name
                )
                lore_entries.extend(lore)

        # Lore for current location
        location = player_ctx.get("location", "")
        if location and location not in seen_names:
            seen_names.add(location)
            lore = self.graph_db.query_lore_for_entity(
                player_ctx.get("session_id", ""), location
            )
            lore_entries.extend(lore)

        # Lore for items
        for item in player_ctx.get("items", []):
            name = item.get("name", "") if isinstance(item, dict) else str(item)
            if name and name not in seen_names:
                seen_names.add(name)
                lore = self.graph_db.query_lore_for_entity(
                    player_ctx.get("session_id", ""), name
                )
                lore_entries.extend(lore)

        # Sort by priority descending
        lore_entries.sort(key=lambda x: x.get("priority", 0), reverse=True)
        return lore_entries

    async def _assemble_suffix(self, session_id, md_cache_data, player_ctx, lorebook_results, active_lorebook, md_prefix, token_budget):
        """Assemble dynamic suffix from all sources."""
        dynamic_parts = []
        remaining = token_budget - count_tokens(md_prefix)

        # State delta (differences between .md and current DB)
        delta = self._format_state_delta(md_cache_data, player_ctx)
        if delta:
            delta_tokens = count_tokens(delta)
            if delta_tokens <= remaining:
                dynamic_parts.append(f"[최신 변경]\n{delta}")
                remaining -= delta_tokens

        # Episode memory (3-stage: recent + important + similar)
        episodes = player_ctx.get("episodes", [])
        if episodes:
            ep_lines = []
            for ep in episodes:
                marker = ""
                if ep["source"] == "important":
                    marker = "[!] "
                elif ep["source"] == "recent":
                    marker = "[R] "
                ep_lines.append(f"- {marker}Turn {ep['turn']}: {ep['text'][:200]}")
            ep_text = "\n".join(ep_lines[:10])
            ep_tokens = count_tokens(ep_text)
            ep_budget = min(remaining * 0.35, 2000)
            if ep_tokens <= ep_budget:
                dynamic_parts.append(f"[에피소드 기억]\n{ep_text}")
                remaining -= ep_tokens

        # Graph structural associations
        if lorebook_results and lorebook_results.get("metadatas"):
            graph_text = self._format_graph_context(lorebook_results)
            if graph_text:
                graph_tokens = count_tokens(graph_text)
                graph_budget = min(remaining * 0.2, self.config.token_budget.graph_context_max)
                if graph_tokens <= graph_budget:
                    dynamic_parts.append(f"[관련 연결]\n{graph_text}")
                    remaining -= graph_tokens

        # Active lorebook entries
        if active_lorebook:
            lorebook_text = "\n".join(f"- {entry}" for entry in active_lorebook[:5])
            lorebook_tokens = count_tokens(lorebook_text)
            if lorebook_tokens <= remaining:
                dynamic_parts.append(f"[관련 로어북]\n{lorebook_text}")
                remaining -= lorebook_tokens

        # Triggered events
        events = await self.sqlite_db.get_triggered_events(session_id)
        if events:
            event_text = "\n".join(f"- [{e.get('event_type')}] {e.get('payload', '')}" for e in events)
            dynamic_parts.append(f"[이벤트]\n{event_text}")

        # State tracking instruction (always added)
        dynamic_parts.append(STATE_BLOCK_INSTRUCTION)

        return "\n\n".join(dynamic_parts)

    def _format_state_delta(self, md_cache_data, player_ctx):
        """Format differences between cached .md and current DB state."""
        if not player_ctx:
            return ""
        parts = []
        if player_ctx.get("location"):
            parts.append(f"위치: {player_ctx['location']}")
        if player_ctx.get("hp") is not None:
            parts.append(f"HP: {player_ctx['hp']}/{player_ctx.get('max_hp', 100)}")
        if player_ctx.get("items"):
            items_str = ", ".join(player_ctx["items"])
            parts.append(f"인벤토리: {items_str}")
        if player_ctx.get("nearby_chars"):
            npcs = ", ".join(c.get("name", "") for c in player_ctx["nearby_chars"])
            parts.append(f"주변 NPC: {npcs}")
        return " | ".join(parts) if parts else ""

    def _format_graph_context(self, lorebook_results):
        """Format graph expansion results."""
        if not lorebook_results or not lorebook_results.get("metadatas"):
            return ""
        metadatas = lorebook_results.get("metadatas", [[]])[0]
        connections = []
        for m in metadatas:
            if m.get("kuzu_node_id"):
                connections.append(f"{m.get('name', 'unknown')}({m.get('type', '')})")
        return " → ".join(connections[:5]) if connections else ""

    async def _build_md_from_db(self, session_id):
        """Fallback: build md content directly from DB when cache is stale."""
        # Return dict with state/relations/story/lore keys
        player = self.graph_db.query_player_context(session_id)
        relationships = self.graph_db.get_relationships(session_id, player.get("name", "")) if player else []

        state_content = self._format_state_md(player, session_id)
        relations_content = self._format_relations_md(relationships, session_id)
        story_content = ""  # Will be built from turn_log
        lore_content = ""   # Will be built from ChromaDB

        return {
            "state": state_content,
            "relations": relations_content,
            "story": story_content,
            "lore": lore_content
        }

    def _format_state_md(self, player, session_id):
        if not player:
            return ""
        items_str = ", ".join(player.get("items", [])) or "없음"
        return f"""## 현재 상태
- **플레이어:** {player.get('name', '?')} | HP: {player.get('hp', '?')}/{player.get('max_hp', 100)} | 위치: {player.get('location', '?')}
- **인벤토리:** {items_str}
- **분위기:** {player.get('mood', 'neutral')}"""

    def _format_relations_md(self, relationships, session_id):
        if not relationships:
            return ""
        lines = ["## 관계 요약"]
        for r in relationships:
            lines.append(f"- {r.get('from', '?')} ↔ {r.get('to', '?')}: {r.get('type', '?')}(강도: {r.get('strength', '?')})")
        return "\n".join(lines)
