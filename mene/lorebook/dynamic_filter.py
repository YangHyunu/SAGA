"""다이나믹 로어북 필터 — Graph × Vector 하이브리드 필터링.

후보 풀(벡터 검색 결과) × 그래프 게이트(Kuzu) × 규칙 게이트 → 활성화 대상 결정
"""
import logging
from mene.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

LAYER_PRIORITY = {"A1": 0, "A2": 1, "A3": 3, "A4": 4}


class DynamicLorebookFilter:
    def __init__(self, config):
        self.config = config
        self._mention_cache = {}  # name -> last_turn

    def update_mention(self, name: str, turn: int):
        self._mention_cache[name] = turn

    def get_turns_since_mention(self, name: str, current_turn: int) -> int:
        last_turn = self._mention_cache.get(name, 0)
        return current_turn - last_turn

    def filter(self, candidates: dict, gate: dict, token_budget: int, current_turn: int = 0) -> list[dict]:
        """
        Filter lorebook candidates using graph gate and rules.

        candidates: ChromaDB query result {"documents": [[...]], "metadatas": [[...]], "ids": [[...]]}
        gate: player context from Kuzu {"location": str, "nearby_chars": [...], "relationships": [...]}
        token_budget: max tokens for lorebook
        current_turn: current turn number for decay
        """
        if not candidates or not candidates.get("documents") or not candidates["documents"][0]:
            return []

        docs = candidates["documents"][0]
        metas = candidates["metadatas"][0]
        ids = candidates["ids"][0]

        entries = []
        for i, (doc, meta, entry_id) in enumerate(zip(docs, metas, ids)):
            entries.append({
                "id": entry_id,
                "document": doc,
                "metadata": meta,
                "score": 0.0
            })

        activated = []
        player_location = gate.get("location", "")
        active_chars = [c.get("name", "") for c in gate.get("nearby_chars", [])]
        related_names = set()
        for rel in gate.get("relationships", []):
            related_names.add(rel.get("to", ""))
            related_names.add(rel.get("from", ""))

        for entry in entries:
            meta = entry["metadata"]
            tags = meta.get("tags", "")
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",")]
            entry_type = meta.get("type", "")
            entry_name = meta.get("name", "")
            layer = meta.get("layer", "A1")

            # Location gate
            if player_location and player_location in tags:
                entry["score"] += 3.0

            # Character gate
            if entry_type == "character" and entry_name in active_chars:
                entry["score"] += 2.0

            # Relationship propagation
            if entry_name in related_names:
                entry["score"] += 1.0

            # Layer decay
            turns_since = self.get_turns_since_mention(entry_name, current_turn)
            decay_config = self.config.dynamic_lorebook

            if layer == "A4" and turns_since > 3:
                continue
            if layer == "A3" and turns_since > 7:
                continue

            # Layer priority boost
            entry["score"] += (4 - LAYER_PRIORITY.get(layer, 4)) * 0.5

            activated.append(entry)

        # Deduplicate
        seen = set()
        unique = []
        for entry in activated:
            if entry["id"] not in seen:
                seen.add(entry["id"])
                unique.append(entry)

        # Sort by score (descending) then layer priority
        unique.sort(key=lambda x: (-x["score"], LAYER_PRIORITY.get(x["metadata"].get("layer", "A1"), 99)))

        # Token budget cutoff
        result = []
        total_tokens = 0
        for entry in unique:
            entry_tokens = count_tokens(entry["document"])
            if total_tokens + entry_tokens > token_budget:
                break
            result.append(entry)
            total_tokens += entry_tokens

        return result
