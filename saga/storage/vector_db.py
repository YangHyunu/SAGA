import chromadb
import logging
from datetime import datetime

from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(
        self,
        db_path: str = "db/chroma",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.db_path = db_path
        self.openai_api_key = openai_api_key
        self.embedding_model = embedding_model
        self.client = None
        self.episodes = None

    def initialize(self):
        self.client = chromadb.PersistentClient(path=self.db_path)

        if not self.openai_api_key:
            raise RuntimeError(
                "VectorDB requires an OpenAI API key for embeddings. "
                "Set api_keys.openai in config.yaml."
            )

        embed_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.openai_api_key,
            model_name=self.embedding_model,
        )

        self.episodes = self.client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
            embedding_function=embed_fn,
        )
        logger.info(
            f"[VectorDB] initialized with OpenAI embedding model={self.embedding_model}"
        )

    # ------------------------------------------------------------------ #
    # Episode operations
    # ------------------------------------------------------------------ #

    def add_episode(
        self,
        session_id: str,
        turn: int,
        summary: str,
        location: str = "unknown",
        episode_type: str = "episode",
        importance: int = 10,
        entities: list[str] = None,
        npcs: list[str] = None,
        status: str = "provisional",
    ):
        """Add a turn episode summary to the episodes collection."""
        episode_id = f"{session_id}_turn_{turn}"
        metadata = {
            "session_id": session_id,
            "turn": turn,
            "location": location,
            "episode_type": episode_type,
            "importance": importance,
            "entities": ",".join(entities) if entities else "",
            "npcs": ",".join(npcs) if npcs else "",
            "status": status,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.episodes.upsert(
            ids=[episode_id],
            documents=[summary],
            metadatas=[metadata],
        )
        logger.debug(f"[VectorDB] upsert episode: id={episode_id} importance={importance} summary_len={len(summary)}")

    def set_episode_status(self, session_id: str, turn: int, status: str):
        """Update an episode's lifecycle status (provisional/confirmed/superseded/quarantined)."""
        episode_id = f"{session_id}_turn_{turn}"
        try:
            existing = self.episodes.get(ids=[episode_id])
            if not existing.get("ids"):
                return
            metadata = existing["metadatas"][0] or {}
            metadata["status"] = status
            self.episodes.update(ids=[episode_id], metadatas=[metadata])
            logger.debug(f"[VectorDB] set_episode_status: id={episode_id} status={status}")
        except Exception as e:
            logger.warning(f"[VectorDB] set_episode_status failed: {e}")

    @staticmethod
    def _drop_inactive(result: dict) -> dict:
        """Filter out superseded/quarantined episodes from a query/get result.

        Done in Python (not a Chroma where-clause) so legacy episodes without
        a status field keep matching.
        """
        ids = result.get("ids") or []
        if not ids:
            return result
        nested = bool(ids and isinstance(ids[0], list))  # query returns [[...]]
        keys = ("ids", "documents", "metadatas", "distances")

        def _filter(ids_l, docs_l, metas_l, dists_l):
            keep = [
                i for i, m in enumerate(metas_l)
                if (m or {}).get("status") not in ("superseded", "quarantined")
            ]
            return (
                [ids_l[i] for i in keep],
                [docs_l[i] for i in keep] if docs_l else docs_l,
                [metas_l[i] for i in keep] if metas_l else metas_l,
                [dists_l[i] for i in keep] if dists_l else dists_l,
            )

        if nested:
            for batch in range(len(ids)):
                filtered = _filter(
                    result["ids"][batch],
                    (result.get("documents") or [[]])[batch],
                    (result.get("metadatas") or [[]])[batch],
                    (result.get("distances") or [[]])[batch] if result.get("distances") else [],
                )
                for key, value in zip(keys, filtered):
                    if result.get(key):
                        result[key][batch] = value
        else:
            filtered = _filter(
                result["ids"],
                result.get("documents") or [],
                result.get("metadatas") or [],
                result.get("distances") or [],
            )
            for key, value in zip(keys, filtered):
                if result.get(key):
                    result[key] = value
        return result

    def search_episodes(
        self, session_id: str, query: str, n_results: int = 20
    ) -> dict:
        """Semantic search over episode summaries filtered by session_id."""
        try:
            result = self.episodes.query(
                query_texts=[query],
                n_results=n_results,
                where={"session_id": session_id},
            )
        except Exception as e:
            logger.warning(f"[VectorDB] search_episodes failed: {e}")
            result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self._drop_inactive(result)

    def search_important_episodes(
        self, session_id: str, min_importance: int = 40, n_results: int = 10
    ) -> dict:
        """Retrieve high-importance episodes (combat, relationship changes, events)."""
        try:
            result = self.episodes.query(
                query_texts=["중요한 사건"],
                n_results=n_results,
                where={
                    "$and": [
                        {"session_id": session_id},
                        {"importance": {"$gte": min_importance}},
                    ]
                },
            )
        except Exception as e:
            logger.warning(f"[VectorDB] search_important_episodes failed: {e}")
            result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return self._drop_inactive(result)

    def get_recent_episodes(self, session_id: str, n_results: int = 20) -> dict:
        """Get the most recent episode entries for a session, ordered by turn desc."""
        try:
            result = self.episodes.get(
                where={"session_id": session_id},
                limit=n_results,
            )
            # Sort by turn descending
            if result.get("metadatas"):
                combined = sorted(
                    zip(
                        result["ids"],
                        result["documents"],
                        result["metadatas"],
                    ),
                    key=lambda x: x[2].get("turn", 0),
                    reverse=True,
                )
                if combined:
                    ids, docs, metas = zip(*combined)
                    result["ids"] = list(ids)
                    result["documents"] = list(docs)
                    result["metadatas"] = list(metas)
        except Exception as e:
            logger.warning(f"[VectorDB] get_recent_episodes failed: {e}")
            result = {"ids": [], "documents": [], "metadatas": []}
        return self._drop_inactive(result)

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def delete_session_data(self, session_id: str):
        """Delete all episodes belonging to a session."""
        if self.episodes is None:
            return
        try:
            self.episodes.delete(where={"session_id": session_id})
        except Exception as e:
            logger.warning(f"[VectorDB] delete_session_data failed: {e}")
