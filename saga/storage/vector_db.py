import chromadb
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class VectorDB:
    def __init__(self, db_path: str = "db/chroma"):
        self.db_path = db_path
        self.client = None
        self.episodes = None

    def initialize(self):
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.episodes = self.client.get_or_create_collection(
            name="episodes", metadata={"hnsw:space": "cosine"}
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
            "created_at": datetime.utcnow().isoformat(),
        }
        self.episodes.upsert(
            ids=[episode_id],
            documents=[summary],
            metadatas=[metadata],
        )
        logger.debug(f"[VectorDB] upsert episode: id={episode_id} importance={importance} summary_len={len(summary)}")

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
        return result

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
        return result

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
        return result

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
