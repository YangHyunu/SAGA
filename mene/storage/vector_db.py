import chromadb
from datetime import datetime


class VectorDB:
    def __init__(self, db_path: str = "db/chroma"):
        self.db_path = db_path
        self.client = None
        self.lorebook = None
        self.episodes = None

    def initialize(self):
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.lorebook = self.client.get_or_create_collection(
            name="lorebook", metadata={"hnsw:space": "cosine"}
        )
        self.episodes = self.client.get_or_create_collection(
            name="episodes", metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------ #
    # Lorebook operations
    # ------------------------------------------------------------------ #

    def add_lorebook_entry(self, entry_id: str, text: str, metadata: dict):
        """Add or update a lorebook entry. Upserts by entry_id."""
        self.lorebook.upsert(
            ids=[entry_id],
            documents=[text],
            metadatas=[metadata],
        )

    def search_lorebook(
        self, session_id: str, query: str, n_results: int = 10
    ) -> dict:
        """Semantic search over lorebook entries filtered by session_id."""
        try:
            result = self.lorebook.query(
                query_texts=[query],
                n_results=n_results,
                where={"session_id": session_id},
            )
        except Exception:
            # If collection is empty or no matching docs, return empty structure
            result = {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        return result

    def search_lorebook_by_ids(self, entry_ids: list[str]) -> dict:
        """Retrieve lorebook entries by their IDs."""
        if not entry_ids:
            return {"ids": [], "documents": [], "metadatas": []}
        try:
            result = self.lorebook.get(ids=entry_ids)
        except Exception:
            result = {"ids": [], "documents": [], "metadatas": []}
        return result

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
    ):
        """Add a turn episode summary to the episodes collection."""
        episode_id = f"{session_id}_turn_{turn}"
        metadata = {
            "session_id": session_id,
            "turn": turn,
            "location": location,
            "episode_type": episode_type,
            "created_at": datetime.utcnow().isoformat(),
        }
        self.episodes.upsert(
            ids=[episode_id],
            documents=[summary],
            metadatas=[metadata],
        )

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
        except Exception:
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
        except Exception:
            result = {"ids": [], "documents": [], "metadatas": []}
        return result

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def delete_session_data(self, session_id: str):
        """Delete all lorebook entries and episodes belonging to a session."""
        for collection in (self.lorebook, self.episodes):
            if collection is None:
                continue
            try:
                collection.delete(where={"session_id": session_id})
            except Exception:
                pass
