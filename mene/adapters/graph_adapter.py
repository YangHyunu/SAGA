"""Graph adapter pattern for Kuzu (primary) and NetworkX (fallback)."""
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class GraphAdapter(ABC):
    @abstractmethod
    def query_relationships(self, session_id: str, character: str, depth: int = 2) -> list[dict]: ...

    @abstractmethod
    def query_adjacent_locations(self, session_id: str, location: str) -> list[dict]: ...

    @abstractmethod
    def find_shortest_path(self, session_id: str, from_loc: str, to_loc: str) -> dict | None: ...

    @abstractmethod
    def detect_contradictions(self, session_id: str) -> list[dict]: ...


class KuzuAdapter(GraphAdapter):
    """Primary graph adapter using Kuzu embedded DB."""
    def __init__(self, graph_db):
        from mene.storage.graph_db import GraphDB
        self.graph_db: GraphDB = graph_db

    def query_relationships(self, session_id, character, depth=2):
        return self.graph_db.get_relationships(session_id, character)

    def query_adjacent_locations(self, session_id, location):
        return self.graph_db.get_adjacent_locations(session_id, location)

    def find_shortest_path(self, session_id, from_loc, to_loc):
        return self.graph_db.find_shortest_path(session_id, from_loc, to_loc)

    def detect_contradictions(self, session_id):
        return self.graph_db.detect_contradictions(session_id)


class NetworkXAdapter(GraphAdapter):
    """Fallback graph adapter using NetworkX (in-memory)."""
    def __init__(self):
        try:
            import networkx as nx
            self.nx = nx
            self.graphs = {}  # session_id -> nx.DiGraph
        except ImportError:
            logger.warning("NetworkX not installed. Fallback graph adapter unavailable.")
            self.nx = None
            self.graphs = {}

    def _get_graph(self, session_id):
        if session_id not in self.graphs:
            self.graphs[session_id] = self.nx.DiGraph() if self.nx else None
        return self.graphs[session_id]

    def query_relationships(self, session_id, character, depth=2):
        g = self._get_graph(session_id)
        if not g or character not in g:
            return []
        results = []
        for neighbor in g.neighbors(character):
            edge_data = g.edges[character, neighbor]
            results.append({"from": character, "to": neighbor, **edge_data})
        return results

    def query_adjacent_locations(self, session_id, location):
        g = self._get_graph(session_id)
        if not g or location not in g:
            return []
        return [{"name": n, **g.edges[location, n]} for n in g.neighbors(location) if g.nodes[n].get("type") == "location"]

    def find_shortest_path(self, session_id, from_loc, to_loc):
        g = self._get_graph(session_id)
        if not g:
            return None
        try:
            path = self.nx.shortest_path(g, from_loc, to_loc)
            return {"nodes": path, "distance": len(path) - 1, "description": " → ".join(path)}
        except (self.nx.NetworkXNoPath, self.nx.NodeNotFound):
            return None

    def detect_contradictions(self, session_id):
        # Simplified: no contradiction detection in NetworkX fallback
        return []
