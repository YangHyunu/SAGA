import kuzu
import json
from datetime import datetime


class GraphDB:
    def __init__(self, db_path: str = "db/graph.kuzu"):
        self.db_path = db_path
        self.db = None
        self.conn = None

    def initialize(self):
        """Create Kuzu DB and all node/edge tables."""
        self.db = kuzu.Database(self.db_path)
        self.conn = kuzu.Connection(self.db)
        self._create_schema()

    def _create_schema(self):
        """Create all node and edge tables. Wrap each CREATE in try/except for idempotency."""

        node_tables = [
            """CREATE NODE TABLE Character (
                id STRING,
                session_id STRING,
                name STRING,
                is_player BOOLEAN,
                hp INT64,
                max_hp INT64,
                location STRING,
                mood STRING,
                status STRING,
                traits STRING,
                custom STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE Location (
                id STRING,
                session_id STRING,
                name STRING,
                loc_type STRING,
                description STRING,
                properties STRING,
                created_at STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE Item (
                id STRING,
                session_id STRING,
                name STRING,
                item_type STRING,
                description STRING,
                properties STRING,
                created_at STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE Event (
                id STRING,
                session_id STRING,
                name STRING,
                event_type STRING,
                description STRING,
                turn INT64,
                importance INT64,
                entities STRING,
                created_at STRING,
                PRIMARY KEY (id)
            )""",
            """CREATE NODE TABLE Lore (
                id STRING,
                session_id STRING,
                name STRING,
                lore_type STRING,
                layer STRING,
                keywords STRING,
                content STRING,
                priority INT64,
                conditions STRING,
                properties STRING,
                auto_generated BOOLEAN,
                source_turns STRING,
                created_at STRING,
                updated_at STRING,
                PRIMARY KEY (id)
            )""",
        ]

        edge_tables = [
            "CREATE REL TABLE RELATES_TO (FROM Character TO Character, rel_type STRING, strength INT64, updated_at STRING)",
            "CREATE REL TABLE LOCATED_AT (FROM Character TO Location, since_turn INT64)",
            "CREATE REL TABLE OWNS (FROM Character TO Item, quantity INT64, equipped BOOLEAN)",
            "CREATE REL TABLE ADJACENT (FROM Location TO Location, direction STRING, cost INT64, conditions STRING)",
            "CREATE REL TABLE INVOLVED_IN (FROM Character TO Event, role STRING)",
            "CREATE REL TABLE CAUSED (FROM Event TO Event, description STRING)",
            "CREATE REL TABLE KNOWS (FROM Character TO Lore, confidence INT64)",
            "CREATE REL TABLE RELATED (FROM Lore TO Lore, relation STRING)",
            "CREATE REL TABLE HAS_LORE (FROM Location TO Lore)",
            "CREATE REL TABLE ITEM_LORE (FROM Item TO Lore, note STRING)",
        ]

        for stmt in node_tables + edge_tables:
            try:
                self.conn.execute(stmt)
            except RuntimeError:
                pass

    def close(self):
        # Kuzu embedded DB does not need explicit close
        pass

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _node_id(session_id: str, name: str) -> str:
        return f"{session_id}_{name}"

    def _char_exists(self, session_id: str, name: str) -> bool:
        node_id = self._node_id(session_id, name)
        result = self.conn.execute(
            "MATCH (c:Character {id: $id}) RETURN c.id", {"id": node_id}
        )
        return result.has_next()

    def _loc_exists(self, session_id: str, name: str) -> bool:
        node_id = self._node_id(session_id, name)
        result = self.conn.execute(
            "MATCH (l:Location {id: $id}) RETURN l.id", {"id": node_id}
        )
        return result.has_next()

    def _item_exists(self, session_id: str, name: str) -> bool:
        node_id = self._node_id(session_id, name)
        result = self.conn.execute(
            "MATCH (i:Item {id: $id}) RETURN i.id", {"id": node_id}
        )
        return result.has_next()

    @staticmethod
    def _result_to_list(result) -> list[dict]:
        rows = []
        while result.has_next():
            rows.append(result.get_next())
        col_names = result.get_column_names()
        return [dict(zip(col_names, row)) for row in rows]

    # ------------------------------------------------------------------ #
    # Character operations
    # ------------------------------------------------------------------ #

    def create_character(
        self,
        session_id: str,
        name: str,
        is_player: bool = False,
        hp: int = 100,
        max_hp: int = 100,
        location: str = "unknown",
        mood: str = "neutral",
        traits: list = None,
        custom: dict = None,
    ):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        traits_json = json.dumps(traits or [], ensure_ascii=False)
        custom_json = json.dumps(custom or {}, ensure_ascii=False)
        if self._char_exists(session_id, name):
            self.conn.execute(
                """
                MATCH (c:Character {id: $id})
                SET c.hp = $hp, c.max_hp = $max_hp, c.location = $location,
                    c.mood = $mood, c.traits = $traits, c.custom = $custom,
                    c.updated_at = $now
                """,
                {
                    "id": node_id,
                    "hp": hp,
                    "max_hp": max_hp,
                    "location": location,
                    "mood": mood,
                    "traits": traits_json,
                    "custom": custom_json,
                    "now": now,
                },
            )
        else:
            self.conn.execute(
                """
                CREATE (c:Character {
                    id: $id, session_id: $session_id, name: $name,
                    is_player: $is_player, hp: $hp, max_hp: $max_hp,
                    location: $location, mood: $mood, status: 'alive',
                    traits: $traits, custom: $custom,
                    created_at: $now, updated_at: $now
                })
                """,
                {
                    "id": node_id,
                    "session_id": session_id,
                    "name": name,
                    "is_player": is_player,
                    "hp": hp,
                    "max_hp": max_hp,
                    "location": location,
                    "mood": mood,
                    "traits": traits_json,
                    "custom": custom_json,
                    "now": now,
                },
            )

    def get_player(self, session_id: str) -> dict | None:
        result = self.conn.execute(
            "MATCH (c:Character {session_id: $sid, is_player: true}) RETURN c.*",
            {"sid": session_id},
        )
        rows = self._result_to_list(result)
        return rows[0] if rows else None

    def get_character(self, session_id: str, name: str) -> dict | None:
        node_id = self._node_id(session_id, name)
        result = self.conn.execute(
            "MATCH (c:Character {id: $id}) RETURN c.*", {"id": node_id}
        )
        rows = self._result_to_list(result)
        return rows[0] if rows else None

    def update_character_hp(self, session_id: str, delta: int):
        """Apply delta to player HP. Clamp to [0, max_hp]."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            MATCH (c:Character {session_id: $sid, is_player: true})
            SET c.hp = CASE
                WHEN c.hp + $delta < 0 THEN 0
                WHEN c.hp + $delta > c.max_hp THEN c.max_hp
                ELSE c.hp + $delta
            END,
            c.updated_at = $now
            """,
            {"sid": session_id, "delta": delta, "now": now},
        )

    def update_character_location(
        self, session_id: str, character_name: str, new_location: str, turn: int
    ):
        node_id = self._node_id(session_id, character_name)
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            MATCH (c:Character {id: $id})
            SET c.location = $loc, c.updated_at = $now
            """,
            {"id": node_id, "loc": new_location, "now": now},
        )
        # Create LOCATED_AT edge if location node exists
        loc_id = self._node_id(session_id, new_location)
        if self._loc_exists(session_id, new_location):
            try:
                self.conn.execute(
                    """
                    MATCH (c:Character {id: $cid}), (l:Location {id: $lid})
                    CREATE (c)-[:LOCATED_AT {since_turn: $turn}]->(l)
                    """,
                    {"cid": node_id, "lid": loc_id, "turn": turn},
                )
            except RuntimeError:
                pass

    def update_character_mood(self, session_id: str, mood: str):
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            MATCH (c:Character {session_id: $sid, is_player: true})
            SET c.mood = $mood, c.updated_at = $now
            """,
            {"sid": session_id, "mood": mood, "now": now},
        )

    def update_character_status(self, session_id: str, name: str, status: str):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            MATCH (c:Character {id: $id})
            SET c.status = $status, c.updated_at = $now
            """,
            {"id": node_id, "status": status, "now": now},
        )

    # ------------------------------------------------------------------ #
    # Location operations
    # ------------------------------------------------------------------ #

    def create_location(
        self,
        session_id: str,
        name: str,
        loc_type: str = "",
        description: str = "",
        properties: dict = None,
    ):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        if self._loc_exists(session_id, name):
            self.conn.execute(
                """
                MATCH (l:Location {id: $id})
                SET l.loc_type = $loc_type, l.description = $description,
                    l.properties = $props
                """,
                {"id": node_id, "loc_type": loc_type, "description": description, "props": props_json},
            )
        else:
            self.conn.execute(
                """
                CREATE (l:Location {
                    id: $id, session_id: $session_id, name: $name,
                    loc_type: $loc_type, description: $description,
                    properties: $props, created_at: $now
                })
                """,
                {
                    "id": node_id,
                    "session_id": session_id,
                    "name": name,
                    "loc_type": loc_type,
                    "description": description,
                    "props": props_json,
                    "now": now,
                },
            )

    def get_adjacent_locations(self, session_id: str, location_name: str) -> list[dict]:
        loc_id = self._node_id(session_id, location_name)
        result = self.conn.execute(
            """
            MATCH (l:Location {id: $id})-[e:ADJACENT]->(adj:Location)
            RETURN adj.name AS name, adj.description AS description,
                   e.direction AS direction, e.cost AS cost, e.conditions AS conditions
            """,
            {"id": loc_id},
        )
        return self._result_to_list(result)

    def create_adjacency(
        self,
        session_id: str,
        from_loc: str,
        to_loc: str,
        direction: str = "",
        cost: int = 1,
        conditions: str = "",
    ):
        from_id = self._node_id(session_id, from_loc)
        to_id = self._node_id(session_id, to_loc)
        try:
            self.conn.execute(
                """
                MATCH (a:Location {id: $from_id}), (b:Location {id: $to_id})
                CREATE (a)-[:ADJACENT {direction: $dir, cost: $cost, conditions: $cond}]->(b)
                """,
                {
                    "from_id": from_id,
                    "to_id": to_id,
                    "dir": direction,
                    "cost": cost,
                    "cond": conditions,
                },
            )
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Item operations
    # ------------------------------------------------------------------ #

    def create_item(
        self,
        session_id: str,
        name: str,
        item_type: str = "",
        description: str = "",
        properties: dict = None,
    ):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        if self._item_exists(session_id, name):
            self.conn.execute(
                """
                MATCH (i:Item {id: $id})
                SET i.item_type = $item_type, i.description = $description,
                    i.properties = $props
                """,
                {"id": node_id, "item_type": item_type, "description": description, "props": props_json},
            )
        else:
            self.conn.execute(
                """
                CREATE (i:Item {
                    id: $id, session_id: $session_id, name: $name,
                    item_type: $item_type, description: $description,
                    properties: $props, created_at: $now
                })
                """,
                {
                    "id": node_id,
                    "session_id": session_id,
                    "name": name,
                    "item_type": item_type,
                    "description": description,
                    "props": props_json,
                    "now": now,
                },
            )

    def add_ownership(
        self,
        session_id: str,
        character_name: str,
        item_name: str,
        quantity: int = 1,
        equipped: bool = False,
    ):
        char_id = self._node_id(session_id, character_name)
        item_id = self._node_id(session_id, item_name)
        try:
            self.conn.execute(
                """
                MATCH (c:Character {id: $cid}), (i:Item {id: $iid})
                CREATE (c)-[:OWNS {quantity: $qty, equipped: $eq}]->(i)
                """,
                {"cid": char_id, "iid": item_id, "qty": quantity, "eq": equipped},
            )
        except RuntimeError:
            pass

    def remove_ownership(self, session_id: str, character_name: str, item_name: str):
        char_id = self._node_id(session_id, character_name)
        item_id = self._node_id(session_id, item_name)
        try:
            self.conn.execute(
                """
                MATCH (c:Character {id: $cid})-[e:OWNS]->(i:Item {id: $iid})
                DELETE e
                """,
                {"cid": char_id, "iid": item_id},
            )
        except RuntimeError:
            pass

    def transfer_item(
        self, session_id: str, item_name: str, from_char: str, to_char: str
    ):
        self.remove_ownership(session_id, from_char, item_name)
        self.add_ownership(session_id, to_char, item_name)

    # ------------------------------------------------------------------ #
    # Relationship operations
    # ------------------------------------------------------------------ #

    def create_relationship(
        self,
        session_id: str,
        from_name: str,
        to_name: str,
        rel_type: str = "met",
        strength: int = 30,
    ):
        from_id = self._node_id(session_id, from_name)
        to_id = self._node_id(session_id, to_name)
        now = datetime.utcnow().isoformat()
        try:
            self.conn.execute(
                """
                MATCH (a:Character {id: $fid}), (b:Character {id: $tid})
                CREATE (a)-[:RELATES_TO {rel_type: $rtype, strength: $str, updated_at: $now}]->(b)
                """,
                {"fid": from_id, "tid": to_id, "rtype": rel_type, "str": strength, "now": now},
            )
        except RuntimeError:
            pass

    def update_relationship(
        self,
        session_id: str,
        from_name: str,
        to_name: str,
        rel_type: str,
        delta: int = 0,
    ):
        from_id = self._node_id(session_id, from_name)
        to_id = self._node_id(session_id, to_name)
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            """
            MATCH (a:Character {id: $fid})-[e:RELATES_TO]->(b:Character {id: $tid})
            SET e.strength = e.strength + $delta, e.rel_type = $rtype, e.updated_at = $now
            """,
            {"fid": from_id, "tid": to_id, "rtype": rel_type, "delta": delta, "now": now},
        )

    def get_relationships(self, session_id: str, character_name: str) -> list[dict]:
        char_id = self._node_id(session_id, character_name)
        result = self.conn.execute(
            """
            MATCH (a:Character {id: $cid})-[e:RELATES_TO]->(b:Character)
            RETURN b.name AS target, e.rel_type AS rel_type, e.strength AS strength
            """,
            {"cid": char_id},
        )
        return self._result_to_list(result)

    # ------------------------------------------------------------------ #
    # Lore operations
    # ------------------------------------------------------------------ #

    def create_lore(
        self,
        session_id: str,
        name: str,
        lore_type: str = "",
        layer: str = "core",
        keywords: str = "",
        content: str = "",
        priority: int = 50,
        conditions: str = "{}",
        properties: str = "{}",
        auto_generated: bool = False,
        source_turns: str = "[]",
    ):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        result = self.conn.execute(
            "MATCH (l:Lore {id: $id}) RETURN l.id", {"id": node_id}
        )
        if result.has_next():
            # Update existing lore
            self.conn.execute(
                """
                MATCH (l:Lore {id: $id})
                SET l.content = $content, l.keywords = $keywords,
                    l.priority = $priority, l.properties = $properties,
                    l.source_turns = $source_turns, l.updated_at = $now
                """,
                {"id": node_id, "content": content, "keywords": keywords,
                 "priority": priority, "properties": properties,
                 "source_turns": source_turns, "now": now},
            )
            return
        self.conn.execute(
            """
            CREATE (l:Lore {
                id: $id, session_id: $session_id, name: $name,
                lore_type: $lore_type, layer: $layer,
                keywords: $keywords, content: $content,
                priority: $priority, conditions: $conditions,
                properties: $properties, auto_generated: $auto_generated,
                source_turns: $source_turns,
                created_at: $now, updated_at: $now
            })
            """,
            {
                "id": node_id, "session_id": session_id, "name": name,
                "lore_type": lore_type, "layer": layer,
                "keywords": keywords, "content": content,
                "priority": priority, "conditions": conditions,
                "properties": properties, "auto_generated": auto_generated,
                "source_turns": source_turns,
                "now": now,
            },
        )

    def get_entities_without_lore(self, session_id: str) -> list[dict]:
        """Find Characters, Locations, Items that have no KNOWS/HAS_LORE/ITEM_LORE edges."""
        entities = []
        # Characters without KNOWS edges
        try:
            result = self.conn.execute(
                """
                MATCH (c:Character {session_id: $sid})
                WHERE NOT EXISTS { MATCH (c)-[:KNOWS]->(:Lore) }
                AND c.is_player = false
                RETURN c.name AS name, 'character' AS entity_type,
                       c.location AS location, c.mood AS mood
                """,
                {"sid": session_id},
            )
            entities.extend(self._result_to_list(result))
        except RuntimeError:
            pass
        # Locations without HAS_LORE edges
        try:
            result = self.conn.execute(
                """
                MATCH (l:Location {session_id: $sid})
                WHERE NOT EXISTS { MATCH (l)-[:HAS_LORE]->(:Lore) }
                RETURN l.name AS name, 'location' AS entity_type,
                       l.description AS description
                """,
                {"sid": session_id},
            )
            entities.extend(self._result_to_list(result))
        except RuntimeError:
            pass
        # Items without ITEM_LORE edges
        try:
            result = self.conn.execute(
                """
                MATCH (i:Item {session_id: $sid})
                WHERE NOT EXISTS { MATCH (i)-[:ITEM_LORE]->(:Lore) }
                RETURN i.name AS name, 'item' AS entity_type,
                       i.description AS description
                """,
                {"sid": session_id},
            )
            entities.extend(self._result_to_list(result))
        except RuntimeError:
            pass
        return entities

    def query_lore_for_entity(self, session_id: str, entity_name: str) -> list[dict]:
        """Get all Lore nodes connected to a named entity via KNOWS/HAS_LORE/ITEM_LORE."""
        node_id = self._node_id(session_id, entity_name)
        lore_entries = []
        for query in [
            "MATCH (:Character {id: $nid})-[:KNOWS]->(l:Lore) RETURN l.name AS name, l.content AS content, l.lore_type AS lore_type, l.priority AS priority, l.keywords AS keywords",
            "MATCH (:Location {id: $nid})-[:HAS_LORE]->(l:Lore) RETURN l.name AS name, l.content AS content, l.lore_type AS lore_type, l.priority AS priority, l.keywords AS keywords",
            "MATCH (:Item {id: $nid})-[:ITEM_LORE]->(l:Lore) RETURN l.name AS name, l.content AS content, l.lore_type AS lore_type, l.priority AS priority, l.keywords AS keywords",
        ]:
            try:
                result = self.conn.execute(query, {"nid": node_id})
                lore_entries.extend(self._result_to_list(result))
            except RuntimeError:
                pass
        return lore_entries

    def get_all_lore(self, session_id: str) -> list[dict]:
        """Get all Lore entries for a session."""
        try:
            result = self.conn.execute(
                """
                MATCH (l:Lore {session_id: $sid})
                RETURN l.name AS name, l.content AS content, l.lore_type AS lore_type,
                       l.priority AS priority, l.keywords AS keywords,
                       l.auto_generated AS auto_generated, l.source_turns AS source_turns
                ORDER BY l.priority DESC
                """,
                {"sid": session_id},
            )
            return self._result_to_list(result)
        except RuntimeError:
            return []

    def link_lore(
        self, session_id: str, entity_type: str, entity_name: str, lore_name: str
    ):
        entity_id = self._node_id(session_id, entity_name)
        lore_id = self._node_id(session_id, lore_name)
        entity_type_upper = entity_type.capitalize()
        if entity_type_upper == "Location":
            rel_table = "HAS_LORE"
            match_clause = "MATCH (e:Location {id: $eid}), (l:Lore {id: $lid})"
        elif entity_type_upper == "Item":
            rel_table = "ITEM_LORE"
            match_clause = "MATCH (e:Item {id: $eid}), (l:Lore {id: $lid})"
        elif entity_type_upper == "Character":
            rel_table = "KNOWS"
            match_clause = "MATCH (e:Character {id: $eid}), (l:Lore {id: $lid})"
        else:
            return
        try:
            self.conn.execute(
                f"{match_clause} CREATE (e)-[:{rel_table}]->(l)",
                {"eid": entity_id, "lid": lore_id},
            )
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Event operations
    # ------------------------------------------------------------------ #

    def create_event(
        self,
        session_id: str,
        name: str,
        event_type: str = "",
        description: str = "",
        turn: int = 0,
        importance: int = 10,
        entities: list = None,
    ):
        node_id = self._node_id(session_id, name)
        now = datetime.utcnow().isoformat()
        entities_json = json.dumps(entities or [], ensure_ascii=False)
        result = self.conn.execute(
            "MATCH (e:Event {id: $id}) RETURN e.id", {"id": node_id}
        )
        if result.has_next():
            self.conn.execute(
                """
                MATCH (e:Event {id: $id})
                SET e.description = $desc, e.importance = $imp, e.entities = $ents
                """,
                {"id": node_id, "desc": description, "imp": importance, "ents": entities_json},
            )
        else:
            self.conn.execute(
                """
                CREATE (e:Event {
                    id: $id, session_id: $session_id, name: $name,
                    event_type: $event_type, description: $description,
                    turn: $turn, importance: $importance, entities: $entities,
                    created_at: $now
                })
                """,
                {
                    "id": node_id, "session_id": session_id, "name": name,
                    "event_type": event_type, "description": description,
                    "turn": turn, "importance": importance, "entities": entities_json,
                    "now": now,
                },
            )

    def get_important_events(self, session_id: str, min_importance: int = 40) -> list[dict]:
        """Get high-importance events for a session."""
        result = self.conn.execute(
            """
            MATCH (e:Event {session_id: $sid})
            WHERE e.importance >= $min_imp
            RETURN e.name AS name, e.event_type AS event_type,
                   e.description AS description, e.turn AS turn,
                   e.importance AS importance
            ORDER BY e.importance DESC, e.turn DESC
            """,
            {"sid": session_id, "min_imp": min_importance},
        )
        return self._result_to_list(result)

    def link_event_to_character(self, session_id: str, event_name: str, char_name: str, role: str = "participant"):
        event_id = self._node_id(session_id, event_name)
        char_id = self._node_id(session_id, char_name)
        try:
            self.conn.execute(
                """
                MATCH (c:Character {id: $cid}), (e:Event {id: $eid})
                CREATE (c)-[:INVOLVED_IN {role: $role}]->(e)
                """,
                {"cid": char_id, "eid": event_id, "role": role},
            )
        except RuntimeError:
            pass

    # ------------------------------------------------------------------ #
    # Query operations (for Sub-A)
    # ------------------------------------------------------------------ #

    def query_player_context(self, session_id: str) -> dict:
        """Get player character with location, items, and nearby NPCs."""
        player = self.get_player(session_id)
        if not player:
            return {}

        player_id = player.get("c.id") or player.get("id", "")

        # Items owned by player
        items_result = self.conn.execute(
            """
            MATCH (c:Character {id: $pid})-[e:OWNS]->(i:Item)
            RETURN i.name AS name, i.description AS description,
                   e.quantity AS quantity, e.equipped AS equipped
            """,
            {"pid": player_id},
        )
        items = self._result_to_list(items_result)

        # NPCs in same location
        location = player.get("c.location") or player.get("location", "unknown")
        npcs_result = self.conn.execute(
            """
            MATCH (c:Character {session_id: $sid, location: $loc, is_player: false})
            RETURN c.name AS name, c.mood AS mood, c.status AS status, c.hp AS hp
            """,
            {"sid": session_id, "loc": location},
        )
        npcs = self._result_to_list(npcs_result)

        # Adjacent locations
        adjacent = self.get_adjacent_locations(session_id, location)

        return {
            "session_id": session_id,
            "player": player,
            "location": location,
            "items": items,
            "nearby_npcs": npcs,
            "adjacent_locations": adjacent,
        }

    def query_graph_expansion(
        self, session_id: str, node_ids: list[str], max_hop: int = 2
    ) -> list[dict]:
        """N-hop graph expansion from given node IDs."""
        results = []
        for node_id in node_ids:
            # Try Character expansion
            try:
                result = self.conn.execute(
                    f"""
                    MATCH p = (start {{id: $nid}})-[*1..{max_hop}]-(end)
                    WHERE start.session_id = $sid
                    RETURN end.id AS id, end.name AS name, labels(end) AS label
                    """,
                    {"nid": node_id, "sid": session_id},
                )
                results.extend(self._result_to_list(result))
            except RuntimeError:
                pass
        return results

    def get_graph_summary(self, session_id: str) -> str:
        """Full graph summary for curator."""
        lines = []

        # Characters
        try:
            result = self.conn.execute(
                """
                MATCH (c:Character {session_id: $sid})
                RETURN c.name AS name, c.is_player AS is_player, c.hp AS hp,
                       c.max_hp AS max_hp, c.location AS location, c.mood AS mood,
                       c.status AS status
                """,
                {"sid": session_id},
            )
            chars = self._result_to_list(result)
            if chars:
                lines.append("## Characters")
                for c in chars:
                    tag = "[PLAYER]" if c.get("is_player") else "[NPC]"
                    lines.append(
                        f"- {tag} {c.get('name')} | HP:{c.get('hp')}/{c.get('max_hp')} "
                        f"| loc:{c.get('location')} | mood:{c.get('mood')} | status:{c.get('status')}"
                    )
        except RuntimeError:
            pass

        # Locations
        try:
            result = self.conn.execute(
                "MATCH (l:Location {session_id: $sid}) RETURN l.name AS name, l.description AS description",
                {"sid": session_id},
            )
            locs = self._result_to_list(result)
            if locs:
                lines.append("\n## Locations")
                for l in locs:
                    lines.append(f"- {l.get('name')}: {l.get('description','')}")
        except RuntimeError:
            pass

        # Relationships
        try:
            result = self.conn.execute(
                """
                MATCH (a:Character {session_id: $sid})-[e:RELATES_TO]->(b:Character)
                RETURN a.name AS from_name, b.name AS to_name,
                       e.rel_type AS rel_type, e.strength AS strength
                """,
                {"sid": session_id},
            )
            rels = self._result_to_list(result)
            if rels:
                lines.append("\n## Relationships")
                for r in rels:
                    lines.append(
                        f"- {r.get('from_name')} --[{r.get('rel_type')}({r.get('strength')}])--> {r.get('to_name')}"
                    )
        except RuntimeError:
            pass

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Contradiction detection
    # ------------------------------------------------------------------ #

    def detect_contradictions(self, session_id: str) -> list[dict]:
        """Rule-based contradiction detection."""
        contradictions = []

        # Dead characters that still have a location set (not 'dead'/'unknown')
        try:
            result = self.conn.execute(
                """
                MATCH (c:Character {session_id: $sid, status: 'dead'})
                WHERE c.location <> 'dead' AND c.location <> 'unknown'
                RETURN c.name AS name, c.location AS location
                """,
                {"sid": session_id},
            )
            for row in self._result_to_list(result):
                contradictions.append({
                    "type": "dead_char_with_location",
                    "character": row.get("name"),
                    "location": row.get("location"),
                    "description": f"Dead character '{row.get('name')}' still has active location '{row.get('location')}'",
                })
        except RuntimeError:
            pass

        # Characters with HP <= 0 but status != 'dead'
        try:
            result = self.conn.execute(
                """
                MATCH (c:Character {session_id: $sid})
                WHERE c.hp <= 0 AND c.status <> 'dead'
                RETURN c.name AS name, c.hp AS hp, c.status AS status
                """,
                {"sid": session_id},
            )
            for row in self._result_to_list(result):
                contradictions.append({
                    "type": "zero_hp_alive",
                    "character": row.get("name"),
                    "hp": row.get("hp"),
                    "status": row.get("status"),
                    "description": f"Character '{row.get('name')}' has HP={row.get('hp')} but status='{row.get('status')}'",
                })
        except RuntimeError:
            pass

        return contradictions

    # ------------------------------------------------------------------ #
    # Path finding
    # ------------------------------------------------------------------ #

    def find_shortest_path(
        self, session_id: str, from_location: str, to_location: str
    ) -> dict | None:
        from_id = self._node_id(session_id, from_location)
        to_id = self._node_id(session_id, to_location)
        try:
            result = self.conn.execute(
                """
                MATCH p = shortestPath(
                    (a:Location {id: $fid})-[:ADJACENT*]->(b:Location {id: $tid})
                )
                RETURN nodes(p) AS path_nodes, length(p) AS path_length
                """,
                {"fid": from_id, "tid": to_id},
            )
            rows = self._result_to_list(result)
            if rows:
                return rows[0]
        except RuntimeError:
            pass
        return None

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    def delete_session_data(self, session_id: str):
        """Delete all nodes and edges belonging to a session."""
        node_types = ["Character", "Location", "Item", "Event", "Lore"]
        for node_type in node_types:
            try:
                self.conn.execute(
                    f"MATCH (n:{node_type} {{session_id: $sid}}) DETACH DELETE n",
                    {"sid": session_id},
                )
            except RuntimeError:
                pass
