import os
import yaml

STABLE_FILE = "stable_prefix.md"  # BP2 cache target
LIVE_FILE = "live_state.md"       # dynamic, no cache


class MdCache:
    def __init__(self, cache_dir: str = "cache/sessions"):
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def get_session_dir(self, session_id: str) -> str:
        return os.path.join(self.cache_dir, session_id)

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    async def read_stable(self, session_id: str) -> str:
        """Read stable_prefix.md. Returns empty string if not found."""
        filepath = os.path.join(self.get_session_dir(session_id), STABLE_FILE)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    async def read_live(self, session_id: str) -> str:
        """Read live_state.md. Returns empty string if not found."""
        filepath = os.path.join(self.get_session_dir(session_id), LIVE_FILE)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return ""

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    async def write_stable(self, session_id: str, content: str):
        """Atomic write of stable_prefix.md. Called rarely (Curator cycle).

        Increments version counter from existing frontmatter, then prepends
        fresh frontmatter before writing.
        """
        current_version = self.get_stable_version(session_id)
        new_version = current_version + 1

        frontmatter = f"---\nversion: {new_version}\nsession_id: {session_id}\n---\n"
        body = self._strip_frontmatter(content)
        full_content = frontmatter + body

        filepath = os.path.join(self.get_session_dir(session_id), STABLE_FILE)
        self._write_atomic(filepath, full_content)

    async def write_live(
        self,
        session_id: str,
        turn: int,
        state_block: dict,
        player_context: dict,
    ):
        """Build and write live_state.md from current game state.
        Called every turn by Post-Turn.

        state_block keys used: location, hp, max_hp, mood, nearby_npcs, recent_events
        player_context keys used: (merged on top of state_block if present)
        """
        merged = {**state_block, **player_context}

        location = merged.get("location", "알 수 없음")
        hp = merged.get("hp", "?")
        max_hp = merged.get("max_hp", "?")
        mood = merged.get("mood", "보통")
        nearby_npcs: list = merged.get("nearby_npcs", [])
        recent_events: list = merged.get("recent_events", [])

        lines = [f"---\nturn: {turn}\n---\n"]

        lines.append("## 현재 상태")
        lines.append(f"- 위치: {location}")
        lines.append(f"- HP: {hp}/{max_hp}")
        lines.append(f"- 기분: {mood}")
        lines.append("")

        lines.append("## 주변 인물")
        if nearby_npcs:
            for npc in nearby_npcs:
                if isinstance(npc, dict):
                    name = npc.get("name", "이름 없음")
                    rel_type = npc.get("rel_type", npc.get("relation", "알 수 없음"))
                    strength = npc.get("strength", npc.get("intimacy", "?"))
                    lines.append(f"- {name} (관계: {rel_type}, 친밀도: {strength})")
                else:
                    lines.append(f"- {npc}")
        else:
            lines.append("- 없음")
        lines.append("")

        lines.append("## 최근 이벤트")
        if recent_events:
            for event in recent_events:
                if isinstance(event, dict):
                    t = event.get("turn", "?")
                    desc = event.get("description", event.get("event", str(event)))
                    lines.append(f"- Turn {t}: {desc}")
                else:
                    lines.append(f"- {event}")
        else:
            lines.append("- 없음")
        lines.append("")

        full_content = "\n".join(lines)
        filepath = os.path.join(self.get_session_dir(session_id), LIVE_FILE)
        self._write_atomic(filepath, full_content)

    async def build_stable_content(
        self,
        characters: list,
        lore_entries: list,
        world_config: str = "",
    ) -> str:
        """Build stable_prefix.md content from DB data.
        Called by Curator or on session init.
        Does NOT write the file — caller should pass result to write_stable().
        """
        lines = []

        lines.append("## 세계관")
        lines.append(world_config.strip() if world_config else "")
        lines.append("")

        lines.append("## 등장인물")
        for char in characters:
            if isinstance(char, dict):
                name = char.get("name", "이름 없음")
                traits = char.get("traits", char.get("description", ""))
                lines.append(f"### {name}")
                if traits:
                    lines.append(f"- 특성: {traits}")
            else:
                lines.append(f"### {char}")
            lines.append("")

        lines.append("## 로어")
        for entry in lore_entries:
            if isinstance(entry, dict):
                name = entry.get("name", entry.get("title", "이름 없음"))
                content = entry.get("content", entry.get("description", ""))
                lines.append(f"### {name}")
                lines.append(content.strip() if content else "")
            else:
                lines.append(str(entry))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def get_stable_version(self, session_id: str) -> int:
        """Return version number from stable_prefix.md frontmatter. -1 if not found."""
        filepath = os.path.join(self.get_session_dir(session_id), STABLE_FILE)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            return -1
        fm = self._parse_frontmatter(content)
        try:
            return int(fm.get("version", -1))
        except (TypeError, ValueError):
            return -1

    def _write_atomic(self, filepath: str, content: str):
        """Write content atomically using tmp + replace pattern."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, filepath)

    # ------------------------------------------------------------------ #
    # Frontmatter (kept for compatibility and internal use)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_frontmatter(content: str) -> dict:
        """Parse YAML frontmatter from content. Returns empty dict if none."""
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return {}
        rest = stripped[3:]
        end_idx = rest.find("---")
        if end_idx == -1:
            return {}
        yaml_str = rest[:end_idx].strip()
        try:
            return yaml.safe_load(yaml_str) or {}
        except yaml.YAMLError:
            return {}

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Remove YAML frontmatter block (between --- markers) from the top of content."""
        if not content:
            return content
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return content
        rest = stripped[3:]
        end_idx = rest.find("---")
        if end_idx == -1:
            return content
        return rest[end_idx + 3:].lstrip("\n")
