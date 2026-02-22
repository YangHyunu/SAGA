import os
import re
import yaml
from datetime import datetime

MD_FILES = ["state.md", "relations.md", "story.md", "lore.md"]
MD_KEYS = ["state", "relations", "story", "lore"]


class MdCache:
    def __init__(self, cache_dir: str = "cache/sessions"):
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------ #
    # Path helpers
    # ------------------------------------------------------------------ #

    def get_session_dir(self, session_id: str) -> str:
        return os.path.join(self.cache_dir, session_id)

    # ------------------------------------------------------------------ #
    # Read / write
    # ------------------------------------------------------------------ #

    async def read_cache(self, session_id: str) -> dict:
        """Read all 4 .md files and return dict with keys state/relations/story/lore.
        Missing files return empty string for that key."""
        session_dir = self.get_session_dir(session_id)
        cache: dict[str, str] = {}
        for filename, key in zip(MD_FILES, MD_KEYS):
            filepath = os.path.join(session_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    cache[key] = f.read()
            except FileNotFoundError:
                cache[key] = ""
        return cache

    async def write_cache_atomic(
        self,
        session_id: str,
        turn: int,
        contents: dict,
        changed: list[str] = None,
    ):
        """Atomic write: write to .tmp then os.replace for each file.
        contents dict has filename keys like 'state.md'."""
        session_dir = self.get_session_dir(session_id)
        os.makedirs(session_dir, exist_ok=True)

        frontmatter = self._make_frontmatter(turn, session_id, changed)

        for filename in MD_FILES:
            if filename not in contents:
                continue
            filepath = os.path.join(session_dir, filename)
            tmp_path = filepath + ".tmp"
            # Strip existing frontmatter from supplied content, then prepend fresh one
            body = self._strip_frontmatter(contents[filename])
            full_content = frontmatter + body
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(full_content)
            os.replace(tmp_path, filepath)

    # ------------------------------------------------------------------ #
    # Frontmatter helpers
    # ------------------------------------------------------------------ #

    def _make_frontmatter(
        self, turn: int, session_id: str, changed: list[str] = None
    ) -> str:
        data: dict = {
            "turn": turn,
            "session_id": session_id,
            "updated_at": datetime.utcnow().isoformat(),
        }
        if changed:
            data["changed"] = changed
        yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False).strip()
        return f"---\n{yaml_str}\n---\n"

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """Remove YAML frontmatter block (between --- markers) from the top of content."""
        if not content:
            return content
        stripped = content.lstrip()
        if not stripped.startswith("---"):
            return content
        # Find closing ---
        rest = stripped[3:]
        end_idx = rest.find("---")
        if end_idx == -1:
            return content
        return rest[end_idx + 3:].lstrip("\n")

    # ------------------------------------------------------------------ #
    # Cache inspection
    # ------------------------------------------------------------------ #

    def get_cache_turn(self, md_cache: dict) -> int:
        """Extract turn from state.md YAML frontmatter."""
        state_content = md_cache.get("state", "")
        if not state_content:
            return -1
        try:
            fm = self._parse_frontmatter(state_content)
            return int(fm.get("turn", -1))
        except Exception:
            return -1

    def is_fresh(self, md_cache: dict, current_turn: int) -> bool:
        """Check if cache is fresh (turn diff <= 1)."""
        cache_turn = self.get_cache_turn(md_cache)
        if cache_turn < 0:
            return False
        return abs(current_turn - cache_turn) <= 1

    def extract_keywords(self, md_cache: dict) -> list[str]:
        """Extract keywords from story.md '핵심 키워드' section."""
        story_content = md_cache.get("story", "")
        if not story_content:
            return []
        body = self._strip_frontmatter(story_content)
        # Look for section header containing '핵심 키워드'
        pattern = re.compile(
            r"#{1,4}\s*핵심\s*키워드[^\n]*\n(.*?)(?=\n#{1,4}\s|\Z)",
            re.DOTALL,
        )
        match = pattern.search(body)
        if not match:
            return []
        section_text = match.group(1)
        keywords = []
        for line in section_text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Support bullet lists (-, *, •) or comma-separated
            line = re.sub(r"^[-*•]\s*", "", line)
            if "," in line:
                keywords.extend(kw.strip() for kw in line.split(",") if kw.strip())
            elif line:
                keywords.append(line)
        return keywords

    def format_as_prefix(self, md_cache: dict, max_tokens: int) -> str:
        """Format 4 .md files as prompt caching prefix.
        Strips frontmatter from each and joins with double newlines.
        Truncates to roughly max_tokens characters (1 token ~= 4 chars)."""
        parts = []
        for key in MD_KEYS:
            content = md_cache.get(key, "")
            if content:
                body = self._strip_frontmatter(content)
                if body.strip():
                    parts.append(body.strip())
        combined = "\n\n".join(parts)
        # Rough token budget: 1 token ~= 4 chars
        char_limit = max_tokens * 4
        if len(combined) > char_limit:
            combined = combined[:char_limit]
        return combined

    # ------------------------------------------------------------------ #
    # Internal
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
