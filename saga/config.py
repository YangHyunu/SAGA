"""
saga/config.py — Configuration loading with Pydantic models.

Loads config.yaml, expands ${ENV_VAR} references, and exposes a global
`config` singleton of type SagaConfig.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str = ""  # Empty = auth disabled; set to enable Bearer token auth


class ModelsConfig(BaseModel):
    narration: str = "claude-sonnet-4-5-20250929"
    extraction: str = "gemini-2.0-flash"
    curator: str = "claude-sonnet-4-5-20250929"
    embedding: str = "text-embedding-3-small"


class ApiKeysConfig(BaseModel):
    anthropic: Optional[str] = None
    openai: Optional[str] = None
    google: Optional[str] = None


class TokenBudgetConfig(BaseModel):
    total_context_max: int = 128000
    dynamic_context_max: int = 1500
    md_cache_max: int = 600
    lorebook_max: int = 800
    state_briefing_max: int = 200
    graph_context_max: int = 300
    state_block_instruction: int = 100


class MdCacheConfig(BaseModel):
    enabled: bool = True
    cache_dir: str = "cache/sessions"
    files: List[str] = Field(default_factory=lambda: ["state.md", "relations.md", "story.md", "lore.md"])
    atomic_write: bool = True


class PromptCachingConfig(BaseModel):
    enabled: bool = True
    strategy: str = "md_prefix"
    stabilize_system: bool = True
    canonical_similarity_threshold: float = 0.30


class CuratorConfig(BaseModel):
    interval: int = 10
    enabled: bool = True
    memory_block_schema: List[str] = Field(
        default_factory=lambda: ["narrative_summary", "curation_decisions", "contradiction_log"]
    )
    compress_story_after_turns: int = 50
    letta_base_url: str = "http://localhost:8283"
    letta_model: str = "anthropic/claude-sonnet-4-5-20250929"
    letta_embedding: str = "openai/text-embedding-3-small"


class DynamicLorebookConfig(BaseModel):
    character_layers: List[str] = Field(default_factory=lambda: ["A1", "A2", "A3", "A4"])
    decay_threshold: int = 5
    propagation_depth: int = 2


class GraphConfig(BaseModel):
    db_path: str = "db/graph.kuzu"
    mode: str = "on-disk"
    max_hop: int = 3
    hybrid_rerank: bool = True


class SessionConfig(BaseModel):
    auto_save: bool = True
    auto_save_interval: int = 5
    default_world: str = "my_world"


class ModuleEntry(BaseModel):
    enabled: bool = False
    config: Optional[str] = None


class ModulesConfig(BaseModel):
    rpg: ModuleEntry = Field(default_factory=ModuleEntry)
    map: ModuleEntry = Field(default_factory=ModuleEntry)

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class SagaConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    md_cache: MdCacheConfig = Field(default_factory=MdCacheConfig)
    prompt_caching: PromptCachingConfig = Field(default_factory=PromptCachingConfig)
    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    dynamic_lorebook: DynamicLorebookConfig = Field(default_factory=DynamicLorebookConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    modules: ModulesConfig = Field(default_factory=ModulesConfig)


# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand ${VAR} references in strings within a parsed YAML structure."""
    if isinstance(value, str):
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return _ENV_VAR_PATTERN.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path = "config.yaml") -> SagaConfig:
    """Load and parse a SAGA config YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        A fully populated :class:`SagaConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path.resolve()}")

    with config_path.open("r", encoding="utf-8") as fh:
        raw: Dict[str, Any] = yaml.safe_load(fh) or {}

    expanded = _expand_env_vars(raw)
    return SagaConfig.model_validate(expanded)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

#: Global config instance.  Populated when the module is first imported via
#: :func:`load_config`, or manually replaced by calling ``load_config``
#: and assigning the result here.
config: SagaConfig = SagaConfig()


def _init_global_config(path: str = "config.yaml") -> None:
    """Attempt to load the global config on module import; silently fall back
    to defaults if the file does not exist (useful during testing/import)."""
    global config
    try:
        config = load_config(path)
    except FileNotFoundError:
        pass  # Use defaults when no file is present


_init_global_config()
