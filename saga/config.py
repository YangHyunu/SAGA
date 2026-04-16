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
    api_key: str | None = ""  # Empty/None = auth disabled; set to enable Bearer token auth


class ModelsConfig(BaseModel):
    narration: str = "claude-haiku-4-5-20251001"
    extraction: str = "gemini-2.5-flash-lite"
    curator: str = "claude-haiku-4-5-20251001"
    embedding: str = "text-embedding-3-small"


class ApiKeysConfig(BaseModel):
    anthropic: Optional[str] = None
    openai: Optional[str] = None
    google: Optional[str] = None


class TokenBudgetConfig(BaseModel):
    total_context_max: int = 180000   # Anthropic 200K 기준 안전 마진 (~90%)
    dynamic_context_max: int = 4000   # 동적 컨텍스트 (state + episodes + lore + instruction)


class MdCacheConfig(BaseModel):
    enabled: bool = True
    cache_dir: str = "cache/sessions"
    atomic_write: bool = True


class PromptCachingConfig(BaseModel):
    enabled: bool = True
    stabilize_system: bool = True
    cache_ttl: str = "1h"  # extended-cache-ttl: "5m" (default) or "1h"
    compress_enabled: bool = True
    compress_threshold_ratio: float = 0.35  # total_context_max * ratio 초과 시 압축 (65K 기준 ~33K)
    min_compress_turns: int = 3             # 최소 압축 단위 (턴)


class CuratorConfig(BaseModel):
    interval: int = 10
    enabled: bool = True
    memory_block_schema: List[str] = Field(
        default_factory=lambda: ["narrative_summary", "curation_decisions", "contradiction_log"]
    )
    compress_story_after_turns: int = 50
    letta_base_url: str = "http://localhost:8283"
    letta_model: str = "anthropic/claude-haiku-4-5-20251001"
    letta_embedding: str = "openai/text-embedding-3-small"


class CacheWarmingConfig(BaseModel):
    enabled: bool = True
    interval: int = 270  # seconds (4.5 minutes — just before 5-min TTL expiry)
    max_warmings: int = 4  # per session before giving up


class LangSmithConfig(BaseModel):
    enabled: bool = False
    project: str = "saga-risu"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class SagaConfig(BaseModel):
    model_config = {"extra": "ignore"}

    server: ServerConfig = Field(default_factory=ServerConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    api_keys: ApiKeysConfig = Field(default_factory=ApiKeysConfig)
    token_budget: TokenBudgetConfig = Field(default_factory=TokenBudgetConfig)
    md_cache: MdCacheConfig = Field(default_factory=MdCacheConfig)
    prompt_caching: PromptCachingConfig = Field(default_factory=PromptCachingConfig)
    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    cache_warming: CacheWarmingConfig = Field(default_factory=CacheWarmingConfig)
    langsmith: LangSmithConfig = Field(default_factory=LangSmithConfig)


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
