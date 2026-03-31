"""Cost Tracker — 토큰 사용량 및 비용 추적, 캐시 절감 수치화.

모델별 단가 테이블 기반으로 매 LLM 호출의 비용을 계산하고 SQLite에 기록.
캐시 hit 토큰은 할인 적용하여 절감액 산출.
"""

import logging
import time
from dataclasses import dataclass, field, asdict

from saga.storage.sqlite_db import SQLiteDB

logger = logging.getLogger(__name__)

# ─── 모델별 단가 (USD per 1M tokens) ───
# https://docs.anthropic.com/en/docs/about-claude/models
# https://ai.google.dev/pricing
# https://openai.com/api/pricing/
PRICING = {
    # Anthropic
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0, "cache_read": 0.30, "cache_create": 3.75},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0, "cache_read": 0.08, "cache_create": 1.0},
    "claude-opus-4-20250514": {"input": 15.0, "output": 75.0, "cache_read": 1.50, "cache_create": 18.75},
    # Google
    "gemini-2.5-flash-lite": {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_create": 0.0},  # free tier
    "gemini-2.5-flash-preview-05-20": {"input": 0.15, "output": 0.60, "cache_read": 0.0375, "cache_create": 0.15},
    "gemini-3-flash-preview": {"input": 0.15, "output": 0.60, "cache_read": 0.0375, "cache_create": 0.15},
    "gemini-2.5-pro-preview-05-06": {"input": 1.25, "output": 10.0, "cache_read": 0.3125, "cache_create": 1.25},
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.0, "cache_read": 1.25, "cache_create": 2.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cache_read": 0.075, "cache_create": 0.15},
    "gpt-4.1": {"input": 2.0, "output": 8.0, "cache_read": 0.50, "cache_create": 2.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60, "cache_read": 0.10, "cache_create": 0.40},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40, "cache_read": 0.025, "cache_create": 0.10},
    # Embedding
    "text-embedding-3-small": {"input": 0.02, "output": 0.0, "cache_read": 0.0, "cache_create": 0.0},
}

# Fallback: match by prefix
_PRICING_PREFIXES = {
    "claude-sonnet": PRICING["claude-sonnet-4-5-20250929"],
    "claude-haiku": PRICING["claude-haiku-4-5-20251001"],
    "claude-opus": PRICING["claude-opus-4-20250514"],
    "gemini-2.5-flash-lite": PRICING["gemini-2.5-flash-lite"],
    "gemini-2.5-flash": PRICING["gemini-2.5-flash-preview-05-20"],
    "gemini-3-flash": PRICING["gemini-3-flash-preview"],
    "gemini-2.5-pro": PRICING["gemini-2.5-pro-preview-05-06"],
    "gpt-4o-mini": PRICING["gpt-4o-mini"],
    "gpt-4o": PRICING["gpt-4o"],
    "gpt-4.1-mini": PRICING["gpt-4.1-mini"],
    "gpt-4.1-nano": PRICING["gpt-4.1-nano"],
    "gpt-4.1": PRICING["gpt-4.1"],
}


def get_pricing(model: str) -> dict:
    """Get pricing for a model. Falls back to prefix matching, then zero."""
    if model in PRICING:
        return PRICING[model]
    model_lower = model.lower()
    for prefix, pricing in _PRICING_PREFIXES.items():
        if model_lower.startswith(prefix):
            return pricing
    logger.warning(f"[CostTracker] Unknown model pricing: {model}, using zero")
    return {"input": 0.0, "output": 0.0, "cache_read": 0.0, "cache_create": 0.0}


@dataclass
class UsageRecord:
    """Single LLM call usage record."""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0
    cost_usd: float = 0.0
    savings_usd: float = 0.0  # how much saved by cache
    session_id: str = ""
    call_type: str = ""  # "main", "sub_b", "curator", etc.
    timestamp: float = field(default_factory=time.time)
    ttft_ms: float = 0.0     # time to first token (ms)
    total_ms: float = 0.0    # total request duration (ms)


class CostTracker:
    """Tracks LLM costs and cache savings."""

    def __init__(self, sqlite_db: SQLiteDB):
        self.sqlite_db = sqlite_db
        self._initialized = False

    async def initialize(self):
        """Create cost_log table if not exists."""
        await self.sqlite_db._db.execute("""
            CREATE TABLE IF NOT EXISTS cost_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                model TEXT NOT NULL,
                call_type TEXT DEFAULT '',
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_create_tokens INTEGER DEFAULT 0,
                cost_usd REAL DEFAULT 0.0,
                savings_usd REAL DEFAULT 0.0,
                timestamp REAL NOT NULL,
                ttft_ms REAL DEFAULT 0.0,
                total_ms REAL DEFAULT 0.0
            )
        """)
        # Migrate existing tables that lack the new columns
        for col in ("ttft_ms", "total_ms"):
            try:
                await self.sqlite_db._db.execute(f"ALTER TABLE cost_log ADD COLUMN {col} REAL DEFAULT 0.0")
            except Exception:
                pass  # column already exists
        await self.sqlite_db._db.commit()
        self._initialized = True

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int,
                       cache_read_tokens: int = 0, cache_create_tokens: int = 0) -> tuple[float, float]:
        """Calculate cost and savings for an LLM call.

        Returns:
            (cost_usd, savings_usd)
        """
        pricing = get_pricing(model)
        per_m = 1_000_000

        # Actual input = total input - cache_read - cache_create (these are billed separately)
        regular_input = max(0, input_tokens - cache_read_tokens - cache_create_tokens)

        cost = (
            (regular_input / per_m) * pricing["input"]
            + (output_tokens / per_m) * pricing["output"]
            + (cache_read_tokens / per_m) * pricing["cache_read"]
            + (cache_create_tokens / per_m) * pricing["cache_create"]
        )

        # Savings: cache_read tokens would have cost full input price without caching
        savings = (cache_read_tokens / per_m) * (pricing["input"] - pricing["cache_read"])

        return round(cost, 6), round(savings, 6)

    async def record(self, record: UsageRecord):
        """Record a usage entry to SQLite."""
        if not self._initialized:
            await self.initialize()

        record.cost_usd, record.savings_usd = self.calculate_cost(
            record.model, record.input_tokens, record.output_tokens,
            record.cache_read_tokens, record.cache_create_tokens,
        )

        await self.sqlite_db._db.execute(
            """
            INSERT INTO cost_log (session_id, model, call_type, input_tokens, output_tokens,
                                  cache_read_tokens, cache_create_tokens, cost_usd, savings_usd,
                                  timestamp, ttft_ms, total_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (record.session_id, record.model, record.call_type,
             record.input_tokens, record.output_tokens,
             record.cache_read_tokens, record.cache_create_tokens,
             record.cost_usd, record.savings_usd, record.timestamp,
             record.ttft_ms, record.total_ms),
        )
        await self.sqlite_db._db.commit()

        logger.info(
            f"[Cost] {record.call_type} model={record.model} "
            f"in={record.input_tokens} out={record.output_tokens} "
            f"cache_read={record.cache_read_tokens} "
            f"cost=${record.cost_usd:.4f} saved=${record.savings_usd:.4f}"
            + (f" ttft={record.ttft_ms:.0f}ms total={record.total_ms:.0f}ms" if record.total_ms else "")
        )

    async def get_session_summary(self, session_id: str) -> dict:
        """Get cost summary for a session."""
        async with self.sqlite_db._db.execute(
            """
            SELECT
                COUNT(*) as total_calls,
                COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                COALESCE(SUM(cache_create_tokens), 0) as total_cache_create,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COALESCE(SUM(savings_usd), 0) as total_savings
            FROM cost_log WHERE session_id = ?
            """,
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        total_cost = row[5]
        total_savings = row[6]
        cost_without_cache = total_cost + total_savings

        return {
            "session_id": session_id,
            "total_calls": row[0],
            "total_input_tokens": row[1],
            "total_output_tokens": row[2],
            "total_cache_read_tokens": row[3],
            "total_cache_create_tokens": row[4],
            "total_cost_usd": round(total_cost, 4),
            "total_savings_usd": round(total_savings, 4),
            "cost_without_cache_usd": round(cost_without_cache, 4),
            "cache_savings_percent": round((total_savings / cost_without_cache * 100) if cost_without_cache > 0 else 0, 1),
        }

    async def get_global_summary(self) -> dict:
        """Get cost summary across all sessions."""
        async with self.sqlite_db._db.execute(
            """
            SELECT
                COUNT(*) as total_calls,
                COUNT(DISTINCT session_id) as total_sessions,
                COALESCE(SUM(input_tokens), 0) as total_input,
                COALESCE(SUM(output_tokens), 0) as total_output,
                COALESCE(SUM(cache_read_tokens), 0) as total_cache_read,
                COALESCE(SUM(cache_create_tokens), 0) as total_cache_create,
                COALESCE(SUM(cost_usd), 0) as total_cost,
                COALESCE(SUM(savings_usd), 0) as total_savings
            FROM cost_log
            """,
        ) as cursor:
            row = await cursor.fetchone()

        total_cost = row[6]
        total_savings = row[7]
        cost_without_cache = total_cost + total_savings

        # Per-model breakdown
        async with self.sqlite_db._db.execute(
            """
            SELECT model,
                   COUNT(*) as calls,
                   COALESCE(SUM(input_tokens), 0) as input_t,
                   COALESCE(SUM(output_tokens), 0) as output_t,
                   COALESCE(SUM(cost_usd), 0) as cost,
                   COALESCE(SUM(savings_usd), 0) as savings
            FROM cost_log GROUP BY model ORDER BY cost DESC
            """,
        ) as cursor:
            model_rows = await cursor.fetchall()

        models = [
            {
                "model": r[0], "calls": r[1],
                "input_tokens": r[2], "output_tokens": r[3],
                "cost_usd": round(r[4], 4), "savings_usd": round(r[5], 4),
            }
            for r in model_rows
        ]

        return {
            "total_calls": row[0],
            "total_sessions": row[1],
            "total_input_tokens": row[2],
            "total_output_tokens": row[3],
            "total_cache_read_tokens": row[4],
            "total_cache_create_tokens": row[5],
            "total_cost_usd": round(total_cost, 4),
            "total_savings_usd": round(total_savings, 4),
            "cost_without_cache_usd": round(cost_without_cache, 4),
            "cache_savings_percent": round((total_savings / cost_without_cache * 100) if cost_without_cache > 0 else 0, 1),
            "by_model": models,
        }
