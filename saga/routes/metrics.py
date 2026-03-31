"""Metrics API routes for SAGA dashboard."""
import time
from fastapi import APIRouter, Depends, Query

from saga.core import dependencies as deps
from saga.middleware.auth import verify_bearer

router = APIRouter(prefix="/api/metrics", dependencies=[Depends(verify_bearer)])


@router.get("/timeline")
async def metrics_timeline(period: str = Query("hour", pattern="^(hour|day)$")):
    """Cost and token aggregation over time.

    period=hour → grouped by hour (last 48 hours)
    period=day  → grouped by day (last 30 days)
    """
    if period == "hour":
        group_expr = "strftime('%Y-%m-%d %H:00', timestamp, 'unixepoch', 'localtime')"
        since = time.time() - 48 * 3600
    else:
        group_expr = "strftime('%Y-%m-%d', timestamp, 'unixepoch', 'localtime')"
        since = time.time() - 30 * 86400

    async with deps.sqlite_db._db.execute(
        f"""
        SELECT
            {group_expr} as period,
            COUNT(*) as calls,
            COALESCE(SUM(input_tokens), 0) as input_tokens,
            COALESCE(SUM(output_tokens), 0) as output_tokens,
            COALESCE(SUM(cache_read_tokens), 0) as cache_read_tokens,
            COALESCE(SUM(cost_usd), 0) as cost_usd,
            COALESCE(SUM(savings_usd), 0) as savings_usd
        FROM cost_log
        WHERE timestamp >= ?
        GROUP BY {group_expr}
        ORDER BY period
        """,
        (since,),
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        "period": period,
        "data": [
            {
                "time": r[0],
                "calls": r[1],
                "input_tokens": r[2],
                "output_tokens": r[3],
                "cache_read_tokens": r[4],
                "cost_usd": round(r[5], 4),
                "savings_usd": round(r[6], 4),
            }
            for r in rows
        ],
    }


@router.get("/cache")
async def metrics_cache(limit: int = Query(100, ge=1, le=1000)):
    """Per-call cache hit ratio time series (most recent calls)."""
    async with deps.sqlite_db._db.execute(
        """
        SELECT
            timestamp,
            session_id,
            model,
            input_tokens,
            cache_read_tokens,
            cache_create_tokens,
            CASE WHEN (input_tokens + cache_read_tokens + cache_create_tokens) > 0
                 THEN ROUND(CAST(cache_read_tokens AS REAL) / (input_tokens + cache_read_tokens + cache_create_tokens) * 100, 1)
                 ELSE 0 END as cache_hit_pct
        FROM cost_log
        WHERE call_type IN ('main', 'main_stream')
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        "data": [
            {
                "timestamp": r[0],
                "session_id": r[1],
                "model": r[2],
                "input_tokens": r[3],
                "cache_read_tokens": r[4],
                "cache_create_tokens": r[5],
                "cache_hit_pct": r[6],
            }
            for r in reversed(rows)  # chronological order
        ],
    }


@router.get("/latency")
async def metrics_latency(limit: int = Query(100, ge=1, le=1000)):
    """TTFT and total latency statistics."""
    async with deps.sqlite_db._db.execute(
        """
        SELECT
            timestamp,
            session_id,
            model,
            call_type,
            ttft_ms,
            total_ms
        FROM cost_log
        WHERE call_type IN ('main', 'main_stream') AND total_ms > 0
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()

    data = [
        {
            "timestamp": r[0],
            "session_id": r[1],
            "model": r[2],
            "call_type": r[3],
            "ttft_ms": r[4],
            "total_ms": r[5],
        }
        for r in reversed(rows)
    ]

    # Compute summary stats
    ttft_values = [d["ttft_ms"] for d in data if d["ttft_ms"] > 0]
    total_values = [d["total_ms"] for d in data if d["total_ms"] > 0]

    def _stats(values):
        if not values:
            return {"avg": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
        s = sorted(values)
        n = len(s)
        return {
            "avg": round(sum(s) / n, 0),
            "p50": round(s[n // 2], 0),
            "p95": round(s[int(n * 0.95)], 0) if n > 1 else round(s[0], 0),
            "min": round(s[0], 0),
            "max": round(s[-1], 0),
        }

    return {
        "summary": {
            "ttft": _stats(ttft_values),
            "total": _stats(total_values),
            "count": len(data),
        },
        "data": data,
    }


@router.get("/sessions")
async def metrics_sessions():
    """Per-session summary with turn count and cost."""
    async with deps.sqlite_db._db.execute(
        """
        SELECT
            c.session_id,
            s.name,
            s.turn_count,
            s.updated_at,
            COUNT(*) as llm_calls,
            COALESCE(SUM(c.input_tokens), 0) as total_input,
            COALESCE(SUM(c.output_tokens), 0) as total_output,
            COALESCE(SUM(c.cost_usd), 0) as total_cost,
            COALESCE(SUM(c.savings_usd), 0) as total_savings,
            COALESCE(AVG(CASE WHEN c.ttft_ms > 0 THEN c.ttft_ms END), 0) as avg_ttft,
            COALESCE(AVG(CASE WHEN c.total_ms > 0 THEN c.total_ms END), 0) as avg_total_ms
        FROM cost_log c
        LEFT JOIN sessions s ON c.session_id = s.id
        GROUP BY c.session_id
        ORDER BY MAX(c.timestamp) DESC
        """,
    ) as cursor:
        rows = await cursor.fetchall()

    return {
        "data": [
            {
                "session_id": r[0],
                "name": r[1] or "",
                "turn_count": r[2] or 0,
                "last_active": r[3] or "",
                "llm_calls": r[4],
                "total_input_tokens": r[5],
                "total_output_tokens": r[6],
                "total_cost_usd": round(r[7], 4),
                "total_savings_usd": round(r[8], 4),
                "avg_ttft_ms": round(r[9], 0),
                "avg_total_ms": round(r[10], 0),
            }
            for r in rows
        ],
    }
