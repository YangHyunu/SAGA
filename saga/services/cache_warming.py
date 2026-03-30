"""Background cache warming loop for Anthropic prompt caching."""
import asyncio
import logging
import time

from saga.core import dependencies as deps

logger = logging.getLogger(__name__)


async def cache_warming_loop():
    """Background task: send lightweight keepalive to maintain Anthropic cache.

    Runs every 30 seconds, fires a max_tokens=1 ping for any Anthropic session
    that has been idle longer than cache_warming.interval seconds and has not
    yet exhausted its max_warmings quota.  Sessions idle for 10+ minutes or
    at max_warmings are evicted from tracking.
    """
    while True:
        await asyncio.sleep(30)
        if not deps.config or not deps.config.cache_warming.enabled:
            continue
        now = time.time()
        expired = []
        async with deps._warming_lock:
            for sid, data in list(deps._warming_data.items()):
                elapsed = now - data["timestamp"]
                if elapsed > 600 or data["count"] >= deps.config.cache_warming.max_warmings:
                    expired.append(sid)
                elif elapsed > deps.config.cache_warming.interval:
                    try:
                        await deps.llm_client.call_llm(
                            model=data["model"],
                            messages=data["messages"],
                            max_tokens=1,
                            temperature=0,
                        )
                        data["timestamp"] = now
                        data["count"] += 1
                        logger.info(f"[CacheWarming] session={sid} warming #{data['count']}")
                    except Exception as e:
                        logger.warning(f"[CacheWarming] Failed for {sid}: {e}")
            for sid in expired:
                del deps._warming_data[sid]
                logger.debug(f"[CacheWarming] Cleaned up session {sid}")
