"""세션 유휴 타이머 (스펙 §3.2) — 유휴 = 캐시 TTL 경과 = 꿈 트리거."""
import asyncio

from dreaming.idle import IdleWatcher


def _run(coro):
    return asyncio.run(coro)


def test_fires_once_after_idle():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.03, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.1)

    _run(scenario())
    assert fired == ["sess1"]


def test_touch_resets_timer():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.05, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.03)
        w.touch("sess1")                 # 리셋 — 아직 안 울려야 함
        await asyncio.sleep(0.03)
        assert fired == []
        await asyncio.sleep(0.05)

    _run(scenario())
    assert fired == ["sess1"]


def test_on_idle_exception_swallowed():
    async def scenario():
        async def on_idle(s):
            raise RuntimeError("dream failed")
        w = IdleWatcher(0.01, on_idle)
        w.touch("sess1")
        await asyncio.sleep(0.05)        # 예외가 새어나오면 여기서 터짐

    _run(scenario())                     # fail-open: 통과하면 성공


def test_cancel_all():
    fired = []

    async def scenario():
        async def on_idle(s):
            fired.append(s)
        w = IdleWatcher(0.02, on_idle)
        w.touch("a")
        w.touch("b")
        w.cancel_all()
        await asyncio.sleep(0.06)

    _run(scenario())
    assert fired == []
