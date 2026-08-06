"""Dreamer 사이클 B-0~B-3 (스펙 §3.2) — FakeLLM으로 1콜 검증."""
import asyncio
import json

from dreaming.dreamer import Dreamer
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore

_EXTRACTION = json.dumps({
    "episodes": [{"start_turn": 0, "end_turn": 0, "title": "포션 흥정",
                  "summary": "가격을 물었다.", "open_threads": []}],
    "facts": [{"claim": "포션은 50골드다", "evidence_turn": 0,
               "numbers": [{"name": "가격", "value": 50, "unit": "골드"}]}],
    "commits": [{"slot": "소지금", "op": "set", "value": 1450, "turn": 0}],
    "actors": [{"names": ["리사"], "profile": "시장 상인", "tier": "main"}],
}, ensure_ascii=False)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def complete(self, system, user):
        self.calls.append((system, user))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def _seed_raw(storage, session="sess1", turns=1):
    for t in range(turns):
        storage.put(f"{session}/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"포션 얼마야? ({t})",
            "assistant_text": "50골드다. 잔액은 1,450골드.",
            "user_hash": f"u{t}", "assistant_hash": f"a{t}"})


def test_dream_full_cycle_advances_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    llm = FakeLLM(_EXTRACTION)
    report = asyncio.run(Dreamer(storage, llm).dream("sess1"))
    assert report["facts"] == 1 and report["commits"] == 1
    assert report["actors"] == 1 and report["episodes"] == 1
    assert report["blocked"] == 0
    assert len(llm.calls) == 1                                    # 사이클당 1콜
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}
    store = MemoryStore(storage, "sess1")
    assert store.list_facts()[0].status == "confirmed"
    assert store.current_state() == {"소지금": 1450.0}


def test_dream_without_backlog_is_noop(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    llm = FakeLLM(_EXTRACTION)
    d = Dreamer(storage, llm)
    asyncio.run(d.dream("sess1"))
    assert asyncio.run(d.dream("sess1")) is None                 # 두 번째: 잔량 없음
    assert len(llm.calls) == 1


def test_llm_failure_discards_cycle_keeps_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(RuntimeError("api down")))
    assert asyncio.run(d.dream("sess1")) is None                 # fail-open
    assert storage.get("sess1/dreamer", "cursor") is None        # 커서 불변
    assert d.has_backlog("sess1")                                # 다음 유휴에 재시도


def test_garbage_json_discards_cycle(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM("JSON 아님"))
    assert asyncio.run(d.dream("sess1")) is None
    assert storage.get("sess1/dreamer", "cursor") is None


def test_concurrent_dream_skips(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))

    async def scenario():
        d._active.add("sess1")                # 꿈꾸는 중 시뮬레이션
        try:
            return await d.dream("sess1")
        finally:
            d._active.discard("sess1")

    assert asyncio.run(scenario()) is None
    assert asyncio.run(d.dream("sess1")) is not None   # 해제 후 정상 진행


def test_has_backlog_and_snapshot_respect_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=3)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))
    storage.put("sess1/dreamer", "cursor", {"next_turn": 2})
    snap = d.snapshot("sess1")
    assert [r["turn_number"] for r in snap] == [2]
    assert d.has_backlog("sess1")
