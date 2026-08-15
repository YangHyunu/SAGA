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
    report = asyncio.run(Dreamer(storage, llm, batch_turns=1).dream("sess1"))
    assert report["facts"] == 1 and report["commits"] == 1
    assert report["actors"] == 1 and report["episodes"] == 1
    assert report["blocked"] == 0
    assert report["chunks"] == 0                          # 전부 꼬리 안 → 플랜 없음
    assert storage.get("sess1/compression", "plan") is None
    assert len(llm.calls) == 1                                    # 사이클당 1콜
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}
    store = MemoryStore(storage, "sess1")
    assert store.list_facts()[0].status == "confirmed"
    assert store.current_state() == {"소지금": 1450.0}


def test_dream_without_backlog_is_noop(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    llm = FakeLLM(_EXTRACTION)
    d = Dreamer(storage, llm, batch_turns=1)
    asyncio.run(d.dream("sess1"))
    assert asyncio.run(d.dream("sess1")) is None                 # 두 번째: 잔량 없음
    assert len(llm.calls) == 1


def test_llm_failure_discards_cycle_keeps_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(RuntimeError("api down")), batch_turns=1)
    assert asyncio.run(d.dream("sess1")) is None                 # fail-open
    assert storage.get("sess1/dreamer", "cursor") is None        # 커서 불변
    assert d.has_backlog("sess1")                                # 다음 유휴에 재시도


def test_garbage_json_discards_cycle(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM("JSON 아님"), batch_turns=1)
    assert asyncio.run(d.dream("sess1")) is None
    assert storage.get("sess1/dreamer", "cursor") is None


def test_concurrent_dream_skips(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage)
    d = Dreamer(storage, FakeLLM(_EXTRACTION), batch_turns=1)

    async def scenario():
        d._active.add("sess1")                # 꿈꾸는 중 시뮬레이션
        try:
            return await d.dream("sess1")
        finally:
            d._active.discard("sess1")

    assert asyncio.run(scenario()) is None
    assert asyncio.run(d.dream("sess1")) is not None   # 해제 후 정상 진행


def test_dream_writes_compression_plan(tmp_path):
    # 압축 경계는 BOUNDARY_STEP(10)턴 단위로만 전진한다 (chunks.py) —
    # 꼬리 TAIL_KEEP(6) 밖에 최소 한 계단이 쌓여야 플랜이 생긴다.
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=20)
    ext = json.dumps({"episodes": [
        {"start_turn": 0, "end_turn": 9, "title": "초반",
         "summary": "만남과 흥정.", "open_threads": []}]}, ensure_ascii=False)
    report = asyncio.run(Dreamer(storage, FakeLLM(ext)).dream("sess1"))
    assert report["chunks"] == 1
    plan = storage.get("sess1/compression", "plan")
    assert plan["covers_until_turn"] == 10
    assert "초반" in plan["messages"][0]["content"]


def test_dream_batches_fixed_turn_quanta(tmp_path):
    """배치 결정론: 완전 양자만, 오래된 순으로, 잔량은 원문 유지.

    트리거가 언제 오든 배치 내용이 같아야 한다 — 유휴 타이밍이 에피소드
    경계를 정하던 병리(같은 100턴이 사실 353 vs 39로 갈림)의 재발 방지.
    """
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=25)
    llm = FakeLLM(_EXTRACTION)
    d = Dreamer(storage, llm, batch_turns=10)
    report = asyncio.run(d.dream("sess1"))
    assert report["cycles"] == 2 and len(llm.calls) == 2     # 양자당 1콜
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 20}
    # 각 콜의 프롬프트가 정확히 자기 양자 구간만 담는다
    assert "[턴 0]" in llm.calls[0][1] and "[턴 9]" in llm.calls[0][1]
    assert "[턴 10]" not in llm.calls[0][1]
    assert "[턴 10]" in llm.calls[1][1] and "[턴 19]" in llm.calls[1][1]
    # 잔량 5턴 < 양자 — 부분 배치 금지, 원문으로 남는다
    assert asyncio.run(d.dream("sess1")) is None
    assert d.has_backlog("sess1") and not d.ready("sess1")


def test_has_backlog_and_snapshot_respect_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed_raw(storage, turns=3)
    d = Dreamer(storage, FakeLLM(_EXTRACTION))
    storage.put("sess1/dreamer", "cursor", {"next_turn": 2})
    snap = d.snapshot("sess1")
    assert [r["turn_number"] for r in snap] == [2]
    assert d.has_backlog("sess1")
