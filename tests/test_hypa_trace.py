"""hypa.py trace 훅 — trace=None이면 기존과 동일, 리스트면 이벤트 기록."""
import math
from benchmarks.eval import hypa


def fake_embed(texts):
    # 결정론 가짜 임베딩: 'ring' 포함 텍스트끼리 유사하게
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _mk_summaries():
    return [
        {"text": "the ring was found", "chatMemos": ["m0", "m1"], "isImportant": False},
        {"text": "weather talk", "chatMemos": ["m2", "m3"], "isImportant": True},
        {"text": "market visit", "chatMemos": ["m4", "m5"], "isImportant": False},
    ]


def _recent():
    return [{"role": "user", "content": "where is the ring", "memo": "m6"}]


def test_select_trace_none_is_default_and_unchanged():
    s = hypa.HypaSettings()
    a = hypa.select_summaries(_mk_summaries(), 1000, _recent(), s, embed=fake_embed)
    b = hypa.select_summaries(_mk_summaries(), 1000, _recent(), s, embed=fake_embed,
                              trace=None)
    assert [x["text"] for x in a] == [x["text"] for x in b]


def test_select_trace_records_stage_events():
    s = hypa.HypaSettings()
    # 마지막 요약을 크게 → recent 예산에 안 들어가 similar 후보가 남는다
    summaries = _mk_summaries()
    summaries[2]["text"] = "market visit " * 400
    tr = []
    hypa.select_summaries(summaries, 1000, _recent(), s, embed=fake_embed,
                          trace=tr)
    evs = {e["ev"] for e in tr}
    assert {"important", "recent", "similar"} <= evs
    sim = next(e for e in tr if e["ev"] == "similar")
    assert sim["queries"] and sim["chunk_scores"]
    imp = next(e for e in tr if e["ev"] == "important")
    assert imp["picked"] == [1]          # isImportant인 index 1


def test_plan_batches_trace_records_batches():
    s = hypa.HypaSettings(max_chats_per_summary=2)
    chats = [{"role": "user", "content": "msg " * 50, "memo": f"m{i}"}
             for i in range(10)]
    tr = []
    # 10_050: 발동(>10_000)하되 배치 1개 소비면 과요약 break에 닿는 값
    batches, _cur, _si, err = hypa.plan_batches(chats, 0, 10_050, 10_000, s,
                                                trace=tr)
    assert err is None and batches
    ev_batches = [e for e in tr if e["ev"] == "batch"]
    assert len(ev_batches) == len(batches)
    assert ev_batches[0]["idx"] == ["m0", "m1"]
