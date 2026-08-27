"""hypaMemoryV3Main 포트 — exp와 갈라지는 지점 고정 (스펙 §3 표)."""
import pytest
from benchmarks.eval import hypa, hypa_main


def fake_embed(texts):
    return [[1.0, 0.0] if "ring" in t else [0.0, 1.0] for t in texts]


def _summaries():
    return [
        {"text": "the ring was found", "chatMemos": ["m0", "m1"], "isImportant": False},
        {"text": "weather talk", "chatMemos": ["m2", "m3"], "isImportant": True},
        {"text": "market visit", "chatMemos": ["m4", "m5"], "isImportant": False},
    ]


def _summaries_with_similar_candidates():
    """recent 예산을 조기 소진시켜 similar 후보를 남기는 픽스처 —
    마지막 요약이 reserved_recent보다 커서 recent가 0개로 break."""
    s = _summaries()
    s[2]["text"] = "market visit " * 400
    return s


def test_main_similar_uses_whole_message_queries():
    """main은 문단 분할 없이 원문 통짜 쿼리 (hypav3.ts:1369) —
    exp의 서브쿼리 분할(680~695)과 갈라지는 지점."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": "first paragraph\n\nsecond ring paragraph",
               "memo": "m6"}]
    tr = []
    hypa_main.select_summaries_main(_summaries_with_similar_candidates(),
                                    1000, recent, s, embed=fake_embed, trace=tr)
    sim = next(e for e in tr if e["ev"] == "similar")
    assert len(sim["queries"]) == 1                      # 문단 2개여도 쿼리 1개
    assert sim["queries"][0]["content"] == "first paragraph\n\nsecond ring paragraph"
    assert sim["queries"][0]["weight"] == 1.0            # n=1 → (0+1)/(1*2/2)


def test_main_similar_weights_follow_simplecc_callback():
    """가중치 (i+1)/(n(n+1)/2) (hypav3.ts:1409~1414) — n=3 → 1/6, 2/6, 3/6."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": f"query {i} ring", "memo": f"m{6+i}"}
              for i in range(3)]
    tr = []
    hypa_main.select_summaries_main(_summaries_with_similar_candidates(),
                                    1000, recent, s, embed=fake_embed, trace=tr)
    sim = next(e for e in tr if e["ev"] == "similar")
    assert [round(q["weight"], 4) for q in sim["queries"]] == [
        round(1 / 6, 4), round(2 / 6, 4), round(3 / 6, 4)]


def test_main_metrics_four_index_lists():
    """main은 4단계 인덱스를 기록 (hypav3.ts:1578~1591)."""
    s = hypa.HypaSettings()
    recent = [{"role": "user", "content": "where is the ring", "memo": "m6"}]
    selected, metrics = hypa_main.select_summaries_main(
        _summaries(), 1000, recent, s, embed=fake_embed)
    assert metrics["lastImportantSummaries"] == [1]
    assert set(metrics) == {"lastImportantSummaries", "lastRecentSummaries",
                            "lastSimilarSummaries", "lastRandomSummaries"}
    assert metrics["lastRandomSummaries"] == []
    # 시간순 반환 (1535~1537)
    texts = [x["text"] for x in _summaries()]
    order = [s_["text"] for s_ in selected]
    assert order == sorted(order, key=texts.index)


def test_main_random_ratio_positive_raises():
    s = hypa.HypaSettings(recent_memory_ratio=0.4, similar_memory_ratio=0.4)
    with pytest.raises(hypa.HypaError):
        hypa_main.select_summaries_main(_summaries(), 1000, [], s, embed=fake_embed)
