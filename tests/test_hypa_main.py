"""hypaMemoryV3Main 포트 — exp와 갈라지는 지점 고정 (스펙 §3 표)."""
import functools

import pytest
from benchmarks.eval import hypa, hypa_main


@pytest.fixture(autouse=True)
def _isolated_summary_cache(monkeypatch, tmp_path):
    """summarize의 기본 cache_dir은 def-시점 바인딩이라 setattr(CACHE_DIR)로는
    안 바뀐다 — 함수 자체를 tmp 캐시로 감싼다. 실캐시 오염 방지 (실측:
    가짜 요약이 dreaming_data/eval/hypa-cache에 박혀 후속 테스트가 히트했다)."""
    monkeypatch.setattr(hypa, "summarize",
                        functools.partial(hypa.summarize, cache_dir=tmp_path))


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


def _mk_chats(n, tok_words=50):
    return [{"role": "user" if i % 2 == 0 else "assistant",
             "content": f"m{i} " + "word " * tok_words} for i in range(n)]


def _fake_send_factory(log):
    def send(messages):
        log.append(messages)
        return f"summary #{len(log)}"
    return send


def test_main_batch_fixed_window_vs_exp_fill():
    """main 배치는 고정 창(hypav3.ts:1059~1062): 창 안에 스킵이 있으면
    배치가 작아지고 다음 배치는 창 끝에서 시작. exp(293~296)는 찰 때까지
    스캔 — 같은 입력에서 배치 구성원이 갈라진다."""
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3,
                          memory_tokens_ratio=0.2, extra_summarization_ratio=0.1)
    hist = _mk_chats(12)
    hist[2]["content"] = "   "            # 창(0..3) 안 스킵 대상
    calls = []
    send = _fake_send_factory(calls)
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, send, max_ctx=tot // 2, max_response=100,
        embed=fake_embed)
    assert err is None
    first = data["summaries"][0]["chatMemos"]
    assert first == ["m0", "m1", "m3"]     # 고정 창 0..3에서 m2 스킵 → 3개
    # exp라면 m4까지 채웠을 것 — 갈라짐 고정.
    # current를 max_ctx 살짝 위로 잡아야 배치 1개 후 target에 닿는다
    # (tot*2면 보호구간 min_messages 에러로 배치가 통째로 버려진다)
    exp_batches, _, _, exp_err = hypa.plan_batches(
        hypa.to_risu_chats(hist), 0, int(tot * 1.1), tot, S)
    assert exp_err is None
    assert [c["memo"] for c in exp_batches[0]] == ["m0", "m1", "m3", "m4"]


def test_main_zero_summary_turn_inserts_empty_card():
    """요약 미발동 턴에도 빈 카드 삽입 (hypav3.ts:1188~1199) — exp는 None."""
    S = hypa.HypaSettings()
    hist = _mk_chats(4, tok_words=5)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, None, max_ctx=100_000, max_response=100)
    assert err is None
    assert mem == hypa.wrap_xml(hypa.MEMORY_TAG, "")
    assert data["summaries"] == []


def test_main_summarize_failure_keeps_earlier_summaries():
    """즉시 요약(1151~1159)이라 중간 실패 시 앞 배치 요약은 data에 남는다 —
    exp의 전량 폐기(계획 후 일괄)와 갈라짐."""
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3)
    hist = _mk_chats(16)
    n_calls = [0]
    def send(messages):
        n_calls[0] += 1
        if n_calls[0] >= 2:
            raise hypa.HypaError("boom")
        return "first summary"
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, send, max_ctx=tot // 3, max_response=100)
    assert err is not None
    assert len(data["summaries"]) == 1     # 첫 배치는 살아남음


def test_main_metrics_stored_in_data():
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3)
    hist = _mk_chats(16)
    calls = []
    tot = sum(hypa.tok_chat({"role": "u", "content": m["content"]}) for m in hist)
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, _fake_send_factory(calls),
        max_ctx=tot // 2, max_response=100, embed=fake_embed)
    assert err is None
    assert set(data["metrics"]) == {
        "lastImportantSummaries", "lastRecentSummaries",
        "lastSimilarSummaries", "lastRandomSummaries"}


def test_main_over_summarize_break_leaves_batch_unsummarized():
    """과요약 직전 중단 (hypav3.ts:1129~1139): current ≤ max_ctx인 반복에서
    current − batch_tokens < target이면 요약 없이 break, startIdx 유지 —
    스펙 §8-2. max_ctx는 조건이 성립하는 값을 탐색으로 잡는다."""
    import math
    S = hypa.HypaSettings(max_chats_per_summary=4, query_chat_count=3,
                          extra_summarization_ratio=0.1, memory_tokens_ratio=0.2)
    hist = _mk_chats(18)
    chats = hypa.to_risu_chats(hist)
    toks = [hypa.tok_chat(c) for c in chats]
    b1 = sum(toks[:4])
    total = 50 + sum(toks)
    # 발동(current0 > max_ctx)하되 배치 1개 후 (target, max_ctx] 구간에 착지
    max_ctx = next(m for m in range(300, 5000)
                   if (c0 := total + math.floor(m * 0.2)) > m
                   and m >= c0 - b1 > 0.9 * m)
    tr = []
    mem, kept, ksm, data, err = hypa_main.hypa_step_main(
        hist, 0, S, {"summaries": []}, _fake_send_factory([]), max_ctx,
        max_response=100, embed=fake_embed, trace=tr)
    assert err is None
    assert len(data["summaries"]) == 1     # 두 번째 배치는 요약되지 않음
    assert ksm == 4                        # startIdx가 두 번째 배치 앞에서 멈춤
    reasons = [e["reason"] for e in tr if e["ev"] == "batch_stop"]
    assert reasons[-1] == "over_summarize_break"
