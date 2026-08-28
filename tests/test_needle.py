"""니들 생존 오라클 (플랜 Task 6)."""

from benchmarks.fixed_haystack import needle


LEDGER = [
    {"fid": "a", "value": "은화 서른 닢", "turn": 1, "probed": True},
    {"fid": "b", "value": "왼팔 흉터", "turn": 2, "probed": True},
    {"fid": "c", "value": "생강과 꿀", "turn": 3, "probed": False},
    {"fid": "d", "value": "미래사건", "turn": 9, "probed": True},
    {"fid": "e", "value": "x", "turn": 1, "probed": True},      # 길이 미달
]


def _result(memory, window_msgs, ksm, turn=5):
    return {"messages": [{"memo": f"m{i}", "role": "user", "content": c}
                         for i, c in enumerate(window_msgs)],
            "turns": [{"turn": turn, "memory_text": memory,
                       "kept_start_msg": ksm}]}


def test_memory_hit_counts_as_covered():
    r = needle.annotate(_result("은화 서른 닢을 약속했다", ["무관"], 0), LEDGER)
    n = r["turns"][0]["needle"]["probed"]
    assert n["in_memory"] == 1 and n["covered"] == 1


def test_window_hit_counts_even_without_memory():
    r = needle.annotate(_result("", ["왼팔 흉터가 보였다"], 0), LEDGER)
    n = r["turns"][0]["needle"]["probed"]
    assert n["in_memory"] == 0 and n["in_window"] == 1 and n["covered"] == 1


def test_kept_start_msg_excludes_dropped_messages():
    """압축으로 창 밖에 나간 원문은 세지 않는다."""
    r = needle.annotate(_result("", ["왼팔 흉터", "무관"], 1), LEDGER)
    assert r["turns"][0]["needle"]["probed"]["in_window"] == 0


def test_future_facts_are_not_eligible():
    r = needle.annotate(_result("미래사건", ["무관"], 0), LEDGER)
    # turn 5 시점 — turn 9 팩트는 후보에서 빠진다
    assert r["turns"][0]["needle"]["probed"]["n_eligible"] == 2


def test_short_values_filtered():
    r = needle.annotate(_result("x", ["x"], 0), LEDGER)
    n = r["turns"][0]["needle"]["probed"]
    assert n["n_eligible"] == 2          # a, b 만 (e는 길이 미달, d는 미래)


def test_all_bucket_includes_unprobed():
    r = needle.annotate(_result("생강과 꿀", ["무관"], 0), LEDGER)
    assert r["turns"][0]["needle"]["all"]["n_eligible"] == 3
    assert r["turns"][0]["needle"]["all"]["in_memory"] == 1


def test_memory_only_isolates_retrieval_contribution():
    """창에도 있으면 검색 기여가 아니다 — memory_only가 진짜 검색 몫."""
    r = needle.annotate(_result("은화 서른 닢", ["은화 서른 닢"], 0), LEDGER)
    n = r["turns"][0]["needle"]["probed"]
    assert n["in_memory"] == 1 and n["memory_only"] == 0


def test_summarize_over_run():
    r = needle.annotate(_result("은화 서른 닢", ["무관"], 0), LEDGER)
    s = needle.summarize(r)
    assert s["turns_scored"] == 1 and s["coverage_last"] == 0.5
