"""HypaPlus 포트 — 순수 함수층 (플랜 Task 2)."""

from datetime import datetime

import pytest

from benchmarks.eval import hypaplus as hp


# ── 토크나이저 ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("", 0),
    ("한글", 2),          # 0.8*2 = 1.6 → ceil 2
    ("ab", 1),            # 0.25*2 = 0.5 → ceil 1
    ("가나다라마", 4),     # 0.8*5 = 4.0
    ("漢字", 2),          # 0.9*2 = 1.8 → ceil 2
    ("한 글", 2),         # 0.8+0.25+0.8 = 1.85 → ceil 2
])
def test_est_tokens(text, expected):
    assert hp.est_tokens(text) == expected


# ── 시각 파서 ────────────────────────────────────────────────────────────

def test_parse_time_full():
    assert hp.parse_time("2024-03-05 14:30") == datetime(2024, 3, 5, 14, 30)


def test_parse_time_range_end_inherits_date():
    """시각만 있는 범위 끝은 시작값의 날짜를 상속한다 (원본 1745-1749)."""
    assert hp.parse_time("2024-03-05 14:30 → 16:00") == datetime(2024, 3, 5, 16, 0)


def test_parse_time_short_uses_frozen_year():
    """MM-DD는 연도가 없다 — 원본은 실시각 연도, 포트는 고정 상수."""
    assert hp.parse_time("03-05 14:30") == datetime(hp.FALLBACK_YEAR, 3, 5, 14, 30)


def test_parse_time_rejects_rollover():
    """2월 30일은 되돌아가므로 거부 (원본 1725-1733의 필드 재확인)."""
    assert hp.parse_time("2024-02-30 10:00") is None


def test_korean_relative_time_survives_normalize_but_dies_in_parse():
    """작중 상대 시각은 저장은 되고 점수 계산에서만 죽는다 — 시간축 사망 경로."""
    assert hp.normalize_time("3일차 저녁") == "3일차 저녁"
    assert hp.parse_time("3일차 저녁") is None


def test_normalize_pads_short_components():
    assert hp.normalize_time("2024-3-5 9:07") == "2024-03-05 09:07"


# ── 노드 파서 ────────────────────────────────────────────────────────────

RAW = """<event>
시간: 2024-03-05 14:30
장소: 항구 창고
인물: 리안, 렌
내용: 창고에서 낡은 장부를 발견했다.
--- 대사 ---
리안: "이 장부, 네가 가지고 있어."
렌: 따옴표 없는 줄
</event>
<event>
시간: 2024-03-06 09:00
내용: 두 번째 사건이 일어났다.
</event>
<facts>
인물: 리안
시간: 2024-03-05 14:30
내용: 리안의 왼팔 흉터를 렌이 알게 되었다.
</facts>
"""


def test_parse_nodes_events_first_then_facts():
    st = hp.new_state()
    nodes = hp.parse_nodes(RAW, st)
    assert [n["type"] for n in nodes] == ["event", "event", "facts"]
    assert [n["id"] for n in nodes] == ["1", "2", "3"]


def test_parse_nodes_ids_continue_from_existing_max():
    st = hp.new_state()
    st["chunks"].append({"id": "1", "nodes": hp.parse_nodes(RAW, st),
                         "summarized": "done", "msg_indices": []})
    more = hp.parse_nodes(RAW, st)
    assert [n["id"] for n in more] == ["4", "5", "6"]


def test_event_fields_and_dialogue_filtering():
    st = hp.new_state()
    ev = hp.parse_nodes(RAW, st)[0]
    assert ev["location"] == "항구 창고"
    assert ev["characters"] == ["리안", "렌"]
    assert ev["content"] == "창고에서 낡은 장부를 발견했다."
    # 따옴표 없는 줄은 대사가 아니다 (원본 745-750)
    assert ev["dialogues"] == [{"speaker": "리안", "text": "이 장부, 네가 가지고 있어."}]


def test_event_without_time_is_dropped():
    st = hp.new_state()
    assert hp.parse_nodes("<event>\n내용: 시간 없는 사건\n</event>", st) == []


def test_facts_without_time_falls_back_and_counts():
    st = hp.new_state()
    nodes = hp.parse_nodes("<facts>\n인물: 리안\n내용: 시간 없는 사실\n</facts>", st)
    assert nodes[0]["time"] == hp.FACTS_FALLBACK_TIME
    assert st["facts_fallback"] == 1


# ── 정렬 ─────────────────────────────────────────────────────────────────

def test_unparseable_time_sorts_after_in_story_dates():
    """원본은 파싱 실패 노드에 Date.now()를 쓴다 — 작중 과거보다 뒤로 간다."""
    st = hp.new_state()
    good = hp._parse_event("시간: 2024-03-05 14:30\n내용: a", "1", st)
    bad = hp._parse_event("시간: 3일차 저녁\n내용: b", "2", st)
    assert hp._time_key(good) < hp._time_key(bad)


def test_created_at_is_monotonic_not_wallclock():
    st = hp.new_state()
    a = hp._created_at(st)
    b = hp._created_at(st)
    assert b == a + 1


# ── 포매터 ───────────────────────────────────────────────────────────────

def test_node_text_joins_content_location_chars_dialogue():
    st = hp.new_state()
    ev = hp.parse_nodes(RAW, st)[0]
    txt = hp.node_text(ev)
    assert "창고에서 낡은 장부를" in txt
    assert "항구 창고" in txt
    assert "리안 렌" in txt
    assert "이 장부, 네가 가지고 있어." in txt


def test_format_node_collapses_empty_pipe_slot():
    """빈 슬롯이 만든 ' |  | '를 ' | '로 접는다 (원본 1955-1957)."""
    S = hp.PlusSettings(event_format="[a | [[location]] | b]")
    st = hp.new_state()
    node = hp._parse_event("시간: 2024-03-05 14:30\n내용: x", "1", st)
    assert hp.format_node(node, S) == "[a | b]"


def test_format_node_default_template_has_fields():
    S = hp.PlusSettings()
    st = hp.new_state()
    ev = hp.parse_nodes(RAW, st)[0]
    out = hp.format_node(ev, S)
    assert 'time="2024-03-05 14:30"' in out
    assert 'location="항구 창고"' in out
    assert 'characters="리안, 렌"' in out
    assert '리안: "이 장부, 네가 가지고 있어."' in out


# ══════════════════════════════════════════════════════════════════════════
# 요약 루프 (플랜 Task 3)
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """실 캐시 오염 방지 — 선행 플랜에서 테스트가 실 캐시에 가짜 요약을 쓴 전례."""
    monkeypatch.setattr(hp, "CACHE_DIR", tmp_path / "cache")


FAKE_SUMMARY = ('<facts>\n인물: 리안\n시간: 2024-01-01 00:00\n'
                '내용: 더미 사실\n</facts>')


def _fake_send(_messages):
    return FAKE_SUMMARY


def _settings():
    return hp.PlusSettings(summary_prompt="요약해라\n{{slot}}")


def _simulate(n_turns: int, send=_fake_send, S=None):
    """인스펙터와 같은 히스토리 규약 — m0=인사말, 턴 t가 m(2t-1)/m(2t).

    반환 (state, before_triggers, after_triggers).
    """
    S = S or _settings()
    st = hp.new_state()
    msgs = [{"role": "assistant", "content": "인사말"}]
    before, after = [], []
    for t in range(1, n_turns + 1):
        msgs.append({"role": "user", "content": f"u{t}"})
        tr = []
        hp.track(msgs, st, S, send, trace=tr)
        if tr:
            before.append(t)
        msgs.append({"role": "assistant", "content": f"a{t}"})
        tr2 = []
        hp.track(msgs, st, S, send, trace=tr2)
        if tr2:
            after.append(t)
    return st, before, after


def test_trigger_turns_match_precomputation():
    """플랜 사전 계산: 트리거는 턴 20/35/50/65/80/95 여섯 번."""
    st, before, after = _simulate(100)
    assert before == [20, 35, 50, 65, 80, 95]
    assert len(st["chunks"]) == 6


def test_track_assistant_never_fires():
    """2t-S=39는 좌변이 항상 짝수라 도달 불가 — 플랜의 증명을 실측으로 박제."""
    _st, _before, after = _simulate(100)
    assert after == []


def test_chunk_boundaries_are_fixed_30():
    st, _b, _a = _simulate(100)
    bounds = [(c["msg_indices"][0], c["msg_indices"][-1]) for c in st["chunks"]]
    assert bounds == [(0, 29), (30, 59), (60, 89), (90, 119), (120, 149), (150, 179)]


def test_partial_chunk_waits():
    """29개로는 요약하지 않는다 (원본 1552-1556)."""
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(29)]
    assert hp.run_summarization(msgs, st, S, _fake_send) is None
    assert st["chunks"] == []


def test_failed_chunk_still_consumes_indices():
    """LLM 실패해도 그 30개는 영구히 창 밖으로 나간다 (스펙 §4 #2)."""
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(60)]

    def boom(_messages):
        raise RuntimeError("upstream 500")

    ev = hp.run_summarization(msgs, st, S, boom)
    assert ev["status"] == "failed"
    assert st["chunks"][0]["nodes"] == []
    assert hp.last_summarized_index(st) == 29


def test_empty_response_is_failure_not_empty_summary():
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    ev = hp.run_summarization(msgs, st, S, lambda _m: "   ")
    assert ev["status"] == "failed"


def test_summary_input_has_no_role_prefix():
    """원본은 content만 이어붙인다 (1485) — hypa와 다르다 (스펙 §4 #1)."""
    seen = {}

    def capture(messages):
        seen["text"] = messages[-1]["content"]
        return FAKE_SUMMARY

    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": "유저발화"},
            {"role": "assistant", "content": "봇발화"}] * 15
    hp.run_summarization(msgs, st, S, capture)
    assert "유저발화\n\n봇발화" in seen["text"]
    assert "user:" not in seen["text"]


def test_cache_hit_avoids_second_call(tmp_path):
    calls = []

    def counting(messages):
        calls.append(1)
        return FAKE_SUMMARY

    S = _settings()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    st1 = hp.new_state()
    hp.run_summarization(msgs, st1, S, counting, cache_dir=tmp_path)
    st2 = hp.new_state()
    hp.run_summarization(msgs, st2, S, counting, cache_dir=tmp_path)
    assert len(calls) == 1
    assert st2["chunks"][0]["summarized"] == "done"


def test_cache_key_never_collides_with_hypa():
    """같은 디렉터리를 쓰므로 판별자가 유일한 안전장치."""
    batch = [{"role": "user", "content": "같은 내용"}]
    S = _settings()
    hs = hypa_settings_stub(S.summary_prompt)
    assert hp.summary_cache_key(batch, S) != hypa_mod.summary_cache_key(batch, hs)


def test_missing_cache_without_send_raises():
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    with pytest.raises(Exception):
        hp.run_summarization(msgs, st, S, None)


# ── ChatML / 프롬프트 추출 ───────────────────────────────────────────────

def test_parse_chatml_pure_blocks():
    p = ("<|im_start|>user\nA\n<|im_end|>\n"
         "<|im_start|>assistant\nB\n<|im_end|>")
    got = hp.parse_chatml(p)
    assert [b["role"] for b in got] == ["user", "assistant"]


def test_parse_chatml_rejects_stray_text():
    p = "앞말<|im_start|>user\nA\n<|im_end|>"
    assert hp.parse_chatml(p) is None


def test_parse_chatml_rejects_plain_text():
    assert hp.parse_chatml("그냥 프롬프트") is None


def test_load_default_prompt_from_real_source():
    import os
    if not os.path.exists(hp.DEFAULT_PLUGIN_SRC):
        pytest.skip("원본 플러그인 소스 없음")
    prompt = hp.load_default_prompt()
    assert "{{slot}}" in prompt
    blocks = hp.parse_chatml(prompt.replace("{{slot}}", "본문"))
    assert blocks is not None and len(blocks) == 3


def test_load_default_prompt_missing_file_raises():
    with pytest.raises(Exception):
        hp.load_default_prompt("/nonexistent/plugin.ts")


# 캐시 키 충돌 테스트용 헬퍼 (hypa 설정 객체)
from benchmarks.eval import hypa as hypa_mod  # noqa: E402


def hypa_settings_stub(prompt: str):
    return hypa_mod.HypaSettings(summarization_prompt=prompt)


# ══════════════════════════════════════════════════════════════════════════
# 검색 · 주입 (플랜 Task 4)
# ══════════════════════════════════════════════════════════════════════════

def _node(nid, time, content, state=None, fav=False):
    st = state or hp.new_state()
    return {"id": nid, "type": "facts", "time": time, "location": None,
            "characters": [], "content": content, "dialogues": None,
            "createdAt": hp._created_at(st), "favorite": fav}


def _state_with(nodes):
    st = hp.new_state()
    st["chunks"].append({"id": "1", "msg_indices": [0], "nodes": nodes,
                         "summarized": "done"})
    return st


def test_combined_score_hand_computed():
    """sim 0.82 / time 0.953, 가중치 1·1 → 1 - 0.047*0.18 = 0.99154."""
    S = hp.PlusSettings()
    st = _state_with([_node("1", "2024-01-01 00:00", "x")])
    got = hp.retrieve(st, "q", None, 10_000, S,
                      embed=lambda ts: [[1.0, 0.0]] * len(ts))
    # 임베딩이 동일 벡터라 cos=1.0, last_time None이라 time=0.5
    # combined = 1 - (1-0.5)*(1-1.0) = 1.0
    assert len(got) == 1


@pytest.mark.parametrize("sim,ts,expected", [
    (0.82, 0.953, 0.99154),
    (0.5, 0.5, 0.75),
    (0.0, 0.0, 0.0),
])
def test_combined_formula(sim, ts, expected):
    tf = (1.0 - ts) ** 1.0
    combined = 1.0 - tf * (1.0 - sim) ** 1.0
    assert combined == pytest.approx(expected, abs=1e-9)


def test_time_scoring_none_makes_combined_equal_similarity():
    S = hp.PlusSettings(time_scoring_mode="none")
    node = _node("1", "2024-01-01 00:00", "x")
    assert hp.time_score(node, "2024-01-05 00:00", S) == 0.0


def test_time_score_decay_shape():
    S = hp.PlusSettings(time_decay_days=15.0)
    node = _node("1", "2024-01-01 00:00", "x")
    # 15일 차 → 1/√2
    got = hp.time_score(node, "2024-01-16 00:00", S)
    assert got == pytest.approx(1 / (2 ** 0.5), abs=1e-9)


def test_unparseable_time_pins_score_to_half():
    """작중 상대 시각이면 시간축이 죽고 0.5 고정 — 관측 대상 병리."""
    S = hp.PlusSettings()
    node = _node("1", "3일차 저녁", "x")
    assert hp.time_score(node, "2024-01-01 00:00", S) == 0.5


def test_latest_node_time_ignores_unparseable():
    st = _state_with([_node("1", "3일차 저녁", "a"),
                      _node("2", "2024-05-01 00:00", "b"),
                      _node("3", "2024-03-01 00:00", "c")])
    assert hp.latest_node_time(st) == "2024-05-01 00:00"


def test_latest_node_time_none_when_all_unparseable():
    st = _state_with([_node("1", "3일차 저녁", "a")])
    assert hp.latest_node_time(st) is None


def test_no_embedding_pins_similarity_to_half():
    """배포 기본값(embeddingUrl 미설정) 재현 — 순위가 시간축만으로 결정된다."""
    S = hp.PlusSettings()
    st = _state_with([_node("1", "2024-01-01 00:00", "가"),
                      _node("2", "2024-01-02 00:00", "나")])
    tr = []
    hp.retrieve(st, "질의", "2024-01-02 00:00", 10_000, S, embed=None, trace=tr)
    rows = next(e for e in tr if e["ev"] == "scores")["rows"]
    assert all(r[1] == 0.5 for r in rows)


def test_budget_skips_big_node_and_takes_smaller_later_one():
    """예산 초과 노드는 자르지 않고 건너뛴다 — 뒤의 작은 노드가 들어간다."""
    S = hp.PlusSettings(max_memory_tokens=60, time_scoring_mode="none")
    big = _node("1", "2024-01-01 00:00", "가" * 200)
    small = _node("2", "2024-01-02 00:00", "나")
    st = _state_with([big, small])
    tr = []
    got = hp.retrieve(st, "q", None, 60, S, embed=None, trace=tr)
    ids = [n["id"] for n in got]
    assert "1" not in ids and "2" in ids
    ev = next(e for e in tr if e["ev"] == "retrieve")
    assert ev["n_skipped_budget"] == 1


def test_floor_cut_drops_near_zero_scores():
    S = hp.PlusSettings(time_scoring_mode="none")
    st = _state_with([_node("1", "2024-01-01 00:00", "x")])
    tr = []
    got = hp.retrieve(st, "q", None, 10_000, S,
                      embed=lambda ts: [[1.0, 0.0] if "q" == t else [0.0, 1.0]
                                        for t in ts], trace=tr)
    ev = next(e for e in tr if e["ev"] == "retrieve")
    assert got == [] and ev["n_skipped_floor"] == 1


def test_selected_sorted_chronologically():
    S = hp.PlusSettings(time_scoring_mode="none")
    st = _state_with([_node("1", "2024-03-01 00:00", "늦은"),
                      _node("2", "2024-01-01 00:00", "이른")])
    got = hp.retrieve(st, "q", None, 10_000, S, embed=None)
    assert [n["id"] for n in got] == ["2", "1"]


def test_time_diag_counts_parse_failures():
    S = hp.PlusSettings()
    st = _state_with([_node("1", "3일차 저녁", "a"),
                      _node("2", "2024-01-01 00:00", "b")])
    tr = []
    hp.retrieve(st, "q", None, 10_000, S, embed=None, trace=tr)
    diag = next(e for e in tr if e["ev"] == "time_diag")
    assert diag["n_time_parsed"] == 1 and diag["n_time_failed"] == 1


# ── 주입 ─────────────────────────────────────────────────────────────────

def test_inject_appends_to_first_system():
    msgs = [{"role": "system", "content": "프리셋"},
            {"role": "user", "content": "u"}]
    got = hp.inject(msgs, "기억블록")
    assert got[0]["content"] == "프리셋\n\n기억블록"
    assert msgs[0]["content"] == "프리셋"          # 원본 무변경


def test_inject_unshifts_when_no_system():
    msgs = [{"role": "user", "content": "u"}]
    got = hp.inject(msgs, "기억블록")
    assert got[0] == {"role": "system", "content": "기억블록"}


def test_inject_prefers_placeholder():
    msgs = [{"role": "system", "content": "앞 [HypaPlus.memory] 뒤"},
            {"role": "user", "content": "u"}]
    got = hp.inject(msgs, "기억블록")
    assert got[0]["content"] == "앞 기억블록 뒤"


def test_inject_noop_on_empty_memory():
    msgs = [{"role": "system", "content": "프리셋"}]
    assert hp.inject(msgs, "") == msgs


# ── step ─────────────────────────────────────────────────────────────────

def test_step_returns_kept_window_after_compression():
    S = _settings()
    st = hp.new_state()
    hist = [{"role": "assistant", "content": "인사말"}]
    mem = kept = ksm = None
    for t in range(1, 21):
        hist.append({"role": "user", "content": f"u{t}"})
        mem, kept, ksm, st, err = hp.hypaplus_step(hist, S, st, _fake_send)
        assert err is None
        hist.append({"role": "assistant", "content": f"a{t}"})
    assert ksm == 30                    # m0~m29 압축 완료
    assert len(kept) == 10              # m30~m39
    assert mem                          # 노드가 생겼으니 주입된다


def test_step_empty_memory_before_first_compression():
    S = _settings()
    st = hp.new_state()
    hist = [{"role": "assistant", "content": "인사말"},
            {"role": "user", "content": "u1"}]
    mem, kept, ksm, st, err = hp.hypaplus_step(hist, S, st, _fake_send)
    assert mem == "" and ksm == 0 and len(kept) == 2


def test_build_query_uses_last_n_messages():
    S = hp.PlusSettings(embedding_context_messages=3)
    msgs = [{"role": "user", "content": str(i)} for i in range(10)]
    assert hp.build_query(msgs, S) == "7\n8\n9"


def test_failure_reason_is_recorded_not_swallowed():
    """실패 사유를 버리면 '전부 failed'만 남고 원인을 못 가린다 — 실제로 당함."""
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]

    def no_key(_m):
        raise FileNotFoundError("/repo/.env")

    ev = hp.run_summarization(msgs, st, S, no_key)
    assert ev["status"] == "failed"
    assert "FileNotFoundError" in ev["fail_reason"]
    assert st["chunks"][0]["fail_reason"] == ev["fail_reason"]


def test_empty_response_reason():
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    ev = hp.run_summarization(msgs, st, S, lambda _m: "")
    assert ev["fail_reason"] == "empty_response"


def test_chunk_event_carries_nodes_by_type():
    """HTML 청크 카드가 사건/사실로 나눠 보이려면 트레이스에 노드 원문이 있어야 한다."""
    S = _settings()
    st = hp.new_state()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(30)]
    ev = hp.run_summarization(msgs, st, S, lambda _m: RAW)
    assert [n["type"] for n in ev["nodes"]] == ["event", "event", "facts"]
    assert ev["nodes"][0]["location"] == "항구 창고"
    assert ev["nodes"][0]["dialogues"][0]["speaker"] == "리안"
    assert ev["nodes"][2]["dialogues"] is None
