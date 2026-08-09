"""평가 하네스 — 대본/오라클/변형 조립 순수 함수 (스펙 §9)."""
import json

from benchmarks.eval.script import BEATS, PROBES, freeze_script, load_script


def test_script_shape_and_probe_positions():
    assert len(BEATS) == 30
    turns = [p.turn for p in PROBES]
    assert turns == [21, 23, 25, 27, 28, 29]
    assert all(0 <= t < len(BEATS) for t in turns)
    # 지뢰(0~15)와 프로브(21~29) 사이 간격이 트림 윈도우(8 pair)보다 큼
    assert min(turns) - 15 >= 6


def test_probe_expectations_are_nonempty():
    for p in PROBES:
        assert p.expect and all(group for group in p.expect)
    recall = [p for p in PROBES if p.recall]
    assert len(recall) == 1 and recall[0].turn == 29
    assert len(recall[0].expect) == 5          # 회상 항목 5개


def test_freeze_and_load_roundtrip(tmp_path):
    turns = [{"turn": 0, "user_text": "안녕, 나는 한결이야."}]
    p = tmp_path / "script.json"
    freeze_script(p, turns)
    assert load_script(p) == turns
    assert json.loads(p.read_text())           # 평문 JSON


# ------------------------------------------------------------------ #
# 결정론 오라클
# ------------------------------------------------------------------ #

from benchmarks.eval.oracle import score_reply
from benchmarks.eval.script import Probe


def test_score_full_partial_miss():
    p = Probe(0, "이름·나이", [["한결"], ["27", "스물일곱", "이십칠"]])
    assert score_reply("한결님, 스물일곱이시죠.", p)["hit"] == "full"
    r = score_reply("한결님이라는 건 기억해요.", p)
    assert r["hit"] == "partial" and r["matched"] == 1 and r["total"] == 2
    assert score_reply("글쎄요, 기억나지 않네요.", p)["hit"] == "miss"


def test_score_ignores_whitespace():
    p = Probe(0, "선물", [["세 개"]])
    assert score_reply("사과를 세\n개 주셨죠.", p)["hit"] == "full"


def test_score_recall_counts_groups():
    p = Probe(0, "회상", [["한결"], ["왼손잡이"], ["보름달"]], recall=True)
    r = score_reply("한결님은 왼손잡이시고...", p)
    assert r["hit"] == "partial" and r["matched"] == 2 and r["total"] == 3


def test_korean_numeral_expectation_matches():
    p = Probe(0, "잔액", [["250", "이백오십"]])
    assert score_reply("이백오십 남으셨을 거예요.", p)["hit"] == "full"


def test_statbar_header_does_not_score():
    # 카드 스탯바([<img=..> | 한결 | 27세 ...])가 모든 응답에 붙어
    # 이름·나이 프로브를 공짜 적중시킨다 — 선두 브래킷 블록은 채점 제외
    p = Probe(0, "이름·나이", [["한결"], ["27", "스물일곱"]])
    bar = '[<img="Land"> | 한결 | 27세 여행자 | 무소속 ] --- '
    assert score_reply(bar + "글쎄요, 기억나지 않네요.", p)["hit"] == "miss"
    assert score_reply(bar + "한결님, 스물일곱이시죠.", p)["hit"] == "full"


# ------------------------------------------------------------------ #
# 변형별 조립
# ------------------------------------------------------------------ #

from benchmarks.eval.variants import prepare_request, retrieve_turns, trim_window


def _hist(pairs):
    h = []
    for i in range(pairs):
        h.append({"role": "user", "content": f"질문{i} 사과 이야기"})
        h.append({"role": "assistant", "content": f"답{i}"})
    h.append({"role": "user", "content": "마지막 질문"})
    return h


def test_trim_window_keeps_last_pairs_and_trailing_user():
    out = trim_window(_hist(12), w=8)
    assert out[0]["content"] == "질문4 사과 이야기"    # 앞 4 pair 잘림
    assert out[-1]["content"] == "마지막 질문"
    assert trim_window(_hist(3), w=8) == _hist(3)      # 짧으면 그대로


def test_retrieve_turns_is_deterministic_topk():
    h = _hist(12)
    h[0]["content"] = "질문0 보름달 축제 약속"
    got = retrieve_turns(h, "보름달 약속 기억해?", k=2)
    assert got == retrieve_turns(h, "보름달 약속 기억해?", k=2)
    assert any("보름달" in g for g in got)
    assert len(got) <= 2
    # 윈도우 안 pair는 검색 대상 아님 (원문이 이미 있음)
    assert not any("질문11" in g for g in got)


def test_retrieve_turns_with_greeting_history():
    # 실전(run2)은 greeting이 history[0]에 assistant로 온다 — 고정 스텝 pair
    # 스캔이 전부 assistant 인덱스에 걸려 발췌 0건이 되던 회귀 케이스.
    h = [{"role": "assistant", "content": "어서 와요."}] + _hist(12)
    h[1]["content"] = "질문0 보름달 축제 약속"
    got = retrieve_turns(h, "보름달 약속 기억해?", k=2)
    assert any("보름달" in g for g in got)


def test_retrieve_turns_honors_caller_window():
    # run2는 token_trim으로 실창을 자른다 — 호출자 창을 넘기면 내부
    # trim_window 대신 그 경계 밖에서만 발췌한다.
    h = _hist(12)
    win = h[-5:]                               # 마지막 2 pair + 진행 중 user
    got = retrieve_turns(h, "질문9", k=3, window=win)
    assert any("질문9" in g for g in got)
    # 기본 경계(8 pair)에서는 질문9가 창 안이라 발췌되지 않는다
    assert not any("질문9" in g for g in retrieve_turns(h, "질문9", k=3))


def test_prepare_request_variants_differ():
    from benchmarks.cardsim.lorebook import Card
    card = Card(name="리사", description="너는 리사다.", post_history="",
                greeting="어서 와요.")
    h = _hist(12)
    full = prepare_request("vanilla", card, h)
    trimmed = prepare_request("trim", card, h)
    retr = prepare_request("retrieval", card, h)
    assert len(full) > len(trimmed)
    assert "질문0" in json.dumps(full, ensure_ascii=False)
    assert "질문0" not in json.dumps(trimmed, ensure_ascii=False)
    assert "[과거 대화 발췌]" in retr[-1]["content"]
    assert prepare_request("dreaming", card, h) == trimmed  # 전송분 동일, 차이는 프록시


def test_prepare_request_merges_leading_systems():
    # 실제 RisuAI 와이어는 선두 system 하나에 설명+로어를 병합한다 (corpus 실측).
    # 분리 전송하면 lore_shift(첫 system만 처리)가 keyed를 못 걷어낸다.
    from benchmarks.cardsim.lorebook import Card, LoreEntry
    card = Card(name="리사", description="너는 리사다.", post_history="",
                greeting="어서 와요.",
                lore=[LoreEntry(name="l1", keys=[], content="상시 설정.",
                                order=0, constant=True, tokens=3)])
    out = prepare_request("trim", card, _hist(2))
    assert out[0]["role"] == "system"
    assert out[1]["role"] != "system"                  # 선두 system은 정확히 1개
    assert "너는 리사다." in out[0]["content"]
    assert "상시 설정." in out[0]["content"]


# ------------------------------------------------------------------ #
# 드라이버 — 결과 조립 (네트워크 없는 순수 함수만)
# ------------------------------------------------------------------ #

from benchmarks.eval.run import build_result


def test_build_result_scores_probes_and_totals():
    turns = []
    for i in range(30):
        reply = "기억해요, 한결님. 스물일곱이시죠." if i == 21 else f"응답{i}"
        turns.append({"turn": i, "user": f"발화{i}", "reply": reply,
                      "prompt": 100, "cached": 90 if i else 0, "write": 0,
                      "cost": 0.001, "sec": 1.0})
    r = build_result("trim", "s1", "m", turns)
    assert r["totals"]["oracle_full"] == 1            # 21번 프로브만 적중
    assert r["totals"]["cost"] == 0.03
    assert abs(r["totals"]["avg_hit_t2"] - 90.0) < 1e-6
    p21 = next(p for p in r["probes"] if p["turn"] == 21)
    assert p21["hit"] == "full" and "한결" in p21["reply"]
    recall = next(p for p in r["probes"] if p["turn"] == 29)
    assert recall["matched"] == 0 and recall["total"] == 5


# ------------------------------------------------------------------ #
# 비교 리포트
# ------------------------------------------------------------------ #

from benchmarks.eval.report import render_report


def test_render_report_table_and_audit_section():
    res = [{"variant": "dreaming", "session": "a", "model": "m",
            "turns": [],
            "probes": [{"label": "이름·나이", "hit": "full", "matched": 2,
                        "total": 2, "turn": 21, "reply": "한결님이시죠"}],
            "totals": {"cost": 0.21, "avg_hit_t2": 93.0, "avg_sec": 2.1,
                       "oracle_full": 1, "oracle_partial": 0,
                       "recall": "4/5"}}]
    md = render_report(res)
    assert "| dreaming |" in md and "4/5" in md and "93.0" in md
    assert "한결님이시죠" in md                     # 수동 감사용 원문 병기
