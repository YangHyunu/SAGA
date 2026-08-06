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
