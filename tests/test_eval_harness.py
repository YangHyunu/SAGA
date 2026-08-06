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
