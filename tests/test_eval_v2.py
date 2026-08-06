"""평가 v2 — 충실도/디렉터 순수 함수 (EVAL2.md, 실캡처 형태 기준)."""
import json

from benchmarks.eval.director import (
    DirFact,
    Ledger,
    eligible,
    extract_facts,
    make_false_premise,
    probe_plan,
)
from benchmarks.eval.fidelity import (
    check_wire_shape,
    compare_with_corpus,
    corpus_signature,
)


def _fake_llm(reply):
    def f(system, user):
        return reply
    return f


def test_extract_facts_parses_lines_and_skips_garbage():
    out = extract_facts(
        _fake_llm("exact|250골드|한결의 남은 소지금은 250골드\n"
                  "relation|연인|한결과 리사는 연인 사이\n"
                  "이상한 줄 형식\n"
                  "event|보름달 축제|보름달에 축제 동행 약속"),
        "u", "a", turn_no=7)
    assert [f.kind for f in out] == ["exact", "relation", "event"]
    assert out[0].value == "250골드" and out[0].turn == 7
    assert all(not f.probed for f in out)


def test_ledger_roundtrip_and_unprobed_filter():
    led = Ledger()
    led.add([DirFact(fid="f1", kind="exact", value="250", text="잔액", turn=3),
             DirFact(fid="f2", kind="relation", value="연인", text="관계", turn=5,
                     probed=True)])
    led2 = Ledger.from_rows(led.to_rows())
    assert [f.fid for f in led2.unprobed()] == ["f1"]
    assert [f.fid for f in led2.unprobed(kind="relation")] == []


def _wire(pairs=2, greeting=True, tail_system=True):
    out = [{"role": "system", "content": "카드 전문 + 로어 + 프리셋"}]
    if greeting:
        out.append({"role": "assistant", "content": "어서 와요."})
    for i in range(pairs):
        out.append({"role": "user", "content": f"질문{i}"})
        out.append({"role": "assistant", "content": f"답{i}"})
    out.append({"role": "user", "content": "새 질문"})
    if tail_system:
        out.append({"role": "system", "content": "PHI"})
    return out


def test_capture_shape_passes():
    assert check_wire_shape(_wire()) == []
    assert check_wire_shape(_wire(greeting=False)) == []      # 트림 후 인사 소멸
    assert check_wire_shape(_wire(tail_system=False)) == []


def test_mid_conversation_system_is_violation():
    msgs = _wire()
    msgs.insert(4, {"role": "system", "content": "중간 주입"})
    assert any("중간에 system" in v for v in check_wire_shape(msgs))


def test_split_leading_systems_is_violation():
    msgs = _wire()
    msgs.insert(1, {"role": "system", "content": "로어 분리"})
    # 선두 system 2개 = 두 번째가 중간 system으로 잡힌다
    assert check_wire_shape(msgs)


def test_last_message_must_be_user():
    msgs = _wire(tail_system=False)
    msgs.append({"role": "assistant", "content": "선응답"})
    assert any("마지막 메시지가 user가 아님" in v for v in check_wire_shape(msgs))


def test_consecutive_same_role_is_violation():
    msgs = _wire(tail_system=False)
    msgs.append({"role": "user", "content": "연속 발화"})
    assert any("역할 연속 중복" in v for v in check_wire_shape(msgs))


def test_client_cache_control_is_violation():
    msgs = _wire()
    msgs[0]["cache_control"] = {"type": "ephemeral"}
    assert any("cache_control" in v for v in check_wire_shape(msgs))


def test_block_content_is_violation():
    msgs = _wire()
    msgs[0]["content"] = [{"type": "text", "text": "블록 형식"}]
    assert any("OpenAI-compat" in v for v in check_wire_shape(msgs))


def test_unresolved_macro_is_violation():
    msgs = _wire()
    msgs[0]["content"] += "\n오늘은 {{date}}"
    assert any("미해석 매크로" in v for v in check_wire_shape(msgs))


def test_empty_messages():
    assert check_wire_shape([]) == ["메시지가 비어 있음"]


def _write_capture(path, lead, tail):
    path.write_text(json.dumps({"body": {"messages": [
        {"role": "system", "content": lead},
        {"role": "user", "content": "안녕"},
        {"role": "system", "content": tail},
    ]}}))


def test_corpus_signature_and_compare(tmp_path):
    for i in range(3):
        _write_capture(tmp_path / f"req-00{i}.json", "선두 프리셋", "꼬리 PHI")
    sig = corpus_signature(str(tmp_path))
    assert sig["n"] == 3
    assert len(sig["leading"]) == 1 and len(sig["trailing"]) == 1

    ok = [{"role": "system", "content": "선두 프리셋"},
          {"role": "user", "content": "질문"},
          {"role": "system", "content": "꼬리 PHI"}]
    assert compare_with_corpus(ok, sig) == []

    bad = [dict(m) for m in ok]
    bad[0]["content"] = "다른 선두"
    assert any("leading" in v for v in compare_with_corpus(bad, sig))


def _led():
    led = Ledger()
    led.add([DirFact(fid=f"f{i}", kind=k, value=f"v{i}", text=f"사실{i}", turn=t)
             for i, (k, t) in enumerate([("exact", 2), ("exact", 30),
                                         ("relation", 4), ("event", 6)])])
    return led


def test_eligible_only_outside_window():
    led = _led()
    got = [f.fid for f in eligible(led, window_start_turn=10)]
    assert got == ["f0", "f2", "f3"]          # turn 30(f1)은 창 안
    assert [f.fid for f in eligible(led, 10, kind="relation")] == ["f2"]


def test_probe_plan_marks_probed_and_respects_want():
    led = _led()
    plan = probe_plan(led, window_start_turn=10,
                      want={"recall": 1, "relation": 1, "false": 1})
    types = [t for t, _ in plan]
    assert types == ["recall", "relation", "false"]
    assert len({f.fid for _, f in plan}) == 3          # 사실 중복 출제 없음
    assert len(eligible(led, 10)) == 0                 # 전부 probed 처리


def test_probe_plan_recent_uses_in_window_facts():
    led = _led()
    plan = probe_plan(led, window_start_turn=10, want={"recent": 2})
    assert [f.fid for _, f in plan] == ["f1"]          # 창 안(turn 30)만


def test_false_premise_corrupts_value():
    q, wrong = make_false_premise(
        _fake_llm("질문: 그때 350골드 남았댔지?\n오염값: 350골드"),
        DirFact(fid="x", kind="exact", value="250골드", text="잔액", turn=1))
    assert "350" in q and wrong == "350골드"
