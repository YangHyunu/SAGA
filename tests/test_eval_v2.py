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
from benchmarks.eval.judge_check import agreement
from benchmarks.eval.scoring import decompose_miss, judge_pass, oracle_pass
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


def test_oracle_pass_ignores_whitespace():
    assert oracle_pass("이백 오십 골드 남았죠", "이백오십")
    assert not oracle_pass("기억 안 나요", "250")


def test_judge_false_premise_inverts():
    ok = judge_pass(_fake_llm("N"), "false", "잔액 250", "250골드",
                    "350골드 남았댔지?", "아뇨, 250골드였어요.",
                    wrong_value="350골드")
    assert ok["pass"] is True                  # 오염값 단정 안 함 = 통과
    bad = judge_pass(_fake_llm("Y"), "false", "잔액 250", "250골드",
                     "350골드 남았댔지?", "맞아요 350골드.",
                     wrong_value="350골드")
    assert bad["pass"] is False


def test_decompose_miss_checks_dreaming_storage(tmp_path):
    d = tmp_path / "sess" / "facts"
    d.mkdir(parents=True)
    (d / "a.json").write_text(json.dumps(
        {"claim": "한결의 잔액은 250골드"}, ensure_ascii=False))
    f_in = DirFact(fid="a", kind="exact", value="250골드", text="잔액", turn=1)
    f_out = DirFact(fid="b", kind="exact", value="은반지", text="선물", turn=2)
    assert decompose_miss(tmp_path, "sess", f_in) == "utilization_fail"
    assert decompose_miss(tmp_path, "sess", f_out) == "storage_fail"


def test_agreement_counts_and_lists_disagreements():
    rows = [
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "250이요", "human": True},
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "몰라요", "human": False},
        {"ptype": "recall", "fact_text": "잔액 250", "expected_value": "250",
         "question": "얼마 남았지?", "reply": "500이요", "human": False},
    ]
    r = agreement(rows, judge=_fake_llm("Y"))     # 전부 Y로 판정하는 가짜 judge
    assert r["n"] == 3 and r["agree"] == 1
    assert r["disagrees"] == [1, 2]
    assert abs(r["rate"] - 1 / 3) < 1e-9


# ---- preset2wire (뮈토스 6.2 조립) ----

def test_resolve_when_tis_tisnot_else_and_nesting():
    from benchmarks.eval.preset2wire import resolve_when
    t = "A{{#when::mode::tis::0}}RP{{:else}}소설{{/when}}B"
    assert resolve_when(t, {"mode": "0"}) == "ARPB"
    assert resolve_when(t, {"mode": "2"}) == "A소설B"
    nested = ("{{#when::a::tis::1}}X{{#when::b::tisnot::0}}Y{{/when}}Z{{/when}}")
    assert resolve_when(nested, {"a": "1", "b": "1"}) == "XYZ"
    assert resolve_when(nested, {"a": "1", "b": "0"}) == "XZ"
    assert resolve_when(nested, {"a": "0", "b": "1"}) == ""


def _mini_preset():
    return {"promptTemplate": [
        {"type": "plain", "role": "system", "text": "규칙."},
        {"type": "plain", "role": "system",
         "text": "{{#when::nsfw::tis::1}}성인 지침{{/when}}"},
        {"type": "description", "role": "system",
         "innerFormat": "### 캐릭터\n{{slot}}"},
        {"type": "chat", "role": "system", "rangeStart": 0, "rangeEnd": -2},
        {"type": "memory", "role": "system", "innerFormat": "### 기억\n{{slot}}"},
        {"type": "chat", "role": "system", "rangeStart": -2, "rangeEnd": "end"},
        {"type": "plain", "role": "user", "text": "프리필 선언"},
        {"type": "plain", "role": "bot", "text": "프리필 승인"},
        {"type": "plain", "role": "user", "text": "렌더링 기준 {{char}}"},
    ]}


def _hist(n=3):
    out = []
    for i in range(n):
        out.append({"role": "user", "content": f"u{i}"})
        out.append({"role": "assistant", "content": f"a{i}"})
    return out[:-1] + [{"role": "user", "content": f"u{n-1}"}] if False else out


def test_assemble_merges_systems_and_splices_history():
    from benchmarks.eval.preset2wire import assemble
    hist = _hist(3)
    hist[-1] = {"role": "user", "content": "u-last"}      # 마지막은 user
    msgs = assemble(_mini_preset(), {"nsfw": "1"}, hist,
                    card={"description": "소연"}, char_name="소연")
    assert msgs[0]["role"] == "system"
    assert "규칙." in msgs[0]["content"] and "성인 지침" in msgs[0]["content"]
    assert "### 캐릭터\n소연" in msgs[0]["content"]      # 선두 병합
    # 히스토리 splice: 앞 4개 + 뒤 2개 사이에 memory 없음(빈 슬롯 드랍)
    roles = [m["role"] for m in msgs]
    assert roles.count("system") == 1                     # memory 드랍 + 병합
    assert msgs[-1]["content"] == "렌더링 기준 소연"     # bot→assistant 프리필 체인
    assert msgs[-2]["role"] == "assistant"


def test_assemble_nsfw_off_drops_section_and_memory_injects_mid():
    from benchmarks.eval.preset2wire import assemble
    hist = _hist(3)
    hist[-1] = {"role": "user", "content": "u-last"}
    msgs = assemble(_mini_preset(), {"nsfw": "0"}, hist, memory="과거 요약")
    assert "성인 지침" not in msgs[0]["content"]
    mems = [m for m in msgs if "### 기억" in m["content"]]
    assert len(mems) == 1 and mems[0]["role"] == "system"


# ---- run2 순수 함수 ----

def test_token_trim_cuts_at_pair_boundary_and_reports_start_turn():
    from benchmarks.eval.run2 import token_trim
    h = []
    for i in range(10):
        h.append({"role": "user", "content": "가" * 100})
        h.append({"role": "assistant", "content": "나" * 100})
    h.append({"role": "user", "content": "새 질문"})
    win, start = token_trim(h, budget=300, count_fn=lambda t: len(t))
    assert win[0]["role"] == "user"                    # pair 경계 절단
    assert start == 10 - (len(win) - 1) // 2           # 남은 pair 수로 역산
    full, s0 = token_trim(h, budget=10**9, count_fn=len)
    assert full == h and s0 == 0


def test_token_trim_drops_greeting_once_trimming():
    from benchmarks.eval.run2 import token_trim
    h = [{"role": "assistant", "content": "인사" * 50}]
    for i in range(5):
        h.append({"role": "user", "content": "가" * 100})
        h.append({"role": "assistant", "content": "나" * 100})
    win, start = token_trim(h, budget=450, count_fn=len)
    assert all(m["content"] != "인사" * 50 for m in win)   # 인사 소멸
    assert win[0]["role"] == "user" and start == 3


def test_probe_schedule_covers_five_types():
    from benchmarks.eval.run2 import probe_schedule
    sched = probe_schedule(40)
    types = [t for t in sched if t]
    assert len(sched) == 40
    assert types.count("recall") >= 8
    for t in ("relation", "false", "update", "recent"):
        assert sched.count(t) >= 2
    assert sched.count(None) >= 10                     # 필러 존재
