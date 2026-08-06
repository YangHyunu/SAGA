"""SyncPath: 동기 경로 오케스트레이터 — 턴당 LLM 0콜 (스펙 §3.1)."""
from dreaming.records import Actor, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, render_knowledge


def _msgs(*texts):
    roles = ["user", "assistant"]
    out = [{"role": "system", "content": "너는 상인 리사다."}]
    out += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
    return out


# ------------------------------------------------------------------ #
# render_knowledge
# ------------------------------------------------------------------ #

def test_render_includes_state_pinned_facts_main_actors(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    ms.save_fact(Fact(claim="리사는 밀수품을 취급한다", pinned=True, status="confirmed"))
    ms.save_actor(Actor(names=["리사"], profile="시장 상인", tier="main"))
    text = render_knowledge(ms)
    assert "소지금: 450" in text
    assert "리사는 밀수품을 취급한다" in text
    assert "리사" in text


def test_render_excludes_provisional_unpinned_and_extras(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.save_fact(Fact(claim="잠정 사실", status="provisional"))
    ms.save_actor(Actor(names=["행인1"], tier="extra"))
    text = render_knowledge(ms)
    assert "잠정 사실" not in text
    assert "행인1" not in text


def test_render_empty_store_is_empty(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    assert render_knowledge(ms) == ""


def test_render_is_deterministic(tmp_path):
    ms = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    ms.save_fact(Fact(claim="사실", status="confirmed", pinned=True))
    assert render_knowledge(ms) == render_knowledge(ms)


# ------------------------------------------------------------------ #
# SyncPath
# ------------------------------------------------------------------ #

def test_process_injects_and_marks(tmp_path):
    storage = JsonDirStorage(tmp_path)
    ms = MemoryStore(storage, "sess1")
    ms.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=1))
    sp = SyncPath(storage, "sess1")
    out, verdict = sp.process(_msgs("안녕"))
    assert verdict.kind == "new_session"
    assert "<dreaming_context>" in out[-1]["content"]     # 지식 주입 (캐시 밖)
    assert out[0].get("cache_control") is not None        # BP1
    assert "소지금: 450" in out[-1]["content"]


def test_full_turn_cycle_then_reroll(tmp_path):
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "sess1")
    msgs1 = _msgs("안녕")
    out1, v1 = sp.process(msgs1)
    sp.record_response(v1, msgs1, "어서 와.")

    msgs2 = _msgs("안녕", "어서 와.", "포션 얼마야?")
    out2, v2 = sp.process(msgs2)
    assert v2.kind == "next_turn"
    sp.record_response(v2, msgs2, "50골드다.")

    # 리롤: 같은 요청 재전송
    out3, v3 = sp.process(_msgs("안녕", "어서 와.", "포션 얼마야?"))
    assert v3.kind == "reroll"
    assert v3.reroll_turn_number == 1


def test_process_never_raises_on_weird_input(tmp_path):
    # fail-open (스펙 §2.6): 판정 불가여도 메시지는 통과시킨다
    sp = SyncPath(JsonDirStorage(tmp_path), "sess1")
    out, verdict = sp.process([{"role": "system", "content": "x"}])
    assert out[-1]["content"] == "x"


# ------------------------------------------------------------------ #
# 격리 버퍼 (스펙 §3.1: 판정 불확실 → fail-open, 기록은 격리 버퍼에)
# ------------------------------------------------------------------ #

def test_stranger_history_is_quarantined(tmp_path):
    # 원장 있는 세션에 전혀 무관한 히스토리 — 본원장 오염 금지 (스펙 §3.1)
    import json
    from dreaming.storage import JsonDirStorage
    from dreaming.sync import SyncPath
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "s")
    m1 = [{"role": "system", "content": "너는 리사다."},
          {"role": "user", "content": "안녕"}]
    _, v1 = sp.process(m1)
    sp.record_response(v1, m1, "어서 와.")
    assert storage.get("s/raw", "000000") is not None

    stranger = [{"role": "system", "content": "너는 리사다."},
                {"role": "user", "content": "전혀 다른 이야기"},
                {"role": "assistant", "content": "낯선 응답"},
                {"role": "user", "content": "다음 질문"}]
    out, v = sp.process(stranger)
    assert v.quarantine
    assert out == stranger                       # 무가공 passthrough
    assert "cache_control" not in json.dumps(out, ensure_ascii=False)
    sp.record_response(v, stranger, "응답")
    assert storage.get("s/quarantine", "000000") is not None
    raws = [k for k, _ in storage.scan("s/raw")]
    assert raws == ["000000"]                    # 본원장 무오염


def test_trimmed_reroll_is_not_quarantined(tmp_path):
    # 트림 직후 리롤: 정렬은 실패해도 trailing user가 출처를 확정 → 격리 금지
    from dreaming.identity import PairLedger
    from dreaming.storage import JsonDirStorage
    ledger = PairLedger(JsonDirStorage(tmp_path), "s")
    pairs = [{"index": i, "user_hash": f"u{7 + i}",
              "assistant_hash": f"a{7 + i}"} for i in range(18)]
    v1 = ledger.analyze_and_apply(pairs, "u25")
    ledger.record_turn(v1, "u25", "유저", "응답", turn_number=v1.position)
    v2 = ledger.analyze_and_apply(pairs, "u25")  # 동일 재전송 = 리롤
    assert v2.kind == "reroll"
    assert not v2.quarantine
