"""리롤/분기 시 지식 강등 + 꿈 커서 되감기 (스펙 §3.1)."""
from dreaming.records import Evidence, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import SyncPath, demote_after


def _seed(storage, session="sess1"):
    for t, (uh, ah) in enumerate([("u0", "a0"), ("u1", "a1")]):
        storage.put(f"{session}/raw", f"{t:06d}", {
            "turn_number": t, "user_text": f"u{t}", "assistant_text": f"a{t}",
            "user_hash": uh, "assistant_hash": ah})
    store = MemoryStore(storage, session)
    store.save_fact(Fact(claim="턴0에서 배움", status="confirmed",
                         evidence=[Evidence(pair_hash="u0")]))
    store.save_fact(Fact(claim="턴1에서 배움", status="confirmed",
                         evidence=[Evidence(pair_hash="u1")]))
    store.save_fact(Fact(claim="유저가 고정", status="confirmed", user_edited=True,
                         evidence=[Evidence(pair_hash="u1")]))
    store.append_commit(StateCommit(slot="소지금", op="set", value=450, turn=0))
    store.append_commit(StateCommit(slot="소지금", op="set", value=400, turn=1))
    storage.put(f"{session}/dreamer", "cursor", {"next_turn": 2})
    return store


def test_demote_after_turn1(tmp_path):
    storage = JsonDirStorage(tmp_path)
    store = _seed(storage)
    demote_after(storage, "sess1", 1)
    by_claim = {f.claim: f for f in store.list_facts()}
    assert by_claim["턴0에서 배움"].status == "confirmed"       # 분기점 이전 무사
    assert by_claim["턴1에서 배움"].status == "provisional"     # 강등
    assert by_claim["유저가 고정"].status == "confirmed"        # 유저 편집 보호
    assert store.current_state() == {"소지금": 450.0}           # 턴1 커밋 격리
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}  # 되감기


def test_demote_keeps_earlier_cursor(tmp_path):
    storage = JsonDirStorage(tmp_path)
    _seed(storage)
    storage.put("sess1/dreamer", "cursor", {"next_turn": 0})   # 아직 안 꿈꿈
    demote_after(storage, "sess1", 1)
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 0}  # 그대로


def test_syncpath_reroll_triggers_demotion(tmp_path):
    storage = JsonDirStorage(tmp_path)
    sp = SyncPath(storage, "sess1")

    def msgs(*texts):
        roles = ["user", "assistant"]
        out = [{"role": "system", "content": "너는 상인 리사다."}]
        out += [{"role": roles[i % 2], "content": t} for i, t in enumerate(texts)]
        return out

    m1 = msgs("안녕")
    _, v1 = sp.process(m1)
    sp.record_response(v1, m1, "어서 와.")
    m2 = msgs("안녕", "어서 와.", "포션 얼마야?")
    _, v2 = sp.process(m2)
    sp.record_response(v2, m2, "50골드다.")

    # 꿈이 턴1까지 처리했고 턴1에서 fact를 배웠다고 시뮬레이션
    store = MemoryStore(storage, "sess1")
    raw1 = storage.get("sess1/raw", "000001")
    store.save_fact(Fact(claim="포션은 50골드다", status="confirmed",
                         evidence=[Evidence(pair_hash=raw1["user_hash"])]))
    storage.put("sess1/dreamer", "cursor", {"next_turn": 2})

    # 리롤: 같은 요청 재전송
    _, v3 = sp.process(msgs("안녕", "어서 와.", "포션 얼마야?"))
    assert v3.kind == "reroll"
    assert store.list_facts()[0].status == "provisional"
    assert storage.get("sess1/dreamer", "cursor") == {"next_turn": 1}
