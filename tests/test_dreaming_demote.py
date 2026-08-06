"""리롤/분기 시 지식 강등 + 꿈 커서 되감기 (스펙 §3.1)."""
from dreaming.records import Episode, Evidence, Fact, StateCommit
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


def test_deep_divergence_invalidates_plan_and_stale_episodes(tmp_path):
    storage = JsonDirStorage(tmp_path)
    store = MemoryStore(storage, "sess1")
    storage.put("sess1/compression", "plan",
                {"covers_until_turn": 4, "messages": [
                    {"role": "assistant", "content": "청크"}]})
    store.save_episode(Episode(range_start="u0", range_end="u1",
                               start_turn=0, end_turn=1,
                               title="보존", summary="분기 전"))
    store.save_episode(Episode(range_start="u2", range_end="u3",
                               start_turn=2, end_turn=3,
                               title="무효", summary="분기 걸침"))
    demote_after(storage, "sess1", from_turn=2)
    assert storage.get("sess1/compression", "plan") is None
    assert [e.title for e in store.list_episodes()] == ["보존"]


def test_late_reroll_keeps_plan_and_episodes(tmp_path):
    # 흔한 케이스: 마지막 턴 리롤 — 압축 구간(0~3) 밖이라 무손상
    storage = JsonDirStorage(tmp_path)
    store = MemoryStore(storage, "sess1")
    plan = {"covers_until_turn": 4,
            "messages": [{"role": "assistant", "content": "청크"}]}
    storage.put("sess1/compression", "plan", plan)
    store.save_episode(Episode(range_start="u0", range_end="u3",
                               start_turn=0, end_turn=3,
                               title="보존", summary="압축 구간"))
    demote_after(storage, "sess1", from_turn=9)
    assert storage.get("sess1/compression", "plan") == plan
    assert len(store.list_episodes()) == 1


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
