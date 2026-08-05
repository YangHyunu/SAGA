"""MemoryStore: 세션 스코프 파사드 — 이후 모든 플랜의 저장 진입점."""
from dreaming.facts import apply_user_edit
from dreaming.records import Actor, Episode, Fact, StateCommit
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore


def _store(tmp_path, session="sess1"):
    return MemoryStore(JsonDirStorage(tmp_path), session_id=session)


# ------------------------------------------------------------------ #
# Fact
# ------------------------------------------------------------------ #

def test_fact_save_get_roundtrip(tmp_path):
    ms = _store(tmp_path)
    f = Fact(claim="포션은 50골드다")
    ms.save_fact(f)
    assert ms.get_fact(f.id) == f
    assert ms.get_fact("missing") is None


def test_list_facts_hides_superseded_by_default(tmp_path):
    ms = _store(tmp_path)
    f = Fact(claim="포션은 50골드다")
    old2, new2 = apply_user_edit(f, claim="포션은 45골드다")
    ms.save_fact(old2)
    ms.save_fact(new2)
    visible = ms.list_facts()
    assert [x.claim for x in visible] == ["포션은 45골드다"]
    assert len(ms.list_facts(include_superseded=True)) == 2


def test_sessions_are_isolated(tmp_path):
    a = _store(tmp_path, "sess_a")
    b = _store(tmp_path, "sess_b")
    a.save_fact(Fact(claim="A만의 사실"))
    assert b.list_facts() == []


# ------------------------------------------------------------------ #
# Episode / Actor
# ------------------------------------------------------------------ #

def test_episode_roundtrip(tmp_path):
    ms = _store(tmp_path)
    e = Episode(range_start="h1", range_end="h2", title="흥정", summary="합의했다")
    ms.save_episode(e)
    assert ms.list_episodes() == [e]


def test_actor_roundtrip(tmp_path):
    ms = _store(tmp_path)
    a = Actor(names=["리사"])
    ms.save_actor(a)
    assert ms.list_actors() == [a]


# ------------------------------------------------------------------ #
# WorldState
# ------------------------------------------------------------------ #

def test_commits_replay_to_current_state(tmp_path):
    ms = _store(tmp_path)
    ms.append_commit(StateCommit(slot="소지금", op="set", value=500, turn=1))
    ms.append_commit(StateCommit(slot="소지금", op="add", value=-50, turn=2))
    assert ms.current_state() == {"소지금": 450}
    assert len(ms.list_commits()) == 2
