"""Pair ledger: content-hash session identity + reroll detection."""
import pytest

from saga.services.pair_ledger import (
    PairLedgerService,
    classify,
    extract_pairs,
    hash_text,
)
from saga.storage.sqlite_db import SQLiteDB


# ------------------------------------------------------------------ #
# hash_text
# ------------------------------------------------------------------ #

def test_hash_text_normalizes_whitespace():
    assert hash_text("hello   world\n") == hash_text("hello world")


def test_hash_text_normalizes_nfc():
    # NFD vs NFC of the same Hangul syllable
    assert hash_text("한") == hash_text("한")


def test_hash_text_differs_on_content():
    assert hash_text("a") != hash_text("b")
    assert hash_text(None) == hash_text("")


# ------------------------------------------------------------------ #
# extract_pairs
# ------------------------------------------------------------------ #

def _msgs(*roles_contents):
    return [{"role": r, "content": c} for r, c in roles_contents]


def test_extract_pairs_basic():
    msgs = _msgs(
        ("system", "sys"),
        ("assistant", "first message"),  # card greeting — skipped
        ("user", "u1"), ("assistant", "a1"),
        ("user", "u2"), ("assistant", "a2"),
        ("user", "u3"),
    )
    pairs, last_user = extract_pairs(msgs)
    assert len(pairs) == 2
    assert pairs[0] == {"index": 0, "user_hash": hash_text("u1"), "assistant_hash": hash_text("a1")}
    assert pairs[1]["user_hash"] == hash_text("u2")
    assert last_user == hash_text("u3")


def test_extract_pairs_merges_consecutive_assistants():
    msgs = _msgs(("user", "u1"), ("assistant", "part1"), ("assistant", "part2"), ("user", "u2"))
    pairs, last_user = extract_pairs(msgs)
    assert pairs[0]["assistant_hash"] == hash_text("part1\npart2")
    assert last_user == hash_text("u2")


def test_extract_pairs_consecutive_users():
    msgs = _msgs(("user", "u1"), ("user", "u2"))
    pairs, last_user = extract_pairs(msgs)
    assert pairs[0] == {"index": 0, "user_hash": hash_text("u1"), "assistant_hash": None}
    assert last_user == hash_text("u2")


def test_extract_pairs_empty_and_greeting_only():
    assert extract_pairs([]) == ([], None)
    pairs, last_user = extract_pairs(_msgs(("assistant", "greeting"), ("user", "hi")))
    assert pairs == []
    assert last_user == hash_text("hi")


# ------------------------------------------------------------------ #
# classify
# ------------------------------------------------------------------ #

def _chain(*entries):
    """entries: (user_text, assistant_text, turn_number)"""
    return [
        {"id": i + 1, "pair_index": i, "user_hash": hash_text(u),
         "assistant_hash": hash_text(a), "status": "confirmed", "turn_number": t}
        for i, (u, a, t) in enumerate(entries)
    ]


def _request(*texts, trailing_user=None):
    """texts: (user, assistant) pairs sent by the client."""
    pairs = [
        {"index": i, "user_hash": hash_text(u), "assistant_hash": hash_text(a)}
        for i, (u, a) in enumerate(texts)
    ]
    return pairs, (hash_text(trailing_user) if trailing_user else None)


def test_classify_empty_chain_is_baseline():
    pairs, last = _request(("u1", "a1"), trailing_user="u2")
    v = classify([], pairs, last)
    assert v["kind"] == "append"
    assert v["position"] == 1
    assert v["aligned"] and v["offset"] == 0


def test_classify_append():
    chain = _chain(("u1", "a1", 1), ("u2", "a2", 2))
    pairs, last = _request(("u1", "a1"), ("u2", "a2"), trailing_user="u3")
    v = classify(chain, pairs, last)
    assert v["kind"] == "append"
    assert v["position"] == 2
    assert (0, hash_text("a1")) in v["confirm"]
    assert (1, hash_text("a2")) in v["confirm"]


def test_classify_reroll_of_last_turn():
    chain = _chain(("u1", "a1", 1), ("u2", "a2", 2), ("u3", "a3", 3))
    # Client popped a3 and resends u3
    pairs, last = _request(("u1", "a1"), ("u2", "a2"), trailing_user="u3")
    v = classify(chain, pairs, last)
    assert v["kind"] == "reroll"
    assert v["position"] == 2
    assert v["reroll_turn_number"] == 3
    assert v["superseded_indices"] == [2]
    assert v["quarantined_indices"] == []


def test_classify_rollback_quarantines_tail():
    chain = _chain(("u1", "a1", 1), ("u2", "a2", 2), ("u3", "a3", 3))
    # Client rolled back to turn 2 and resends u2
    pairs, last = _request(("u1", "a1"), trailing_user="u2")
    v = classify(chain, pairs, last)
    assert v["kind"] == "reroll"
    assert v["position"] == 1
    assert v["reroll_turn_number"] == 2
    assert v["superseded_indices"] == [1]
    assert v["quarantined_indices"] == [2]


def test_classify_sliding_window_offset():
    chain = _chain(("u1", "a1", 1), ("u2", "a2", 2), ("u3", "a3", 3), ("u4", "a4", 4))
    # Front-truncated request: only pairs 3-4 visible, new user input
    pairs, last = _request(("u3", "a3"), ("u4", "a4"), trailing_user="u5")
    v = classify(chain, pairs, last)
    assert v["kind"] == "append"
    assert v["position"] == 4
    assert v["offset"] == 2
    assert (2, hash_text("a3")) in v["confirm"]


def test_classify_edited_user_input_supersedes():
    chain = _chain(("u1", "a1", 1), ("u2", "a2", 2))
    # Client kept pair 0, replaced u2 with an edited text
    pairs, last = _request(("u1", "a1"), trailing_user="u2-edited")
    v = classify(chain, pairs, last)
    assert v["kind"] == "reroll"
    assert v["position"] == 1
    assert v["reroll_turn_number"] == 2


def test_classify_new_session_shape():
    v = classify([], [], None)
    assert v["kind"] == "new"


# ------------------------------------------------------------------ #
# PairLedgerService (real SQLite)
# ------------------------------------------------------------------ #

@pytest.fixture
async def db(tmp_path):
    db = SQLiteDB(db_path=str(tmp_path / "test.db"))
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
def service(db):
    return PairLedgerService(db)


async def test_full_turn_lifecycle(db, service):
    session_id = "sess1"
    await db.create_session(session_id)

    # Turn 1: fresh chat
    pairs, last = extract_pairs(_msgs(("assistant", "greet"), ("user", "u1")))
    v = await service.analyze_and_apply(session_id, pairs, last)
    assert v["kind"] == "append"
    await service.record_turn(session_id, v, last, "a1", turn_number=1)

    # Turn 2: client returns with (u1, a1) confirmed
    pairs, last = extract_pairs(_msgs(
        ("assistant", "greet"), ("user", "u1"), ("assistant", "a1"), ("user", "u2")))
    v = await service.analyze_and_apply(session_id, pairs, last)
    assert v["kind"] == "append"
    rows = await db.get_pair_ledger(session_id)
    assert rows[0]["status"] == "confirmed"
    await service.record_turn(session_id, v, last, "a2", turn_number=2)

    # Reroll of turn 2: client popped a2, resends u2
    pairs, last = extract_pairs(_msgs(
        ("assistant", "greet"), ("user", "u1"), ("assistant", "a1"), ("user", "u2")))
    v = await service.analyze_and_apply(session_id, pairs, last)
    assert v["kind"] == "reroll"
    assert v["reroll_turn_number"] == 2
    statuses = {(r["pair_index"], r["status"]) for r in await db.get_pair_ledger(session_id)}
    assert (1, "superseded") in statuses


async def test_reroll_marks_turn_log_superseded(db, service):
    session_id = "sess2"
    await db.create_session(session_id)
    await db.insert_turn_log(session_id, 1, {}, user_input="u1", assistant_output="a1")
    await db.insert_pair(session_id, 0, hash_text("u1"), hash_text("a1"),
                         status="provisional", turn_number=1)

    pairs, last = extract_pairs(_msgs(("user", "u1")))  # a1 absent = reroll
    v = await service.analyze_and_apply(session_id, pairs, last)
    assert v["kind"] == "reroll"

    logs = await db.get_turn_logs(session_id)
    assert logs == []  # superseded turns excluded by default
    logs_all = await db.get_turn_logs(session_id, include_inactive=True)
    assert logs_all[0]["status"] == "superseded"


async def test_resolve_session_by_pair_overlap(db, service):
    await db.create_session("story_a")
    await db.create_session("story_b")
    await db.insert_pair("story_a", 0, hash_text("u1"), hash_text("resp-a"), turn_number=1)
    await db.insert_pair("story_b", 0, hash_text("u1"), hash_text("resp-b"), turn_number=1)

    # Same opening user text, different generated assistant → assistant hash decides
    pairs, _ = extract_pairs(_msgs(("user", "u1"), ("assistant", "resp-b"), ("user", "next")))
    assert await service.resolve_session(pairs) == "story_b"


async def test_resolve_rejects_single_user_hash_collision(db, service):
    await db.create_session("story_a")
    await db.insert_pair("story_a", 0, hash_text("..."), hash_text("unique-a"), turn_number=1)

    # Only one weak user-hash overlap ("..."), assistant unknown to ledger
    pairs = [{"index": 0, "user_hash": hash_text("..."), "assistant_hash": hash_text("other")}]
    assert await service.resolve_session(pairs) is None


async def test_fresh_chat_never_inherits(db, service):
    """Same card, new chat: no pair overlap → no session resolved."""
    await db.create_session("old_story")
    await db.insert_pair("old_story", 0, hash_text("u1"), hash_text("a1"), turn_number=1)

    pairs, last = extract_pairs(_msgs(("assistant", "same greeting"), ("user", "brand new opening")))
    assert pairs == []  # greeting skipped, no completed pairs yet
    assert await service.resolve_session(pairs) is None


async def test_backfill_heals_ledger_holes(db, service):
    session_id = "sess3"
    await db.create_session(session_id)
    # Ledger only knows pair 0; client history shows pairs 0-2
    await db.insert_pair(session_id, 0, hash_text("u1"), hash_text("a1"),
                         status="confirmed", turn_number=1)
    pairs, last = extract_pairs(_msgs(
        ("user", "u1"), ("assistant", "a1"),
        ("user", "u2"), ("assistant", "a2"),
        ("user", "u3"), ("assistant", "a3"),
        ("user", "u4")))
    v = await service.analyze_and_apply(session_id, pairs, last)
    assert v["kind"] == "append"
    rows = await db.get_pair_ledger(session_id)
    assert {r["pair_index"] for r in rows} == {0, 1, 2}
