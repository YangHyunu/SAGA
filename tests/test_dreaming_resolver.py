"""세션 해석: 해시 역색인 (saga resolve_session의 KV 이식)."""
from dreaming.resolver import SessionResolver
from dreaming.storage import JsonDirStorage

from saga.services.pair_ledger import hash_text


def _pairs(*uas):
    return [{"index": i, "user_hash": hash_text(u), "assistant_hash": hash_text(a)}
            for i, (u, a) in enumerate(uas)]


# ------------------------------------------------------------------ #
# resolve
# ------------------------------------------------------------------ #

def test_resolves_by_assistant_hash(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("안녕"), hash_text("어서 와."))
    assert r.resolve(_pairs(("안녕", "어서 와."))) == "sess1"


def test_single_user_hash_match_is_not_enough(tmp_path):
    # "계속" 같은 흔한 입력의 user 해시 하나로는 세션 확정 금지 (saga 규칙)
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("계속"), None)
    assert r.resolve([{"index": 0, "user_hash": hash_text("계속"),
                       "assistant_hash": hash_text("다른 응답")}]) is None


def test_two_user_matches_resolve(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess1", hash_text("안녕"), None)
    r.index_pair("sess1", hash_text("포션 얼마야?"), None)
    got = r.resolve([
        {"index": 0, "user_hash": hash_text("안녕"), "assistant_hash": hash_text("x")},
        {"index": 1, "user_hash": hash_text("포션 얼마야?"), "assistant_hash": hash_text("y")},
    ])
    assert got == "sess1"


def test_no_match_returns_none(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    assert r.resolve(_pairs(("안녕", "어서 와."))) is None
    assert r.resolve([]) is None


def test_best_scoring_session_wins(tmp_path):
    r = SessionResolver(JsonDirStorage(tmp_path))
    r.index_pair("sess_a", hash_text("안녕"), hash_text("어서 와."))
    r.index_pair("sess_b", hash_text("안녕"), hash_text("어서 와."))
    r.index_pair("sess_b", hash_text("포션 얼마야?"), hash_text("50골드다."))
    got = r.resolve(_pairs(("안녕", "어서 와."), ("포션 얼마야?", "50골드다.")))
    assert got == "sess_b"
