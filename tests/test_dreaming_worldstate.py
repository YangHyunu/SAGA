"""WorldState: typed commit 원장 + 리플레이 (스펙 §4.3 — 서술은 상태를 못 바꾼다)."""
import pytest
from pydantic import ValidationError

from dreaming.records import StateCommit
from dreaming.worldstate import replay


def _c(slot, op, value, turn, status="applied", recorded_at="2026-08-04T00:00:00+00:00"):
    return StateCommit(
        slot=slot, op=op, value=value, turn=turn, status=status, recorded_at=recorded_at
    )


# ------------------------------------------------------------------ #
# replay
# ------------------------------------------------------------------ #

def test_replay_set_then_add():
    # 소지금 500 → -50 → +300 = 750: 왜 750인지 커밋 히스토리로 추적 가능 (스펙 §7.1)
    state = replay([
        _c("소지금", "set", 500, turn=1),
        _c("소지금", "add", -50, turn=2),
        _c("소지금", "add", 300, turn=3),
    ])
    assert state == {"소지금": 750}


def test_replay_orders_by_turn_not_input_order():
    state = replay([
        _c("소지금", "add", -50, turn=2),
        _c("소지금", "set", 500, turn=1),
    ])
    assert state == {"소지금": 450}


def test_replay_excludes_pending_contradiction():
    # 모순 커밋은 확정 전까지 상태에 반영 금지 (스펙 §3.2 B-3, Stubbornness 2502.04390)
    state = replay([
        _c("소지금", "set", 500, turn=1),
        _c("소지금", "set", 9999, turn=2, status="pending_contradiction"),
    ])
    assert state == {"소지금": 500}


def test_replay_add_on_missing_slot_starts_from_zero():
    assert replay([_c("빚", "add", 100, turn=1)]) == {"빚": 100}


def test_replay_string_slots_set_only():
    state = replay([_c("현재_장소", "set", "시장", turn=1),
                    _c("현재_장소", "set", "여관", turn=2)])
    assert state == {"현재_장소": "여관"}


def test_replay_add_to_string_raises():
    with pytest.raises(TypeError):
        replay([_c("현재_장소", "set", "시장", turn=1),
                _c("현재_장소", "add", 1, turn=2)])


def test_replay_manual_commits_apply():
    # 유저 직접 수정 = manual 커밋으로 기록, 상태에 반영 (스펙 §7.2)
    state = replay([_c("소지금", "set", 500, turn=1),
                    _c("소지금", "set", 100, turn=2, status="manual")])
    assert state == {"소지금": 100}


# ------------------------------------------------------------------ #
# StateCommit 검증
# ------------------------------------------------------------------ #

def test_commit_rejects_unknown_op():
    with pytest.raises(ValidationError):
        _c("소지금", "multiply", 2, turn=1)
