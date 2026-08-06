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


def test_replay_add_to_string_skipped():
    # 나쁜 커밋 하나가 파이프라인 전체를 fail-open시키면 안 된다 —
    # raise 대신 skip하고 기존 값을 보존한다
    state = replay([_c("현재_장소", "set", "시장", turn=1),
                    _c("현재_장소", "add", 1, turn=2)])
    assert state == {"현재_장소": "시장"}


def test_replay_skips_add_on_nonnumeric_slot(caplog):
    # 코퍼스 #1 재현: Dreamer가 문자열 슬롯 '반S2.소지품'에 add를 기록
    with caplog.at_level("WARNING", logger="dreaming.worldstate"):
        state = replay([
            _c("반S2.소지품", "set", "반S2의 페르소나 소지품", turn=0),
            _c("반S2.소지품", "add", "300 세스테르티우스 (쓸모없음)", turn=9),
        ])
    assert state == {"반S2.소지품": "반S2의 페르소나 소지품"}
    assert any("반S2.소지품" in r.message for r in caplog.records)


def test_replay_skips_nonnumeric_add_value_on_missing_slot():
    # 슬롯이 없어도 add 값이 비숫자면 skip — 0 + str 시도 금지
    assert replay([_c("소지품", "add", "단검", turn=1)]) == {}


def test_replay_bad_commit_does_not_block_later_commits():
    # 오염 커밋 뒤의 정상 커밋은 계속 적용된다
    state = replay([
        _c("소지품", "set", "단검", turn=1),
        _c("소지품", "add", "물약", turn=2),
        _c("소지금", "set", 500, turn=3),
        _c("소지금", "add", -50, turn=4),
    ])
    assert state == {"소지품": "단검", "소지금": 450}


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
