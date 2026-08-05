"""Fact 버전 체인: 덮어쓰기 금지, invalidate-and-append (스펙 §4.1, WISE 2405.14768)."""
from dreaming.facts import apply_user_edit, dreamer_can_modify, supersede
from dreaming.records import Fact


# ------------------------------------------------------------------ #
# supersede
# ------------------------------------------------------------------ #

def test_supersede_links_chain_and_keeps_inputs_immutable():
    old = Fact(claim="포션은 50골드다")
    new = Fact(claim="포션은 30골드다 (할인)")
    old2, new2 = supersede(old, new)
    assert old2.status == "superseded"
    assert new2.supersedes == old.id
    # 원본 불변 — 저장은 호출자(MemoryStore) 몫
    assert old.status == "provisional"
    assert new.supersedes is None


# ------------------------------------------------------------------ #
# 유저 편집 보호 (스펙 §2.7, §7.2)
# ------------------------------------------------------------------ #

def test_dreamer_cannot_modify_user_edited_fact():
    f = Fact(claim="유저가 고친 사실", user_edited=True)
    assert dreamer_can_modify(f) is False


def test_dreamer_can_modify_ordinary_fact():
    assert dreamer_can_modify(Fact(claim="평범한 사실")) is True


def test_apply_user_edit_creates_protected_new_version():
    f = Fact(claim="포션은 50골드다")
    old2, new2 = apply_user_edit(f, claim="포션은 45골드다")
    assert old2.status == "superseded"
    assert new2.claim == "포션은 45골드다"
    assert new2.user_edited is True          # 이후 Dreamer가 못 덮음
    assert new2.supersedes == f.id
    assert new2.id != f.id                   # 새 버전 = 새 레코드
    assert new2.recorded_at != ""


def test_apply_user_edit_preserves_untouched_fields():
    f = Fact(claim="포션은 50골드다", entities=["리사"], pinned=True)
    _, new2 = apply_user_edit(f, claim="포션은 45골드다")
    assert new2.entities == ["리사"]
    assert new2.pinned is True
