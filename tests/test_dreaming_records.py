"""레코드 4종 직렬화 (스펙 §4). 이 파일은 Task 2~3에 걸쳐 자란다."""
import pytest
from pydantic import ValidationError

from dreaming.records import Evidence, Fact, TypedNumber


# ------------------------------------------------------------------ #
# Fact (스펙 §4.1)
# ------------------------------------------------------------------ #

def test_fact_minimal_defaults():
    f = Fact(claim="리사는 포션을 50골드에 판다")
    assert f.status == "provisional"          # 꿈이 확정하기 전까지 잠정 (스펙 §4.1)
    assert f.user_edited is False
    assert f.pinned is False
    assert f.supersedes is None
    assert len(f.id) == 32                    # uuid4 hex
    assert "T" in f.recorded_at               # ISO 8601


def test_fact_typed_numbers_and_evidence():
    f = Fact(
        claim="리사는 포션을 50골드에 판다",
        entities=["리사"],
        numbers=[TypedNumber(name="포션 가격", value=50, unit="골드")],
        evidence=[Evidence(pair_hash="abc123", offset=140)],
        learned_by=["user"],
        recorded_at="2026-08-04T00:00:00+00:00",
    )
    assert f.numbers[0].value == 50
    assert f.evidence[0].pair_hash == "abc123"


def test_fact_json_roundtrip():
    f = Fact(
        claim="리사는 포션을 50골드에 판다",
        numbers=[TypedNumber(name="포션 가격", value=50, unit="골드")],
        recorded_at="2026-08-04T00:00:00+00:00",
    )
    data = f.model_dump(mode="json")          # Storage에 넣는 dict
    assert isinstance(data, dict)
    restored = Fact.model_validate(data)
    assert restored == f


def test_fact_rejects_unknown_status():
    with pytest.raises(ValidationError):
        Fact(claim="x", status="deleted")      # 삭제는 상태가 아님 — supersede만 (스펙 §4.1)


def test_fact_ids_are_unique():
    assert Fact(claim="a").id != Fact(claim="a").id


def test_fact_default_lists_are_not_shared():
    a, b = Fact(claim="a"), Fact(claim="b")
    a.entities.append("리사")
    assert b.entities == []
