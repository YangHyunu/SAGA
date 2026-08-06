"""한국어 수사 표기 생성 — B-3 숫자 검증 보조 (실카드 실측 결함 A)."""
from dreaming.numerals import korean_spellings


def test_native_small_numbers():
    assert {"셋", "세"} <= set(korean_spellings(3))
    assert {"스물", "스무"} <= set(korean_spellings(20))
    assert "스물일곱" in korean_spellings(27)
    assert "쉰" in korean_spellings(50)


def test_sino_numbers():
    assert "삼" in korean_spellings(3)
    assert "이십칠" in korean_spellings(27)
    assert "삼백" in korean_spellings(300)
    assert "천이백삼십사" in korean_spellings(1234)


def test_out_of_range_is_empty():
    assert korean_spellings(0) == []
    assert korean_spellings(-3) == []
    assert korean_spellings(10000) == []
