"""charx2card keyed 로어 추출 테스트. 픽스처는 인메모리 book dict."""
import pytest

from benchmarks.eval.charx2card import _split_lore, _strip_deco


def _book(entries):
    return {"entries": entries}


def test_unknown_decorator_dropped_not_fatal():
    # RisuAI 본체: 미인식 @@Depth는 줄만 제거, 일반 블록 배치 (switch default)
    body, depth = _strip_deco("@@Depth 0\n본문이다")
    assert body == "본문이다"
    assert depth is None


def test_known_unreproducible_decorator_still_fatal():
    with pytest.raises(SystemExit):
        _strip_deco("@@priority 5\n본문")


def test_keyed_entries_extracted_with_order():
    book = _book([
        {"constant": True, "content": "상시로어", "insertion_order": 10},
        {"constant": False, "content": "길드로어", "keys": ["Guilds", "길드"],
         "insertion_order": 860, "name": "길드", "enabled": True},
        {"constant": False, "content": "", "keys": [], "name": "폴더"},
    ])
    block, post, keyed, orders = _split_lore(book)
    assert block == ["상시로어"] and orders == [10]
    assert len(keyed) == 1
    assert keyed[0]["keys"] == ["Guilds", "길드"]
    assert keyed[0]["order"] == 860
    assert keyed[0]["depth"] is None


def test_keyed_depth0_marked():
    book = _book([
        {"constant": False, "content": "@@depth 0\n지침", "keys": ["k"],
         "insertion_order": 1, "enabled": True},
    ])
    _, _, keyed, _ = _split_lore(book)
    assert keyed[0]["depth"] == 0 and keyed[0]["content"] == "지침"


def test_disabled_keyed_skipped():
    book = _book([
        {"constant": False, "content": "죽은로어", "keys": ["k"],
         "insertion_order": 1, "enabled": False},
    ])
    _, _, keyed, _ = _split_lore(book)
    assert keyed == []
