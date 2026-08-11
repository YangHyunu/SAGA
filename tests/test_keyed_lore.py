"""RisuAI 키워드 활성화 에뮬 테스트 (lorebook.svelte.ts 시맨틱)."""
from benchmarks.eval.keyed_lore import activate


def _card(**kw):
    base = {"lore": ["상시A", "상시B"], "lore_orders": [100, 200],
            "keyed_lore": [], "lore_settings": {"scan_depth": 5}}
    base.update(kw)
    return base


def test_no_keyed_card_passthrough():
    card = {"lore": ["상시A"]}          # 소연형 구카드 — 신규 필드 자체가 없음
    blocks, post = activate(card, ["아무 말"])
    assert blocks == ["상시A"] and post == ""


def test_key_match_case_insensitive_substring():
    card = _card(keyed_lore=[{"name": "길드", "keys": ["Guilds", "길드"],
                              "content": "길드로어", "depth": None, "order": 150}])
    blocks, _ = activate(card, ["오늘 GUILDS 얘기를 했다"])
    assert "길드로어" in blocks


def test_key_match_ignores_whitespace():
    # RisuAI는 공백 제거 후 substring (lorebook.svelte.ts:206-222)
    card = _card(keyed_lore=[{"name": "", "keys": ["황금 사자"],
                              "content": "문장로어", "depth": None, "order": 1}])
    blocks, _ = activate(card, ["그 황금\n사자 문양을 보았다"])
    assert "문장로어" in blocks


def test_scan_depth_window():
    card = _card(lore_settings={"scan_depth": 2},
                 keyed_lore=[{"name": "", "keys": ["오래된키"],
                              "content": "X", "depth": None, "order": 1}])
    blocks, _ = activate(card, ["오래된키 언급", "중간", "최근"])
    assert "X" not in blocks           # 최근 2개 밖이라 미활성


def test_merge_order_risu_rule():
    # 정렬: 합쳐서 sort(-order) 후 reverse → order 오름차순 (동점은 역순)
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "키드150",
                              "depth": None, "order": 150}])
    blocks, _ = activate(card, ["k"])
    assert blocks == ["상시A", "키드150", "상시B"]


def test_depth0_goes_to_post():
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "지침",
                              "depth": 0, "order": 1}])
    blocks, post = activate(card, ["k"])
    assert "지침" not in blocks and post == "지침"
