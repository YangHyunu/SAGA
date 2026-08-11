"""RisuAI 키워드 활성화 에뮬 테스트 (lorebook.svelte.ts 시맨틱)."""
import copy

from benchmarks.eval.keyed_lore import activate


def _card(**kw):
    base = {"lore": ["상시A", "상시B"], "lore_orders": [100, 200],
            "lore_indices": [0, 1], "keyed_lore": [],
            "lore_settings": {"scan_depth": 5}}
    base.update(kw)
    return base


def test_no_keyed_card_passthrough():
    card = {"lore": ["상시A"]}          # 소연형 구카드 — 신규 필드 자체가 없음
    blocks, post = activate(card, ["아무 말"])
    assert blocks == ["상시A"] and post == ""


def test_key_match_case_insensitive_substring():
    card = _card(keyed_lore=[{"name": "길드", "keys": ["Guilds", "길드"],
                              "content": "길드로어", "depth": None,
                              "order": 150, "index": 2}])
    blocks, _ = activate(card, ["오늘 GUILDS 얘기를 했다"])
    assert "길드로어" in blocks


def test_key_match_ignores_only_space():
    # RisuAI는 스페이스(U+0020)만 지운다 — replace(/ /g,'')
    # (lorebook.svelte.ts:206,208). 2026-08-11 리뷰: \s+로 개행·탭까지
    # 지우면 245키 중 공백 포함 11개가 과활성된다.
    card = _card(keyed_lore=[{"name": "", "keys": ["황금 사자"],
                              "content": "문장로어", "depth": None,
                              "order": 1, "index": 2}])
    blocks, _ = activate(card, ["그 황금 사자 문양을 보았다"])
    assert "문장로어" in blocks


def test_key_match_newline_does_not_activate():
    # 개행은 스페이스가 아니라 지워지지 않는다 — "황금\n사자"는 미활성이
    # 정답 (리뷰 실증: 키 "White Lotus"가 "White\nLotus" 발화에도 붙던
    # 과활성 버그의 반대 케이스).
    card = _card(keyed_lore=[{"name": "", "keys": ["황금 사자"],
                              "content": "문장로어", "depth": None,
                              "order": 1, "index": 2}])
    blocks, _ = activate(card, ["그 황금\n사자 문양을 보았다"])
    assert "문장로어" not in blocks


def test_key_match_does_not_bleed_across_messages():
    # \x00 메시지 구분자는 공백이 아니므로 정규식을 스페이스로 좁혀도
    # 크로스 메시지 매칭 차단은 그대로 유지된다.
    card = _card(keyed_lore=[{"name": "", "keys": ["가나"],
                              "content": "붙은로어", "depth": None,
                              "order": 1, "index": 2}])
    blocks, _ = activate(card, ["...가", "나..."])
    assert "붙은로어" not in blocks


def test_scan_depth_window():
    card = _card(lore_settings={"scan_depth": 2},
                 keyed_lore=[{"name": "", "keys": ["오래된키"],
                              "content": "X", "depth": None,
                              "order": 1, "index": 2}])
    blocks, _ = activate(card, ["오래된키 언급", "중간", "최근"])
    assert "X" not in blocks           # 최근 2개 밖이라 미활성


def test_merge_order_risu_rule():
    # 정렬: constant·keyed 합쳐 priority desc, 동점은 북 원위치로 재정렬 후
    # 전체 reverse. 이 케이스는 order가 전부 달라 동점이 없다 — 동점 재역전
    # 검증은 test_merge_ties_preserved_across_reinsertion이 별도로 지킨다.
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "키드150",
                              "depth": None, "order": 150, "index": 2}])
    blocks, _ = activate(card, ["k"])
    assert blocks == ["상시A", "키드150", "상시B"]


def test_depth0_goes_to_post():
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "지침",
                              "depth": 0, "order": 1, "index": 2}])
    blocks, post = activate(card, ["k"])
    assert "지침" not in blocks and post == "지침"


def _tied_card(**kw):
    # 동점(order=100) 재현 픽스처 — 원시 북 순서 [C@0(50), A@1(100), B@2(100)]에
    # charx2card의 T(priority desc, 동점은 원위치, 전체 reverse)를 한 번 적용한
    # 결과가 lore=[C,B,A]다 (2026-08-11 리뷰가 재현한 Critical 버그의 픽스처).
    base = {"lore": ["C", "B", "A"], "lore_orders": [50, 100, 100],
            "lore_indices": [0, 2, 1], "keyed_lore": [],
            "lore_settings": {"scan_depth": 5}}
    base.update(kw)
    return base


def test_merge_no_hit_is_identity_even_with_ties():
    # keyed_lore가 비어있지 않아도(=early-return 경로를 안 타도) 적중이
    # 0건이면 T를 원시 데이터부터 재계산한 결과가 입력 lore와 정확히
    # 같아야 한다 — T∘T ≠ T 버그가 있으면 동점 [B,A]가 [A,B]로 뒤집힌다.
    card = _tied_card(keyed_lore=[{"name": "", "keys": ["무적중키"],
                                   "content": "안뜸", "depth": None,
                                   "order": 1, "index": 3}])
    blocks, _ = activate(card, ["아무 관련 없는 발화"])
    assert blocks == card["lore"]


def test_merge_ties_preserved_across_reinsertion():
    # 동점(order=100) 그룹 한가운데 새 keyed 히트(order=100)가 끼어들어도
    # 기존 동점 상대 순서(B,A)가 유지된 채 원시 북 순서로 삽입돼야 한다.
    card = _tied_card(keyed_lore=[{"name": "", "keys": ["k"], "content": "K",
                                   "depth": None, "order": 100, "index": 3}])
    blocks, _ = activate(card, ["k"])
    assert blocks == ["C", "K", "B", "A"]


def test_activate_does_not_mutate_card():
    # 플랜이 못박은 계약: activate는 읽기만 한다 — 다음 턴 스캔 오염 금지.
    card = _tied_card(keyed_lore=[{"name": "", "keys": ["k"], "content": "K",
                                   "depth": None, "order": 100, "index": 3}])
    before = copy.deepcopy(card)
    activate(card, ["k"])
    assert card == before
