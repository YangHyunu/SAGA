"""지식 주입: 마지막 user prepend — 캐시 밖 (스펙 §3.1, §5)."""
from dreaming.assembly import HOT_ZONE_CHAR_BUDGET, clip_knowledge, inject_knowledge


def _msgs():
    return [
        {"role": "system", "content": "너는 상인 리사다."},
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "어서 와."},
        {"role": "user", "content": "포션 얼마야?"},
    ]


# ------------------------------------------------------------------ #
# inject_knowledge
# ------------------------------------------------------------------ #

def test_injects_into_last_user_only():
    msgs = _msgs()
    out = inject_knowledge(msgs, "소지금: 450골드")
    assert out[3]["content"].startswith("<dreaming_context>\n소지금: 450골드\n</dreaming_context>\n\n")
    assert out[3]["content"].endswith("포션 얼마야?")
    # 다른 메시지 무변경 — 프리픽스(캐시 계층) 불가침 (스펙 §5)
    assert out[0] == msgs[0] and out[1] == msgs[1] and out[2] == msgs[2]


def test_original_list_is_not_mutated():
    msgs = _msgs()
    inject_knowledge(msgs, "소지금: 450골드")
    assert msgs[3]["content"] == "포션 얼마야?"


def test_empty_knowledge_is_noop():
    msgs = _msgs()
    assert inject_knowledge(msgs, "") == msgs


def test_no_user_message_is_noop():
    msgs = [{"role": "system", "content": "x"}]
    assert inject_knowledge(msgs, "지식") == msgs


# ------------------------------------------------------------------ #
# clip_knowledge
# ------------------------------------------------------------------ #

def test_clip_respects_budget():
    text = "가" * (HOT_ZONE_CHAR_BUDGET + 100)
    assert len(clip_knowledge(text)) == HOT_ZONE_CHAR_BUDGET


def test_clip_short_text_unchanged():
    assert clip_knowledge("짧다") == "짧다"
