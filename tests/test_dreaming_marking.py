"""3-BP 캐시 마킹 (스펙 §3.1). RisuAI가 cachePoint를 지우므로 마킹은 우리 몫(§0.1)."""
from dreaming.marking import mark_cache


def _msgs():
    return [
        {"role": "system", "content": "너는 상인 리사다."},
        {"role": "system", "content": "세계관: 판타지 시장."},
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "어서 와."},
        {"role": "user", "content": "포션 얼마야?"},
    ]


# ------------------------------------------------------------------ #
# BP1 / BP3
# ------------------------------------------------------------------ #

def test_marks_last_system_and_last_assistant():
    out = mark_cache(_msgs())
    assert out[1]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}   # BP1
    assert out[3]["cache_control"] == {"type": "ephemeral", "ttl": "5m"}   # BP3
    assert "cache_control" not in out[0]
    assert "cache_control" not in out[2]
    assert "cache_control" not in out[4]


def test_ttl_configurable():
    out = mark_cache(_msgs(), ttl="1h")
    assert out[1]["cache_control"]["ttl"] == "1h"


def test_strips_preexisting_marks():
    msgs = _msgs()
    msgs[2]["cache_control"] = {"type": "ephemeral"}   # 낯선 마킹 — 제거돼야 함
    out = mark_cache(msgs)
    assert "cache_control" not in out[2]


def test_original_not_mutated():
    msgs = _msgs()
    mark_cache(msgs)
    assert "cache_control" not in msgs[1]


def test_no_system_no_assistant_is_safe():
    out = mark_cache([{"role": "user", "content": "안녕"}])
    assert "cache_control" not in out[0]
