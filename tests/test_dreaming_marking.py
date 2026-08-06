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


def test_bp1_stays_in_leading_system_run():
    # charx PHI는 globalNote로 꼬리 system에 붙는다 (prompt.ts:427) —
    # BP1이 거기 찍히면 지식 주입이 캐시 span 안에 들어간다 (스펙 §3.1 위반)
    msgs = [{"role": "system", "content": "본문"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답"},
            {"role": "system", "content": "PHI 꼬리"}]
    marked = mark_cache(msgs)
    assert "cache_control" in marked[0]              # BP1 = 선두 연속 system 끝
    assert "cache_control" not in marked[3]          # 꼬리 system 금지


def test_no_bp_after_injected_last_user():
    msgs = [{"role": "system", "content": "본문"},
            {"role": "user", "content": "질문"},
            {"role": "assistant", "content": "답"},
            {"role": "user",
             "content": "<dreaming_context>지식</dreaming_context>\n\n새 질문"},
            {"role": "system", "content": "globalNote"}]
    marked = mark_cache(msgs)
    idxs = [i for i, m in enumerate(marked) if "cache_control" in m]
    assert max(idxs) == 2                            # 마지막 user 뒤엔 BP 없음
