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


# ------------------------------------------------------------------ #
# 꼬리 system (RisuAI globalNote/PHI) — BP1은 선두 구간에만 찍혀야 한다
# ------------------------------------------------------------------ #

def _risu_msgs():
    """RisuAI 기본 promptTemplate 레이아웃 (prompt.ts:427).

    description → lorebook → chat → globalNote 순이라 **히스토리 뒤에 system이
    온다**. charx의 post_history_instructions가 globalNote로 들어간다
    (characterCards.ts:992).
    """
    return [
        {"role": "system", "content": "설명: 너는 상인 리사다."},      # 0
        {"role": "system", "content": "로어북: 판타지 시장."},          # 1
        {"role": "assistant", "content": "어서 와."},                   # 2
        {"role": "user", "content": "포션 얼마야?"},                    # 3
        {"role": "assistant", "content": "금화 10닢."},                 # 4
        {"role": "user", "content": "<dreaming_context>…</>\n두 개 줘"},  # 5
        {"role": "system", "content": "PHI: 항상 한국어로 답해라."},     # 6
    ]


def test_bp1_stays_in_leading_system_run():
    out = mark_cache(_risu_msgs())
    assert "cache_control" in out[1]        # 선두 연속 system의 마지막
    assert "cache_control" not in out[6]    # 꼬리 system은 BP1 후보가 아니다


def test_no_bp_after_injected_last_user():
    """지식 주입은 캐시 밖이어야 한다 (assembly.py §3.1).

    마지막 user 뒤에 BP가 찍히면 매 턴 변하는 지식이 캐시 span에 들어가
    전체 프롬프트가 매번 재작성된다.
    """
    msgs = _risu_msgs()
    last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    out = mark_cache(msgs)
    marked = [i for i, m in enumerate(out) if "cache_control" in m]
    assert marked, "BP가 하나도 없다"
    assert max(marked) < last_user
