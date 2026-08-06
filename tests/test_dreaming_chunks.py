"""청크 조립 — 결정론 템플릿 (스펙 §6.1) + Tier 계층 (§6.2)."""
from dreaming.chunks import assemble_tier1, assemble_tier2
from dreaming.records import Episode

_EP = Episode(range_start="u0", range_end="u3", start_turn=0, end_turn=3,
              title="포션 흥정", summary="리사와 가격을 흥정해 50골드에 샀다.",
              open_threads=["잔액의 출처"])


def test_tier1_template_is_deterministic():
    text = assemble_tier1(_EP)
    assert text == assemble_tier1(_EP.model_copy())     # 같은 입력 → 같은 바이트
    assert "포션 흥정" in text and "50골드" in text
    assert "잔액의 출처" in text                          # open_threads 포함


def test_tier1_without_threads_has_no_thread_line():
    ep = _EP.model_copy(update={"open_threads": []})
    assert "실마리" not in assemble_tier1(ep)


def test_tier2_is_one_line_per_episode():
    ep2 = _EP.model_copy(update={"title": "여관 투숙",
                                 "summary": "방을 80골드에\n잡았다."})
    text = assemble_tier2([_EP, ep2])
    lines = text.splitlines()
    assert len(lines) == 3                               # 헤더 + 에피소드 2
    assert "여관 투숙" in lines[2] and "\n" not in lines[2]
