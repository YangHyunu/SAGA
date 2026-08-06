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


# ------------------------------------------------------------------ #
# 압축 플랜 빌드 (B-4)
# ------------------------------------------------------------------ #

import json

from dreaming.chunks import TAIL_KEEP, build_compression
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore


def _ep(start, end, title="에피"):
    return Episode(range_start=f"u{start}", range_end=f"u{end}",
                   start_turn=start, end_turn=end, title=f"{title}{start}",
                   summary=f"{start}~{end} 요약.")


def _store_with(tmp_path, episodes):
    store = MemoryStore(JsonDirStorage(tmp_path), "sess1")
    for e in episodes:
        store.save_episode(e)
    return store


def test_contiguous_prefix_outside_tail_is_chunked(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
    plan = build_compression(store, last_turn=7 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 8
    assert [m["role"] for m in plan["messages"]] == ["assistant", "assistant"]
    assert "에피0" in plan["messages"][0]["content"]


def test_tail_and_gap_stop_chunking(tmp_path):
    # 꼬리 안 에피소드는 미압축, 턴 갭(4 누락)에서 중단
    store = _store_with(tmp_path, [_ep(0, 3), _ep(5, 6)])
    plan = build_compression(store, last_turn=6 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 4                 # 갭 앞까지만
    store2 = _store_with(tmp_path / "b", [_ep(0, 3)])
    assert build_compression(store2, last_turn=3) is None  # 전부 꼬리 안


def test_overlapping_redream_episode_is_skipped(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(0, 2, title="중복"),
                                   _ep(4, 5)])
    plan = build_compression(store, last_turn=5 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 6
    assert len(plan["messages"]) == 2                     # 중복 스킵


def test_legacy_episode_without_turns_is_ignored(tmp_path):
    legacy = Episode(range_start="u0", range_end="u3", title="구버전",
                     summary="턴 없음")
    store = _store_with(tmp_path, [legacy])
    assert build_compression(store, last_turn=99) is None


def test_tier2_promotion_is_stable(tmp_path):
    # T1_MAX(8) 초과 시 오래된 것부터 CHAPTER_SIZE(5) 고정 블록으로 승격 —
    # 에피소드가 늘어도 기존 챕터 바이트는 불변 (프리픽스 안정)
    eps = [_ep(i * 2, i * 2 + 1) for i in range(10)]      # 10개 → 챕터 1 + T1 5개
    store = _store_with(tmp_path, eps)
    plan = build_compression(store, last_turn=19 + TAIL_KEEP)
    assert len(plan["messages"]) == 1 + 5
    assert plan["messages"][0]["content"].startswith("[지난 장 요약]")

    store.save_episode(_ep(20, 21))                       # 11개 → 그룹 경계 불변
    plan2 = build_compression(store, last_turn=21 + TAIL_KEEP)
    assert plan2["messages"][0] == plan["messages"][0]


def test_plan_is_deterministic(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
    a = build_compression(store, last_turn=20)
    b = build_compression(store, last_turn=20)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)
