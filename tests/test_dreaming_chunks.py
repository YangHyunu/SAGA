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

from dreaming.chunks import BOUNDARY_STEP, TAIL_KEEP, build_compression
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
    store = _store_with(tmp_path, [_ep(0, 4), _ep(5, 9)])
    plan = build_compression(store, last_turn=9 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 10
    assert [m["role"] for m in plan["messages"]] == ["assistant", "assistant"]
    assert "에피0" in plan["messages"][0]["content"]


def test_padded_baseline_turns_are_chunked(tmp_path):
    # 프로덕션 턴 번호는 0이 아니라 _BASELINE_PAD(1024)부터 시작한다
    # (identity.py:95-99). 0-베이스 픽스처만 있어서 압축이 프로덕션에서
    # 100% 실패하는 걸 테스트가 못 잡았다 (docs/DREAMING_FLAW.md §2).
    store = _store_with(tmp_path, [_ep(1025, 1029), _ep(1030, 1034)])
    plan = build_compression(store, last_turn=1034 + TAIL_KEEP)
    assert plan is not None
    assert plan["covers_until_turn"] == 1035
    assert len(plan["messages"]) == 2


def test_tail_and_gap_stop_chunking(tmp_path):
    # 꼬리 안 에피소드는 미압축, 턴 갭(5 누락)에서 중단
    store = _store_with(tmp_path, [_ep(0, 4), _ep(6, 9)])
    plan = build_compression(store, last_turn=9 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 5                  # 갭 앞까지만
    store2 = _store_with(tmp_path / "b", [_ep(0, 4)])
    assert build_compression(store2, last_turn=4) is None   # 전부 꼬리 안


def test_overlapping_redream_episode_is_skipped(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 4), _ep(0, 2, title="중복"),
                                   _ep(5, 9)])
    plan = build_compression(store, last_turn=9 + TAIL_KEEP)
    assert plan["covers_until_turn"] == 10
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


def test_boundary_moves_in_steps_not_every_turn(tmp_path):
    # 경계가 매 턴 움직이면 플랜이 매 턴 바뀌고 그 뒤 꼬리가 통째로 새
    # 바이트가 된다 → 프리픽스 캐시 상시 파괴 (fix-drm-r0 실측 47%).
    # 계단 안에서는 바이트가 동일해야 하고, 계단을 넘을 때만 자라야 한다.
    eps = [_ep(1025 + i, 1025 + i) for i in range(40)]      # 프로덕션 패딩
    store = _store_with(tmp_path, eps)
    plans = {t: build_compression(store, last_turn=t)
             for t in range(1030, 1070)}
    covers = sorted({p["covers_until_turn"] for p in plans.values() if p})
    assert len(covers) >= 2                                  # 실제로 전진함
    assert all(b - a == BOUNDARY_STEP for a, b in zip(covers, covers[1:]))

    changes = sum(1 for t in range(1031, 1070)
                  if json.dumps(plans[t], ensure_ascii=False)
                  != json.dumps(plans[t - 1], ensure_ascii=False))
    assert changes == len(covers)          # 계단 수만큼만 바뀜 (39턴 아님)


def test_emitted_chapters_keep_their_bytes_across_steps(tmp_path):
    # 이미 낸 챕터가 다시 쓰이면 그 지점부터 캐시가 죽는다 — 계단을
    # 넘어가도 앞 챕터 바이트는 불변이어야 한다.
    eps = [_ep(1025 + i, 1025 + i) for i in range(40)]
    store = _store_with(tmp_path, eps)
    early = build_compression(store, last_turn=1050)
    late = build_compression(store, last_turn=1069)
    assert early is not None and late is not None
    chapters = [m for m in early["messages"]
                if m["content"].startswith("[지난 장 요약]")]
    assert chapters                                          # 승격이 실제로 일어남
    assert late["messages"][:len(chapters)] == chapters


def test_plan_is_deterministic(tmp_path):
    store = _store_with(tmp_path, [_ep(0, 3), _ep(4, 7)])
    a = build_compression(store, last_turn=20)
    b = build_compression(store, last_turn=20)
    assert json.dumps(a, ensure_ascii=False) == json.dumps(b, ensure_ascii=False)


# ------------------------------------------------------------------ #
# 압축 적용 (동기 경로)
# ------------------------------------------------------------------ #

from dreaming.chunks import apply_compression

_PLAN = {"covers_until_turn": 2,
         "messages": [{"role": "assistant", "content": "[지난 이야기 · 초반]"}]}


def _msgs(pairs, greeting=True):
    out = [{"role": "system", "content": "너는 리사다."}]
    if greeting:
        out.append({"role": "assistant", "content": "어서 와요."})
    for i in range(pairs):
        out.append({"role": "user", "content": f"질문{i}"})
        out.append({"role": "assistant", "content": f"답{i}"})
    out.append({"role": "user", "content": "새 질문"})
    return out


def test_apply_replaces_first_k_pairs_keeps_system_and_greeting():
    msgs = _msgs(4)
    out, bp2 = apply_compression(msgs, _PLAN)
    assert out[0]["content"] == "너는 리사다."
    assert out[1]["content"] == "어서 와요."               # 인사 보존
    assert out[2]["content"] == "[지난 이야기 · 초반]"      # 청크
    assert bp2 == 2
    texts = [m["content"] for m in out]
    assert "질문0" not in texts and "질문2" in texts        # 꼬리 보존
    assert out[-1]["content"] == "새 질문"
    assert msgs[2]["content"] == "질문0"                    # 원본 불변


def test_apply_short_history_fails_open():
    msgs = _msgs(1)                                        # pair 1 < K=2
    out, bp2 = apply_compression(msgs, _PLAN)
    assert out is msgs and bp2 is None


def test_mark_cache_bp2():
    from dreaming.marking import mark_cache
    out, bp2 = apply_compression(_msgs(4), _PLAN)
    marked = mark_cache(out, bp2_index=bp2)
    assert marked[0]["cache_control"]["type"] == "ephemeral"   # BP1
    assert marked[bp2]["cache_control"]["type"] == "ephemeral"  # BP2
    last_asst = max(i for i, m in enumerate(marked)
                    if m["role"] == "assistant")
    assert marked[last_asst]["cache_control"]["type"] == "ephemeral"  # BP3
    assert sum(1 for m in marked if "cache_control" in m) == 3


# ------------------------------------------------------------------ #
# 윈도우 앵커 — 트림 정상상태 (corpus3 실증 결함 ②)
# ------------------------------------------------------------------ #

def test_window_past_covers_restores_chunks_without_drop():
    # 트림이 이미 압축 구간을 지나감 (window_start 5 ≥ covers 2) —
    # 드롭 0 + 청크 prepend = 사라진 컨텍스트 복원 (이 기능의 본래 가치)
    msgs = _msgs(3)
    out, bp2 = apply_compression(msgs, _PLAN, window_start_turn=5)
    texts = [m["content"] for m in out]
    assert "[지난 이야기 · 초반]" in texts
    assert all(f"질문{i}" in "".join(texts) for i in range(3))   # 전량 보존
    assert bp2 == 2                                # system+인사 다음
    assert len(out) == len(msgs) + 1


def test_window_inside_covers_drops_remainder_only():
    # 윈도우 시작 1, covers 2 → 윈도우에 남은 압축 대상은 1 pair뿐
    msgs = _msgs(4)
    out, bp2 = apply_compression(msgs, _PLAN, window_start_turn=1)
    texts = [m["content"] for m in out]
    assert "질문0" not in "".join(texts)           # 첫 pair(턴1)만 드롭
    assert "질문1" in "".join(texts)               # 턴2부터 보존
    assert bp2 == 2


def test_default_window_start_keeps_existing_behavior():
    msgs = _msgs(4)
    assert apply_compression(msgs, _PLAN) == \
        apply_compression(msgs, _PLAN, window_start_turn=0)


# ------------------------------------------------------------------ #
# 발췌 렌더링 (Task 5)
# ------------------------------------------------------------------ #

def test_tier1_발췌_포함():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.",
                 key_excerpts=["은검을 300골드에 팔았다"])
    out = assemble_tier1(ep)
    assert '원문: "은검을 300골드에 팔았다"' in out


def test_tier1_발췌_없으면_기존_바이트_그대로():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.")
    assert "원문:" not in assemble_tier1(ep)   # byte-stable — 구 에피소드 불변


def test_tier2_병합_발췌_5개_캡():
    # 스펙 §6.2: "유닛당 3, 병합 시 5" — 챕터로 접혀도 발췌 5개는 생존
    eps = [Episode(range_start=f"a{i}", range_end=f"b{i}", title=f"장{i}",
                   summary="요약.", key_excerpts=[f"발췌{i}-1", f"발췌{i}-2"])
           for i in range(4)]                   # 총 발췌 8개
    out = assemble_tier2(eps)
    assert out.count("원문:") == 5              # 연대순 선착 5개
    assert '원문: "발췌0-1"' in out and '원문: "발췌2-1"' in out
    assert "발췌3-2" not in out


def test_tier2_발췌_없으면_기존_바이트_그대로():
    ep = Episode(range_start="a", range_end="b", title="거래", summary="요약.")
    assert "원문:" not in assemble_tier2([ep])
