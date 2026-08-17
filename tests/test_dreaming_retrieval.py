"""tests/test_dreaming_retrieval.py — 어휘 랭킹 + 장면 쿼리 (스펙 §3.1-2)."""
from dreaming.records import Fact
from dreaming.retrieval import features, rank_facts, scene_query


def _fact(claim, **kw):
    kw.setdefault("status", "confirmed")
    return Fact(claim=claim, **kw)


def test_features_토큰과_bigram():
    fs = features("잿빛 강돌은 열쇠다")
    assert "강돌은" in fs          # 토큰
    assert "강돌" in fs            # bigram — 조사 변형 흡수의 핵심
    assert "잿빛" in fs


def test_쿼리_고유명사가_최신성을_이긴다():
    old = _fact("잿빛 강돌은 돌 관의 십자 표식 중앙에 끼워 넣는 열쇠 역할을 한다",
                recorded_at="2026-01-01T00:00:00+00:00")
    new = _fact("위지소연은 마을 사람들이 바치는 공물로 생활을 유지한다",
                recorded_at="2026-06-01T00:00:00+00:00")
    ranked = rank_facts([new, old], "그 강돌을 관에 끼우면 어떻게 되는 거야?")
    assert ranked[0] is old


def test_조사_변형은_bigram으로_매칭된다():
    f1 = _fact("설한초의 독은 일반 약초로 해독되지 않는다")
    f2 = _fact("보자기 속 강돌은 짐승의 뼈에서 떼어낸 것이다")
    ranked = rank_facts([f2, f1], "설한초를 먹으면 어떻게 돼?")  # "설한초의"≠"설한초를"
    assert ranked[0] is f1


def test_entities도_매칭에_참여():
    f1 = _fact("그 검은 왕가의 유물이다", entities=["은검", "유리"])
    f2 = _fact("마을 축제는 보름마다 열린다")
    ranked = rank_facts([f2, f1], "은검 얘기 좀 해줘")
    assert ranked[0] is f1


def test_빈_쿼리는_최신순_폴백():
    old = _fact("옛 사실", recorded_at="2026-01-01T00:00:00+00:00")
    new = _fact("새 사실", recorded_at="2026-06-01T00:00:00+00:00")
    assert rank_facts([old, new], "")[0] is new


def test_pinned는_점수_무관_선두():
    pin = _fact("핀 사실", pinned=True, recorded_at="2026-01-01T00:00:00+00:00")
    hit = _fact("강돌은 열쇠다", recorded_at="2026-06-01T00:00:00+00:00")
    assert rank_facts([hit, pin], "강돌 어디 씀?")[0] is pin


def test_동점은_최신순():
    a = _fact("무관한 사실 하나", recorded_at="2026-01-01T00:00:00+00:00")
    b = _fact("무관한 사실 둘", recorded_at="2026-06-01T00:00:00+00:00")
    ranked = rank_facts([a, b], "강돌")   # 둘 다 점수 0
    assert ranked[0] is b


def test_scene_query_마지막유저_플러스_직전응답600():
    msgs = [{"role": "system", "content": "카드"},
            {"role": "user", "content": "이전 질문"},
            {"role": "assistant", "content": "긴 응답 " * 300},   # 1500자+
            {"role": "user", "content": "강돌 얘기"}]
    q = scene_query(msgs)
    assert "카드" not in q and "이전 질문" not in q
    assert q.endswith("강돌 얘기")
    assert len(q) <= 600 + 1 + len("강돌 얘기")   # 직전 응답은 600자 캡


def test_scene_query_유저턴만_있으면_그것만():
    assert scene_query([{"role": "user", "content": "안녕"}]) == "안녕"


def test_scene_query_비문자열_content_무시():
    msgs = [{"role": "user", "content": [{"type": "text"}]},
            {"role": "user", "content": "진짜 질문"}]
    assert scene_query(msgs) == "진짜 질문"


import os
from pathlib import Path

import pytest

_DATA = Path(os.environ.get("DREAMING_DATA_DIR", "dreaming_data"))


@pytest.mark.skipif(not (_DATA / "fix-drm-r0" / "facts").is_dir(),
                    reason="로컬 실측 데이터 없음")
def test_retrieval_lab_실데이터_로드():
    from dreaming.storage import JsonDirStorage
    from dreaming.store import MemoryStore
    ms = MemoryStore(JsonDirStorage(_DATA), "fix-drm-r0")
    assert len(ms.list_facts()) > 300


# ------------------------------------------------------------------ #
# rank_episodes (RAG 수리 ①)
# ------------------------------------------------------------------ #

def _ep(title, summary, **kw):
    from dreaming.records import Episode
    kw.setdefault("range_start", "a")
    kw.setdefault("range_end", "b")
    return Episode(title=title, summary=summary, **kw)


def test_rank_episodes_관련_장면이_점수와_함께_온다():
    from dreaming.retrieval import rank_episodes
    hit = _ep("눈 내리는 산장의 밤", "렌이 하룻밤 머물기로 했다")
    other = _ep("시장의 아침", "쌀값을 흥정했다")
    out = rank_episodes([other, hit], "저녁만 먹고 내려가라 하셨지만 하룻밤만 더")
    assert out and out[0][0].title == "눈 내리는 산장의 밤"
    assert out[0][1] > 0


def test_rank_episodes_점수_0은_반환하지_않는다():
    # 사실(최신순 폴백)과 다른 결정 — 에피소드는 통이 커서 무관 주입은
    # 예산 낭비 + 프리픽스 청크와 중복이다 (rank_episodes 주석)
    from dreaming.retrieval import rank_episodes
    eps = [_ep("시장의 아침", "쌀값을 흥정했다")]
    assert rank_episodes(eps, "zzz qqq") == []
    assert rank_episodes(eps, "") == []


def test_rank_episodes_발췌도_매칭에_참여():
    from dreaming.retrieval import rank_episodes
    ep = _ep("옛 무덤터", "두 사람은 능선으로 향했다",
             key_excerpts=["12년 전, 이 산에 나와 함께 살던 이가 있었지."])
    out = rank_episodes([ep], "12년 전에 같이 살던 분 이야기")
    assert out and out[0][0].title == "옛 무덤터"


# ------------------------------------------------------------------ #
# expand_query (RAG 수리 ②)
# ------------------------------------------------------------------ #

def test_expand_query_별칭이_쿼리에_합류():
    from dreaming.records import Actor
    from dreaming.retrieval import expand_query
    actors = [Actor(names=["위지소연", "소연"], tier="main")]
    out = expand_query("소연 님이 그때 뭐라고 하셨죠", actors)
    assert "위지소연" in out
    assert out.startswith("소연 님이")      # 원 쿼리 보존


def test_expand_query_한글자_이름은_대조하지_않는다():
    # "준"이 "준비"에 걸리면 무관한 별칭이 쿼리를 오염시킨다
    from dreaming.records import Actor
    from dreaming.retrieval import expand_query
    actors = [Actor(names=["준", "어린도령"], tier="main")]
    q = "출발 준비는 끝났나요"
    assert expand_query(q, actors) == q


def test_expand_query_매칭_없으면_원본_그대로():
    from dreaming.records import Actor
    from dreaming.retrieval import expand_query
    actors = [Actor(names=["위지소연", "소연"], tier="main")]
    q = "날씨가 참 좋네요"
    assert expand_query(q, actors) == q
