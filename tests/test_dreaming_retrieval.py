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
