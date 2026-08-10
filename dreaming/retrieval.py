"""dreaming/retrieval.py — 지식 검색: 어휘 랭킹 + 장면 쿼리 (스펙 §3.1-2).

fix-drm-r0 오프라인 실측(docs/FINDINGS-2026-08-10-fix-run.md §4)에서
어휘(IDF+bigram)가 덴스(bge-m3)·RRF 하이브리드를 이겼다 — 한국어 RP는
고유명사·아이템이 리터럴로 등장하고, 덴스는 어휘 상위를 오히려 깎았다.
그래서 어휘 단독이다. LLM 0콜, 영속 인덱스 없음 — fact 수백 개 규모에선
요청 시 즉석 계산이 충분히 싸다 (329개 기준 ~ms).
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Set

from dreaming.records import Fact

_RUN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

# 직전 assistant 응답의 쿼리 참여 상한. 플랜 리뷰 실측: 유저 턴 평균 206자
# vs 어시스턴트 1,925자라 장면 전체(2400자)를 쿼리로 쓰면 ~90%가 모델
# 산문이 되어 정답 순위가 오히려 밀렸다 (T39 1위→33위). 유저턴+직전응답
# 600자 캡이 4개 프로브 중 3개에서 최선이었다.
_PREV_ASSISTANT_CAP = 600


def features(text: str) -> Set[str]:
    """토큰 + 문자 bigram. 한국어는 조사가 붙어 토큰 일치가 잘 안 되므로
    bigram이 주력이고, 토큰 자체는 영단어·숫자 리터럴 일치를 담당한다."""
    out: Set[str] = set()
    for run in _RUN_RE.findall(text):
        run = run.lower()
        out.add(run)
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
    return out


def rank_facts(facts: List[Fact], query: str) -> List[Fact]:
    """pinned 먼저, 그 안에서 어휘 점수 내림차순, 동점은 최신순.

    점수 0인 사실은 자동으로 최신순 꼬리가 된다 — 예산이 남으면 기존
    최신순 동작이 바닥값으로 유지된다 (쿼리가 빈약한 초반 턴 방어).
    빈 쿼리면 계산 없이 기존 동작(핀 우선+최신순) 그대로.
    """
    if not query.strip():
        return sorted(facts, key=lambda f: (f.pinned, f.recorded_at),
                      reverse=True)
    q = features(query)
    n = len(facts) or 1
    df: Dict[str, int] = {}
    docs: List[Set[str]] = []
    for f in facts:
        fs = features(f.claim + " " + " ".join(f.entities))
        docs.append(fs)
        for feat in fs:
            df[feat] = df.get(feat, 0) + 1
    scores = [sum(math.log(1.0 + n / (1 + df[feat])) for feat in (q & docs[i]))
              for i in range(len(facts))]
    order = sorted(range(len(facts)),
                   key=lambda i: (facts[i].pinned, scores[i],
                                  facts[i].recorded_at),
                   reverse=True)
    return [facts[i] for i in order]


def scene_query(messages: List[Dict]) -> str:
    """지식 검색 쿼리 = 마지막 유저 턴 + 직전 assistant 응답 600자 캡.

    직전 응답을 붙이는 이유: 대명사 질문("그거 어디 씀?")의 선행사가
    직전 응답에 있다. 600자 캡인 이유: 위 _PREV_ASSISTANT_CAP 주석.
    """
    cur = ""
    prev_a = ""
    for m in reversed(messages):
        c = m.get("content")
        if not isinstance(c, str) or not c.strip():
            continue
        role = m.get("role")
        if not cur:
            if role == "user":
                cur = c
            continue
        if role == "assistant":
            prev_a = c[-_PREV_ASSISTANT_CAP:]
            break
    return (prev_a + "\n" + cur) if prev_a else cur
