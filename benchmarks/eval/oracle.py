"""결정론 오라클 — 프로브 응답 채점, LLM 0콜 (스펙 §9 "숫자/상태").

공백 제거 후 substring 매칭 — 한글 수사("이백오십")는 대본 단계에서
korean_spellings로 기대답 그룹에 이미 포함돼 있다.
"""

from __future__ import annotations

import re
from typing import Dict, List

from benchmarks.eval.script import Probe
from dreaming.numerals import korean_spellings

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
# 카드 스탯바 헤더 — 선두 "[ ... ]" 블록(+구분선). 이름·나이가 상시 박혀
# 있어 채점에 넣으면 모든 변형이 공짜 적중한다.
_STATBAR = re.compile(r"^\s*\[[^\]]*\]\s*(?:-{2,}\s*)?")


def _norm(text: str) -> str:
    return _WS.sub("", text)


def expect_alternatives(expected_value: str) -> List[str]:
    """기대값 하나 → 동치 표현 목록. 안에 든 정수마다 한글 수사를 붙인다.

    대본(script.PROBES)은 기대 그룹을 손으로 적어 두지만 디렉터 벤치의
    DirFact는 값이 문자열 하나뿐이라 여기서 만들어 쓴다.
    """
    alts = [expected_value]
    for digits in _DIGITS.findall(expected_value):
        alts += korean_spellings(int(digits))
    return alts


def score_reply(reply: str, probe: Probe) -> Dict:
    hay = _norm(_STATBAR.sub("", reply))
    matched = sum(1 for group in probe.expect
                  if any(_norm(c) in hay for c in group))
    total = len(probe.expect)
    if matched == total:
        hit = "full"
    elif matched > 0:
        hit = "partial"
    else:
        hit = "miss"
    return {"label": probe.label, "hit": hit,
            "matched": matched, "total": total}
