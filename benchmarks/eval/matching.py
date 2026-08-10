"""값 동치 매칭 — 최하층: eval 내부 import 금지.

오라클(scoring.py)과 프로브 누출 게이트(lucid.probe_leaks_value)가 함께 쓰는
문자열 비교 유틸. scoring.py가 이미 lucid를 import하므로(DirFact/LlmFn),
lucid가 scoring을 import하면 lucid → scoring → lucid 순환이 생긴다 — 그래서
이 최하층을 따로 판다 (config.py와 같은 규약). 의존은 stdlib +
dreaming.numerals뿐.
"""

from __future__ import annotations

import re
from typing import List

from dreaming.numerals import korean_spellings

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")


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


def value_hit(hay: str, expected_value: str) -> bool:
    """expected_value(한글 수사 변형 포함)가 hay 안에 있는지.

    계약: hay는 호출자가 이미 _norm()으로 공백을 제거한 문자열이어야 한다
    — 이 함수는 expected_value 쪽 대안만 정규화한다 (구 scoring._hit의
    계약을 그대로 승계).
    """
    return any(_norm(alt) in hay for alt in expect_alternatives(expected_value))
