"""dreaming/numerals.py — 한국어 수사 표기 생성 (스펙 §3.2 B-3 검증 보조).

실카드 실측: "세 개"·"스물일곱"·"삼백"처럼 원문이 한글 수사면 아라비아
숫자 문자열 검증이 실패해 사실이 과잉 격리된다. 1~9999 정수의 한자어
표기(전 구간)와 고유어 표기(1~99)를 결정론적으로 생성한다.
"""

from __future__ import annotations

from typing import List

_NATIVE_ONES = {1: ["하나", "한"], 2: ["둘", "두"], 3: ["셋", "세"],
                4: ["넷", "네"], 5: ["다섯"], 6: ["여섯"], 7: ["일곱"],
                8: ["여덟"], 9: ["아홉"]}
_NATIVE_TENS = {1: ["열"], 2: ["스물", "스무"], 3: ["서른"], 4: ["마흔"],
                5: ["쉰"], 6: ["예순"], 7: ["일흔"], 8: ["여든"], 9: ["아흔"]}
_SINO_ONES = {1: "일", 2: "이", 3: "삼", 4: "사", 5: "오",
              6: "육", 7: "칠", 8: "팔", 9: "구"}


def _native(n: int) -> List[str]:
    tens, ones = divmod(n, 10)
    if tens == 0:
        return list(_NATIVE_ONES[ones])
    if ones == 0:
        return list(_NATIVE_TENS[tens])
    # 결합형은 기본형만: 스물일곱 (관형형 "스무"는 단독 20에서만)
    return [_NATIVE_TENS[tens][0] + o for o in _NATIVE_ONES[ones]]


def _sino(n: int) -> str:
    s = ""
    for unit, name in ((1000, "천"), (100, "백"), (10, "십")):
        d, n = divmod(n, unit)
        if d:
            s += ("" if d == 1 else _SINO_ONES[d]) + name
    if n:
        s += _SINO_ONES[n]
    return s


def korean_spellings(value: int) -> List[str]:
    if not 1 <= value <= 9999:
        return []
    out = [_sino(value)]
    if value <= 99:
        out += _native(value)
    return out
