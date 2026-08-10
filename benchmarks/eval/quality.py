"""응답 품질 게이트 — 거부·언어 드리프트·루프 판정 + 리롤."""

from __future__ import annotations

import difflib
import re
from typing import Callable, Dict, List, Sequence, Tuple

# 실유저가 리롤로 걷어내는 응답 — 남겨두면 디렉터가 캐릭터 대사를 지어내며
# 사칭하기 시작한다 (파일럿50 T3 거부 → T4 디렉터가 소연 대사 작성).
REFUSAL_MARKS = ("죄송합니다만", "처리할 수 없습니", "수행할 수 없습니",
                  "I cannot", "I can't", "I'm not able to")
# 거부는 아닌데 프리셋 내부 방어 문구가 나레이션 말미에 새는 경우 —
# 본문이 정상 한국어 산문이라 refusal·language_drift 어디에도 안 걸려
# 리롤 없이 통과했다 (night-drm-r0 T70 실측, docs/DREAMING_FLAW.md §4).
# 프리셋 중립 문구만 — 특정 프리셋 고유어를 넣지 마라.
GUARD_MARKS = ("프롬프트를 위반", "프롬프트에 위배", "시스템 프롬프트",
               "지침을 위반", "정책을 위반")
_HANGUL = re.compile(r"[가-힣]")

LOOP_LOOKBACK = 3      # 직전 몇 개 응답과 비교할지
LOOP_RATIO = 0.97      # 이 이상이면 사실상 동일 (실측: 972자/1159자 완전일치)


def reply_flaw(reply: str, prior_replies: Sequence[str] = ()) -> str:
    """리롤 사유. 정상이면 빈 문자열.

    guard_leak: 산문은 멀쩡한데 프리셋 방어 문구만 말미에 붙는 유형.
    한글 비율도 정상이고 거부 문구도 아니라 이 마커가 없으면 그냥 통과한다.
    한글 비율 임계 0.3: 파일럿 실측에서 병리 턴(영어 드리프트·프리셋 지시문
    에코)은 전부 0.09 이하, 정상 턴은 전부 0.64 이상 — 사이가 비어 있다.
    loop: 직전 lookback개 응답과 SequenceMatcher ratio>=0.97 — 실측(trim
    런 T85=T86, T91=T92) 완전 동일 응답 재현 방지.
    """
    if any(m in reply for m in REFUSAL_MARKS):
        return "refusal"
    if any(m in reply for m in GUARD_MARKS):
        return "guard_leak"
    if len(_HANGUL.findall(reply)) / max(len(reply), 1) < 0.3:
        return "language_drift"
    for prior in prior_replies[-LOOP_LOOKBACK:]:
        if difflib.SequenceMatcher(None, reply, prior).ratio() >= LOOP_RATIO:
            return "loop"
    return ""


def reroll_until_clean(call: Callable[[], Dict],
                        prior_replies: Sequence[str] = (),
                        max_rerolls: int = 2) -> Tuple[Dict, List[str]]:
    """flaw 있으면 재호출 최대 max_rerolls회. 반환: (최종 st, 시도별 flaw 이력).

    flaw_history[0]은 첫 시도, 이후는 리롤 시도 순 — 폐기된 세대의 사유도
    남긴다 (이전엔 최종 flaw만 남아 리롤 원인 분석이 불가능했다).
    prior_replies는 직전 턴 응답들 — 중복 응답(loop) 판정에 쓴다.
    """
    st = call()
    flaw = reply_flaw(st["reply"], prior_replies)
    flaw_history = [flaw]
    rerolls = 0
    while flaw and rerolls < max_rerolls:
        st2 = call()
        st2["cost"] += st["cost"]
        st = st2
        rerolls += 1
        flaw = reply_flaw(st["reply"], prior_replies)
        flaw_history.append(flaw)
    st["rerolls"], st["flaw"], st["flaw_history"] = rerolls, flaw, flaw_history
    return st, flaw_history


def abort_reroll_count(flaw_history: Sequence[str]) -> int:
    """중단 게이트(MAX_REROLL_STREAK)에 셀 리롤 수.

    마지막 항목은 최종 상태지 리롤이 아니므로 제외한다. "loop"은 거부
    반복(비용 소각)과 다른 병리라 게이트 오탐을 막기 위해 카운트에서 뺀다.
    """
    return sum(1 for f in flaw_history[:-1] if f != "loop")
