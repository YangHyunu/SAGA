"""dreaming/assembly.py — 지식 주입 조립 (스펙 §3.1, §5).

지식 계층은 아무리 바뀌어도 캐시를 깨지 않아야 한다 → 주입 위치는
오직 마지막 user 메시지 prepend (프리픽스 밖). system 주입 금지 —
hypaV3의 선두 system 삽입(hypav3.ts:1593)이 반면교사다.

**형식은 내용물처럼, 지시처럼 보이면 안 된다.** 예전엔 지식을
`<dreaming_context>` 태그로 감쌌는데, 이게 지시문으로 읽혀 프리셋의 주입
방어를 건드렸다 (night-drm-r0 실측: T36부터 dreaming에만 "——" 더듬기 붕괴,
T70에 프리셋 내부 역할 가드 문구가 나레이션에 유출 — 나머지 3변형 0건).
지금은 로어북과 같은 평문 서술 블록이다. RisuAI가 로어북을 평문으로 끼워
넣고 모든 프리셋이 그와 공존하도록 설계돼 있다는 게 근거다 — 생태계에서
이미 검증된 유일한 주입 형식. 명령문·역할 주장·XML 태그를 다시 넣지 마라.
(docs/DREAMING_FLAW.md §4.5)
"""

from __future__ import annotations

import copy
from typing import Dict, List

# ≈2K tokens (스펙 §3.1 hot zone). 한/영 혼합 보수 추정 — 후속 플랜에서
# 토크나이저 기반으로 교체한다.
HOT_ZONE_CHAR_BUDGET = 6000

# 지식 블록과 유저 발화의 경계. 마크다운 수평선 — 문서 서식이지 지시가 아니다.
KNOWLEDGE_SEP = "\n\n---\n\n"


def clip_knowledge(text: str, budget: int = HOT_ZONE_CHAR_BUDGET) -> str:
    return text[:budget]


def inject_knowledge(messages: List[Dict], knowledge: str) -> List[Dict]:
    if not knowledge:
        return messages
    last_user = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user = i
            break
    if last_user is None:
        return messages
    out = [copy.deepcopy(m) for m in messages]
    out[last_user]["content"] = (
        f"{knowledge}{KNOWLEDGE_SEP}{out[last_user]['content']}")
    return out
