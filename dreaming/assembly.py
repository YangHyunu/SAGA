"""dreaming/assembly.py — 지식 주입 조립 (스펙 §3.1, §5).

지식 계층은 아무리 바뀌어도 캐시를 깨지 않아야 한다 → 주입 위치는
오직 마지막 user 메시지 prepend (프리픽스 밖). system 주입 금지 —
hypaV3의 선두 system 삽입(hypav3.ts:1593)이 반면교사다.
"""

from __future__ import annotations

import copy
from typing import Dict, List

# ≈2K tokens (스펙 §3.1 hot zone). 한/영 혼합 보수 추정 — 후속 플랜에서
# 토크나이저 기반으로 교체한다.
HOT_ZONE_CHAR_BUDGET = 6000


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
        f"<dreaming_context>\n{knowledge}\n</dreaming_context>\n\n"
        f"{out[last_user]['content']}"
    )
    return out
