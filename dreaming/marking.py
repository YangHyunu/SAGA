"""dreaming/marking.py — 3-BP 캐시 마킹 (스펙 §3.1, §5).

BP1 = 마지막 system (Anthropic 변환은 선두 연속 system만 병합 — anthropic.ts:209),
BP3 = 마지막 assistant (원문 꼬리 끝).
BP2 = 첫 청크 assistant — 청크가 생기는 Plan 4에서 추가된다.
RisuAI는 cachePoint를 전송 직전 제거하므로(requests.ts:141) 마킹 주체는
프록시/프로바이더인 우리다. 기존 마킹은 전부 제거 후 재마킹한다.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional


def mark_cache(messages: List[Dict], ttl: str = "5m",
               bp2_index: Optional[int] = None) -> List[Dict]:
    out = [copy.deepcopy(m) for m in messages]
    for m in out:
        m.pop("cache_control", None)

    # BP1 후보는 **선두 연속 system 구간**뿐이다. RisuAI 기본 템플릿은
    # globalNote/PHI를 히스토리 뒤에 두므로(prompt.ts:427, charx PHI→globalNote는
    # characterCards.ts:992) 전체에서 마지막 system을 잡으면 꼬리에 찍힌다.
    # 그러면 마지막 user에 prepend한 지식(assembly.py)이 캐시 span 안으로 들어가
    # 매 턴 전체 프롬프트가 재작성된다. 업스트림 변환도 선두 밖 system은
    # user로 강등하므로(anthropic.ts:233) 어차피 system 블록이 아니다.
    last_system = None
    for i, m in enumerate(out):          # BP1 후보는 선두 연속 system만 —
        if m.get("role") != "system":    # 꼬리 PHI(globalNote)는 캐시 밖
            break
        last_system = i
    last_assistant = None
    for i, m in enumerate(out):
        if m.get("role") == "assistant":
            last_assistant = i

    mark = {"type": "ephemeral", "ttl": ttl}
    if last_system is not None:
        out[last_system]["cache_control"] = dict(mark)     # BP1
    if last_assistant is not None:
        out[last_assistant]["cache_control"] = dict(mark)  # BP3
    if bp2_index is not None and 0 <= bp2_index < len(out):
        out[bp2_index]["cache_control"] = dict(mark)       # BP2 = 첫 청크
    return out
