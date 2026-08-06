"""결정론 오라클 — 프로브 응답 채점, LLM 0콜 (스펙 §9 "숫자/상태").

공백 제거 후 substring 매칭 — 한글 수사("이백오십")는 대본 단계에서
korean_spellings로 기대답 그룹에 이미 포함돼 있다.
"""

from __future__ import annotations

import re
from typing import Dict

from benchmarks.eval.script import Probe

_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub("", text)


def score_reply(reply: str, probe: Probe) -> Dict:
    hay = _norm(reply)
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
