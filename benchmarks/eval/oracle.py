# [v1 하네스] 현행은 run2 계열 — 이 파일은 구 테스트·PR 호환용으로 보존.
"""결정론 오라클 — 프로브 응답 채점, LLM 0콜 (스펙 §9 "숫자/상태").

공백 제거 후 substring 매칭 — 한글 수사("이백오십")는 대본 단계에서
korean_spellings로 기대답 그룹에 이미 포함돼 있다.
"""

from __future__ import annotations

from typing import Dict

from benchmarks.eval.script import Probe
# expect_alternatives는 이 파일 안에서는 안 쓰임 — 구 임포터 호환용 재노출.
from benchmarks.eval.scoring import _STATBAR, _norm, expect_alternatives  # noqa: F401


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
