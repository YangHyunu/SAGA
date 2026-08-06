"""judge를 쓰기 전에 judge부터 검증 (EVAL2).

PersonaEval 경고(최고 judge도 69% vs 인간 90.8%)에 따라, 사람 라벨 표본과의
일치율 95%+를 확인해야 본채점에 judge를 쓴다.

usage: python3 -m benchmarks.eval.judge_check dreaming_data/eval/judge-labels.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

from benchmarks.eval.director import LlmFn
from benchmarks.eval.scoring import judge_pass

GATE = 0.95


def agreement(rows: List[Dict], judge: LlmFn) -> Dict:
    agree, disagrees = 0, []
    for i, r in enumerate(rows):
        got = judge_pass(judge, r["ptype"], r["fact_text"],
                         r["expected_value"], r["question"], r["reply"],
                         wrong_value=r.get("wrong_value", ""))["pass"]
        if got == bool(r["human"]):
            agree += 1
        else:
            disagrees.append(i)
    n = len(rows)
    return {"n": n, "agree": agree,
            "rate": agree / n if n else 0.0, "disagrees": disagrees}


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = [json.loads(line) for line in
            pathlib.Path(argv[0]).read_text().splitlines() if line.strip()]
    from benchmarks.eval.run2 import make_judge_llm       # 실 judge (Sonnet)
    r = agreement(rows, make_judge_llm())
    print(f"judge-사람 일치 {r['agree']}/{r['n']} = {r['rate']:.1%} "
          f"(게이트 {GATE:.0%}) 불일치 행: {r['disagrees']}")
    return 0 if r["rate"] >= GATE else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
