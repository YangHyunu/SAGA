"""judge를 쓰기 전에 judge부터 검증 (EVAL2).

참조답을 준 이진 사실채점에서 GPT-4급 judge는 사람과 잘 맞는다 —
LongMemEval은 유형당 30개씩 210개 표본에서 평균 0.97~0.98 일치를 보고했고
(arXiv:2410.10813), Thakur et al.은 TriviaQA에서 Cohen's κ 0.84를 얻었다
(arXiv:2406.12624, 사람끼리는 κ 0.9636). 그래서 기준선은 상수가 아니라
**같은 라벨에서 잰 사람끼리의 일치율**이다 — 두 명이 라벨하면 공짜로 나온다.

원시 일치율만 보면 안 된다. Thakur et al.의 핵심 지적: 원시 일치율 90%를
넘겨도 시스템 점수는 10점 넘게 갈릴 수 있어서 κ를 같이 봐야 한다. 반대로
합격률이 한쪽으로 쏠리면 κ는 실제보다 나빠 보인다(prevalence paradox) —
그래서 혼동행렬 원수까지 같이 낸다.

라벨 파일: 한 줄에 JSON 하나.
  {"ptype": "recall", "fact_text": ..., "expected_value": ..., "question": ...,
   "reply": ..., "human": true, "human2": false}
human2는 선택 — 있으면 사람끼리 일치율/κ를 같이 계산해 기준선으로 쓴다.
5개 프로브 유형에 고르게 150~210개를 권장한다 (LongMemEval 유형당 30,
ARES 최소 150 / 실사용 300).

usage: python3 -m benchmarks.eval.judge_check dreaming_data/eval/judge-labels.jsonl
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List, Tuple

from benchmarks.eval.director import LlmFn
from benchmarks.eval.scoring import judge_pass

GATE = 0.95           # human2가 없을 때만 쓰는 대체 기준선


def kappa(pairs: List[Tuple[bool, bool]]) -> float:
    """이진 Cohen's κ. 두 라벨이 완전히 한쪽으로 쏠리면 1.0으로 본다."""
    n = len(pairs)
    if not n:
        return 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    pa = sum(1 for a, _ in pairs if a) / n
    pb = sum(1 for _, b in pairs if b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe >= 1.0 else (po - pe) / (1 - pe)


def agreement(rows: List[Dict], judge: LlmFn) -> Dict:
    """judge를 라벨과 대조. 파싱 실패는 불일치가 아니라 따로 센다."""
    pairs: List[Tuple[bool, bool]] = []
    disagrees, unparsed = [], []
    matrix = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for i, r in enumerate(rows):
        got = judge_pass(judge, r["ptype"], r["fact_text"],
                         r["expected_value"], r["question"], r["reply"],
                         wrong_value=r.get("wrong_value", ""))["pass"]
        if got is None:
            unparsed.append(i)
            continue
        human = bool(r["human"])
        pairs.append((got, human))
        matrix["TP" if got and human else "FP" if got else
               "FN" if human else "TN"] += 1
        if got != human:
            disagrees.append(i)
    n = len(pairs)
    out = {"n": n, "agree": n - len(disagrees),
           "rate": (n - len(disagrees)) / n if n else 0.0,
           "kappa": kappa(pairs), "matrix": matrix,
           "disagrees": disagrees, "unparsed": unparsed}
    human_pairs = [(bool(r["human"]), bool(r["human2"])) for r in rows
                   if r.get("human2") is not None]
    if human_pairs:
        hit = sum(1 for a, b in human_pairs if a == b)
        out["human_rate"] = hit / len(human_pairs)
        out["human_kappa"] = kappa(human_pairs)
    return out


def _baseline(r: Dict) -> Tuple[float, str]:
    if "human_rate" in r:
        return r["human_rate"], "사람끼리"
    return GATE, "고정 게이트"


def main(argv: List[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = [json.loads(line) for line in
            pathlib.Path(argv[0]).read_text().splitlines() if line.strip()]
    from benchmarks.eval.run2 import make_judge_llm       # 실 judge (Sonnet)
    r = agreement(rows, make_judge_llm())
    bar, label = _baseline(r)
    print(f"judge-사람 일치 {r['agree']}/{r['n']} = {r['rate']:.1%}, "
          f"κ={r['kappa']:.3f}  기준선({label}) {bar:.1%}"
          + (f", κ={r['human_kappa']:.3f}" if "human_kappa" in r else ""))
    print(f"혼동행렬 {r['matrix']}  불일치 행 {r['disagrees']}"
          + (f"  파싱실패 {r['unparsed']}" if r["unparsed"] else ""))
    return 0 if r["rate"] >= bar else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
