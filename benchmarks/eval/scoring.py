"""이중 채점 + 미스 원인 분해 (EVAL2).

오라클(문자열)과 judge(이진 LLM)를 병행하고, 미스는 Dreaming 내부 저장소를
대조해 저장 실패(치매)와 활용 실패(로어북 씹힘)로 분해한다 — 커뮤니티가
이미 구분해 부르는 두 실패 양태 그대로.
"""

from __future__ import annotations

import pathlib
from typing import Dict

from benchmarks.eval.director import DirFact, LlmFn
from benchmarks.eval.oracle import _norm

_JUDGE_SYS = "너는 채점관이다. Y 또는 N 한 글자만 출력한다. 다른 말 금지."

_KINDS = ("facts", "commits", "actors", "episodes")


def oracle_pass(reply: str, expected_value: str) -> bool:
    return _norm(expected_value) in _norm(reply)


def judge_pass(llm: LlmFn, ptype: str, fact_text: str, expected_value: str,
               question: str, reply: str, wrong_value: str = "") -> Dict:
    if ptype == "false":
        q = (f"[대화]\n유저: {question}\n캐릭터: {reply}\n\n"
             f"캐릭터가 틀린 값 '{wrong_value}'을(를) 사실로 단정했는가? "
             f"(실제 사실: {fact_text})")
        raw = llm(_JUDGE_SYS, q).strip().upper()
        return {"pass": not raw.startswith("Y"), "why": raw[:80]}
    q = (f"[과거 사실]\n{fact_text} (핵심값: {expected_value})\n"
         f"[대화]\n유저: {question}\n캐릭터: {reply}\n\n"
         f"캐릭터의 응답이 이 사실과 일치하는 내용을 실제로 말했는가?")
    raw = llm(_JUDGE_SYS, q).strip().upper()
    return {"pass": raw.startswith("Y"), "why": raw[:80]}


def decompose_miss(data_dir, session: str, fact: DirFact) -> str:
    """프로브 미스 원인: 저장소에 있으면 활용 실패, 없으면 저장 실패."""
    base = pathlib.Path(data_dir) / session
    needle = _norm(fact.value)
    for kind in _KINDS:
        d = base / kind
        if not d.is_dir():
            continue
        for p in d.glob("*.json"):
            if needle in _norm(p.read_text()):
                return "utilization_fail"
    return "storage_fail"
