"""이중 채점 + 미스 원인 분해 (EVAL2).

오라클(결정론)과 judge(이진 LLM)를 병행하고, 미스는 Dreaming 내부 저장소를
대조해 저장 실패(치매)와 활용 실패(로어북 씹힘)로 분해한다 — 커뮤니티가
이미 구분해 부르는 두 실패 양태 그대로.

judge 프롬프트 설계 근거:
- 참조답 제시(reference-guided). MT-Bench §3.3에서 추론 과제 오판 70%→15%로
  가장 효과 큰 개입. Prometheus·LongMemEval도 동일하게 정답을 프롬프트에 준다.
- 이진 판정. 사실 재현 채점은 전부 이진이 관행 (LongMemEval, mem0, ARES).
  RAGAS Table 1은 같은 데이터에서 이진 0.95 vs Likert 0.72.
- 유형별 프롬프트. LongMemEval은 5개 유형에 각각 다른 템플릿을 쓴다. 특히
  knowledge-update는 "이전 값이 같이 나와도 갱신값이 맞으면 정답" 조항이,
  abstention은 부정형이 아니라 **긍정형** 질문이 필요하다.
- 파싱 실패에 기본값을 두지 않는다. 예전 구현은 startswith("Y")라서 파싱
  실패가 일반 프로브에선 오답, false 프로브에선(부정 반전) 정답이 됐다 —
  같은 노이즈가 유형별로 반대 방향 편향을 만든다. 이제 None을 반환하고
  집계에서 분모에서 뺀다.
"""

from __future__ import annotations

import pathlib
import re
from typing import Dict, List, Optional

from benchmarks.eval import prompts
from benchmarks.eval.lucid import DirFact, LlmFn
from dreaming.numerals import korean_spellings

_WS = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
# 카드 스탯바 헤더 — 선두 "[ ... ]" 블록(+구분선). 이름·나이가 상시 박혀
# 있어 채점에 넣으면 모든 변형이 공짜 적중한다.
_STATBAR = re.compile(r"^\s*\[[^\]]*\]\s*(?:-{2,}\s*)?")


def _norm(text: str) -> str:
    return _WS.sub("", text)


def expect_alternatives(expected_value: str) -> List[str]:
    """기대값 하나 → 동치 표현 목록. 안에 든 정수마다 한글 수사를 붙인다.

    대본(script.PROBES)은 기대 그룹을 손으로 적어 두지만 디렉터 벤치의
    DirFact는 값이 문자열 하나뿐이라 여기서 만들어 쓴다.
    """
    alts = [expected_value]
    for digits in _DIGITS.findall(expected_value):
        alts += korean_spellings(int(digits))
    return alts


_KINDS = ("facts", "commits", "actors", "episodes")

_Q_BASE = """[과거 사실]
{fact} (핵심값: {value})

[대화]
유저: {question}
캐릭터: {reply}

캐릭터가 이 사실과 일치하는 내용을 실제로 말했는가?"""

_Q_UPDATE = """[과거 사실 — 값이 갱신됐다]
{fact} (갱신된 현재값: {value})

[대화]
유저: {question}
캐릭터: {reply}

캐릭터가 갱신된 현재값을 말했는가? 이전 값이 함께 언급되더라도 갱신값이
최종 답으로 제시됐다면 Y다. 이전 값만 말했으면 N이다."""

_Q_FALSE = """[실제 사실]
{fact} (참값: {value})

[대화 — 유저가 '{wrong}'이라는 틀린 전제를 깔았다]
유저: {question}
캐릭터: {reply}

캐릭터가 틀린 전제 '{wrong}'을(를) 적극적으로 바로잡거나 부정했는가?
정정·부정의 명시적 근거가 있어야 Y다. 회피·화제 전환·무시·침묵·얼버무림은
전부 N이다."""


def _hit(hay: str, expected_value: str) -> bool:
    return any(_norm(alt) in hay for alt in expect_alternatives(expected_value))


def oracle_pass(reply: str, expected_value: str,
                wrong_value: str = "", char_name: str = "") -> Optional[bool]:
    """결정론 채점. 스탯바를 벗기고 한글 수사 동치를 허용한다.

    스탯바에는 이름·나이가 상시 박혀 있어 그대로 두면 모든 변형이 공짜로
    적중한다 (oracle.score_reply와 같은 처리 — 예전 구현은 이걸 빼먹었다).
    wrong_value가 주어지면(false 프로브) 오염값 복창은 즉시 실패다.

    기대값이 캐릭터 이름의 일부면 None — 나레이션이 캐릭터 이름을 상시
    언급하므로("소연은 눈을 감은 채…") 문자열 포함으로는 판정 불가다.
    파일럿 실측: relation 프로브(값 "소연")가 오라클 공짜 적중, judge는 N.
    """
    if char_name and _norm(expected_value) in _norm(char_name):
        return None
    hay = _norm(_STATBAR.sub("", reply))
    if wrong_value and _hit(hay, wrong_value):
        return False
    return _hit(hay, expected_value)


def _verdict(raw: str) -> Optional[bool]:
    """마지막 단독 Y/N 줄만 읽는다. 못 읽으면 None — 기본값을 두지 않는다."""
    for line in reversed(raw.strip().splitlines()):
        token = re.sub(r"[^A-Za-z]", "", line).upper()
        if token in ("Y", "N"):
            return token == "Y"
    return None


def judge_pass(llm: LlmFn, ptype: str, fact_text: str, expected_value: str,
               question: str, reply: str, wrong_value: str = "") -> Dict:
    """이진 judge. pass는 True/False, 파싱 실패 시 None.

    why에는 근거 문장을 남긴다 — 불일치 감사를 하려면 판정 이유가 필요하다.
    """
    if ptype == "false":
        q = _Q_FALSE.format(fact=fact_text, value=expected_value,
                            question=question, reply=reply, wrong=wrong_value)
    elif ptype == "update":
        q = _Q_UPDATE.format(fact=fact_text, value=expected_value,
                             question=question, reply=reply)
    else:
        q = _Q_BASE.format(fact=fact_text, value=expected_value,
                           question=question, reply=reply)
    raw = llm(prompts.JUDGE_SYS, q).strip()
    why = " ".join(raw.split())[:200]
    return {"pass": _verdict(raw), "why": why}


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
