"""디렉터: 동적 사실 추출 + 거리 게이팅 프로브.

The Seed DIRECTOR 방식 (EVAL2 §3): 사실을 미리 심지 않고, 롤플레이가 자연히
만든 사실(가격·인명·관계·사건)을 턴마다 추출해 원장에 쌓고, 가시 창 밖으로
밀려난 사실만 자연스러운 질문으로 되묻는다.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

from benchmarks.eval import prompts
# 별칭 재노출 — 기존 테스트(director._PROBE_SYS 등)가 이 이름으로 내용을
# 검증한다. 실제 호출부는 override_from이 반영되도록 prompts.X(점 접근)를
# 쓴다 — 이 별칭은 이 파일 안에서는 안 쓰인다 (재노출 목적).
from benchmarks.eval.prompts import (EXTRACT_SYS as _EXTRACT_SYS,  # noqa: F401
                                     FALSE_SYS as _FALSE_SYS,  # noqa: F401
                                     PROBE_SYS as _PROBE_SYS)  # noqa: F401

LlmFn = Callable[[str, str], str]          # (system, user) -> 응답 텍스트


@dataclass
class DirFact:
    fid: str
    kind: str
    value: str
    text: str
    turn: int
    probed: bool = False


def extract_facts(llm: LlmFn, user_text: str, reply_text: str,
                  turn_no: int) -> List[DirFact]:
    # prompts.EXTRACT_SYS (dotted, not 모듈 상단 별칭) — override_from은 이
    # 모듈이 아니라 prompts 모듈 전역을 재바인딩하므로, 호출 시점에 다시
    # 읽어야 A/B 오버라이드가 실제로 반영된다.
    raw = llm(prompts.EXTRACT_SYS,
              f"[유저]\n{user_text[-600:]}\n[캐릭터]\n{reply_text[-1200:]}")
    out: List[DirFact] = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 3 or parts[0] not in ("exact", "relation", "event"):
            continue
        out.append(DirFact(fid=uuid.uuid4().hex[:8], kind=parts[0],
                           value=parts[1], text=parts[2], turn=turn_no))
    return out


class Ledger:
    def __init__(self) -> None:
        self.facts: List[DirFact] = []

    def add(self, facts: List[DirFact]) -> None:
        self.facts.extend(facts)

    def unprobed(self, kind: Optional[str] = None) -> List[DirFact]:
        return [f for f in self.facts
                if not f.probed and (kind is None or f.kind == kind)]

    def to_rows(self) -> List[Dict]:
        return [asdict(f) for f in self.facts]

    @classmethod
    def from_rows(cls, rows: List[Dict]) -> "Ledger":
        led = cls()
        led.facts = [DirFact(**r) for r in rows]
        return led


# LITM 분리 게이팅: 창 밖 여부가 아니라 사실 나이로 출제한다. evict-전용
# 게이팅은 창 안 원거리 실패(lost in the middle)를 못 재고 창이 클수록
# 벤치가 관대해진다. evict 여부는 run2가 프로브에 in_window로 별도 기록.
# 파일럿 실측 — 풀컨텍스트 vanilla가 dist 19+에서 실패, dist 9는 통과.
MIN_PROBE_AGE = 15   # 이 나이(현재턴-기록턴)부터 원거리 프로브 출제
RECENT_MAX_AGE = 8   # 단기 대조군(recent) 상한


def eligible(ledger: Ledger, turn_now: int, kind: Optional[str] = None,
             min_age: int = MIN_PROBE_AGE) -> List[DirFact]:
    """나이가 min_age 이상인 미출제 사실."""
    return [f for f in ledger.unprobed(kind)
            if turn_now - f.turn >= min_age]


def _probe_user(fact: DirFact, scene: str, style: str) -> str:
    parts = []
    if scene:
        parts.append(f"[직전 캐릭터 응답 — 여기에 이어서 말한다]\n{scene[-800:]}")
    if style:
        parts.append(f"[유저 문체 예시]\n{style}")
    parts.append(f"[과거 사실]\n{fact.text} (핵심값: {fact.value})")
    return "\n".join(parts)


_PARTICLES = ("에게서", "에서", "으로", "이라서", "이지만", "하고", "까지",
             "부터", "이나", "라도", "은", "는", "이", "가", "을", "를",
             "의", "와", "과", "도", "만", "에", "로")
_STOPWORDS = {"있다", "한다", "했다", "됐다"}


def _strip_particle(word: str) -> str:
    for p in _PARTICLES:
        if word.endswith(p) and len(word) > len(p):
            return word[:-len(p)]
    return word


def _probe_mentions_fact_object(fact: DirFact, utext: str) -> bool:
    """생성된 프로브가 사실의 핵심 대상 명사를 담았는지 대략 확인.

    완벽한 개체명 인식이 아니라 명백한 대상 치환(실측: '저고리'→'옷감')만
    걸러낸다 — 정밀도 낮음, 야간 로그로 재보정 필요.
    """
    words = {_strip_particle(w) for w in re.findall(r"[가-힣]{2,}", fact.text)}
    words -= _STOPWORDS
    words = {w for w in words if w not in fact.value and fact.value not in w}
    return not words or any(w in utext for w in words)


def make_probe(llm: LlmFn, fact: DirFact, scene: str = "",
               style: str = "") -> str:
    return llm(prompts.PROBE_SYS, _probe_user(fact, scene, style)).strip()


def make_false_premise(llm: LlmFn, fact: DirFact, scene: str = "",
                       style: str = "") -> Tuple[str, str]:
    raw = llm(prompts.FALSE_SYS, _probe_user(fact, scene, style))
    q, wrong = "", ""
    for line in raw.splitlines():
        if line.startswith("질문:"):
            q = line[3:].strip()
        elif line.startswith("오염값:"):
            wrong = line[4:].strip()
    return q, wrong


_PTYPE_KIND = {"recall": "exact", "relation": "relation",
               "false": None, "update": "exact", "recent": None}


def probe_plan(ledger: Ledger, turn_now: int, want: Dict[str, int],
               min_age: int = MIN_PROBE_AGE) -> List[Tuple[str, DirFact]]:
    """유형별 수만큼 뽑고 probed 마킹.

    recent만 젊은 사실(단기 대조군), 나머지는 나이 min_age 이상에서 뽑는다.
    """
    plan: List[Tuple[str, DirFact]] = []
    for ptype, n in want.items():
        if ptype == "recent":
            pool = [f for f in ledger.unprobed(_PTYPE_KIND[ptype])
                    if turn_now - f.turn <= RECENT_MAX_AGE]
        else:
            pool = eligible(ledger, turn_now, kind=_PTYPE_KIND[ptype],
                            min_age=min_age)
        pool.sort(key=lambda f: f.turn)
        for f in pool[:n]:
            f.probed = True
            plan.append((ptype, f))
    return plan
