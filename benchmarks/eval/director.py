"""디렉터: 동적 사실 추출 + 거리 게이팅 프로브.

The Seed DIRECTOR 방식 (EVAL2 §3): 사실을 미리 심지 않고, 롤플레이가 자연히
만든 사실(가격·인명·관계·사건)을 턴마다 추출해 원장에 쌓고, 가시 창 밖으로
밀려난 사실만 자연스러운 질문으로 되묻는다.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple

LlmFn = Callable[[str, str], str]          # (system, user) -> 응답 텍스트

_EXTRACT_SYS = (
    "너는 RP 대화 감독관이다. 방금 턴에서 나중에 기억력 시험에 쓸 수 있는 "
    "구체적 사실만 추출한다. 한 줄에 하나, 형식: kind|핵심값|한 문장 서술.\n"
    "kind는 exact(숫자·고유명사·시각), relation(인물 관계·호칭), "
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 짧은 문자열. "
    "추출할 게 없으면 빈 출력. 다른 말 금지.")


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
    raw = llm(_EXTRACT_SYS,
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
