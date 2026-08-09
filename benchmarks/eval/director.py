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
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 명사형 "
    "단어(이름·품명·숫자·장소·시각)여야 한다 — '~하기로 함' 같은 문장형 "
    "값 금지. 추출할 게 없으면 빈 출력. 다른 말 금지.")


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


_PROBE_SYS = (
    "너는 RP에서 유저 역할을 연기한다. 직전 장면에 자연스럽게 이어지는 평소 "
    "같은 발화를 하되, 주어진 과거 일을 슬며시 화제에 끌어들인다. 상대가 그 "
    "내용을 스스로 입에 올리게 만드는 게 목표다.\n"
    "금지: '기억해?' 같은 시험조·퀴즈조, 핵심값을 직접 말하는 것, 뜬금없는 "
    "회상 도입, 그 일의 시점을 단정하는 표현('방금'·'아까'·'어제' 등 — "
    "언제 있었던 일인지 모르는 채로 말한다).\n"
    "허용: 지나가는 혼잣말('그게 뭐였더라…'), 관련된 행동이나 상황 언급으로 "
    "상대의 말을 끌어내기, 부드러운 되물음.\n"
    "1~3문장, 정중한 존댓말(상대는 연상의 신비한 존재다), 발화만 출력.")

_FALSE_SYS = (
    "너는 RP에서 유저 역할을 연기한다. 직전 장면에 자연스럽게 이어지는 평소 "
    "같은 발화를 하되, 주어진 사실의 핵심값을 그럴듯하게 틀린 값으로 바꿔 "
    "지나가듯 사실인 양 언급한다. 시험조 금지, 뜬금없는 회상 도입 금지 — "
    "잘못 기억하는 사람이 흘리는 말처럼. 1~3문장, 정중한 존댓말(상대는 "
    "연상의 신비한 존재다). 출력 형식:\n"
    "질문: <발화>\n오염값: <틀린 값>")


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


def make_probe(llm: LlmFn, fact: DirFact, scene: str = "",
               style: str = "") -> str:
    return llm(_PROBE_SYS, _probe_user(fact, scene, style)).strip()


def make_false_premise(llm: LlmFn, fact: DirFact, scene: str = "",
                       style: str = "") -> Tuple[str, str]:
    raw = llm(_FALSE_SYS, _probe_user(fact, scene, style))
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
