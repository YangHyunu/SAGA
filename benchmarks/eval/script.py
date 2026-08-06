"""평가 대본 — 지뢰 심기 + 트림 윈도우 밖 프로브 (스펙 §9 드라이버).

지뢰는 0~15턴에 심고 프로브는 21턴 이후에 되묻는다. 트림 윈도우 W=8 pair
기준으로 모든 프로브 시점에 해당 지뢰 원문이 윈도우 밖 — 대본 자체가
장기기억 시험이 되도록 설계됐다. 리롤은 없다 (기억 비교에 집중).

유저 턴 동결: 첫 실행이 시뮬레이터 발화를 freeze_script로 저장하고, 이후
변형은 load_script로 같은 텍스트를 재생한다 — 변형 간 입력 공정성.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List

from dreaming.numerals import korean_spellings

BEATS: List[str] = [
    "정중히 자기소개를 한다. 이름은 '한결', 나이는 '스물일곱'이라고 명확히 밝힌다.",
    "상대에 대해 물어본다 — 어떻게 불러야 할지, 어떤 사람인지.",
    "지금 있는 장소/상황에 대해 자연스럽게 묻는다.",
    "자신의 소지금이 정확히 300(이 세계관의 화폐 단위)뿐이라고 대화 중에 언급한다. 숫자 300을 명시.",
    "상대의 취향이나 좋아하는 것을 묻는다.",
    "작은 선물로 먹을 것(세계관에 어울리는 것) '세 개'를 건넨다. 개수 '세 개'를 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "오늘은 물러가겠다며, '내일 자정'에 다시 오겠다고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "짧게 작별 인사를 한다.",
    # -- pause: dream #1 --
    "약속대로 다시 찾아왔다고 인사한다.",
    "값으로 50(화폐)을 치르고 마실 것이나 먹을 것을 산다. 숫자 50을 명시.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "자신이 사실 '왼손잡이'라는 것을 고백한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "'다음 보름달'에 함께 축제나 나들이를 가자고 명확히 약속한다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "지난 며칠을 회상하며 짧게 감상을 말한다.",
    "짧게 작별 인사를 한다.",
    # -- pause: dream #2 --
    "시간이 지나 다시 찾아왔다고 인사한다.",
    "자신의 이름과 나이를 기억하고 있는지 상대에게 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "장부를 잃어버렸다며, 처음 소지금과 그간 쓴 돈을 감안하면 지금 얼마가 남았을지 아는지 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "처음 만난 날 자신이 건넨 선물이 무엇이었고 몇 개였는지 기억하는지 묻는다.",
    "직전 응답에 자연스럽게 반응하며 이야기를 이어간다.",
    "예전에 자신이 '몇 시'에 다시 오겠다고 약속했었는지 묻는다.",
    "언제 어디에 함께 가자고 약속했었는지 묻는다.",
    "지금까지 자신(한결)에 대해 알게 된 것을 전부 말해달라고 한다.",
]

PAUSES: Dict[int, int] = {9: 12, 19: 12}       # beat index → idle 초 (꿈 트리거)


@dataclass
class Probe:
    turn: int                        # beat index (0-based)
    label: str
    expect: List[List[str]]          # 그룹 간 AND, 그룹 내 OR
    recall: bool = False             # True면 그룹 적중 수 m/n으로 채점


def _num(value: int, *extra: str) -> List[str]:
    return [str(value)] + korean_spellings(value) + list(extra)


PROBES: List[Probe] = [
    Probe(21, "이름·나이", [["한결"], _num(27)]),
    Probe(23, "잔액 250", [_num(250)]),
    Probe(25, "선물 세 개", [["세 개", "세개", "3개", "셋"]]),
    Probe(27, "약속 시각", [["자정", "밤 12", "12시"]]),
    Probe(28, "약속 행사", [["보름달", "보름"]]),
    Probe(29, "종합 회상", [["한결"], _num(27), ["왼손잡이", "왼손"],
                          _num(250) + _num(300), ["보름달", "보름"]],
          recall=True),
]


def freeze_script(path, turns: List[Dict]) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(turns, ensure_ascii=False, indent=1))


def load_script(path) -> List[Dict]:
    return json.loads(pathlib.Path(path).read_text())
