"""프롬프트 10종 집결 — A/B 오버라이드 + 런 기록.

run2.py(디렉터 유저 페르소나·필러 비트), lucid.py(추출·프로브·false),
scoring.py(judge)에 흩어져 있던 시스템 프롬프트를 한곳에 모은다. 각 원본
모듈은 별칭 import로 기존 언더스코어 이름(`_DIRECT_SYS` 등)을 유지한다.
config는 최하층이라 여기서 import해도 순환이 없다.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Dict

from benchmarks.eval.config import NPC_NAME

_LAYER_DIR = pathlib.Path(__file__).parent / "prompts"


def _read(name: str) -> str:
    return (_LAYER_DIR / name).read_text(encoding="utf-8").strip()


# Lucid 2층 — 페르소나(누구인지)와 규칙(어떻게 행동하는지)을 분리해
# 독립적으로 버전관리·A/B할 수 있게 한다. 원문은 run2.run_once 매 턴
# system(유저 발화 생성)으로 쓰이던 단일 DIRECT_SYS를 문장 단위로 재배치한
# 것 — 새로 추가된 문구는 없다.
LUCID_PERSONA = _read("lucid_persona.md")
LUCID_RULES = _read("lucid_rules.md")
# 조립 메타 템플릿 — compose_lucid_sys()가 두 층을 여기 꽂는다.
DIRECT_SYS = "{persona}\n\n{rules}"
# 지식갱신 강제 비트 — run2.pick_beat이 UPDATE_EVENTS 턴에 사용
UPDATE_BEAT = ("이번 발화에서 이전에 언급된 수치나 소지품 상태를 명확히 바꾸는 "
               "행동을 한다 (지불, 획득, 분실 중 하나). 새 값이 드러나게.")
# 5턴마다 이야기를 미는 지시 — run2.pick_beat 회전용 (없으면 디렉터가 같은
# 장면을 맴돈다, 30턴 파일럿 실측: 한 장소 하룻밤에서 정체)
BEATS = ("장면이나 장소를 바꾸는 행동을 한다 — 되도록 사람들이 있는 "
         "곳으로, 볼일을 만들어 움직인다.",
         "외부에서 온 사건이나 소식을 장면에 들여온다 — 대화 소재가 아니라 "
         "실제로 벌어지는 일로.",
         "시간을 흘려보낸다 — 다음 날이나 몇 시간 뒤로 넘어갔음이 발화에 "
         "드러나게 한다.",
         "갈등이나 외부 압력을 만든다 — 이번 턴 안에서 풀어버리지 마라.",
         "구체적인 수치·이름·약속이 나올 만한 행동을 한다 — 거래, 날짜 잡기, "
         "무언가를 세거나 값을 치르기.")

# NPC 등장 유도 비트 — run2.pick_beat이 npc_due 턴에 사용
NPC_BEAT = (f"{NPC_NAME}(이)가 자연스럽게 장면에 합류할 상황을 만든다 — "
            f"찾아가거나, 우연히 마주치거나, 이름을 언급하며 소식을 묻는다. "
            f"'{NPC_NAME}'이라는 이름은 말해도 되지만 그 외 설정은 지어내지 "
            f"마라. {NPC_NAME}이 직접 등장해 말과 행동을 하게 만들어라 — "
            f"이름만 언급하고 지나가지 마라. 위지소연과의 장면 흐름은 "
            f"유지한다.")

# 사실 추출 페르소나 — lucid.extract_facts 매 턴 system
EXTRACT_SYS = (
    "너는 RP 대화 감독관이다. 방금 턴에서 나중에 기억력 시험에 쓸 수 있는 "
    "구체적 사실만 추출한다. 한 줄에 하나, 형식: kind|핵심값|한 문장 서술.\n"
    "kind는 exact(숫자·고유명사·시각), relation(인물 관계·호칭), "
    "event(약속·사건) 중 하나. 핵심값은 응답에 그대로 나올 법한 명사형 "
    "단어(이름·품명·숫자·장소·시각)여야 한다 — '~하기로 함' 같은 문장형 "
    "값 금지. 캐릭터 자신의 외모·신체 특징(눈동자 색·머리색·체형 등)은 "
    "추출하지 않는다 — 나레이션에 상시 노출돼 시험이 무의미하다. "
    "추출할 게 없으면 빈 출력. 다른 말 금지.")

# 거리 게이팅 프로브 페르소나 — lucid.make_probe system
PROBE_SYS = (
    "너는 RP에서 유저 역할을 연기한다. 직전 장면에 자연스럽게 이어지는 평소 "
    "같은 발화를 하되, 주어진 과거 일을 슬며시 화제에 끌어들인다. 상대가 그 "
    "내용을 스스로 입에 올리게 만드는 게 목표다.\n"
    "금지: '기억해?' 같은 시험조·퀴즈조, 핵심값을 직접 말하는 것, 뜬금없는 "
    "회상 도입, 그 일의 시점을 단정하는 표현('방금'·'아까'·'어제' 등 — "
    "언제 있었던 일인지 모르는 채로 말한다), 지시대상이 모호한 표현("
    "'그게 누구였더라'·'그거 뭐였지' 등 — 무엇/누구를 가리키는지 이 발화"
    "만 보고 알 수 없는 질문. 사실의 대상을 최소한으로 특정할 수 있는 "
    "실마리를 함께 준다).\n"
    "허용: 지나가는 혼잣말('그게 뭐였더라…'), 관련된 행동이나 상황 언급으로 "
    "상대의 말을 끌어내기, 부드러운 되물음. 발화에는 '그때'·'처음에' 같은 "
    "막연한 과거 지시어를 하나 반드시 포함한다 — 언제인지 단정하지 않으면서 "
    "과거의 일임은 분명히 한다.\n"
    "1~3문장, 정중한 존댓말(상대는 연상의 신비한 존재다), 발화만 출력.")

# 거짓 전제 프로브 페르소나 — lucid.make_false_premise system
FALSE_SYS = (
    "너는 RP에서 유저 역할을 연기한다. 직전 장면에 자연스럽게 이어지는 평소 "
    "같은 발화를 하되, 주어진 사실의 핵심값을 그럴듯하게 틀린 값으로 바꿔 "
    "지나가듯 사실인 양 언급한다. 시험조 금지, 뜬금없는 회상 도입 금지 — "
    "잘못 기억하는 사람이 흘리는 말처럼. 1~3문장, 정중한 존댓말(상대는 "
    "연상의 신비한 존재다). 출력 형식:\n"
    "질문: <발화>\n오염값: <틀린 값>")

# 이진 채점관 페르소나 — scoring.judge_pass system
JUDGE_SYS = """너는 채점관이다.

출력 형식: 한 문장 근거를 쓰고, 줄을 바꿔 마지막 줄에 Y 또는 N 한 글자만
단독으로 쓴다. 마지막 줄에 다른 문자를 붙이지 않는다.

공통 규칙:
- 캐릭터 응답은 롤플레이 산문이다. 상단 스탯바·나레이션·행동 묘사·말투는
  채점 대상이 아니다. 주장된 내용만 본다.
- 응답 길이는 판정에 영향을 주지 않는다. 길다고 맞는 것이 아니다.
- 표현이 달라도 같은 내용이면 일치로 본다 — 패러프레이즈, 아라비아 숫자와
  한글 수사("250"/"이백오십"), 날짜·시각 표기 차이는 모두 동일하게 취급한다.
- 요구된 정보 중 일부만 담겼으면 N이다."""


_NAMES = ("DIRECT_SYS", "UPDATE_BEAT", "BEATS", "NPC_BEAT",
          "EXTRACT_SYS", "PROBE_SYS", "FALSE_SYS", "JUDGE_SYS",
          "LUCID_RULES", "LUCID_PERSONA")


def active() -> Dict[str, object]:
    """현재 프롬프트 세트 스냅샷 — 결과 JSON에 기록해 A/B 추적용."""
    return {n: globals()[n] for n in _NAMES}


def compose_lucid_sys(user: str) -> str:
    """Lucid 시스템 프롬프트 단일 조립 지점 — PERSONA·RULES 층을 합친다.

    2단계로 조립한다: 먼저 str.replace로 두 층 본문을 메타 템플릿에 심고,
    그 다음에야 .format(user=...)을 건다. 순서를 바꿔 .format을 먼저 쓰면
    층 본문 안의 {user}를 만나 KeyError가 난다. globals() 접근은
    override_from()이 런타임에 재바인딩한 값을 즉시 반영하기 위함 —
    import 시점 별칭은 스냅샷이라 오버라이드가 조용히 무시된다
    (active()와 동일 패턴).
    """
    tpl = (globals()["DIRECT_SYS"]
           .replace("{rules}", globals()["LUCID_RULES"])
           .replace("{persona}", globals()["LUCID_PERSONA"]))
    return tpl.format(user=user)


def layer_hashes() -> Dict[str, str]:
    """`active()`의 각 항목을 sha256 앞 12자로 — 결과 JSON `prompt_hashes` 필드용.

    이름 목록을 따로 두지 않고 active()가 반환한 키를 그대로 따른다 — 층이
    늘어나(_NAMES 확장) active()가 항목을 더 반환하게 되면 이 함수도 수정
    없이 자동으로 늘어난 항목을 포함한다.
    """
    return {n: hashlib.sha256(str(v).encode("utf-8")).hexdigest()[:12]
            for n, v in active().items()}


def override_from(path: str) -> None:
    """A/B 실험: JSON({이름: 프롬프트})로 이름 붙은 프롬프트를 교체한다.

    BEATS처럼 tuple인 항목은 JSON 배열로 넘긴다. 미지의 키는 KeyError —
    조용히 무시하면 오타가 A/B 결과를 침묵 속에 무효화한다.
    """
    with open(path, "r", encoding="utf-8") as f:
        overrides = json.load(f)
    for name in overrides:
        if name not in _NAMES:
            raise KeyError(f"unknown prompt: {name} (valid: {', '.join(_NAMES)})")
    for name, value in overrides.items():
        globals()[name] = tuple(value) if isinstance(value, list) else value
