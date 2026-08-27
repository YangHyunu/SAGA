"""헌터스 카드 동결 전처리 — card-hunters-v1.json을 벤치용으로 결정론화.

우리 eval 조립은 RisuAI 런타임(채팅 변수 getvar, 턴별 roll)을 에뮬하지 않는다.
미해석 매크로가 와이어에 그대로 실리면 나레이터가 조건부 지시문을 전부
무조건 지시문으로 읽는다. 그래서 freeze 시점에 한 번 해석한다:

1. lang 게이트: {{#if {{equal::{{getvar::lang}}::N}}}}...{{/if}}
   → lang=1(한국어) 고정. N==1 본문 유지, N==0 제거.
2. roll 게이트: {{#if {{? {{roll::500}}<=N}} }}...{{/if}}
   → 전부 제거 ("주사위가 한 번도 안 걸린" 결정론 궤적).
3. replace_globalnote의 이미지 지침 예시 자리표시자 {{Character Image Command}}
   → CharacterName_expression 으로 치환 (진짜 매크로가 아니라 문서 예시인데,
   fidelity.check_wire_shape가 미해석 매크로로 잡는다).
4. user_name/persona 주입 (렌 — 소연 런과 동일 구조).

사용: python3 scripts/hunters_card_prep.py dreaming_data/eval/card-hunters-v1.json
결과는 제자리 덮어쓰기 + 처리 리포트 stdout. 멱등.
"""
import json
import re
import sys

LANG = re.compile(r"\{\{#if \{\{equal::\{\{getvar::lang\}\}::(\d)\}\}\}\}\n?(.*?)\{\{/if\}\}", re.S)
ROLL = re.compile(r"\{\{#if \{\{\? \{\{roll::500\}\}<=\d+\}\} \}\}\n?(.*?)\{\{/if\}\}", re.S)

PERSONA_NAME = "렌"
PERSONA = (
    "렌. 24세 남성. 2주 전 지하철 2호선 게이트 브레이크 현장에서 각성한 신규 헌터. "
    "보유 스킬 [측정] — 대상의 마력 수치·등급을 숫자로 읽는 감정계. 본인 랭크는 E."
)

stats = {"lang_keep": 0, "lang_drop": 0, "roll_drop": 0}


def resolve(text: str) -> str:
    def lang_sub(m):
        if m.group(1) == "1":
            stats["lang_keep"] += 1
            return m.group(2)
        stats["lang_drop"] += 1
        return ""

    def roll_sub(m):
        stats["roll_drop"] += 1
        return ""

    text = ROLL.sub(roll_sub, text)
    text = LANG.sub(lang_sub, text)
    return text


def main(path: str) -> None:
    card = json.load(open(path))
    card["greeting"] = resolve(card.get("greeting", ""))
    card["description"] = resolve(card.get("description", ""))
    card["lore"] = [
        resolve(e) if isinstance(e, str)
        else {**e, "content": resolve(e.get("content", ""))}
        for e in card.get("lore", [])
    ]
    card["keyed_lore"] = [
        {**e, "content": resolve(e["content"])} for e in card.get("keyed_lore", [])
    ]
    pe = card.get("post_everything", "")
    card["post_everything"] = resolve(pe if isinstance(pe, str) else "".join(pe))
    card["replace_globalnote"] = resolve(
        card.get("replace_globalnote", "")
    ).replace("{{Character Image Command}}", "CharacterName_expression")
    card["user_name"] = PERSONA_NAME
    card["persona"] = PERSONA

    residual = sum(
        t.count("{{#if")
        for t in [card["greeting"], card["description"], card["post_everything"]]
        + [e if isinstance(e, str) else e.get("content", "") for e in card["lore"]]
        + [e["content"] for e in card["keyed_lore"]]
    )
    json.dump(card, open(path, "w"), ensure_ascii=False, indent=1)
    print(f"lang 유지 {stats['lang_keep']} / 제거 {stats['lang_drop']}, "
          f"roll 제거 {stats['roll_drop']}, 잔여 {{{{#if {residual}")
    if residual:
        sys.exit(f"잔여 {{#if}} {residual}개 — 패턴 미매치, 수동 확인 필요")


if __name__ == "__main__":
    main(sys.argv[1])
