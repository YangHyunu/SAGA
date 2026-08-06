"""dreaming/lore_shift.py — 1안: keyed 로어를 프리픽스 밖으로 (스펙 §5).

RisuAI는 로어북을 첫 system 메시지에 병합해 보낸다. keyed(non-constant)
엔트리가 턴마다 켜졌다 꺼지며 프리픽스를 깬다 — 실측 파손 9/9가 로어북 구획
(docs/dreaming/HANDOFF-capture-verification.md §1.2).

해법: 카드(.charx)에서 keyed 본문을 미리 뽑아두고, 매 요청 첫 system에서
string-match로 들어내 마지막 user 앞에 옮긴다. 모델은 같은 로어를 보되
위치만 뒤로 간다. 프리픽스는 constant만 남아 byte-stable해진다.

주의 — 본문만 replace하면 구분자 개행이 남아 프리픽스가 계속 흔들린다.
반드시 strip 매칭 + 빈 줄 정규화 (같은 문서 §4.2, 실제로 한 번 틀렸던 지점).
"""

from __future__ import annotations

import copy
import json
import re
import zipfile
from typing import Dict, List, Tuple

_MIN_BODY = 40  # 너무 짧은 본문은 오매칭 위험 — 프리픽스에 그냥 둔다 (fail-open)


def load_keyed(card_path: str, user_name: str) -> List[str]:
    """카드에서 keyed(non-constant) 로어 본문을 매크로 치환해 뽑는다.

    {{user}}/{{char}}만 치환한다 — 그 외 매크로가 든 엔트리는 와이어와
    바이트가 안 맞아 매칭에 안 걸리고, 그 경우 프리픽스에 남을 뿐이다.
    """
    with zipfile.ZipFile(card_path) as z:
        card = json.loads(z.read("card.json"))["data"]
    char_name = card["name"]
    out = []
    for e in card.get("character_book", {}).get("entries", []):
        if e.get("constant") or e.get("enabled") is False:
            continue
        body = (e.get("content") or "")
        if not body.strip():
            continue
        out.append(body.replace("{{user}}", user_name)
                       .replace("{{char}}", char_name))
    return out


def shift_keyed(messages: List[Dict], keyed: List[str]) -> Tuple[List[Dict], int]:
    """첫 system에서 keyed 본문을 들어내 마지막 user 앞에 붙인다.

    반환: (새 메시지 리스트, 옮긴 엔트리 수). 옮길 게 없으면 원본 그대로.
    """
    if not keyed or not messages:
        return messages, 0

    first_system = next((i for i, m in enumerate(messages)
                         if m.get("role") == "system"
                         and isinstance(m.get("content"), str)), None)
    last_user = next((i for i in range(len(messages) - 1, -1, -1)
                      if messages[i].get("role") == "user"
                      and isinstance(messages[i].get("content"), str)), None)
    if first_system is None or last_user is None or first_system >= last_user:
        return messages, 0

    s = messages[first_system]["content"]
    moved: List[Tuple[int, str]] = []          # (원래 위치, 본문) — 순서 보존
    for body in keyed:
        b = body.strip()
        if len(b) < _MIN_BODY:
            continue
        pos = s.find(b)
        if pos < 0:
            continue
        moved.append((pos, b))
        s = s.replace(b, "")
    # 공백 낀 빈 줄 연쇄를 접는다 — strip 매칭이 남긴 " \n\n" 잔류가
    # 제거 개수에 따라 달라지면 프리픽스가 흔들린다 (테스트로 재현됨).
    # 이동 0개여도 반드시 돌린다: 원문에 이미 있던 \n{3,}이 정규화 여부로
    # 턴 간 바이트가 갈린다 (실캡처 턴1에서 재현 — 69자 차이).
    s = re.sub(r"(?:[ \t]*\n){3,}", "\n\n", s)

    out = [copy.deepcopy(m) for m in messages]
    out[first_system]["content"] = s
    if moved:
        moved.sort(key=lambda t: t[0])
        lore_block = "\n\n".join(b for _, b in moved)
        out[last_user]["content"] = (
            f"<active_lorebook>\n{lore_block}\n</active_lorebook>\n\n"
            f"{out[last_user]['content']}"
        )
    return out, len(moved)
