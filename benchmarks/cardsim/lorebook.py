"""RisuAI 카드 로드 + 로어북 활성화 재현 (`lorebook.svelte.ts:75` 서브셋).

재현 범위 — 캐시 파괴 경로를 밟는 데 필요한 만큼만:

- constant 엔트리는 항상 활성, keyed 엔트리는 최근 `scan_depth` 메시지 스캔
- 매칭 규칙은 원본 그대로: 양쪽 lowercase + 공백 전부 제거 후 substring
  (`lorebook.svelte.ts:206`), `{{//...}}`/`{{comment:...}}` 선제거
- priority(=insertion_order) 내림차순으로 token_budget 컷 → order 내림차순
  재정렬 → 최종 reverse (`:608~662`)
- 삽입 위치는 `pos=''` 기본값 → lorebook 버킷. 기본 promptTemplate에서
  description·persona 뒤, **chat 앞** (`prompt.ts:427`)

미재현 (대상 카드가 안 쓰거나 캐시 거동과 무관):
recursive_scanning(realm 카드 False), `@@` 데코레이터, selective/secondary_keys,
use_regex, folder mode, keep/dontActivateAfterMatch 상태 플래그.
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List

_COMMENT_RE = re.compile(r"\{\{//.+?\}\}|\{\{comment:.+?\}\}")
_SPACE_RE = re.compile(r"\s+")

# RisuAI 기본값 (database.svelte.ts:77,80). 카드가 character_book에서 덮어씀.
DEFAULT_SCAN_DEPTH = 5
DEFAULT_TOKEN_BUDGET = 800

# 로어 예산 컷용 토큰 근사. RisuAI는 자체 토크나이저를 쓰므로 정확히는 못 맞춘다.
# 한/영 혼합 보수 추정 — 예산 컷이 실제로 걸리면 bench가 경고를 찍는다.
_CHARS_PER_TOKEN = 2.5


@dataclass
class LoreEntry:
    name: str
    keys: List[str]
    content: str
    order: int
    constant: bool
    tokens: int


@dataclass
class Card:
    name: str
    description: str
    post_history: str
    greeting: str
    lore: List[LoreEntry] = field(default_factory=list)
    scan_depth: int = DEFAULT_SCAN_DEPTH
    token_budget: int = DEFAULT_TOKEN_BUDGET


def _norm_for_match(text: str) -> str:
    """매칭용 정규화 — lowercase + 주석 제거 + 공백 전부 삭제."""
    return _SPACE_RE.sub("", _COMMENT_RE.sub("", text).lower())


def _est_tokens(text: str) -> int:
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def load_card(path: str, user_name: str) -> Card:
    with zipfile.ZipFile(path) as z:
        data = json.loads(z.read("card.json"))["data"]
    name = data["name"]

    def macro(text: str) -> str:
        text = re.sub(r"\{\{user\}\}", user_name, text or "", flags=re.I)
        return re.sub(r"\{\{char\}\}", name, text, flags=re.I)

    book = data.get("character_book") or {}
    lore: List[LoreEntry] = []
    for i, e in enumerate(book.get("entries") or []):
        content = (e.get("content") or "").strip()
        keys = [k.strip() for k in (e.get("keys") or []) if k.strip()]
        constant = bool(e.get("constant"))
        if not content or e.get("enabled") is False:
            continue
        if not constant and not keys:
            continue                     # 폴더/빈 엔트리
        rendered = macro(content)
        lore.append(LoreEntry(
            name=e.get("name") or e.get("comment") or f"lorebook {i}",
            keys=keys,
            content=rendered,
            order=e.get("insertion_order") or 0,
            constant=constant,
            tokens=_est_tokens(rendered),
        ))

    greet = data.get("first_mes") or (data.get("alternate_greetings") or [""])[0]
    return Card(
        name=name,
        description=macro(data.get("description") or ""),
        post_history=macro(data.get("post_history_instructions") or ""),
        greeting=macro(greet),
        lore=lore,
        scan_depth=book.get("scan_depth") or DEFAULT_SCAN_DEPTH,
        token_budget=book.get("token_budget") or DEFAULT_TOKEN_BUDGET,
    )


def activate(card: Card, history: List[Dict]) -> List[LoreEntry]:
    """이번 턴의 활성 로어를 RisuAI 최종 순서로 돌려준다.

    history: greeting을 제외한 (user/assistant) 메시지 — RisuAI가 스캔하는
    `char.chats[page].message`에 대응한다.
    """
    scanned = history[-card.scan_depth:] if card.scan_depth > 0 else []
    haystack = [_norm_for_match(m.get("content") or "") for m in scanned]

    matched = [
        e for e in card.lore
        if e.constant
        or any(_norm_for_match(k) in h for h in haystack for k in e.keys)
    ]

    # priority 내림차순 → 예산 컷 (lorebook.svelte.ts:608)
    used, kept = 0, []
    for e in sorted(matched, key=lambda x: -x.order):
        if used + e.tokens <= card.token_budget:
            used += e.tokens
            kept.append(e)

    # order 내림차순 재정렬 후 reverse = order 오름차순 (:622, :662)
    kept.sort(key=lambda x: -x.order)
    kept.reverse()
    return kept


def build_messages(card: Card, actives: List[LoreEntry],
                   history: List[Dict]) -> List[Dict]:
    """기본 promptTemplate 순서로 조립.

    plain(main) → description → persona → **lorebook** → chat → globalNote.
    persona/authornote/globalNote는 프리셋 영역이라 비운다. post_history는
    RisuAI가 히스토리 뒤에 두므로 chat 뒤 system으로 붙인다.
    """
    msgs: List[Dict] = [{"role": "system", "content": card.description}]
    if actives:
        msgs.append({"role": "system",
                     "content": "\n\n".join(e.content for e in actives)})
    if card.greeting:
        msgs.append({"role": "assistant", "content": card.greeting})
    msgs.extend(history)
    if card.post_history:
        msgs.append({"role": "system", "content": card.post_history})
    return msgs
