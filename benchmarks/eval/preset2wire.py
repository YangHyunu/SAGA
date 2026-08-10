"""RisuAI 프리셋(.risup) → 실와이어 조립 (뮈토스 6.2 등 promptTemplate 프리셋).

RisuAI 클라이언트가 하는 일을 재현한다: promptTemplate 순서대로 섹션을 펼치고,
{{#when::토글::tis::N}} 조건문을 토글 값으로 해석하고, 연속 system을 하나로
병합해 실캡처와 같은 형태(선두 system 1개 + 교대 히스토리)를 만든다.

.risup 디코드 체인 (external/risuai/src/ts/storage/database.svelte.ts):
  RPack 바이트 치환 → deflate → msgpack → AES-GCM(SHA256("risupreset"), IV=0) → msgpack
"""

from __future__ import annotations

import gzip
import hashlib
import re
import zlib
from typing import Dict, List, Optional

RPACK_MAP = ("external/risuai/src/ts/rpack/rpack_map.bin")

# 미설정 전역변수의 값. getGlobalChatVar는 `?? 'null'`이고(chatVar.svelte.ts:36)
# tis/tisnot·toggle도 전부 이 전역을 읽는다(parser.svelte.ts:1256,1286,1296) —
# templateDefaultVariables 폴백은 getChatVar 전용이라 여기엔 적용되지 않는다.
UNSET = "null"

# chat 아이템 자리표 — 비어 있어도 system 병합을 끊는다 (assemble 주석 참조)
_BARRIER: Dict = {"role": "\x00barrier"}

_WHEN = re.compile(r"\{\{#when::([^{}]*)\}\}", re.S)
_SCALAR = re.compile(
    r"\{\{(getglobalvar|equal|notequal|and|or|any|not|greater|less|length)"
    r"::([^{}]*)\}\}")


def _truthy(v: str) -> bool:
    return v not in ("", "0", "false", "null")


def _reduce_scalars(text: str, toggles: Dict[str, str]) -> str:
    """중첩 CBS 스칼라({{getglobalvar}}, {{and}}, …)를 안쪽부터 값으로 축약."""
    while True:
        m = _SCALAR.search(text)
        if m is None:
            return text
        fn, args = m.group(1), m.group(2).split("::")
        if fn == "getglobalvar":
            name = args[0]
            name = name[len("toggle_"):] if name.startswith("toggle_") else name
            val = str(toggles.get(name, UNSET))
        elif fn == "equal":
            val = "1" if len(args) == 2 and args[0] == args[1] else "0"
        elif fn == "notequal":
            val = "0" if len(args) == 2 and args[0] == args[1] else "1"
        elif fn in ("and",):
            val = "1" if all(_truthy(a) for a in args) else "0"
        elif fn in ("or", "any"):
            val = "1" if any(_truthy(a) for a in args) else "0"
        elif fn in ("greater", "less"):
            try:
                a, b = float(args[0]), float(args[1])
                val = "1" if (a > b if fn == "greater" else a < b) else "0"
            except (ValueError, IndexError):
                val = "0"
        elif fn == "length":
            val = str(len(args[0]) if args else 0)
        else:                                    # not
            val = "0" if _truthy(args[0]) else "1"
        text = text[:m.start()] + val + text[m.end():]


def _cond(cond: str, toggles: Dict[str, str]) -> bool:
    """when 조건 해석: 값 하나 / toggle::VAR / [not::]VAR::tis|tisnot::값."""
    args = cond.split("::")
    neg = False
    if args and args[0] == "not":
        neg, args = True, args[1:]
    if len(args) == 1:
        hit = _truthy(args[0])
    elif len(args) == 2 and args[0] == "toggle":
        hit = _truthy(str(toggles.get(args[1], UNSET)))
    elif len(args) == 3 and args[1] in ("tis", "tisnot"):
        hit = str(toggles.get(args[0], UNSET)) == args[2]
        if args[1] == "tisnot":
            hit = not hit
    else:
        hit = False
    return not hit if neg else hit


def decode_risup(path: str, rpack_map: str = RPACK_MAP) -> Dict:
    import msgpack
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = open(path, "rb").read()
    dec_map = open(rpack_map, "rb").read()[256:512]
    data = bytes(dec_map[b] for b in raw)
    for fn in (gzip.decompress, zlib.decompress,
               lambda d: zlib.decompress(d, -15)):
        try:
            data = fn(data)
            break
        except Exception:
            continue
    outer = msgpack.unpackb(data, raw=False)
    enc = outer.get("preset") or outer.get("pres")
    key = hashlib.sha256(b"risupreset").digest()
    plain = AESGCM(key).decrypt(b"\x00" * 12, bytes(enc), None)
    return msgpack.unpackb(plain, raw=False)


def _newif(body: str, hit: bool) -> str:
    """#when 본문 처리 — parser.svelte.ts:1452-1497 (newif / newif-falsy).

    한 줄짜리 본문은 {{:else}}를 위치로 자르고 그대로 돌려준다. 여러 줄이면
    {{:else}}는 **자기 줄 전체**여야 인식되고, 고른 쪽의 앞뒤 빈 줄이 깎인다
    (`type2 !== 'keep'` 기본 경로). 이 깎기를 빼면 false 분기가 남긴 빈 줄이
    그대로 쌓여 실캡처보다 길어진다.
    """
    lines = body.split("\n")
    if len(lines) == 1:
        i = body.find("{{:else}}")
        if i >= 0:
            return body[:i] if hit else body[i + len("{{:else}}"):]
        return body if hit else ""
    else_at = next((k for k, v in enumerate(lines)
                    if v.strip() == "{{:else}}"), -1)
    if else_at < 0 and not hit:
        return ""
    if else_at >= 0:
        lines = lines[:else_at] if hit else lines[else_at + 1:]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def resolve_when(text: str, toggles: Dict[str, str]) -> str:
    """{{#when::…}} … {{:else}} … {{/when}} 해석 (중첩 지원).

    조건은 _reduce_scalars로 CBS 스칼라를 먼저 값으로 축약한 뒤 _cond로 판정.
    블록은 **안쪽부터** 푼다 — 바깥 블록의 빈 줄 깎기가 안쪽 결과를 보고
    일어나야 RisuAI 파서(재귀 하향)와 같은 결과가 나온다.
    """
    text = _reduce_scalars(text, toggles)
    while True:
        opens = list(_WHEN.finditer(text))
        if not opens:
            return text
        m = opens[-1]                            # 가장 안쪽(=마지막) 여는 태그
        close = text.find("{{/when}}", m.end())
        if close < 0:
            return text                          # 짝 안 맞음 — fidelity가 잡는다
        text = (text[:m.start()]
                + _newif(text[m.end():close], _cond(m.group(1), toggles))
                + text[close + len("{{/when}}"):])


def _slice(history: List[Dict], start, end) -> List[Dict]:
    n = len(history)
    s = start if isinstance(start, int) else 0
    e = n if end in ("end", None) else end
    return history[s if s >= 0 else max(0, n + s): e if e >= 0 else n + e]


def _fill(item: Dict, slot: str, toggles: Dict[str, str]) -> str:
    inner = item.get("innerFormat") or ""
    if inner:
        return resolve_when(inner, toggles).replace("{{slot}}", slot)
    return slot


def assemble(preset: Dict, toggles: Dict[str, str], history: List[Dict],
             card: Optional[Dict] = None, memory: str = "",
             char_name: str = "", user_name: str = "") -> List[Dict]:
    """promptTemplate → OpenAI-compat 메시지 목록 (연속 system 병합 포함).

    card: description/persona/lore/post_everything는 슬롯에 채워지고,
    system_prompt/replace_globalnote는 main·globalNote 아이템을 통째로 덮어쓴다
    ({{original}}에 원래 텍스트). authornote는 카드가 아니라 채팅방 필드
    (currentChat.note, index.svelte.ts:446)라 새 채팅에서는 비어 있다.
    """
    card = card or {}
    out: List[Dict] = []
    for item in preset.get("promptTemplate", []):
        t, role = item.get("type"), item.get("role", "system")
        if t == "chat":
            out.extend(_slice(history, item.get("rangeStart", 0),
                              item.get("rangeEnd", "end")))
            out.append(_BARRIER)
            continue
        if t == "plain":
            # plain/jailbreak/cot는 {{slot}} 치환을 하지 않는다 —
            # index.svelte.ts:1337-1377에 그 코드가 없다. 뮈토스의 Global Note
            # 아이템에 든 리터럴 {{slot}}은 실캡처에도 그대로 실려 온다.
            content = resolve_when(item.get("text", ""), toggles)
            override = {"main": card.get("system_prompt", ""),
                        "globalNote": card.get("replace_globalnote", "")
                        }.get(item.get("type2", ""), "")
            if override:                          # 카드가 프리셋 아이템을 덮어쓴다
                content = override.replace("{{original}}", content)
        elif t == "description":
            content = _fill(item, card.get("description", ""), toggles)
        elif t == "persona":
            content = _fill(item, card.get("persona", ""), toggles)
        elif t == "lorebook":
            content = _fill(item, "\n\n".join(card.get("lore", [])), toggles)
        elif t == "memory":
            content = _fill(item, memory, toggles) if memory else ""
        elif t == "authornote":
            note = card.get("authornote", "")
            content = _fill(item, note, toggles) if note else ""
        elif t == "postEverything":
            # @@depth 0 로어가 실리는 자리 (index.svelte.ts:582-590 —
            # pos==='depth' && depth===0 && role!=='assistant' → postEverything)
            post = card.get("post_everything", "")
            content = _fill(item, post, toggles) if post else ""
        else:
            content = resolve_when(item.get("text", ""), toggles)
        if not content.strip():
            continue
        role = {"bot": "assistant"}.get(role, role)
        content = content.replace("{{char}}", char_name)
        content = content.replace("{{user}}", user_name)
        out.append({"role": role, "content": content})

    # 연속 system은 합치되 chat 아이템은 비어 있어도 경계로 둔다.
    #
    # ⚠ 이 규칙은 **관측에서 나왔고 소스와 아직 화해되지 않았다**. RisuAI의
    # pushPrompts(index.svelte.ts:1235-1254)는 memo/name이 같은 연속 system을
    # 무조건 합치고, plain 아이템은 둘 다 undefined라 합쳐져야 한다. 그런데
    # 실캡처는 첫 턴(Previous Context Data 범위가 빈 턴)에도 본체(48403)와
    # Current Request(464)를 별도 메시지로 보낸다 — req-001/002/005/006 전부.
    # 벤더된 소스는 2026-07-30 스냅샷이라 배포본과 다를 수 있다.
    # 재현 대상은 실제 클라이언트이므로 관측을 따르되, 다른 프리셋으로 확장할
    # 때는 이 가정을 먼저 캡처로 다시 확인할 것.
    merged: List[Dict] = []
    for m in out:
        if m is _BARRIER:
            if merged:
                merged.append(_BARRIER)
            continue
        if (merged and merged[-1] is not _BARRIER
                and m["role"] == "system" == merged[-1]["role"]):
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    # 병합이 끝난 뒤 모든 메시지를 trim한다 (index.svelte.ts:1471-1474).
    # false로 사라진 {{#when}} 블록이 남긴 양끝 빈 줄이 여기서 없어진다.
    return [{**m, "content": m["content"].strip()}
            for m in merged if m is not _BARRIER]


def reformat(msgs: List[Dict], fold_mid_system: bool = True,
             alternate: bool = True) -> List[Dict]:
    """RisuAI request.ts reformater 재현.

    선두 system 병합과 중간 system 접기는 request.ts:355에서 **같은 조건**
    (`!hasFullSystemPrompt`) 아래 묶여 있다. 그래서 fold_mid_system 하나가
    둘 다 지배한다 — hasFullSystemPrompt 프로바이더는 선두 system도 안 합친다
    (뮈토스 캡처 req-005: system 48403 + system 464가 별도로 전송됨).

    alternate는 별개 플래그(requiresAlternateRole)다. Custom API 경로는 꺼져
    있다 — 캡처 req-006에 user가 연속 2개 그대로 실렸다.
    """
    if not msgs:
        return msgs
    head: List[Dict] = []
    rest = list(msgs)
    if fold_mid_system:
        while rest and rest[0]["role"] == "system":
            head.append(rest.pop(0))
    lead = {"role": "system",
            "content": "\n\n".join(m["content"] for m in head)} if head else None
    if fold_mid_system:
        rest = [{"role": "user", "content": f"system: {m['content']}"}
                if m["role"] == "system" else dict(m) for m in rest]
    if alternate:
        packed: List[Dict] = []
        for m in rest:
            if packed and packed[-1]["role"] == m["role"]:
                packed[-1]["content"] += "\n" + m["content"]
            else:
                packed.append(dict(m))
        rest = packed
    return ([lead] if lead else []) + rest
