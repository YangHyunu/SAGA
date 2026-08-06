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


def resolve_when(text: str, toggles: Dict[str, str]) -> str:
    """{{#when::…}} … {{:else}} … {{/when}} 해석 (중첩 지원).

    조건은 _reduce_scalars로 CBS 스칼라를 먼저 값으로 축약한 뒤 _cond로 판정.
    """
    text = _reduce_scalars(text, toggles)
    while True:
        m = _WHEN.search(text)
        if m is None:
            return text
        # 대응하는 {{/when}}을 중첩 카운트로 찾는다
        depth, i = 1, m.end()
        else_at = None
        while depth > 0:
            nxt_open = _WHEN.search(text, i)
            nxt_close = text.find("{{/when}}", i)
            nxt_else = text.find("{{:else}}", i)
            if nxt_close < 0:
                return text                      # 짝 안 맞음 — 그대로 두면 fidelity가 잡는다
            if depth == 1 and nxt_else >= 0 and nxt_else < nxt_close and (
                    nxt_open is None or nxt_else < nxt_open.start()):
                else_at = nxt_else
            if nxt_open is not None and nxt_open.start() < nxt_close:
                depth += 1
                i = nxt_open.end()
            else:
                depth -= 1
                i = nxt_close + len("{{/when}}")
        close_at = i - len("{{/when}}")
        hit = _cond(m.group(1), toggles)
        if else_at is not None:
            body_true = text[m.end():else_at]
            body_false = text[else_at + len("{{:else}}"):close_at]
        else:
            body_true, body_false = text[m.end():close_at], ""
        text = text[:m.start()] + (body_true if hit else body_false) + text[i:]


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

    card: {"description": str, "persona": str, "lore": [str], "authornote": str}
    """
    card = card or {}
    out: List[Dict] = []
    for item in preset.get("promptTemplate", []):
        t, role = item.get("type"), item.get("role", "system")
        if t == "chat":
            out.extend(_slice(history, item.get("rangeStart", 0),
                              item.get("rangeEnd", "end")))
            continue
        if t == "plain":
            content = resolve_when(item.get("text", ""), toggles)
            content = content.replace("{{slot}}", card.get("globalnote", ""))
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
        else:                                    # postEverything 등 빈 항목
            content = resolve_when(item.get("text", ""), toggles)
        if not content.strip():
            continue
        role = {"bot": "assistant"}.get(role, role)
        content = content.replace("{{char}}", char_name)
        content = content.replace("{{user}}", user_name)
        out.append({"role": role, "content": content})

    merged: List[Dict] = []
    for m in out:
        if merged and m["role"] == "system" and merged[-1]["role"] == "system":
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def reformat(msgs: List[Dict], fold_mid_system: bool = True,
             alternate: bool = True) -> List[Dict]:
    """RisuAI request.ts reformater 재현 (DeepSeek 네이티브 flags 기준).

    hasFirstSystemPrompt: 선두 system 연쇄를 하나로 떼어두고,
    나머지 system은 `system: …` user로 접은 뒤(requiresAlternateRole)
    연속 같은 역할을 병합한다. 플러그인 프로바이더가 hasFullSystemPrompt로
    동작하면(캡처로 확인 시) fold_mid_system=False로 끈다.
    """
    if not msgs:
        return msgs
    head: List[Dict] = []
    rest = list(msgs)
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
