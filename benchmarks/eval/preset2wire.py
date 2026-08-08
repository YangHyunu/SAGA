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
            out.append(_BARRIER)
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
    return [m for m in merged if m is not _BARRIER]


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
