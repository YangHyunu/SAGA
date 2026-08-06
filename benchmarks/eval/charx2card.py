"""charx(chara_card_v3) → 벤치 카드 JSON.

preset2wire.assemble이 먹는 필드(description/persona/lore/globalnote/greeting)로
평탄화한다. 유저 페르소나는 캐릭터 카드에 없어서 RisuAI 페르소나 export(PNG의
tEXt persona 청크)에서 따로 읽는다. style_examples는 어느 쪽에도 없으니
기존 파일 것을 보존한다.

usage:
    python3 -m benchmarks.eval.charx2card 카드.charx card-x.json [페르소나.png]
"""

from __future__ import annotations

import base64
import json
import pathlib
import struct
import sys
import zipfile
from typing import Dict, List, Tuple


def _lore(book: Dict) -> List[str]:
    """항상 활성(constant) 엔트리를 insertion_order 순으로."""
    entries = [e for e in book.get("entries", []) if e.get("constant")]
    entries.sort(key=lambda e: e.get("insertion_order", 0))
    return [e["content"] for e in entries if e.get("content")]


def extract(charx_path: str) -> Dict:
    with zipfile.ZipFile(charx_path) as z:
        card = json.loads(z.read("card.json"))
    d = card["data"]
    greetings = [g for g in [d.get("first_mes", "")]
                 + list(d.get("alternate_greetings", [])) if g.strip()]
    return {
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "greeting": greetings[0] if greetings else "",
        "lore": _lore(d.get("character_book") or {}),
        "globalnote": d.get("system_prompt", ""),
        "authornote": d.get("post_history_instructions", ""),
    }


def extract_persona(png_path: str) -> Tuple[str, str]:
    """RisuAI 페르소나 export → (이름, 페르소나 본문).

    PNG tEXt 청크 'persona'에 base64 JSON({name, personaPrompt, note})이 들어 있다.
    """
    raw = pathlib.Path(png_path).read_bytes()
    i = 8                                  # PNG 시그니처 뒤부터 청크 순회
    while i < len(raw):
        size = struct.unpack(">I", raw[i:i + 4])[0]
        typ = raw[i + 4:i + 8]
        if typ == b"tEXt":
            key, _, val = raw[i + 8:i + 8 + size].partition(b"\x00")
            if key == b"persona":
                obj = json.loads(base64.b64decode(val))
                return obj.get("name", ""), obj.get("personaPrompt", "")
        if typ == b"IEND":
            break
        i += 12 + size
    raise SystemExit(f"{png_path}: persona 청크 없음")


def main(argv: List[str]) -> int:
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    src, dst = argv[0], argv[1]
    out = extract(src)
    path = pathlib.Path(dst)
    if path.exists():                      # persona/user_name/few-shot 보존
        old = json.loads(path.read_text())
        for k in ("persona", "user_name", "style_examples"):
            if k in old:
                out[k] = old[k]
    if len(argv) == 3:
        out["user_name"], out["persona"] = extract_persona(argv[2])
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{dst}: " + ", ".join(
        f"{k}={len(v) if isinstance(v, (str, list)) else v}"
        for k, v in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
