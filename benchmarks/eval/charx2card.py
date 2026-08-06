"""charx(chara_card_v3) → 벤치 카드 JSON.

preset2wire.assemble이 먹는 필드(description/persona/lore/globalnote/greeting)로
평탄화한다. persona·user_name·style_examples는 카드에 없는 값이라
기존 파일 것을 보존한다.

usage:
    python3 -m benchmarks.eval.charx2card "카드.charx" dreaming_data/eval/card-x.json
"""

from __future__ import annotations

import json
import pathlib
import sys
import zipfile
from typing import Dict, List


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


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    src, dst = argv
    out = extract(src)
    path = pathlib.Path(dst)
    if path.exists():                      # persona/user_name/few-shot 보존
        old = json.loads(path.read_text())
        for k in ("persona", "user_name", "style_examples"):
            if k in old:
                out[k] = old[k]
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"{dst}: " + ", ".join(
        f"{k}={len(v) if isinstance(v, (str, list)) else v}"
        for k, v in out.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
