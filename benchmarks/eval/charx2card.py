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
import re
import struct
import sys
import zipfile
from typing import Dict, List, Optional, Tuple

_DECO = re.compile(r"^\s*@@(\w+)([^\n]*)\n?", re.M)
# RisuAI가 인식해 content에서 제거하는 데코레이터 중 우리가 배치를 재현할 수
# 있는 것들. 그 외 데코레이터는 활성 조건·순서·역할을 바꾸므로 조용히 지나가면
# 안 된다 (lorebook.svelte.ts:300-514 switch 전수).
_HANDLED = {"depth", "end"}
# RisuAI lorebook.svelte.ts:302-507 switch가 인식하는 전체 목록 (case-sensitive).
# 여기 있는데 _HANDLED에 없으면 = 본체는 처리하지만 우리는 재현 불가 → 중단.
# 여기도 없으면 = 본체도 몰라서 줄만 지우고 무시 (511-513 default) → 미러.
_RISU_KNOWN = {
    "end", "activate_only_after", "activate_only_every",
    "keep_activate_after_match", "dont_activate_after_match", "depth",
    "reverse_depth", "instruct_depth", "reverse_instruct_depth",
    "instruct_scan_depth", "role", "scan_depth", "is_greeting", "position",
    "inject_lore", "inject_at", "inject_replace", "inject_prepend",
    "ignore_on_max_context", "additional_keys", "exclude_keys",
    "exclude_keys_all", "match_full_word", "match_partial_word",
    "is_user_icon", "activate", "dont_activate", "disable_ui_prompt",
    "probability", "priority", "unrecursive", "recursive",
    "no_recursive_search",
}


def _strip_deco(content: str) -> Tuple[str, Optional[int]]:
    """데코레이터 줄을 떼고 (본문, depth)를 돌려준다.

    RisuAI는 CCardLib.decorator.parse로 인식한 데코레이터 줄을 content에서
    제거한 뒤 프롬프트에 싣는다(lorebook.svelte.ts:300). 실캡처의
    postEverything 블록에도 `@@depth 0` 줄이 없다 — 남기면 와이어가 어긋난다.
    """
    depth = None
    for m in _DECO.finditer(content):
        name, arg = m.group(1), m.group(2).strip()
        if name in _HANDLED:
            depth = 0 if name == "end" else int(arg)
        elif name in _RISU_KNOWN:
            raise SystemExit(
                f"@@{name} 데코레이터는 배치·활성 조건을 바꾼다 — 평탄한 카드 "
                f"필드로 옮길 수 없다 (lorebook.svelte.ts:300-514)")
        else:
            print(f"경고: 미인식 데코레이터 @@{name} — RisuAI 본체도 무시하므로 "
                  f"줄만 제거 (lorebook.svelte.ts:511-513)", file=sys.stderr)
    return _DECO.sub("", content).lstrip("\n"), depth


def _split_lore(book: Dict, budget: Optional[int] = None
                ) -> Tuple[List[str], str, List[Dict], List[int], List[int]]:
    """엔트리를 (로어북 블록, postEverything 주입분, keyed 원형, block의 order,
    block의 북 원위치)로 가른다.

    constant 엔트리는 항상 활성이라 지금 배치를 확정할 수 있다. keyed(=
    constant가 아닌) 엔트리는 활성 조건이 걸려 있어 여기서는 원형만 뽑아
    keyed_lore로 넘긴다 — 활성화 판정은 별도(keyed_lore.py)가 매 턴 담당.

    @@depth 0 (role != assistant) 엔트리는 로어북 블록이 아니라 postEverything
    슬롯으로 간다 — index.svelte.ts:582-590이 `pos==='depth' && depth===0 &&
    role!=='assistant'`를 골라 unformated.postEverything에 넣는다.
    depth>0은 히스토리 중간으로 splice되므로(index.svelte.ts:1188-1194) 여기서
    다룰 수 없다. 만나면 조용히 잘못 놓지 말고 멈춘다.

    budget=None이면 토큰 예산을 적용하지 않는다. RisuAI 기본값은 800이고
    (database.svelte.ts:80) 예산 필터는 실재하지만(lorebook.svelte.ts:613-620,
    필터 결과가 그대로 actives로 반환된다), 캡처를 뜬 설정은 3,400토큰 분량을
    전부 통과시켰다 — 즉 그 클라이언트의 loreBookToken이 크다. 카드의
    character_book.token_budget은 비어 있어(이 카드 기준) 카드에서 알 수 없다.
    다른 설정을 재현하려면 budget을 명시해 넘겨라.

    lore_indices(block과 병렬)와 keyed 각 원소의 "index"는 book["entries"]의
    정렬 전 원위치다. keyed_lore.activate가 이미 한 번 이 함수의 정렬규칙(T)을
    적용받은 block 리스트에 keyed 히트를 합쳐 T를 재적용할 때, order만으로
    재정렬하면 동점 그룹이 T∘T에서 원시 순서로 되돌아간다(2026-08-11 리뷰
    발견 — stable sort + 전체 reverse는 그 자체로 멱등이 아니다). 원위치를
    같이 넘겨야 매번 원시 데이터부터 T를 다시 계산할 수 있다.
    """
    raw = list(book.get("entries", []))
    entries = [(i, e) for i, e in enumerate(raw)
               if e.get("constant") and e.get("content")]
    # RisuAI 최종 순서: priority 내림차순 → 예산 필터 → order 내림차순 →
    # .reverse() (lorebook.svelte.ts:608-662). charx 임포트는 order·priority가
    # 둘 다 insertion_order라(lorebook.svelte.ts:273-274, characterCards.ts:1122)
    # 두 정렬이 같은 키다. 그래서 결과는 오름차순 순정렬이 아니라 **동점 그룹이
    # 카드 기재 역순으로 뒤집힌 것** — JS sort가 안정 정렬이라 마지막 reverse가
    # 동점 안쪽 순서까지 뒤집는다. 위지소연의 NPC 5명이 전부 order=100이라
    # 이게 실제로 캡처와 갈렸다.
    entries.sort(key=lambda t: -t[1].get("insertion_order", 0))
    if budget is not None:
        import tiktoken
        # reverse_proxy의 기본 토크나이저는 tik → o200k_base다
        # (tokenizer.ts:105-133의 default 분기, database.svelte.ts:482).
        enc = tiktoken.get_encoding("o200k_base")
        kept, used = [], 0
        for i, e in entries:                       # priority 내림차순 그대로
            n = len(enc.encode(e["content"]))
            if used + n <= budget:
                used += n
                kept.append((i, e))                # 안 맞는 건 건너뛰고 계속
        entries = kept
    entries.reverse()
    block, post, orders, indices = [], [], [], []
    for i, e in entries:
        body, depth = _strip_deco(e["content"])
        if depth is None:
            block.append(body)
            orders.append(e.get("insertion_order", 0))
            indices.append(i)
        elif depth == 0:
            post.append(body)
        else:
            raise SystemExit(
                f"@@depth {depth} 엔트리는 히스토리 중간으로 splice된다 — "
                f"평탄한 카드 필드로 옮길 수 없다: {e.get('name', '')}")
    # keyed(constant=False) 엔트리: 활성 조건이 있으므로 여기서 활성 판정은
    # 하지 않고 원형만 보존한다 — 런타임 활성화는 keyed_lore.py(신규)가
    # 매 턴 최근 메시지를 스캔해 담당한다 (lorebook.svelte.ts:174-222).
    keyed = []
    for i, e in enumerate(raw):
        if e.get("constant") or not e.get("content"):
            continue
        if not e.get("enabled", True) or not [k for k in e.get("keys", []) if k]:
            continue
        body, depth = _strip_deco(e["content"])
        if depth is not None and depth > 0:            # block/post 루프와 동일 거부
            raise SystemExit(
                f"@@depth {depth} 엔트리는 히스토리 중간으로 splice된다 — "
                f"평탄한 카드 필드로 옮길 수 없다: {e.get('name', '')}")
        keyed.append({"name": e.get("name", ""),
                      "keys": [k for k in e["keys"] if k],
                      "content": body, "depth": depth,
                      "order": e.get("insertion_order", 0),
                      "index": i})
    return block, "\n\n".join(post), keyed, orders, indices


def extract(charx_path: str) -> Dict:
    with zipfile.ZipFile(charx_path) as z:
        card = json.loads(z.read("card.json"))
    d = card["data"]
    greetings = [g for g in [d.get("first_mes", "")]
                 + list(d.get("alternate_greetings", [])) if g.strip()]
    book = d.get("character_book") or {}
    lore, post_everything, keyed_lore, lore_orders, lore_indices = \
        _split_lore(book)
    return {
        "name": d.get("name", ""),
        "description": d.get("description", ""),
        "greeting": greetings[0] if greetings else "",
        "lore": lore,
        "keyed_lore": keyed_lore,
        "lore_orders": lore_orders,
        # lore와 병렬 — 각 블록의 book["entries"] 원위치. keyed_lore.activate가
        # 재정렬 시 T를 원시 데이터부터 다시 계산하는 데 쓴다(동점 재역전 방지).
        "lore_indices": lore_indices,
        # character_book 레벨 스캔 설정 — 카드에 없으면 RisuAI 기본값
        # (lorebook.svelte.ts:174-222 스캔, database.svelte.ts 기본치).
        "lore_settings": {
            "scan_depth": book.get("scan_depth", 5),
            "recursive_scanning": book.get("recursive_scanning", False),
            "token_budget": book.get("token_budget", 99999),
        },
        # charx의 두 필드는 프롬프트 슬롯이 아니라 프리셋 아이템 **덮어쓰기**다.
        # system_prompt → char.systemPrompt: main 아이템 텍스트를 대체하고
        #   {{original}}에 원래 텍스트가 들어간다 (index.svelte.ts:411).
        # post_history_instructions → char.replaceGlobalNote: globalNote 아이템에
        #   같은 방식으로 적용된다 (characterCards.ts:992, index.svelte.ts:1350).
        # authornote(= currentChat.note)는 채팅방 필드라 카드에서 나오지 않는다.
        "system_prompt": d.get("system_prompt", ""),
        "replace_globalnote": d.get("post_history_instructions", ""),
        "post_everything": post_everything,
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
