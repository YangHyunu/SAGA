# 키워드 로어 임포트 + 활성화 에뮬 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Alternate Hunters.charx(키워드 트리거 로어 63개, 콘텐츠 79%)를 eval 하네스가 충실 재현할 수 있게 — 변환기가 keyed 엔트리를 보존하고, 와이어 조립이 RisuAI 시맨틱으로 턴마다 활성화한다.

**Architecture:** `charx2card.py`가 keyed 엔트리를 `keyed_lore` 필드로 추출(+ constant 순서 보존용 `lore_orders`), 신규 `benchmarks/eval/keyed_lore.py`가 최근 메시지를 스캔해 활성 엔트리를 constant 블록에 RisuAI 정렬로 병합, `run2.py`가 턴마다 호출.

**Tech Stack:** Python 3.13, pytest. LLM 호출 없음 (전부 결정론).

## Global Constraints

- RisuAI 시맨틱 준거 (2026-08-11 소스 분석 확정, lorebook.svelte.ts):
  - 데코레이터 파서는 **case-sensitive**. 본체 인식 목록(전부 소문자): `end, activate_only_after, activate_only_every, keep_activate_after_match, dont_activate_after_match, depth, reverse_depth, instruct_depth, reverse_instruct_depth, instruct_scan_depth, role, scan_depth, is_greeting, position, inject_lore, inject_at, inject_replace, inject_prepend, ignore_on_max_context, additional_keys, exclude_keys, exclude_keys_all, match_full_word, match_partial_word, is_user_icon, activate, dont_activate, disable_ui_prompt, probability, priority, unrecursive, recursive, no_recursive_search` (lorebook.svelte.ts:302-507)
  - 미인식 데코레이터(`@@Depth` 등): **줄만 제거하고 무시** (ccardlib decorator.parse가 `@@` 줄을 무조건 소비, switch default → false; lorebook.svelte.ts:511-513)
  - 키 매칭: 항상 lowercase, 공백 제거 후 substring (lorebook.svelte.ts:174-222). 스캔 = 최근 `scan_depth`개 메시지 (기본 5)
  - 정렬: constant·keyed 합쳐서 `sort(key=-insertion_order)` 후 `.reverse()` (lorebook.svelte.ts:608-662; charx2card.py:67-73과 동일 규칙)
  - `@@depth 0` 엔트리는 로어 블록이 아니라 postEverything (index.svelte.ts:582-590)
- 이 카드 미사용 기능은 구현하지 않는다 (실측 0건): selective/secondary, regex 키, case_sensitive, match_full_word, 재귀 스캔(카드가 `recursive_scanning:false`), depth>0, 토큰 예산(카드 99999)
- 회귀: 소연 카드 경로(전량 constant)는 기존 출력 필드가 **모두 동일**해야 함 — 신규 필드(`keyed_lore=[]`, `lore_orders`)만 추가
- python3 (python 없음). 테스트: `python3 -m pytest tests/test_charx2card_keyed.py -v`
- 원본 charx·기존 card json·dreaming_data 세션 데이터 수정 금지

---

### Task 1: charx2card.py — 데코레이터 3분류 + keyed_lore 추출

**Files:**
- Modify: `benchmarks/eval/charx2card.py`
- Test: `tests/test_charx2card_keyed.py` (신규)

**Interfaces:**
- Produces: `extract(charx_path)` 반환 dict에 신규 키 3개 —
  `keyed_lore: List[{"name": str, "keys": List[str], "content": str, "depth": Optional[int], "order": int}]`,
  `lore_orders: List[int]` (기존 `lore` 리스트와 같은 길이·순서),
  `lore_settings: {"scan_depth": int, "recursive_scanning": bool, "token_budget": int}` (character_book 레벨 값, 없으면 scan_depth=5, recursive_scanning=False, token_budget=99999)
- 기존 키(`lore`, `post_everything` 등) 값 불변

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""charx2card keyed 로어 추출 테스트. 픽스처는 인메모리 book dict."""
import pytest

from benchmarks.eval.charx2card import _split_lore, _strip_deco


def _book(entries):
    return {"entries": entries}


def test_unknown_decorator_dropped_not_fatal():
    # RisuAI 본체: 미인식 @@Depth는 줄만 제거, 일반 블록 배치 (switch default)
    body, depth = _strip_deco("@@Depth 0\n본문이다")
    assert body == "본문이다"
    assert depth is None


def test_known_unreproducible_decorator_still_fatal():
    with pytest.raises(SystemExit):
        _strip_deco("@@priority 5\n본문")


def test_keyed_entries_extracted_with_order():
    book = _book([
        {"constant": True, "content": "상시로어", "insertion_order": 10},
        {"constant": False, "content": "길드로어", "keys": ["Guilds", "길드"],
         "insertion_order": 860, "name": "길드", "enabled": True},
        {"constant": False, "content": "", "keys": [], "name": "폴더"},
    ])
    block, post, keyed, orders = _split_lore(book)
    assert block == ["상시로어"] and orders == [10]
    assert len(keyed) == 1
    assert keyed[0]["keys"] == ["Guilds", "길드"]
    assert keyed[0]["order"] == 860
    assert keyed[0]["depth"] is None


def test_keyed_depth0_marked():
    book = _book([
        {"constant": False, "content": "@@depth 0\n지침", "keys": ["k"],
         "insertion_order": 1, "enabled": True},
    ])
    _, _, keyed, _ = _split_lore(book)
    assert keyed[0]["depth"] == 0 and keyed[0]["content"] == "지침"


def test_disabled_keyed_skipped():
    book = _book([
        {"constant": False, "content": "죽은로어", "keys": ["k"],
         "insertion_order": 1, "enabled": False},
    ])
    _, _, keyed, _ = _split_lore(book)
    assert keyed == []
```

- [ ] **Step 2: 실행해 실패 확인** — `python3 -m pytest tests/test_charx2card_keyed.py -v` → FAIL (`_split_lore` 반환값 2개, `@@Depth`는 SystemExit)

- [ ] **Step 3: 구현**

`_strip_deco` 3분류 (기존 `_HANDLED` 유지, 본체 인식 목록 상수 추가):

```python
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
```

`_strip_deco` 분기 교체 (기존 for 루프 내부):

```python
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
```

`_split_lore` 반환 4개로 확장 — constant 경로는 기존 로직 그대로 두고 `orders` 병렬 수집, keyed 수집 추가:

```python
    keyed = []
    for e in book.get("entries", []):
        if e.get("constant") or not e.get("content"):
            continue
        if not e.get("enabled", True) or not [k for k in e.get("keys", []) if k]:
            continue
        body, depth = _strip_deco(e["content"])
        keyed.append({"name": e.get("name", ""),
                      "keys": [k for k in e["keys"] if k],
                      "content": body, "depth": depth,
                      "order": e.get("insertion_order", 0)})
```

(depth>0은 `_strip_deco` 기존 경로가 이미 SystemExit — keyed에도 동일 적용됨.)
`extract()`에서 `keyed_lore`/`lore_orders`/`lore_settings` 키 추가. `lore_settings`는 `book.get("scan_depth", 5)` 등 book 레벨에서.

- [ ] **Step 4: 테스트 통과 확인** — `python3 -m pytest tests/test_charx2card_keyed.py -v` → PASS. 이어서 `python3 -m pytest tests/ -q` 전체 — `_split_lore` 반환값 개수가 바뀌므로 기존 호출부·테스트 파손 여부를 여기서 잡는다
- [ ] **Step 5: 소연 회귀** — `python3 -m benchmarks.eval.charx2card "/Users/yanghyeon-u/Downloads/위지소연 (1).charx" /tmp/soyeon-regress.json` 실행 후, 기존 `dreaming_data/eval/card-soyeon-v2.json`과 **기존 키들의 값 동일** 확인(신규 3키 제외; persona/user_name/style_examples는 기존 파일 보존 로직이라 신규 출력엔 없음이 정상). 확인 코드는 커밋하지 않는 일회성 파이썬으로.
- [ ] **Step 6: Commit** — `git add benchmarks/eval/charx2card.py tests/test_charx2card_keyed.py && git commit`

### Task 2: keyed_lore.py 활성화 에뮬 + run2 배선

**Files:**
- Create: `benchmarks/eval/keyed_lore.py`
- Modify: `benchmarks/eval/run2.py` (build_wire 호출부 — card의 lore/post_everything을 턴마다 증강)
- Test: `tests/test_keyed_lore.py` (신규)

**Interfaces:**
- Consumes: Task 1의 `keyed_lore`/`lore_orders`/`lore_settings` 카드 필드
- Produces: `activate(card: dict, recent_texts: List[str]) -> Tuple[List[str], str]`
  — (병합 완료된 로어 블록 리스트, post_everything 추가분["" 가능]).
  keyed_lore 없는 카드(소연)면 `(card["lore"], "")` 그대로 반환 (동작 불변)

- [ ] **Step 1: 실패하는 테스트 작성**

```python
"""RisuAI 키워드 활성화 에뮬 테스트 (lorebook.svelte.ts 시맨틱)."""
from benchmarks.eval.keyed_lore import activate


def _card(**kw):
    base = {"lore": ["상시A", "상시B"], "lore_orders": [100, 200],
            "keyed_lore": [], "lore_settings": {"scan_depth": 5}}
    base.update(kw)
    return base


def test_no_keyed_card_passthrough():
    card = {"lore": ["상시A"]}          # 소연형 구카드 — 신규 필드 자체가 없음
    blocks, post = activate(card, ["아무 말"])
    assert blocks == ["상시A"] and post == ""


def test_key_match_case_insensitive_substring():
    card = _card(keyed_lore=[{"name": "길드", "keys": ["Guilds", "길드"],
                              "content": "길드로어", "depth": None, "order": 150}])
    blocks, _ = activate(card, ["오늘 GUILDS 얘기를 했다"])
    assert "길드로어" in blocks


def test_key_match_ignores_whitespace():
    # RisuAI는 공백 제거 후 substring (lorebook.svelte.ts:206-222)
    card = _card(keyed_lore=[{"name": "", "keys": ["황금 사자"],
                              "content": "문장로어", "depth": None, "order": 1}])
    blocks, _ = activate(card, ["그 황금\n사자 문양을 보았다"])
    assert "문장로어" in blocks


def test_scan_depth_window():
    card = _card(lore_settings={"scan_depth": 2},
                 keyed_lore=[{"name": "", "keys": ["오래된키"],
                              "content": "X", "depth": None, "order": 1}])
    blocks, _ = activate(card, ["오래된키 언급", "중간", "최근"])
    assert "X" not in blocks           # 최근 2개 밖이라 미활성


def test_merge_order_risu_rule():
    # 정렬: 합쳐서 sort(-order) 후 reverse → order 오름차순 (동점은 역순)
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "키드150",
                              "depth": None, "order": 150}])
    blocks, _ = activate(card, ["k"])
    assert blocks == ["상시A", "키드150", "상시B"]


def test_depth0_goes_to_post():
    card = _card(keyed_lore=[{"name": "", "keys": ["k"], "content": "지침",
                              "depth": 0, "order": 1}])
    blocks, post = activate(card, ["k"])
    assert "지침" not in blocks and post == "지침"
```

- [ ] **Step 2: 실행해 실패 확인** — `python3 -m pytest tests/test_keyed_lore.py -v` → FAIL (모듈 없음)

- [ ] **Step 3: 구현** — `benchmarks/eval/keyed_lore.py`:

```python
"""RisuAI 키워드 로어 활성화 에뮬 (lorebook.svelte.ts 준거).

키 매칭: lowercase + 공백 제거 substring (174-222행). 스캔: 최근 scan_depth개
메시지 (84행, 기본 5). 정렬: constant·활성 keyed 합쳐 sort(-order) 후
reverse (608-662행) — charx2card._split_lore가 constant에 이미 적용한 규칙과
동일해, lore_orders로 원래 order를 복원해 재병합한다. depth 0 활성 엔트리는
로어 블록이 아니라 postEverything (index.svelte.ts:582-590).

이 카드(Alternate Hunters)가 안 쓰는 기능은 미구현: selective/regex/
case_sensitive 플래그/재귀 스캔/토큰 예산 (2026-08-11 실측 0건).
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub("", s).lower()


def activate(card: Dict, recent_texts: List[str]) -> Tuple[List[str], str]:
    keyed = card.get("keyed_lore") or []
    lore = list(card.get("lore", []))
    if not keyed:
        return lore, ""
    depth = int(card.get("lore_settings", {}).get("scan_depth", 5))
    scan = _norm("\x00".join(recent_texts[-depth:]))
    hits = [e for e in keyed
            if any(_norm(k) and _norm(k) in scan for k in e["keys"])]
    post = [e["content"] for e in hits if e.get("depth") == 0]
    block_hits = [e for e in hits if e.get("depth") != 0]
    orders = card.get("lore_orders") or [0] * len(lore)
    merged = ([(o, b) for o, b in zip(orders, lore)]
              + [(e["order"], e["content"]) for e in block_hits])
    merged.sort(key=lambda t: -t[0])
    merged.reverse()
    return [b for _, b in merged], "\n\n".join(post)
```

주의: `lore` 리스트는 이미 정렬 완료 상태라 `(order, body)` 쌍으로 되돌려 재정렬해도 동점 그룹의 상대 순서가 유지된다(파이썬 sort 안정성 + 동일 규칙 재적용). 동점 재역전이 일어나지 않음을 `test_merge_order_risu_rule`이 지킨다.

- [ ] **Step 4: run2 배선** — `run2.py`의 `build_wire`(run2.py:96-123)가 배선 지점이다. `window`(메시지 히스토리, 이번 유저 발화 포함)를 이미 인자로 받고 있고, assemble에 넘길 카드 dict를 그 자리에서 만든다. import에 `from benchmarks.eval.keyed_lore import activate`를 추가하고, `msgs = assemble(...)` 직전에:

```python
    lore, extra_post = activate(card, [m.get("content", "") for m in window])
    post = card.get("post_everything", "")
    if extra_post:
        post = f"{post}\n\n{extra_post}" if post else extra_post
```

그리고 assemble의 카드 dict에서 `"lore": card.get("lore", [])` → `"lore": lore`, `"post_everything": card.get("post_everything", "")` → `"post_everything": post`로 교체. 원본 `card` dict는 절대 변경하지 않는다(다음 턴 스캔 오염 금지). keyed_lore 없는 카드는 activate가 passthrough라 기존 런과 와이어가 동일하다.
- [ ] **Step 5: 전체 테스트** — `python3 -m pytest tests/ -q` → 기존 729 + 신규 전부 PASS
- [ ] **Step 6: Commit**

### Task 3: Alternate Hunters 실변환 + 수치 검증

**Files:**
- 산출: `dreaming_data/eval/card-hunters-v1.json` (gitignore 영역 — 커밋 없음)

- [ ] **Step 1: 변환 실행**

```bash
python3 -m benchmarks.eval.charx2card "/Users/yanghyeon-u/Downloads/Alternate Hunters.charx" dreaming_data/eval/card-hunters-v1.json
```

- [ ] **Step 2: 수치 검증** — 일회성 파이썬으로: `keyed_lore` 길이 63, `lore`+`post_everything` 유래 constant 30개 반영, stderr 경고 1건(@@Depth), `lore_settings.scan_depth == 5`. 어긋나면 Task 1로 회귀.
- [ ] **Step 3: 활성화 스모크 (LLM 없음)** — `activate(card, ["길드에 가입하려면?"])`가 길드 로어를 포함하는지, `activate(card, ["안녕"])`이 constant 30개만 내는지 일회성 확인.

## 실행 후기 (2026-08-11)

결과: 3태스크 + 수정 3라운드 완료, 747 passed. Alternate Hunters 변환 실측 —
keyed 63 / lore 블록 12 + depth0→postEverything 18 (constant 30 정합) /
@@Depth 경고 1건. 활성화 스모크: 무적중 항등·한글 키·영문 키 전부 정상.

**플랜 결함 4건 — 전부 리뷰 게이트가 잡음 (구현자는 플랜을 충실히 따랐고, 틀린
건 플랜이었다):**

1. **괄호주 사실 오류** (Task 1): "depth>0은 `_strip_deco` 기존 경로가 이미
   SystemExit"라고 적었으나 실제 raise는 호출부 block/post 루프에만 있었다 —
   keyed 경로에 조용한 통과 구멍. 교훈: 기존 코드의 동작을 플랜에 인용할 때는
   함수 내부인지 호출부인지까지 확인.
2. **병합 알고리즘 T∘T 비멱등** (Task 2): 이미 sort+reverse된 `lore`에 같은
   변환을 재적용하는 알고리즘을 플랜이 직접 제공 — 동점 그룹(소연 NPC 5명
   전원 order=100이 실재 사례)에서 순서 파손, 무적중 경로에서도 발생. 교정:
   북 원위치(`lore_indices`/`index`) 내보내서 원시 데이터에 변환 1회.
3. **플랜 테스트가 오동작을 스펙으로 박제** (Task 2): `\s+` 정규화(개행·탭까지
   제거)를 테스트가 정답으로 명시 — RisuAI 본체는 스페이스(U+0020)만 제거
   (lorebook.svelte.ts:206,208). 교훈: 준거 소스의 정규식을 인용할 때 문자
   클래스 범위를 그대로 옮길 것 (`/ /g` ≠ `\s`).
4. **예산 파급 미고려** (전역): 로어를 턴 가변으로 만들면서 run2의 1회성
   고정비 산정(빈 윈도우 기준)이 깨진다는 것을 플랜이 못 봄 — trim/hypa가
   설정보다 넓은 컨텍스트를 받아 변형 비교 편향. 최종 전체 리뷰(Opus)가 태스크
   경계 관점에서 발견, 턴당 재계산으로 교정. 교훈: 공유 자원(토큰 예산)을
   가변화하는 변경은 그 자원의 모든 산정 지점을 플랜 단계에서 grep.

기록만 남긴 한계: keyed depth0 활성분은 constant post 뒤 통짜 append (RisuAI는
order 정렬로 섞음 — 이 카드 depth0 keyed 0건이라 무영향, 코드 주석 있음),
charx2card 설정 3종이 all-or-nothing 대신 독립 기본값 (두 실카드 무영향).
