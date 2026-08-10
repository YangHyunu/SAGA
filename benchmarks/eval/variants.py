# [v1 하네스] 현행은 run2 계열 — 이 파일은 구 테스트·PR 호환용으로 보존.
"""대조군 변형별 요청 조립 (스펙 §9 대조군).

- vanilla   : 전체 히스토리, 마킹 없음 — Risu 순정 하한 근사
- trim      : 마지막 W pair만 — 클라이언트 트림 치매 베이스라인
- retrieval : trim + 잘린 구간에서 top-k 어휘 검색 발췌 prepend
              (2511.17208 단순 turn-retrieval — Dreaming이 이겨야 하는 선)
- dreaming  : 전송분은 trim과 동일 — 주입/압축/마킹은 프록시가 한다

HypaV3 대조군은 RisuAI 클라이언트가 필요해 플러그인 단계로 미룬다.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from benchmarks.cardsim.lorebook import Card, activate, build_messages

TRIM_WINDOW = 8


def trim_window(history: List[Dict], w: int = TRIM_WINDOW) -> List[Dict]:
    starts = [i for i, m in enumerate(history) if m["role"] == "user"]
    if history and history[-1]["role"] == "user":
        starts = starts[:-1]                   # 진행 중인 현재 턴은 카운트 제외
    if len(starts) <= w:
        return history
    return history[starts[len(starts) - w]:]


def _bigrams(text: str) -> set:
    t = "".join(text.split())
    return {t[i:i + 2] for i in range(len(t) - 1)}


def retrieve_turns(history: List[Dict], query: str, k: int = 3,
                   window: Optional[List[Dict]] = None) -> List[str]:
    """트림으로 잘릴 과거 pair에서 질의와 겹치는 top-k 발췌 (결정론).

    window: 호출자가 실제 전송하는 창 (history의 suffix). 넘기면 그 경계
    밖을 잘린 구간으로 본다 — run2처럼 token_trim으로 창을 자르는 호출자는
    반드시 넘겨야 발췌가 실창과 겹치지 않는다. 없으면 trim_window 기준.
    """
    kept = trim_window(history) if window is None else window
    cut = history[:len(history) - len(kept)]
    q = _bigrams(query)
    scored = []
    for i in range(len(cut) - 1):
        if cut[i]["role"] != "user" or cut[i + 1]["role"] != "assistant":
            continue
        text = f"유저: {cut[i]['content']}\n캐릭터: {cut[i + 1]['content']}"
        scored.append((len(q & _bigrams(text)), -i, text))
    scored.sort(reverse=True)
    return [t for s, _, t in scored[:k] if s > 0]


def _merge_leading_systems(msgs: List[Dict]) -> List[Dict]:
    """선두 system 연쇄를 하나로 병합 — 실제 RisuAI 와이어 형태 (corpus 실측).

    분리 전송하면 lore_shift(첫 system만 처리)가 keyed를 못 걷어내
    프리픽스가 keyed churn마다 깨진다.
    """
    i = 0
    while i < len(msgs) and msgs[i]["role"] == "system":
        i += 1
    if i <= 1:
        return msgs
    merged = "\n\n".join(m["content"] for m in msgs[:i])
    return [{"role": "system", "content": merged}] + msgs[i:]


def prepare_request(variant: str, card: Card, history: List[Dict]) -> List[Dict]:
    window = history if variant == "vanilla" else trim_window(history)
    actives = activate(card, window)
    msgs = _merge_leading_systems(build_messages(card, actives, window))
    if variant == "retrieval":
        query = history[-1]["content"]
        excerpts = retrieve_turns(history, query)
        if excerpts:
            block = "[과거 대화 발췌]\n" + "\n---\n".join(excerpts)
            msgs = copy.deepcopy(msgs)
            for m in reversed(msgs):
                if m["role"] == "user":
                    m["content"] = block + "\n\n" + m["content"]
                    break
    return msgs
