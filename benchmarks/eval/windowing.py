"""토큰 예산 기반 히스토리 창 관리 — 트림·전송 히스토리 결정."""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import tiktoken

_ENC = tiktoken.get_encoding("o200k_base")


def count(text: str) -> int:
    # RisuAI reverse_proxy 기본 토크나이저와 동일 (tokenizer.ts:105-133 →
    # o200k_base). len/2.5 근사는 한국어를 ~40% 과소평가해 12K 예산에서
    # eviction이 아예 안 일어났다 (파일럿 실측 18,917 vs 근사 11,816).
    return len(_ENC.encode(text))


def token_trim(history: List[Dict], budget: int,
               count_fn: Callable[[str], int] = count
               ) -> Tuple[List[Dict], int]:
    """토큰 예산 기반 트림 — 메시지 단위 FIFO (index.svelte.ts:1143-1154).

    RisuAI는 페어 정렬 없이 chats[0]부터 하나씩 제거한다 — greeting도
    이 큐의 일부라 예산 판정에 포함되고, 남는 첫 메시지가 assistant일 수
    있다. 반환: (윈도우, win_start). win_start는 "이 턴 번호부터의 사실이
    창내" 의미 — 창의 첫 메시지가 턴 k의 user면 win_start=k, 턴 k의
    assistant면(반 잘린 턴) win_start=k+1.
    """
    if not history:
        return history, 0
    total = sum(count_fn(m["content"]) for m in history)
    start = 0
    while total > budget and len(history) - start > 1:
        total -= count_fn(history[start]["content"])
        start += 1
    window = history[start:]
    if start == 0:
        return window, 0
    has_greeting = history[0]["role"] == "assistant"
    offset = start - (1 if has_greeting else 0)
    turn, half = divmod(offset, 2)
    return window, turn + half


# 풀 히스토리를 그대로 보내는 변형. dreaming은 창 관리(압축)가 프록시 책임이라
# 벤치가 미리 자르면 프록시가 기억해야 할 턴을 아예 못 본다 — night2에서
# dreaming이 "trim 3회차"가 된 원인 중 하나.
FULL_HISTORY = ("vanilla", "dreaming")


def wire_history(variant: str, history: List[Dict],
                  window: List[Dict]) -> List[Dict]:
    """변형별 전송 히스토리 — 트림 여부 단일 결정점."""
    return history if variant in FULL_HISTORY else window


def hypa_in_window(fact_turn: int, kept_start_msg: int,
                   has_greeting: bool) -> bool:
    """hypa가 실제로 보낸 창에 턴 fact_turn의 발화가 남아 있는가.

    hypa는 턴이 아니라 **메시지 인덱스**로 자른다 (chats.slice(startIdx),
    hypav3.ts:934). greeting이 있으면 턴 t의 user 메시지는 인덱스 1+2t다.
    """
    return (1 if has_greeting else 0) + 2 * fact_turn >= kept_start_msg
