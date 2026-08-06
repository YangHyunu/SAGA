"""dreaming/scaffold.py — 프리셋 프리필 꼬리 분리.

RisuAI 프리셋은 chat 블록 뒤에 정적 프리필 체인을 붙일 수 있다. 뮈토스 6.2의
▶️ 도메인 중립 렌더링 프리필이 그 예로, 가짜 user/assistant 왕복 4~5개가
매 요청 꼬리에 그대로 실린다. 벗기지 않으면 셋 다 깨진다:

  - extract_pairs가 프리필을 대화 쌍으로 세고, last_user_hash가 매 턴
    바이트 동일해져 리롤/분기 판정이 무력화된다
  - 지식 주입이 실제 유저 입력이 아니라 프리필 마지막 메시지 앞에 붙는다
  - mark_cache의 BP3(마지막 assistant)가 프리필 안쪽을 가리켜, 매 턴 바뀌는
    유저 입력이 캐시 span 안으로 들어가 캐시가 안 붙는다

꼬리는 프리셋 설정이 안 바뀌는 한 턴마다 바이트 동일하다는 성질로 찾는다.
직전 요청과의 공통 접미를 재고, 한 번 찾으면 그 지문을 세션에 기억한다.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from saga.services.pair_ledger import hash_text


def fingerprint(messages: List[Dict]) -> List[str]:
    return [f"{m.get('role')}:{hash_text(m.get('content', ''))}"
            for m in messages]


def _cut(messages: List[Dict], n: int) -> Tuple[List[Dict], List[Dict]]:
    if n <= 0:
        return list(messages), []
    return list(messages[:-n]), list(messages[-n:])


def _shrink_to_user(messages: List[Dict], n: int) -> int:
    """core가 user로 끝나도록 꼬리를 줄인다.

    리롤은 assistant만 pop하고 재전송하므로 직전 요청과의 공통 접미가 실제
    유저 턴까지 먹는다. chat 블록은 항상 유저 입력으로 끝나므로 이 경계가
    프리필과 대화를 가른다.
    """
    while n > 0 and (n >= len(messages)
                     or messages[len(messages) - n - 1].get("role") != "user"):
        n -= 1
    return n


def learn(messages: List[Dict],
          prev_fp: Optional[List[str]]) -> Optional[List[str]]:
    """직전 요청과의 공통 접미에서 꼬리 지문을 뽑는다 (없으면 None)."""
    if not prev_fp:
        return None
    fp = fingerprint(messages)
    n = 0
    while n < len(fp) and n < len(prev_fp) and fp[-1 - n] == prev_fp[-1 - n]:
        n += 1
    if n >= len(prev_fp) or n >= len(fp):
        # 두 요청이 통째로 같다 = 리롤. 어디까지가 프리필인지 가르는 정보가
        # 없으므로 배우지 않고 다음 턴을 기다린다.
        return None
    n = _shrink_to_user(messages, n)
    return fp[-n:] if n else None


def split(messages: List[Dict],
          tail_fp: Optional[List[str]]) -> Tuple[List[Dict], List[Dict]]:
    """기억해 둔 꼬리 지문으로 (core, tail) 분리. 안 맞으면 통째로 core."""
    if not tail_fp or len(tail_fp) >= len(messages):
        return list(messages), []
    if fingerprint(messages)[-len(tail_fp):] != tail_fp:
        return list(messages), []
    return _cut(messages, len(tail_fp))
