"""dreaming/chunks.py — 청크 압축 (스펙 §6).

압축은 결정론적 템플릿 조립이다 — 이해(요약문)는 꿈이 이미 끝냈고,
여기서는 같은 입력에 같은 바이트만 만든다 (§6.1). LLM 0콜.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Tuple

from dreaming.records import Episode
from dreaming.store import MemoryStore

logger = logging.getLogger(__name__)

TAIL_KEEP = 6      # 원문 꼬리로 남길 최근 pair 수 (스펙 §5)
T1_MAX = 8         # Tier1 청크 상한 — 초과분은 챕터로 승격 (§6.2)
CHAPTER_SIZE = 5   # 챕터 1개로 묶을 에피소드 수 — 고정 블록이라 승격이 안정


def _one_line(text: str) -> str:
    return " ".join(text.split())


def assemble_tier1(ep: Episode) -> str:
    """에피소드 청크 (~70% 압축): 제목 + 요약 + 미회수 복선."""
    lines = [f"[지난 이야기 · {ep.title}]", ep.summary.strip()]
    if ep.open_threads:
        lines.append("남은 실마리: " + " / ".join(ep.open_threads))
    return "\n".join(lines)


def assemble_tier2(episodes: List[Episode]) -> str:
    """챕터 청크 (~90% 압축): 에피소드당 한 줄."""
    lines = ["[지난 장 요약]"]
    for ep in episodes:
        lines.append(f"- {ep.title}: {_one_line(ep.summary)[:100]}")
    return "\n".join(lines)


def build_compression(store: MemoryStore, last_turn: int) -> Optional[Dict]:
    """에피소드 → 압축 플랜 (B-4, 꿈 안에서만 호출 — §6.3 TTL 창구).

    **가장 이른 에피소드 턴부터의** 연속 구간만 압축한다 (치환이 위치
    기반이라 프리픽스 연속성이 전제). 갭·꼬리(최근 TAIL_KEEP pair)에서
    중단, 재드림 중복 구간은 스킵.

    시작점을 0으로 박으면 안 된다 — 프로덕션 턴 번호는 _BASELINE_PAD로
    1024부터 시작하므로(identity.py:95-99) 첫 에피소드에서 즉시 break 되어
    압축이 영구 무효화된다 (docs/DREAMING_FLAW.md §2, 실측 청크 0개).
    """
    eps = [e for e in store.list_episodes()
           if e.start_turn is not None and e.end_turn is not None]
    if not eps:
        logger.info("[chunks] 압축 없음: 턴 범위를 가진 에피소드가 0개")
        return None
    eps.sort(key=lambda e: (e.start_turn, e.recorded_at))
    cutoff = last_turn - TAIL_KEEP
    chain: List[Episode] = []
    next_turn = eps[0].start_turn          # 패드된 턴 공간을 그대로 승계
    for e in eps:
        if e.start_turn < next_turn:
            continue                      # 이미 덮인 구간 (재드림 중복)
        if e.start_turn > next_turn or e.end_turn > cutoff:
            break                         # 갭 또는 꼬리 진입
        chain.append(e)
        next_turn = e.end_turn + 1
    if not chain:
        # 조용한 실패 금지 — 이 결함이 오래 안 보인 유일한 이유가 침묵이었다
        logger.info("[chunks] 압축 없음: 에피소드 %d개, 시작턴 %d, 꼬리 cutoff %d "
                    "— 전부 꼬리 안이거나 첫 구간이 불연속",
                    len(eps), eps[0].start_turn, cutoff)
        return None

    n_chapters = 0
    if len(chain) > T1_MAX:
        n_chapters = -(-(len(chain) - T1_MAX) // CHAPTER_SIZE)  # ceil
    messages: List[Dict] = []
    idx = 0
    for _ in range(n_chapters):
        group = chain[idx: idx + CHAPTER_SIZE]
        messages.append({"role": "assistant", "content": assemble_tier2(group)})
        idx += len(group)
    for e in chain[idx:]:
        messages.append({"role": "assistant", "content": assemble_tier1(e)})
    return {"covers_until_turn": next_turn, "messages": messages}


def apply_compression(messages: List[Dict], plan: Dict,
                      window_start_turn: int = 0
                      ) -> Tuple[List[Dict], Optional[int]]:
    """히스토리 선두의 압축 대상 pair를 청크로 치환 (스펙 §5 레이아웃).

    window_start_turn = 요청 첫 pair의 세션 턴 번호 (트림 시 0이 아님).
    드롭 수 = 압축 구간 중 윈도우에 아직 남아 있는 pair 수 — 트림이 이미
    구간을 지나갔으면 0이고, 그때 청크 prepend는 사라진 컨텍스트의 복원이다.
    선두 system 블록과 인사(첫 user 이전 assistant)는 보존한다.
    히스토리가 드롭 수보다 짧으면 원본 그대로 — fail-open.
    반환: (메시지, 첫 청크 인덱스 | None).
    """
    to_drop = plan["covers_until_turn"] - window_start_turn
    if to_drop < 0:
        to_drop = 0
    i = 0
    while i < len(messages) and messages[i].get("role") != "user":
        i += 1                             # 첫 user 앞(system·인사)은 보존
    pairs, j = 0, i
    while j < len(messages) and pairs < to_drop:
        if messages[j].get("role") == "user":
            if (j + 1 < len(messages)
                    and messages[j + 1].get("role") == "assistant"):
                j += 2
                pairs += 1
            else:
                break                      # 미완 pair(현재 턴) — 압축 불가
        else:
            j += 1
    if pairs < to_drop:
        return messages, None
    out = messages[:i] + copy.deepcopy(plan["messages"]) + messages[j:]
    return out, i
