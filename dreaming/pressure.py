"""dreaming/pressure.py — 임계 강제압축 폴백 판정 (스펙 §6.3).

유휴 타이머(idle.py)는 매 요청마다 리셋된다 — 5분 미만 간격 연속 채팅에서는
꿈이 영원히 안 돌고 원문 히스토리가 무한 누적된다. §6.3: "유휴 전 예산 임계
도달 시에만 캐시 파괴를 감수 (턴당 상각 ~2%)".

측정 대상은 **조립 후 프롬프트**(압축·주입 적용된 out)다 — 압축이 못 줄인
잔여 크기가 곧 압박이고, 압축이 잘 돌면 원문이 쌓여도 트리거되지 않는다.

상각 보호 2중:
- 플랜 바이트 변경(=캐시 파괴)은 chunks.BOUNDARY_STEP 계단이 이미
  10턴당 1회로 묶는다 — 강제 꿈이 매번 새 플랜을 만들지 않는다.
- MIN_BACKLOG_TURNS가 강제 Flash 콜을 backlog 5턴당 1회로 묶는다 —
  임계 초과가 지속돼도 매 턴 콜이 되지 않는다.
"""

from __future__ import annotations

from typing import Dict, List

DEFAULT_THRESHOLD_CHARS = 120_000   # ~40K+ tokens — 커뮤니티 실사용 하한 근처
MIN_BACKLOG_TURNS = 5


def prompt_chars(messages: List[Dict]) -> int:
    """메시지 content 문자 수 합 — 결정론·0콜 크기 프록시 (토큰카운터 불요)."""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):        # 비전 등 멀티파트
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += len(part["text"])
    return total


def should_force(size: int, threshold: int, backlog_turns: int,
                 min_backlog: int = MIN_BACKLOG_TURNS) -> bool:
    """임계 도달 ∧ 꿈꿀 미처리 구간이 충분할 때만 강제 꿈 (0 이하 = 비활성)."""
    if threshold <= 0:
        return False
    return size >= threshold and backlog_turns >= min_backlog
