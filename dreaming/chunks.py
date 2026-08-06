"""dreaming/chunks.py — 청크 압축 (스펙 §6).

압축은 결정론적 템플릿 조립이다 — 이해(요약문)는 꿈이 이미 끝냈고,
여기서는 같은 입력에 같은 바이트만 만든다 (§6.1). LLM 0콜.
"""

from __future__ import annotations

from typing import List

from dreaming.records import Episode


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
