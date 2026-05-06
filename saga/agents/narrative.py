"""Sub-B 서사 요약 출력 타입.

Flash가 반환하는 4필드 dict를 dataclass로 표현. scene_type enum 검증과
importance 계산을 한 곳에 묶어 dict + .get() 패턴을 제거한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SceneType = Literal["combat", "dialogue", "exploration", "event"]
_VALID_SCENE_TYPES: frozenset[str] = frozenset({"combat", "dialogue", "exploration", "event"})


@dataclass
class NarrativeSummary:
    summary: str = ""
    npcs_mentioned: list[str] = field(default_factory=list)
    scene_type: SceneType = "dialogue"
    key_event: str | None = None

    @classmethod
    def empty(cls) -> "NarrativeSummary":
        return cls()

    @classmethod
    def from_llm_dict(cls, data: dict | None) -> "NarrativeSummary":
        """Build from raw LLM JSON output. Defensive: any field can be missing or wrong type."""
        if not data:
            return cls.empty()
        scene = data.get("scene_type", "dialogue")
        if scene not in _VALID_SCENE_TYPES:
            scene = "dialogue"
        npcs_raw = data.get("npcs_mentioned") or []
        npcs = [str(n) for n in npcs_raw if isinstance(n, str)] if isinstance(npcs_raw, list) else []
        key_event = data.get("key_event")
        if key_event is not None and not isinstance(key_event, str):
            key_event = str(key_event) if key_event else None
        return cls(
            summary=str(data.get("summary") or ""),
            npcs_mentioned=npcs,
            scene_type=scene,  # type: ignore[arg-type]
            key_event=key_event or None,
        )

    def to_dict(self) -> dict:
        """Serialize for turn_log persistence (backward-compat with state_changes JSON)."""
        return {
            "summary": self.summary,
            "npcs_mentioned": list(self.npcs_mentioned),
            "scene_type": self.scene_type,
            "key_event": self.key_event,
        }

    def importance(self) -> int:
        """Episode importance score (0-100)."""
        score = 10
        if self.scene_type == "combat":
            score += 40
        elif self.scene_type == "event":
            score += 35
        elif self.scene_type == "exploration":
            score += 15
        if self.key_event:
            score += 30
        if self.npcs_mentioned:
            score += 10 * min(2, len(self.npcs_mentioned))
        return min(score, 100)
