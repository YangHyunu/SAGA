---
risk_code: R6  # Domain Model Distortion
severity: suggestion
title: Sub-B narrative dict 4필드가 anemic
affected_files:
  - saga/agents/extractors.py
  - saga/agents/post_turn.py
order: 8
---

# 🟢 Suggestion — Sub-B `narrative` dict 4필드가 anemic

## Symptom

`saga/agents/extractors.py:38-46`의 Sub-B가 다음 dict를 반환합니다.

```python
{
    "summary": "2-3문장 요약",
    "npcs_mentioned": ["..."],
    "scene_type": "combat|dialogue|exploration|event",
    "key_event": "핵심 사건 한 줄 또는 null",
}
```

소비처(`saga/agents/post_turn.py`)가 `.get(...)` 패턴으로 안전 접근:

| 라인 | 접근 패턴 |
|------|-----------|
| 55 | `narrative = {"summary": "", "npcs_mentioned": [], "scene_type": "dialogue", "key_event": None}` (fallback) |
| 62 | `narrative.get("npcs_mentioned") or []` |
| 108 | `narrative.get("summary", "")` |
| 112 | `narrative.get("scene_type", "dialogue")` |
| 113 | `narrative.get("npcs_mentioned") or []` |
| 295-310 | `_calculate_importance(narrative)` — `narrative.get("scene_type", "dialogue")`, `narrative.get("key_event")` |

`scene_type` enum 값(`"combat"`, `"event"`, `"exploration"`, `"dialogue"`)이 코드 곳곳에 string literal로 흩어져 있습니다.

## Source

- Evans — *Domain-Driven Design*, Anemic Domain Model
- Fowler — *Refactoring*, Primitive Obsession (도메인 개념을 raw dict/string으로 표현)

## Consequence

- 필드 누락/타입 변경 시 런타임에서야 노출 (예: Flash가 `npcs_mentioned`를 `null`로 반환 시 fallback 잘못 작성하면 silent break)
- `scene_type` 값 enum이 6곳에 흩어져 있어 새 값 추가 시 grep 누락 가능
- `_calculate_importance` 같은 도메인 로직이 외부 함수로 빠져 있음 — 도메인 객체에 메서드로 들어가는 게 자연스러움

## Remedy

### 변경 1: dataclass + Literal 정의

```python
# saga/agents/narrative.py (신규)
from dataclasses import dataclass, field
from typing import Literal

SceneType = Literal["combat", "dialogue", "exploration", "event"]


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
        """Defensive: any field can be missing or wrong type."""
        if not data:
            return cls.empty()
        scene = data.get("scene_type", "dialogue")
        if scene not in {"combat", "dialogue", "exploration", "event"}:
            scene = "dialogue"
        return cls(
            summary=str(data.get("summary", "") or ""),
            npcs_mentioned=list(data.get("npcs_mentioned") or []),
            scene_type=scene,
            key_event=data.get("key_event") or None,
        )

    def importance(self) -> int:
        """Calculate episode importance (formerly PostTurnExtractor._calculate_importance)."""
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
```

### 변경 2: 호출자 변경

```python
# saga/agents/extractors.py
async def narrative_extract(...) -> NarrativeSummary:
    parsed = parse_llm_json(result)
    return NarrativeSummary.from_llm_dict(parsed)
```

```python
# saga/agents/post_turn.py
narrative = await self.extract_fn(response_text, session_id)
# narrative is always NarrativeSummary, never None — extract_fn handles fallback
importance = narrative.importance()
for npc in narrative.npcs_mentioned:
    ...
await self.sqlite_db.insert_turn_log(
    session_id, turn_number,
    state_changes={
        "summary": narrative.summary,
        "npcs_mentioned": narrative.npcs_mentioned,
        "scene_type": narrative.scene_type,
        "key_event": narrative.key_event,
    },
    ...
)
```

`turn_log.state_changes`는 JSON 문자열로 저장하므로 `narrative.__dict__` 또는 `dataclasses.asdict()`로 직렬화.

## Acceptance Criteria

- [ ] `NarrativeSummary` dataclass 정의됨
- [ ] `narrative_extract` 반환 타입이 `NarrativeSummary`
- [ ] `_calculate_importance` 정적 메서드 → `NarrativeSummary.importance()`로 이동
- [ ] `scene_type` 검증이 `from_llm_dict`에서 한 곳에서 처리
- [ ] `narrative.get(...)` 패턴이 `post_turn.py`에서 사라짐 (속성 접근으로 통일)
- [ ] turn_log JSON 직렬화/역직렬화 회귀 없음 (`state_changes` 형식 유지)
- [ ] `pytest tests/` 통과

## 검증 방법

```bash
grep -n "narrative.get\|narrative\[" saga/agents/
# 결과: 0건이어야 함

pytest tests/test_post_turn.py -v
pytest tests/test_extractors.py -v
```

## 관련 finding

- 독립 작업. 다른 finding과 의존성 없음.
- 향후 `scene_type` 추가/변경 시 한 곳만 수정 (Literal + `from_llm_dict` validation).
