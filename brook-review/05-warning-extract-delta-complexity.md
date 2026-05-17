---
risk_code: R1  # Cognitive Overload
severity: warning
title: SystemStabilizer._extract_delta 4-way 분기 + nested loop
affected_files:
  - saga/system_stabilizer.py
order: 5
---

# 🟡 Warning — `SystemStabilizer._extract_delta` 4-way 분기 + nested loop

## Symptom

`saga/system_stabilizer.py:125-211` (87줄) — 한 함수가 다음 4가지 결과를 한꺼번에 분류·생성합니다.

| 분기 | 판정 기준 | 결과 |
|------|-----------|------|
| `inject_append` | `a_para.startswith(r_para)` | 추가된 부분만 delta |
| `inject_shrink` | `a_para in r_para` | 제외 (rare) |
| `inject_replace` | 단어 overlap > 50%, no substring | canonical 갱신 신호 |
| pure-add | 위 3개 매칭 안 됨 | delta에 그대로 추가 |

코드 구조:

```python
def _extract_delta(self, canonical, current) -> tuple[str, str, bool]:
    # ... paragraphs split ...
    for r_para in removed:           # nested loop
        for a_para in added:
            if a_para in matched_added:
                continue
            if a_para.startswith(r_para):  # inject_append
                ...
                break
            if a_para in r_para:           # inject_shrink
                ...
                break
            # inject_replace 단어 overlap 계산
            if overlap > 0.5:
                ...
                break
    # pure_added 처리
    ...
    return (canonical, delta_text, canonical_needs_update)
```

함수가 3-튜플을 반환하고, 호출자(line 86-119)가 다시 4-way 분기로 처리합니다.

## Source

- McConnell — *Code Complete* Ch. 7 (routine length, single level of abstraction)
- Fowler — *Refactoring*, Long Method

## Consequence

- 최근 `inject_append → startswith` 버그(`8e99db8 fix: inject_append substring → startswith`)가 잡힌 위치 — 분류 의도가 코드에서 명확하지 않았던 것이 원인
- 다음 inject 패턴(예: `inject_after`, `inject_before`)이 들어오면 또 같은 함수에 분기 추가 → 함수가 더 두꺼워짐
- 단위 테스트 시 4가지 분기 × edge case (empty/whitespace/multiple matches)를 한 함수 단위로 작성해야 함 — 분류 단계만 따로 테스트하기 어려움

## Remedy

### 변경 1: 분류 단계와 빌드 단계 분리

```python
# saga/system_stabilizer.py
from typing import Literal
from dataclasses import dataclass

InjectKind = Literal["append", "shrink", "replace", "none"]

@dataclass
class InjectMatch:
    r_para: str
    a_para: str
    kind: InjectKind
    appended: str = ""  # only for "append"

def _classify_inject(r_para: str, a_para: str) -> InjectMatch:
    if a_para.startswith(r_para):
        return InjectMatch(r_para, a_para, "append", appended=a_para[len(r_para):].strip())
    if a_para in r_para:
        return InjectMatch(r_para, a_para, "shrink")
    r_words = set(r_para.lower().split())
    a_words = set(a_para.lower().split())
    if r_words and a_words:
        overlap = len(r_words & a_words) / max(len(r_words), len(a_words))
        if overlap > 0.5:
            return InjectMatch(r_para, a_para, "replace")
    return InjectMatch(r_para, a_para, "none")
```

### 변경 2: `_extract_delta`는 분류 결과를 모아 delta 빌드만

```python
def _extract_delta(self, canonical, current) -> tuple[str, str, bool]:
    canonical_paras = self._split_paragraphs(canonical)
    current_paras = self._split_paragraphs(current)
    canonical_counts = Counter(canonical_paras)
    current_counts = Counter(current_paras)

    added = set(p for p in current_paras if current_counts[p] > canonical_counts[p])
    removed = set(p for p in canonical_paras if canonical_counts[p] > current_counts[p])

    if not added:
        return canonical, "", False
    if not removed:
        return canonical, "\n\n".join(p for p in current_paras if p in added), False

    matches = self._match_inject_pairs(removed, added)
    matched_added = {m.a_para for m in matches if m.kind != "none"}
    canonical_needs_update = any(m.kind == "replace" for m in matches)
    inject_delta_parts = [m.appended for m in matches if m.kind == "append" and m.appended]

    pure_added = added - matched_added
    delta_parts = [p for p in current_paras if p in pure_added] + inject_delta_parts
    return canonical, "\n\n".join(delta_parts), canonical_needs_update

def _match_inject_pairs(self, removed: set[str], added: set[str]) -> list[InjectMatch]:
    matches = []
    used = set()
    for r in removed:
        for a in added:
            if a in used:
                continue
            m = _classify_inject(r, a)
            if m.kind != "none":
                matches.append(m)
                used.add(a)
                break
    return matches
```

### 변경 3: 분류 단위 테스트

```python
# tests/test_system_stabilizer.py
@pytest.mark.parametrize("r, a, expected_kind", [
    ("hello", "hello world", "append"),
    ("hello world", "hello", "shrink"),
    ("the quick brown fox", "the quick red fox", "replace"),
    ("foo", "bar", "none"),
])
def test_classify_inject(r, a, expected_kind):
    assert _classify_inject(r, a).kind == expected_kind
```

## Acceptance Criteria

- [ ] `_classify_inject` 단일 책임 함수로 추출
- [ ] `_extract_delta` 본문 50줄 이하
- [ ] nested for 루프 제거 (`_match_inject_pairs`로 분리)
- [ ] `_classify_inject` 단위 테스트 추가 (4가지 kind × edge case)
- [ ] 기존 inject_append/replace 동작 회귀 없음
- [ ] `pytest tests/` 통과

## 검증 방법

```bash
pytest tests/test_system_stabilizer.py -v

# inject 시나리오 회귀
pytest tests/ -k "stabilize or inject" -v
```

## 관련 finding

- 독립 작업. 다른 finding과 의존성 없음 — 사이드 트랙으로 진행 가능.
- `_split_paragraphs`는 그대로 유지 (이미 단일 책임).
