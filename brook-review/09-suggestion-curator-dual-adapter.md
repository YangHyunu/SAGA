---
risk_code: R4  # Accidental Complexity
severity: suggestion
title: Curator dual-adapter (Letta + Direct) speculative complexity
affected_files:
  - saga/agents/curator.py
  - saga/adapters/curator_adapter.py
order: 9
---

# 🟢 Suggestion — Curator dual-adapter (Letta + Direct) speculative complexity

## Symptom

`saga/agents/curator.py:34-53` — Letta primary + Direct fallback 두 어댑터 동시 보유.

```python
self.letta_adapter = LettaCuratorAdapter(config)
self.fallback_adapter = DirectLLMCuratorAdapter(llm_client, config)
self._use_letta = False
```

`saga/adapters/curator_adapter.py:122-172` — Letta context overflow 시 agent 삭제 후 block 보존·복원하는 `_recreate_agent` 메서드(50줄).

`curator.py:118-128` — Letta 실패 시 `fallback_adapter` 재시도.

`curator.py:280-302` — `_sync_letta_memory` 메서드는 `_use_letta`만 체크. **Direct 어댑터로 fallback된 경우 narrative_summary가 stable_prefix에 동기화되지 않음** — 두 경로의 행동 비대칭.

## Source

- Fowler — *Refactoring*, Speculative Generality
- Brooks — *The Mythical Man-Month* Ch. 5 (Second-System Effect: 한 번에 두 경로를 다 지원하려는 욕심)

## Consequence

- 같은 dict shape을 반환하는 두 어댑터의 유지비 (curator_adapter.py 384줄 중 절반 이상이 Letta 경로)
- Direct fallback의 결과가 stable_prefix에 반영 안 되어 메모리 품질 비대칭
- 메모리 노트(`MEMORY.md`): "SAGA 종료 + Plan A 피벗 (2026-04-20 결정)" — 즉, SAGA는 셸브 단계인데 dual path 정리 비용을 들일 가치가 줄어듦
- 새 contributors가 이해할 코드 경로가 2배

## Remedy (옵션은 SAGA 정책에 따름)

### 옵션 A — Letta 단일화 (메모리 품질 우선)

- `DirectLLMCuratorAdapter` 삭제
- `curator.py`의 `_use_letta` 분기 제거
- Letta unavailable이면 명시적 에러 (curator 스킵)

장점: 코드 50% 축소, narrative_summary 동기화 일관성 보장.
단점: Letta 의존성 강화.

### 옵션 B — Direct 단일화 (의존성 최소화)

- `LettaCuratorAdapter`, `_recreate_agent`, `_sync_letta_memory` 삭제
- Memory Block 기능 포기 (narrative_summary는 매 큐레이션마다 LLM이 새로 생성)

장점: letta-client 의존성 제거, 단일 경로
단점: 큐레이션 연속성 약화 (이전 큐레이션 판단을 매번 다시 read해야 함)

### 옵션 C — Direct 어댑터에 narrative_summary 동기화 책임 추가 (현 dual path 유지)

- `DirectLLMCuratorAdapter.run` 결과에 `narrative_summary` 필드 추가
- `_sync_letta_memory`를 `_sync_curator_memory`로 일반화하여 두 어댑터 모두 처리

장점: 호환성 유지하며 비대칭만 해소.
단점: 복잡도 그대로.

**현재 SAGA 셸브 상태에서는 옵션 C(최소 변경)** 또는 코드 동결.
**Plan A 피벗(Cache Keeper)이 SAGA 코드를 재사용한다면 옵션 A**가 깔끔.

## Acceptance Criteria

옵션 결정 후 적용:

### 옵션 A 선택 시
- [ ] `DirectLLMCuratorAdapter` 클래스 삭제
- [ ] `curator.py`의 `fallback_adapter`, `_use_letta`, fallback 재시도 분기 제거
- [ ] Letta unavailable 시 curator 스킵 + WARNING 로그
- [ ] `pytest tests/` 통과

### 옵션 B 선택 시
- [ ] `LettaCuratorAdapter` 삭제
- [ ] `letta-client` 의존성 제거 (`requirements.txt` / `pyproject.toml`)
- [ ] `_sync_letta_memory` 삭제
- [ ] Memory Block 관련 config 제거

### 옵션 C 선택 시
- [ ] `DirectLLMCuratorAdapter.run`이 `narrative_summary` 필드 반환
- [ ] `_sync_letta_memory` → `_sync_curator_narrative` 일반화
- [ ] 두 경로 모두 stable_prefix 동기화 회귀 테스트

## 검증 방법

```bash
pytest tests/test_curator.py -v

# 옵션 A·B 시 LOC 감소 확인
wc -l saga/adapters/curator_adapter.py saga/agents/curator.py
```

## 관련 finding

- 독립 작업. 다른 finding과 의존성 없음.
- **결정 의존성:** 사용자가 SAGA 향후 운영 방향(셸브 / Plan A 흡수 / 단순 유지)을 정한 뒤 옵션 선택.
- 현 시점 우선순위 낮음 (suggestion).
