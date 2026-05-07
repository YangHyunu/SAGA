---
risk_code: R2  # Change Propagation
severity: critical
title: LLMClient `_last_usage` race on concurrent calls
affected_files:
  - saga/llm/client.py
  - saga/services/chat_handler.py
  - saga/services/stream.py
  - saga/agents/extractors.py
  - saga/agents/post_turn.py
  - saga/agents/curator.py
  - saga/cost_tracker.py
order: 1
---

# 🔴 Critical — LLMClient `_last_usage` race on concurrent calls

## Symptom

`saga/llm/client.py:71-72`에서 `_last_usage`/`_last_cache_stats`를 인스턴스 dict로 보관하고, 매 LLM 호출이 끝날 때 덮어씁니다.

```python
# saga/llm/client.py
self._last_cache_stats = {"cache_read": 0, "cache_create": 0}
self._last_usage = {"model": "", "input_tokens": 0, ...}
```

호출자들이 **호출 직후** 인스턴스 상태를 읽어 cost 기록에 사용합니다.

| 호출자 | 라인 | 패턴 |
|--------|------|------|
| `services/chat_handler.py` | 190 | `usage = deps.llm_client._last_usage` |
| `services/stream.py` | 96 | `usage = deps.llm_client._last_usage` |
| `agents/extractors.py` | 57 | `usage = llm_client._last_usage` |
| `agents/post_turn.py` | 240 | `usage = self.llm_client._last_usage` (NPC dedup) |
| `agents/curator.py` | 82 | `usage = self.llm_client._last_usage` (Curator) |

Sub-B / Curator / NPC dedup이 백그라운드 task로 동시에 LLM을 부르면, 메인 호출의 usage 값을 백그라운드 호출이 덮어씁니다.

## Source

- Hunt & Thomas — *The Pragmatic Programmer*, Orthogonality (한 변경의 영향이 무관한 dimension에 미침)
- Winters et al. — *Software Engineering at Google*, Hyrum's Law (다수 caller가 "method 호출 직후 인스턴스 상태가 자기 호출의 결과"라는 비공식 계약에 의존)

## Consequence

- 비용 로그(`cost_log` 테이블)가 다른 호출의 토큰으로 뒤섞여 기록됩니다.
- Sub-B / Curator가 active일수록 wrong-attribution이 누적되며, 캐시 절감액(`savings_usd`) 정확도가 무너집니다.
- 디버깅 시 "왜 sub_b가 50K input?" 같은 헛 추적을 유발합니다.
- LangSmith 트레이스와 cost_log 사이 mismatch.

## Remedy

### 변경 1: `LLMClient` 시그니처에 usage 반환 추가

```python
# saga/llm/client.py
from dataclasses import dataclass

@dataclass
class LLMUsage:
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_create_tokens: int = 0

async def call_llm(self, ...) -> tuple[str, LLMUsage]: ...
async def call_llm_stream(self, ...) -> AsyncIterator[tuple[str, LLMUsage | None]]: ...
```

스트리밍은 마지막 chunk에 `(text=None, usage=LLMUsage)` 또는 별도 `get_final_usage()` 콜백.

### 변경 2: 인스턴스 mutable state 제거

`self._last_usage`, `self._last_cache_stats` 필드 삭제.

### 변경 3: 호출자 수정 (5곳)

```python
# Before
llm_response = await deps.llm_client.call_llm(...)
usage = deps.llm_client._last_usage  # ← race
await deps.cost_tracker.record(UsageRecord(model=usage["model"], ...))

# After
llm_response, usage = await deps.llm_client.call_llm(...)
await deps.cost_tracker.record(UsageRecord(
    model=usage.model, input_tokens=usage.input_tokens, ...
))
```

## Acceptance Criteria

- [ ] `LLMClient`에서 `_last_usage`, `_last_cache_stats` 필드 제거
- [ ] `call_llm`, `call_llm_stream`이 usage를 반환값에 포함
- [ ] 5개 호출자 모두 새 시그니처로 수정 (chat_handler, stream, extractors, post_turn, curator)
- [ ] `pytest tests/` 통과
- [ ] `ruff check saga/` 통과
- [ ] 동시 호출 race 테스트 추가 (`asyncio.gather`로 메인+Sub-B 동시 호출 후 cost_log 정확성 검증)

## 검증 방법

```bash
# 회귀 테스트
pytest tests/ -v

# 새 race 테스트 작성 후
pytest tests/test_llm_usage_isolation.py -v

# 실제 세션으로 cost_log 확인
sqlite3 db/state.db "SELECT call_type, model, input_tokens FROM cost_log ORDER BY id DESC LIMIT 20"
```

## 관련 메모리/이슈

- `saga/cost_tracker.py`의 `UsageRecord` dataclass는 그대로 유지 (LLMUsage → UsageRecord 변환은 호출자 책임).
- 같은 race가 LangSmith `@traceable` 데코레이터에는 영향 없음 (LangSmith는 자체 wrapper로 usage 추적).
