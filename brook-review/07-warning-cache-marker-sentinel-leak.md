---
risk_code: R2  # Change Propagation (Information Leakage)
severity: warning
title: cache_marker가 MessageCompressor 내부 sentinel을 import
affected_files:
  - saga/services/cache_marker.py
  - saga/message_compressor.py
order: 7
---

# 🟡 Warning — `cache_marker`가 `MessageCompressor` 내부 sentinel을 import

## Symptom

`saga/services/cache_marker.py:35`:

```python
from saga.message_compressor import _CHUNK_USER_PREFIX
summary_asst_indices = [
    idx for idx in assistant_indices
    if idx > 0 and messages[idx - 1].get("role") == "user"
    and str(messages[idx - 1].get("content", "")).startswith(_CHUNK_USER_PREFIX)
]
```

`_CHUNK_USER_PREFIX = "[SAGA: 이전 대화 요약"` (`message_compressor.py:23`)는 **단일 underscore prefix**로 "내부 전용" 관용을 따르지만, 다른 모듈이 그것에 의존합니다.

cache_marker는 메시지 배열에서 "이게 chunk-pair user 메시지인가?"를 판별해야 BP2를 첫 chunk assistant에 고정할 수 있는데, 그 판단을 위해 compressor의 내부 prefix 문자열을 알아야 합니다.

## Source

- Ousterhout — *A Philosophy of Software Design* Ch. 5 (Information Leakage: 한 모듈의 design decision이 다른 모듈에 노출되어 변경 시 양쪽이 묶임)
- Fowler — *Refactoring*, Inappropriate Intimacy

## Consequence

- compressor가 prefix 문자열을 바꾸면 (예: 한국어 → 영어로 변경, 또는 더 짧게 정리) cache_marker가 silently 깨집니다 — BP2가 잘못된 위치에 찍혀 캐시 hit이 0%로 떨어져도 에러는 안 남.
- 두 모듈 사이의 비공식 계약이 grep으로만 발견됨.
- 메모리 노트에 "BP2를 첫 번째 chunk assistant에 고정"이 핵심 캐싱 결정인데, 그 핵심이 string 비교에 묶여 있음.

## Remedy

### 옵션 A — 메시지 메타데이터 마커 (권장)

compressor가 chunk pair를 만들 때 `_saga_chunk: True` 메타데이터 추가, cache_marker는 그것만 검사.

```python
# saga/message_compressor.py
def _rebuild_with_chunks(self, system_msgs, non_system, chunks, compressed_through):
    result = list(system_msgs)
    for i, chunk in enumerate(chunks):
        chunk_num = i + 1
        result.append({
            "role": "user",
            "content": f"{_CHUNK_USER_PREFIX} {chunk_num}: Turn {chunk['from_turn']}-{chunk['to_turn']}]",
            "_saga_chunk": True,  # ← 메타데이터
        })
        result.append({
            "role": "assistant",
            "content": chunk["summary_text"],
            "_saga_chunk": True,
        })
    ...
```

```python
# saga/services/cache_marker.py
summary_asst_indices = [
    idx for idx in assistant_indices
    if idx > 0
    and messages[idx - 1].get("role") == "user"
    and messages[idx - 1].get("_saga_chunk") is True
]
```

주의: LLM client `_prepare_anthropic_messages`(`saga/llm/client.py:168`) 등에서 `_saga_chunk` 키를 누락시키도록 확인 (Anthropic API에 보내면 안 됨).

### 옵션 B — 공개 메서드 노출

```python
# saga/message_compressor.py
class MessageCompressor:
    @staticmethod
    def is_chunk_user_message(msg: dict) -> bool:
        content = msg.get("content", "")
        return isinstance(content, str) and content.startswith(_CHUNK_USER_PREFIX)
```

cache_marker는 `MessageCompressor.is_chunk_user_message(msg)` 호출. prefix는 여전히 module-private.

### 옵션 C — sentinel 상수를 공개 모듈에 빼기 (가장 가벼움)

```python
# saga/message_compressor.py
CHUNK_USER_PREFIX = "[SAGA: 이전 대화 요약"  # 공개로 명시
_CHUNK_USER_PREFIX = CHUNK_USER_PREFIX  # 하위 호환
```

또는 `saga/constants.py`로 분리.

**권장:** 옵션 A. 메타데이터 기반이 가장 robust. prefix 문자열 변경 자유도 확보.

## Acceptance Criteria

- [ ] cache_marker가 compressor 내부 sentinel(`_CHUNK_USER_PREFIX`) 직접 import 안 함
- [ ] prefix 문자열을 변경해도 cache_marker 동작 유지 (회귀 테스트로 확인)
- [ ] LLM provider별 메시지 변환에서 `_saga_chunk` 키 누락 (Anthropic/Google/OpenAI API에 unknown field 안 보내짐)
- [ ] `pytest tests/` 통과
- [ ] BP2 위치 검증 단위 테스트 추가 (chunk 있을 때 첫 chunk assistant에 BP2 찍히는지)

## 검증 방법

```bash
# 옵션 A 선택 시
grep -n "_CHUNK_USER_PREFIX" saga/services/
# 결과: 0건이어야 함

# 옵션 A·B 모두 적용 가능한 검증
grep -n "_saga_chunk\|is_chunk_user_message" saga/

pytest tests/test_cache_marker.py -v
pytest tests/test_message_compressor.py -v
```

## 관련 finding

- **Finding 03** (world_state facade) 및 **Finding 06** (deps 글로벌)와 동일 테마 (의존성 누수 정리).
- 옵션 A를 채택하면 향후 chunk 외에 다른 SAGA 메시지 타입(예: `_saga_lorebook_delta`, `_saga_dynamic`) 마킹에도 일관된 패턴으로 확장 가능.
