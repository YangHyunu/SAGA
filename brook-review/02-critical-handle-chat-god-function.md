---
risk_code: R1  # Cognitive Overload
severity: critical
title: handle_chat god function (215줄, 9개 책임)
affected_files:
  - saga/services/chat_handler.py
order: 2
---

# 🔴 Critical — `handle_chat` god function (215줄, 9개 책임)

## Symptom

`saga/services/chat_handler.py:37-252`의 `handle_chat()` 단일 함수가 다음 9가지 책임을 모두 직렬로 처리합니다.

| # | 책임 | 라인 |
|---|------|------|
| 1 | 트레이스 헤더 로깅 (sys/usr/asst 카운트, gen params) | 37-50 |
| 2 | 세션 식별 + scriptstate 영속 | 52-62 |
| 3 | 토큰 예산 계산 | 64-67 |
| 4 | 캐시 진단 해시 (디버그용) | 69-74 |
| 5 | 멀티모달 part 보존 | 76-82 |
| 6 | Stabilizer → Compressor → ContextBuilder 파이프라인 호출 | 83-117 |
| 7 | cacheable 메시지 빌드 + 멀티모달 복원 + BP 카운트 디버그 로그 | 118-141 |
| 8 | last_user_input 추출 | 143-147 |
| 9 | stream/sync 분기 + finalize_turn + state block strip + 응답 직렬화 | 149-252 |

변수 `_multimodal_parts`(76-82)·`last_user_input`(143-147)·`stream_ctx`(150) 등이 한 스코프에 공존하며, 분기 후에도 사용됩니다.

## Source

- Fowler — *Refactoring*, Long Method
- McConnell — *Code Complete* Ch. 7 (High-Quality Routines: 단일 추상화 수준)

## Consequence

- 메모리 노트(`MEMORY.md`)에 "chat_handler.py 분해 — god object → 세션/캐시마킹/후처리 분리"가 진행 중으로 적혀 있는데, **서비스 레이어로 추출은 했지만 호출 그래프는 여전히 한 함수에 직렬**되어 있습니다 (567줄 → 252줄로 줄었으나 여전히 god function).
- 변경 시 모든 단계 영향을 머릿속에 들고 있어야 합니다.
- 각 단계가 독립 단위 테스트 불가 — 현재는 e2e 외에는 stub mocking 외엔 검증이 어렵습니다.
- 새 단계(예: 사용자 의도 감지, A/B 라우팅) 추가 시 함수가 더 두꺼워짐.

## Remedy

다음 4단계로 분리합니다.

```python
# saga/services/chat_handler.py

@dataclass
class RequestContext:
    session_id: str
    session: dict
    scriptstate: dict | None
    last_user_input: str
    multimodal_parts: dict[int, list]
    messages_dicts: list[dict]
    is_continuation: bool
    gen_params: dict
    token_budgets: TokenBudgets

@dataclass
class TokenBudgets:
    messages_tokens: int
    remaining: int
    dynamic: int

@dataclass
class PipelineResult:
    augmented_messages: list[dict]
    md_prefix: str
    dynamic_suffix: str
    lorebook_delta: str


async def _extract_request_context(request, raw_request) -> RequestContext: ...
    # session/scriptstate/budget/multimodal/continuation 추출
    # ~50 lines

async def _build_pipeline_input(ctx: RequestContext) -> PipelineResult: ...
    # Stabilizer → Compressor → ContextBuilder → cache_marker
    # ~40 lines

async def _run_inference_sync(ctx, pipeline) -> tuple[str, LLMUsage]: ...
async def _run_inference_stream(ctx, pipeline) -> StreamingResponse: ...

async def _build_response(ctx, llm_response, usage, turn_number) -> ChatCompletionResponse: ...

# 결과
async def handle_chat(request, raw_request) -> Response:
    ctx = await _extract_request_context(request, raw_request)
    pipeline = await _build_pipeline_input(ctx)
    if request.stream:
        return await _run_inference_stream(ctx, pipeline, request)
    llm_response, usage = await _run_inference_sync(ctx, pipeline, request)
    final_response, turn_number = await finalize_turn(...)
    return _build_response(ctx, final_response, usage, turn_number)
```

`handle_chat` 본문은 ~30줄 셸이 됩니다.

## Acceptance Criteria

- [ ] `handle_chat` 본문이 50줄 이하
- [ ] 4개 헬퍼 함수가 각각 독립 단위 테스트 가능 (FastAPI Request mock만 있으면 됨)
- [ ] `pytest tests/` 통과 (e2e 시나리오 회귀 없음)
- [ ] 로깅 출력 형식이 동일 (트레이스 grep이 깨지지 않음)
- [ ] `_extract_request_context` 단위 테스트 신규 추가 (헤더/유저필드/시스템해시 3경로)
- [ ] `_build_pipeline_input` 단위 테스트 신규 추가 (stabilizer/compressor mock)

## 검증 방법

```bash
pytest tests/ -v
ruff check saga/

# 로컬 server 띄우고 RisuAI 연결 후 1턴 정상 흐름 확인
python -m saga
# 로그에 [Trace] 항목들이 기존과 동일하게 출력되는지 grep 비교
```

## 의존성

- **Finding 01과 직렬:** Finding 01 먼저 처리 (LLMClient 시그니처 변경)하면, `_run_inference_sync`가 `tuple[str, LLMUsage]`를 자연스럽게 받습니다. 반대로 진행하면 두 번 수정해야 합니다.

## 관련 코드

- `saga/services/post_turn_pipeline.py` `finalize_turn`은 이미 분리되어 있어 그대로 활용.
- `saga/services/cache_marker.py` `build_cacheable_messages`는 Finding 07과 함께 정리될 예정 — 현재 시점에는 그대로 호출.
