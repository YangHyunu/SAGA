---
risk_code: R3  # Knowledge Duplication
severity: warning
title: LLM 프로바이더별 파라미터 변환 4중 복제
affected_files:
  - saga/llm/client.py
order: 4
---

# 🟡 Warning — LLM 프로바이더별 파라미터 변환 4중 복제

## Symptom

`saga/llm/client.py`의 5개 메서드가 동일한 `top_p` / `frequency_penalty` / `presence_penalty` / `stop` 매핑 로직을 반복합니다.

| 메서드 | 라인 |
|--------|------|
| `_call_anthropic` | 260-263 |
| `_stream_anthropic` | 304-308 |
| `_call_google` | 345-355 |
| `_call_openai` | 408-415 |
| `_stream_openai` | 445-452 |

각 프로바이더가 받는 키 이름이 다르긴 합니다.
- Anthropic: `top_p`, `stop_sequences` (list로 변환)
- Google: `top_p`, `frequency_penalty`, `presence_penalty`, `stop_sequences`, `response_mime_type`
- OpenAI: `top_p`, `frequency_penalty`, `presence_penalty`, `stop`

하지만 **들어온 kwargs가 None이 아닌 경우만 forward** 패턴은 동일하고, 5번 반복됩니다.

```python
# 반복되는 패턴 (provider마다 살짝 다름)
if kwargs.get("top_p") is not None:
    create_kwargs["top_p"] = kwargs["top_p"]
if kwargs.get("stop") is not None:
    stop = kwargs["stop"]
    create_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop
```

## Source

- Hunt & Thomas — *Pragmatic Programmer*, DRY (Don't Repeat Yourself)
- Fowler — *Refactoring*, Duplicate Code / Shotgun Surgery

## Consequence

- 새 gen 파라미터(예: `seed`, `top_k`) 추가 시 5곳을 동시에 수정해야 함
- Forget 시 일부 프로바이더만 지원되어 silent breakage
- Stop 파라미터의 str-vs-list 처리 로직이 Anthropic·Google에 두 번 반복 — 한쪽 버그 수정 시 다른 쪽 빠뜨릴 수 있음

## Remedy

### 변경 1: 프로바이더별 파라미터 매핑 정의

```python
# saga/llm/client.py

_PROVIDER_PARAM_MAP = {
    "anthropic": {
        "top_p": "top_p",
        "stop": ("stop_sequences", lambda v: [v] if isinstance(v, str) else v),
    },
    "google": {
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop": ("stop_sequences", lambda v: [v] if isinstance(v, str) else v),
        "response_mime_type": "response_mime_type",
    },
    "openai": {
        "top_p": "top_p",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "stop": "stop",
    },
}


def _apply_gen_params(target: dict, kwargs: dict, provider: str) -> None:
    """Forward only present kwargs, mapping each to provider-specific key."""
    mapping = _PROVIDER_PARAM_MAP[provider]
    for src_key, dest in mapping.items():
        v = kwargs.get(src_key)
        if v is None:
            continue
        if isinstance(dest, tuple):
            dest_key, transform = dest
            target[dest_key] = transform(v)
        else:
            target[dest] = v
```

### 변경 2: 5개 메서드 모두 한 줄로 단순화

```python
# Before
create_kwargs = {"model": model, ...}
if kwargs.get("top_p") is not None:
    create_kwargs["top_p"] = kwargs["top_p"]
if kwargs.get("stop") is not None:
    stop = kwargs["stop"]
    create_kwargs["stop_sequences"] = [stop] if isinstance(stop, str) else stop

# After
create_kwargs = {"model": model, ...}
_apply_gen_params(create_kwargs, kwargs, provider="anthropic")
```

## Acceptance Criteria

- [ ] `_PROVIDER_PARAM_MAP` 정의됨
- [ ] `_apply_gen_params` 헬퍼가 5개 메서드에서 사용됨
- [ ] 각 메서드의 if 분기 라인 수가 1줄로 축소
- [ ] `pytest tests/` 통과
- [ ] 신규 단위 테스트: 4개 프로바이더 × 각 파라미터 forward 검증
- [ ] `ruff check saga/` 통과

## 검증 방법

```bash
# 중복 패턴이 모두 사라졌는지
grep -n "kwargs.get(\"top_p\") is not None" saga/llm/client.py
# 결과: 0건이어야 함

pytest tests/test_llm_client.py -v
```

## 관련 finding

- **Finding 01** (LLMClient 시그니처 변경)과 동일 파일 — 같은 PR로 처리하면 한 번에 client.py 정리 가능.
- 이 작업은 OpenAI/Google/Anthropic SDK 버전 업그레이드 시 진단 비용을 줄여줍니다 (한 곳만 보면 됨).
