---
risk_code: R5  # Dependency Disorder
severity: warning
title: deps 글로벌 모듈이 ambient service locator
affected_files:
  - saga/core/dependencies.py
  - saga/services/chat_handler.py
  - saga/services/stream.py
  - saga/services/cache_marker.py
  - saga/services/post_turn_pipeline.py
  - saga/services/session_extractor.py
  - saga/core/lifespan.py
order: 6
---

# 🟡 Warning — `deps` 글로벌 모듈이 ambient service locator

## Symptom

`saga/core/dependencies.py`가 다음 14개 가량의 모듈 변수를 노출하고, 전 services / agents가 `from saga.core import dependencies as deps`로 직접 접근합니다.

| 카테고리 | 변수 |
|----------|------|
| config | `config` |
| 서비스 | `session_mgr`, `sqlite_db`, `system_stabilizer`, `message_compressor`, `context_builder`, `llm_client`, `cost_tracker`, `curator`, `post_turn` |
| mutable globals | `_pending_responses`, `_background_tasks`, `_warming_data`, `_warming_lock` |
| 정규식 sentinels | `_CONTINUE_PATTERN`, `_SESSION_ID_RE` |

직접 import 사용처:

```bash
$ grep -rn "from saga.core import dependencies as deps" saga/
saga/services/chat_handler.py:12
saga/services/stream.py:7
saga/services/cache_marker.py:2
saga/services/post_turn_pipeline.py:6
saga/services/session_extractor.py:8
# ... etc
```

## Source

- Martin — *Clean Architecture*, DIP (Dependency Inversion Principle)
- Hunt & Thomas — Orthogonality (글로벌 mutable state는 무관한 코드 사이에 결합 만듦)

## Consequence

- 함수 시그니처가 의존성을 숨기므로 단위 테스트 시 `monkeypatch.setattr(deps, ...)` 외에는 mock이 어렵습니다.
- 모듈 의존 그래프가 별 모양 — `deps`가 hub. 한 service가 deps에 새 변수를 추가하면 다른 모듈에서 그 변수를 사용할지 안 할지 정적으로 판단 불가.
- 동시 테스트 시 글로벌 mutable state(`_pending_responses`, `_warming_data`)가 누설됩니다.
- `chat_handler.py`만 봐도 `deps.session_mgr`, `deps.sqlite_db`, `deps.config`, `deps.system_stabilizer`, `deps.message_compressor`, `deps.context_builder`, `deps.llm_client`, `deps.cost_tracker` 8개를 사용 — 이 함수의 진짜 의존성이 무엇인지 시그니처에서 보이지 않음.

## Remedy

### 변경 1: ServiceRegistry 클래스 도입

```python
# saga/core/registry.py
from dataclasses import dataclass

@dataclass
class ServiceRegistry:
    config: Config
    session_mgr: SessionManager
    sqlite_db: SQLiteDB
    llm_client: LLMClient
    system_stabilizer: SystemStabilizer
    message_compressor: MessageCompressor
    context_builder: ContextBuilder
    post_turn: PostTurnExtractor
    curator: CuratorRunner
    cost_tracker: CostTracker
    session_state: SessionState  # ← _pending_responses, _background_tasks, _warming_data 흡수

@dataclass
class SessionState:
    pending_responses: dict[str, dict] = field(default_factory=dict)
    background_tasks: set = field(default_factory=set)
    warming_data: dict = field(default_factory=dict)
    warming_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
```

### 변경 2: FastAPI lifespan에서 인스턴스화 + app.state에 보관

```python
# saga/core/lifespan.py
@asynccontextmanager
async def lifespan(app: FastAPI):
    services = await build_service_registry(config_path)
    app.state.services = services
    yield
    await services.llm_client.close()
    await services.sqlite_db.close()


def get_services(request: Request) -> ServiceRegistry:
    return request.app.state.services
```

### 변경 3: 라우트에서 명시적 Depends

```python
# saga/routes/chat.py
@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    services: ServiceRegistry = Depends(get_services),
    _auth: None = Depends(verify_bearer),
):
    return await handle_chat(request, raw_request, services)
```

### 변경 4: services를 함수 인자로 전달

```python
async def handle_chat(request, raw_request, services: ServiceRegistry):
    session = await services.session_mgr.get_or_create_session(...)
    ...
```

### 마이그레이션 전략

- **Phase 1:** ServiceRegistry 클래스만 만들고 `deps` 모듈은 ServiceRegistry 인스턴스를 노출하는 facade로 유지 (호환성).
- **Phase 2:** Top-level 라우트(`chat.py`, `metrics.py`, `sessions.py`, `admin.py`)를 Depends 패턴으로 전환.
- **Phase 3:** 내부 서비스에서 `deps.*` 접근 제거하고 인자로 전달.
- **Phase 4:** `deps` 모듈 삭제.

각 phase가 독립 commit/PR로 가능.

## Acceptance Criteria

- [ ] `ServiceRegistry`, `SessionState` dataclass 정의
- [ ] FastAPI lifespan이 `ServiceRegistry` 인스턴스를 `app.state.services`에 저장
- [ ] 라우트가 `Depends(get_services)`로 services 받음
- [ ] `from saga.core import dependencies as deps` import 0건 (`grep -c` 결과)
- [ ] 단위 테스트가 mock ServiceRegistry로 작성 가능
- [ ] `pytest tests/` 통과
- [ ] 동시 요청 통합 테스트 추가 (warming_data isolation 확인)

## 검증 방법

```bash
grep -rn "from saga.core import dependencies as deps" saga/
# 결과: 0건이어야 함 (Phase 4 완료 시)

grep -rn "deps\." saga/
# 결과: 0건 또는 saga/core/* 내부에서만

pytest tests/ -v
```

## 관련 finding

- **Finding 03** (world_state facade)와 동일한 테마 — 의존성 명시화. 가능하면 같은 작업자가 처리.
- **Finding 02** (handle_chat 분해) 후에 진행하면 함수가 작아져 `services` 전달이 자연스러움.
- mutable globals(`_pending_responses`, `_warming_data`) 캡슐화는 race condition 예방에 직접 기여.
