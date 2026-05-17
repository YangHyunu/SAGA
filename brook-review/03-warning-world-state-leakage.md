---
risk_code: R2  # Change Propagation (Information Leakage)
severity: warning
title: world_state KV가 무관한 도메인의 dumping ground
affected_files:
  - saga/storage/sqlite_db.py
  - saga/system_stabilizer.py
  - saga/message_compressor.py
  - saga/services/chat_handler.py
  - saga/agents/post_turn.py
order: 3
---

# 🟡 Warning — `world_state` KV가 무관한 도메인의 dumping ground

## Symptom

`saga/storage/sqlite_db.py:28-34`의 `world_state(session_id, key, value)` 테이블이 다음 모듈에서 직접 string key로 읽고 씁니다.

| 모듈 | 키 | 라인 |
|------|----|------|
| `system_stabilizer.py` | `canonical_system_prompt`, `canonical_system_hash` | 62-66, 222-229 |
| `message_compressor.py` | `compressed_chunks`, `compressed_through_turn` | 19-20, 116, 140-156 |
| `services/chat_handler.py` | `scriptstate` | 61 |
| `agents/post_turn.py` | `scriptstate` (재조회) | 86 |

키 충돌 방지·스키마 진화·로컬리티가 깨져 있습니다.

## Source

- Ousterhout — *A Philosophy of Software Design* Ch. 5 (Information Hiding and Leakage)
- Hunt & Thomas — DRY (key 문자열이 두 모듈에 흩어짐)

## Consequence

- 한 모듈이 키 명칭을 바꾸려면 다른 모듈도 같이 수정해야 함 (Shotgun Surgery)
- 새 캐시 모듈을 추가할 때 키 충돌 위험을 매번 체크해야 함
- 마이그레이션 시 어떤 KV가 어떤 모듈 소유인지 외부에서 보이지 않음
- `scriptstate` 키가 `chat_handler`(쓰기)와 `post_turn`(읽기) 두 곳에 흩어져 있어 schema 변경 시 한쪽만 수정하면 silent break

## Remedy

### 옵션 A — Facade 패턴 (권장, 변경 최소)

각 도메인별로 facade 클래스를 도입하고 `world_state`는 그대로 유지하되 외부 접근은 facade를 통해서만.

```python
# saga/storage/canonical_system_store.py
class CanonicalSystemStore:
    def __init__(self, sqlite_db):
        self._db = sqlite_db

    async def get(self, session_id) -> tuple[str | None, str | None]:
        text = await self._db.get_world_state_value(session_id, "canonical_system_prompt")
        h = await self._db.get_world_state_value(session_id, "canonical_system_hash")
        return text, h

    async def save(self, session_id, text: str, text_hash: str) -> None:
        await self._db.upsert_world_state(session_id, "canonical_system_prompt", text)
        await self._db.upsert_world_state(session_id, "canonical_system_hash", text_hash)
```

마찬가지로 `CompressedChunkStore`, `ScriptStateStore`를 생성.

`SystemStabilizer`, `MessageCompressor` 등은 facade만 의존.

### 옵션 B — 별도 테이블 분리 (더 깨끗하지만 마이그레이션 필요)

```sql
CREATE TABLE prompt_cache_state (
    session_id TEXT PRIMARY KEY,
    canonical_prompt TEXT,
    canonical_hash TEXT,
    updated_at TIMESTAMP
);

CREATE TABLE compressed_chunks (
    session_id TEXT PRIMARY KEY,
    chunks TEXT,  -- JSON
    compressed_through_turn INTEGER,
    updated_at TIMESTAMP
);

CREATE TABLE script_state (
    session_id TEXT PRIMARY KEY,
    state TEXT,
    updated_at TIMESTAMP
);
```

기존 `world_state`는 deprecated로 표시 후 마이그레이션 스크립트 작성. SAGA 셸브 전환 상태에서는 옵션 A로 충분.

## Acceptance Criteria

- [ ] `world_state` 테이블에 직접 접근하는 모듈 = `sqlite_db.py` + facade 3개로 한정 (Stabilizer/Compressor/chat_handler/post_turn 전부 facade 경유)
- [ ] 도메인별 키 prefix 또는 namespace가 facade 내부에 캡슐화됨
- [ ] 키 문자열이 모듈에 흩어져 있지 않음 (grep `"canonical_system_prompt"` → 단일 파일에서만)
- [ ] `pytest tests/` 통과
- [ ] facade별 단위 테스트 추가

## 검증 방법

```bash
# 키 문자열이 facade 외 다른 곳에 없는지 확인
grep -rn "canonical_system_prompt\|canonical_system_hash\|compressed_chunks\|compressed_through_turn" saga/
# 결과는 saga/storage/canonical_system_store.py + compressed_chunk_store.py 만 나와야 함

pytest tests/ -v
ruff check saga/
```

## 관련 finding

- **Finding 06** (`deps` 글로벌)과 함께 처리하면 facade를 `deps` 컨테이너로 자연스럽게 흡수 가능.
- **Finding 07** (cache_marker chunk prefix leak)과 동일 테마 — 모듈 간 sentinel/key 의존을 끊어내는 작업.
