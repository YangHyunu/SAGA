# Brooks-Lint Review — SAGA `saga/` 코드베이스

**리뷰 일자:** 2026-05-06
**Mode:** PR Review (sampled — 전체 코드베이스 적용)
**Scope:** `saga/` 5,954 LOC, 44 파일. hot path 13개 모듈 깊게 분석.
**Health Score:** 42/100
**Iron Law:** 모든 finding은 Symptom → Source → Consequence → Remedy 순서로 기술됨.

---

## TL;DR

가장 시급한 것은 `LLMClient._last_usage` race condition 제거입니다 — 백그라운드 Sub-B/Curator가 동시에 LLM을 부르는 구조에서 cost 로그가 silently 오염되고 있을 가능성이 높습니다. 두 번째는 `handle_chat` 215줄 god function 분해 (메모리에 진행 중으로 적힌 작업의 마무리). 그 다음은 `world_state` KV·`deps` 글로벌·LLM 파라미터 중복으로 묶인 *의존성 누수* 그룹을 한 번에 정리하면 다음 기능 추가 시 변경 비용이 크게 줄어듭니다.

추세는 좋습니다 — 최근 커밋(`71f909f`, `ba02690`, `e1fc34c`)의 dead-code 제거·단일 소스화 흐름이 그대로 이어지면 됩니다.

---

## Findings 목록

### 🔴 Critical (severity: −15 each)
| # | Title | File |
|---|-------|------|
| 01 | LLMClient `_last_usage` race on concurrent calls | [01-critical-llm-usage-race.md](./01-critical-llm-usage-race.md) |
| 02 | `handle_chat` god function (215줄, 9개 책임) | [02-critical-handle-chat-god-function.md](./02-critical-handle-chat-god-function.md) |

### 🟡 Warning (severity: −5 each)
| # | Title | File |
|---|-------|------|
| 03 | `world_state` KV가 무관한 도메인의 dumping ground | [03-warning-world-state-leakage.md](./03-warning-world-state-leakage.md) |
| 04 | LLM 프로바이더별 파라미터 변환 4중 복제 | [04-warning-llm-param-duplication.md](./04-warning-llm-param-duplication.md) |
| 05 | `SystemStabilizer._extract_delta` 4-way 분기 + nested loop | [05-warning-extract-delta-complexity.md](./05-warning-extract-delta-complexity.md) |
| 06 | `deps` 글로벌 모듈이 ambient service locator | [06-warning-deps-service-locator.md](./06-warning-deps-service-locator.md) |
| 07 | `cache_marker`가 `MessageCompressor` 내부 sentinel을 import | [07-warning-cache-marker-sentinel-leak.md](./07-warning-cache-marker-sentinel-leak.md) |

### 🟢 Suggestion (severity: −1 each)
| # | Title | File |
|---|-------|------|
| 08 | Sub-B `narrative` dict 4필드가 anemic | [08-suggestion-narrative-dataclass.md](./08-suggestion-narrative-dataclass.md) |
| 09 | Curator dual-adapter (Letta + Direct) speculative complexity | [09-suggestion-curator-dual-adapter.md](./09-suggestion-curator-dual-adapter.md) |
| 10 | 압축/큐레이터 내 magic numbers | [10-suggestion-magic-numbers.md](./10-suggestion-magic-numbers.md) |

---

## Recommended Fix Order

```
1 (race) → 2 (god function) → 3 (world_state facade) → 4 (LLM param map) → 5–7 (의존성 정리) → 8–10 (cleanup)
```

**근거:**
- 1번은 silent data corruption — 다른 작업과 무관하게 즉시.
- 2번은 메모리에 "진행 중"으로 적힌 작업의 마무리 — 이후 모든 변경의 진입점이 가벼워짐.
- 3·4·6·7은 같은 *의존성 누수* 테마 — 한 번에 잡으면 일관성 유지가 쉬움.
- 5번은 최근 inject 버그(`8e99db8`)가 잡힌 위치 — 다음 inject 패턴 추가 전에 분리.
- 8–10은 risk가 낮아 사이드 트랙으로 처리 가능.

---

## ralph / superpowers 사용 가이드

- **각 finding 파일은 독립 task**로 picking 가능합니다. frontmatter에 `risk_code`, `severity`, `affected_files`, `acceptance` 필드를 둡니다.
- **순서 의존성:** README의 fix order를 따르되, 1·2·5·8·9·10은 다른 task와 충돌 없이 병렬 진행 가능. 3·4·6·7은 `deps`/`world_state`/`cache_marker`를 공유하므로 한 사람(또는 한 ralph 루프)이 순차 처리 권장.
- **검증:** 각 task 완료 후 `pytest`(`tests/`)·`ruff check saga/` 통과 + 해당 finding 재발 여부 확인.
- **참고 자료:** `_shared/decay-risks.md`(Brooks-Lint), `_shared/source-coverage.md`(인용 디시플린), 본 리뷰 모든 인용은 책 + 챕터/Principle 단위.

---

## Health Score 계산

```
Base:                100
2 × 🔴 Critical    : −30
5 × 🟡 Warning     : −25
3 × 🟢 Suggestion  :  −3
─────────────────────────
Total              :  42 / 100
```

목표: Critical 2건 + Warning 4건 처리 시 → 42 + 30 + 20 = **92/100**.

---

## 셸브 마무리 적용 결과 (2026-05-06 ~ 05-07)

운영 종료 + Plan A(Cache Keeper) 피벗 결정에 따라, 회귀 위험이 낮고 포트폴리오에서 정리된 모습으로 남길 가치가 있는 finding만 선택 적용.

| # | finding | 적용 여부 | commit |
|---|---------|----------|--------|
| 01 | LLM `_last_usage` race | ✅ 적용 | `dddbabd` |
| 02 | `handle_chat` god function | ⏭ 스킵 — 셸브 코드 회귀 risk > 정리 가치 |
| 03 | `world_state` 누수 | ⏭ 스킵 — 새 기능 무관, sunk cost |
| 04 | LLM 파라미터 중복 | ⏭ 스킵 — 동작 멀쩡, 새 파라미터 추가 없음 |
| 05 | `_extract_delta` 분리 | ✅ 적용 | `6f34171` |
| 06 | `deps` 글로벌 | ⏭ 스킵 — 큰 마이그레이션, sunk cost |
| 07 | `cache_marker` sentinel | ⏭ 스킵 — prefix 안 바꾸면 문제 없음 |
| 08 | `NarrativeSummary` dataclass | ✅ 적용 | `872fcca` |
| 09 | Curator dual adapter | ✅ 적용 (옵션 A — Letta 단일) | `0f1eaf3` |
| 10 | Magic numbers → config | ✅ 적용 | `c2b9649` |

스킵한 finding들은 이 폴더에 그대로 보관 — 향후 SAGA 코드를 부분 재사용하거나 같은 패턴이 다른 프로젝트에 나타날 때의 참고 자료로 가치가 있습니다.
