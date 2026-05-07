---
risk_code: R1  # Cognitive Overload
severity: suggestion
title: 압축/큐레이터 내 magic numbers
affected_files:
  - saga/message_compressor.py
  - saga/agents/curator.py
  - saga/adapters/curator_adapter.py
  - saga/config.py
order: 10
---

# 🟢 Suggestion — 압축/큐레이터 내 magic numbers

## Symptom

설명이나 config 없이 코드에 박힌 숫자들.

| 위치 | 값 | 의미 (코드 주석 기반 추정) |
|------|----|----------------------------|
| `message_compressor.py:83` | `0.50` | "aim for 50% of max context to give ~15-20 turns of headroom" |
| `message_compressor.py:252` | `5` | `min_remaining_turns` — 압축 후 최소로 남길 실제 대화 턴 수 |
| `message_compressor.py:245-246` | `+ 4` | 메시지당 토큰 오버헤드 estimate |
| `agents/curator.py:73` | `// 3` | Letta input token 추정 (3자 ≈ 1 token) |
| `agents/curator.py:78` | `500` | Letta output token 추정 |
| `adapters/curator_adapter.py:251` | `1500` | `MAX_SECTION` per-section char cap |
| `adapters/curator_adapter.py:258` | `5` | 최근 5턴만 포함 |
| `agents/curator.py:180` | `[:3]` | 큐레이션당 lore 생성 entity 최대 3개 |
| `agents/curator.py:206` | `[:8]` | episodes line cap |
| `services/cache_marker.py` | (없음 — `cache_ttl`은 config에 있음) | OK |

## Source

- McConnell — *Code Complete* Ch. 12 (Fundamental Data Types: Magic Numbers)

## Consequence

- 50% 타겟이나 5턴 보존이 어디서 왔는지 다음 사람이 모름
- 토큰 예산 튜닝 시 어디를 만져야 하는지 grep 필요
- 메모리 노트(`MEMORY.md`): "compress_threshold_ratio 조정 — config는 0.70, 유저 max context(55K~65K) 고려 필요" — 이 작업이 미착수 상태인데, `target_tokens = int(... * 0.50)`은 *지금도* 코드 안에 magic number로 박혀 있음

## Remedy

### 변경 1: `saga/config.py`에 새 필드 추가

```python
# saga/config.py
@dataclass
class PromptCachingConfig:
    enabled: bool = True
    compress_enabled: bool = True
    compress_threshold_ratio: float = 0.85
    compress_target_ratio: float = 0.50  # ← 신규
    min_compress_turns: int = 2
    min_keep_turns: int = 5  # ← 신규 (실제 대화 최소 보존)
    msg_token_overhead: int = 4  # ← 신규
    cache_ttl: str = "5m"
    stabilize_system: bool = True

@dataclass
class CuratorConfig:
    enabled: bool = False
    interval: int = 5
    compress_story_after_turns: int = 30
    letta_base_url: str = "..."
    letta_model: str = "..."
    letta_embedding: str = "..."
    letta_token_estimate_chars: int = 3  # ← 신규
    letta_output_estimate_tokens: int = 500  # ← 신규
    prompt_section_char_cap: int = 1500  # ← 신규
    prompt_recent_turns_cap: int = 5  # ← 신규
    lore_per_curation_cap: int = 3  # ← 신규
    episodes_per_lore_cap: int = 8  # ← 신규
    memory_block_schema: list = field(default_factory=lambda: [...])
```

### 변경 2: 코드에서 config 참조

```python
# message_compressor.py:83
target_tokens = int(
    self.config.token_budget.total_context_max
    * self.config.prompt_caching.compress_target_ratio
)

# message_compressor.py:245-246
user_tokens = count_tokens(user_msg.get("content", "")) + self.config.prompt_caching.msg_token_overhead
asst_tokens = count_tokens(asst_msg.get("content", "")) + self.config.prompt_caching.msg_token_overhead

# message_compressor.py:252
min_remaining_turns = self.config.prompt_caching.min_keep_turns
```

### 변경 3: `config.example.yaml` 업데이트

```yaml
prompt_caching:
  compress_threshold_ratio: 0.85
  compress_target_ratio: 0.50
  min_compress_turns: 2
  min_keep_turns: 5
  msg_token_overhead: 4
  cache_ttl: 5m

curator:
  interval: 5
  compress_story_after_turns: 30
  letta_token_estimate_chars: 3
  letta_output_estimate_tokens: 500
  prompt_section_char_cap: 1500
  prompt_recent_turns_cap: 5
  lore_per_curation_cap: 3
  episodes_per_lore_cap: 8
```

## Acceptance Criteria

- [ ] 위 7개 magic number가 모두 config로 이동
- [ ] `config.example.yaml`에 새 필드 추가 + 한 줄 주석으로 의미 설명
- [ ] 기존 동작 변화 없음 (기본값이 현재 박힌 값과 동일)
- [ ] `pytest tests/` 통과
- [ ] config 단위 테스트가 새 필드 검증

## 검증 방법

```bash
# magic number grep으로 회귀 확인
grep -nE "\* 0\.50|// 3|MAX_SECTION = 1500|min_remaining_turns = 5" saga/
# 결과: 0건이어야 함

pytest tests/ -v
```

## 관련 finding

- 독립 작업.
- 메모리 노트의 "compress_threshold_ratio 조정" 미착수 작업과 연계 — config로 끌어내면 0.70 vs 0.85 실험이 코드 변경 없이 가능.
- Suggestion 등급이지만 운영 튜닝 측면에서는 가성비 좋음.
