# 고정 건초더미(Fixed Haystack) 벤치 설계 — 2026-08-10

> 코드 작성 없음 — 설계 문서만. 브랜치 `dreaming/track-a-bench`, 커밋 안 함.
> 선행 문서: [HANDOFF](../../HANDOFF-2026-08-10.md), [eval v2 플랜](../plans/2026-08-06-dreaming-eval-v2.md), [회수축 복원 플랜(정직성 조항)](../plans/2026-08-10-retrieval-axis.md).
> `benchmarks/eval/`는 다른 세션이 진행 중이라 **읽기만** — 아래 설계는 전부 그 표면을 import로 재사용하고 신규 코드는 밖에 둔다.

## 0. 문제 재확인

현행 eval v2(`benchmarks/eval/run2.py`)는 변형(dreaming/vanilla/trim/hypa)마다 Lucid 디렉터가 진행 구간을 **라이브로 각자 생성**한다(`run2.py:126-169` `pick_beat`/`_filler_turn`, 나레이터 응답이 매 턴 `history`에 누적되고 다음 디렉터 발화가 그 응답을 참조 — `run2.py:142-147` `recent_dialogue`). 그 결과:

1. 건초더미(진행 구간의 서사)가 변형마다 달라 점수 차가 "기억력 차이"인지 "스토리가 우연히 달라진 차이"인지 분리 불가.
2. 프로브 채점 대상 응답이 그 변형 고유의 서사 흐름 속에서 나온 것이라, 잔여 원문 꼬리(LITM)에서 맞춘 건지 메모리 시스템이 회수해서 맞춘 건지 실험적으로 분리 안 됨.

[회수축 플랜의 정직성 조항](../plans/2026-08-10-retrieval-axis.md#L855-859)이 이미 명시: "기억 개선 주장은 벤치 재설계(고정 건초더미) 이후에만 가능." 이 문서는 그 재설계안이다.

## 1. 핵심 설계 결정 (5줄 요약은 문서 끝)

**아이디어**: 진행 구간(비-프로브 턴)의 (user, assistant) 텍스트 쌍을 전부 얼린다. 4변형은 이 동일한 텍스트를 서로 다른 방식으로만 "컨텍스트에 조립"한다. **오직 프로브 턴에서만** 그 변형의 조립 규칙으로 만든 요청을 실제 나레이터에 라이브로 보내 응답을 받고 채점한다. 프로브 응답은 채점 후 버리고, 다음 턴은 다시 얼린 원문으로 이어간다 — 그래야 4변형의 히스토리가 첫 프로브 이후에도 계속 동일하게 유지된다(안 지키면 §0의 문제가 그대로 재발).

이 구조가 성립하는 근거는 이미 프로덕션에 있다: `benchmarks/longmemeval/run_dreaming.py:70-90` `_ingest_and_dream`이 LongMemEval의 고정 haystack 세션을 `SyncPath.process`+`record_response`로 주입하면서 assistant 텍스트는 **데이터셋 원문을 그대로** 쓰고 나레이터를 호출하지 않는다. 같은 패턴을 dreaming 변형뿐 아니라 4변형 전체의 "얼린 구간 소비"에 적용한다.

무상태(trim/retrieval/vanilla) 변형은 `(얼린 히스토리, 프로브 위치)`의 순수 함수라 조립을 즉석에서 계산하면 되고, 유상태(dreaming/hypa) 변형은 얼린 히스토리 전체를 1회 순차 리플레이해 내부 저장소(사실/에피소드 또는 요약)를 만든 뒤 그 저장소를 반복(runs)마다 재사용한다 — LLM 비용이 드는 "인제스천"이 haystack당 1회로 상각된다(§6).

## 2. 히스토리 동결

### 2.1 소스 선택

기존 `dreaming_data/eval/v2-fix-drm-r0-run0.json`(100턴 dreaming 런, 이미 프로브 9개 내장 + `ledger` 365개 사실)을 haystack v0로 **재사용**할 것을 제안한다. 실물 구조 확인 완료:

```
turns: 100개  {turn, user, reply, finish, prompt, cached, completion, cost, sec, rerolls, flaw, flaw_history, sec_lucid/sec_director, sec_extract, ptype}
probes: 9개   {turn, ptype, fact, value, wrong, question, reply, oracle, judge, miss_cause}  — turn=19,29,39,49,59,69,79,89,99
ledger: 365개 {fid, kind, value, text, turn, probed}
```

재사용 근거:
- 프로브 지뢰(9곳)가 이미 [회수축 플랜](../plans/2026-08-10-retrieval-axis.md)의 실측 기준점(T39 게이트 등)과 같은 파일이라 신·구 비교가 가능하다.
- 원문 assistant 텍스트는 나레이터(`config.MODEL` = `deepseek/deepseek-v4-pro`)가 쓴 것이지 변형별 조립 알고리즘이 아니다 — 즉 "이 텍스트가 dreaming 변형이 만든 런에서 나왔다"는 사실이 vanilla/trim/retrieval에 불공정을 주지 않는다(4변형 비교는 오직 프로브 턴 재생성에서만 갈리므로, 진행 구간 원출처 변형이 무엇이었든 무관). 단, 이 가정 자체가 §8 미결 질문 3에 걸린다.

**정직하게 밝혀야 할 괴리**: [2026-08-06 플랜](../plans/2026-08-06-dreaming-eval-v2.md)은 "5유형 40체크"를 설계 목표로 잡았지만(`Task 6` `probe_schedule` 테스트), 실제 구현(`run2.py:80-93`)은 `probe_every`(기본 10, `config.py:33`)마다 1개씩만 배치한다 — 80~100턴 런엔 8~10개가 실측 한계다(실물 9개로 확인). "기존 40체크 재사용"은 존재하지 않는 것을 재사용할 수 없으므로 이 문서는 재사용 대상을 **9개 실물 프로브**로 바로잡고, 확장 여부는 §8로 넘긴다.

### 2.2 저장 포맷

`dreaming_data/fixed_haystack/{haystack_id}.json` (gitignore 승계 — `dreaming_data/` 전체가 이미 제외 대상):

```json
{
  "haystack_id": "fix-drm-r0",
  "preset_path": "...", "card_path": "dreaming_data/eval/card-soyeon-v2.json",
  "turns": [{"turn": 0, "user": "...", "assistant": "...", "is_probe": false}, ...],
  "probes": [{"turn": 19, "ptype": "relation", "fact": "...", "value": "...", "wrong": "", "question": "..."}, ...],
  "ledger": [{"fid": "...", "kind": "exact", "value": "...", "text": "...", "turn": 0, "probed": true}, ...]
}
```

`turns[i].assistant`는 원본 JSON의 `turns[i].reply`를 그대로 옮긴 것 — 프로브 턴(9개)도 `assistant` 필드는 채워 두되(원문 진행에 필요), 조립 단계(§3)에서 그 9곳만 라이브 재생성으로 **대체**하고 원문은 채점 참고용으로만 남긴다. `probes[].question`은 `turns[].user`와 동일 텍스트 — 프로브가 자연스러운 위장 발화(`lucid.make_probe`/`make_false_premise`, `benchmarks/eval/lucid.py:172-188`)로 이미 만들어진 것이라 원문 그대로 쓴다.

`ledger`를 통째로 얼려 두는 이유: 프로브가 "잔여 원문 꼬리"가 아니라 사전에 고정된 목록에서 나왔음을 보장하는 것이 §0 문제의 두 번째 갈래(정답 출처 모호성)에 대한 답이다.

## 3. 변형 4종 정의

| 변형 | 조립 규칙 | 상태 | 재사용 코드 |
|---|---|---|---|
| **(a) trim** (Risu 순정 슬라이딩) | `windowing.token_trim(frozen_history[:probe_turn], trim_budget)` → `run2.build_wire` 동등 조립 | 무상태, 순수함수 | `benchmarks/eval/windowing.py:19-43`, `run2.py:96-123` |
| **(b) HypaV3 재현** | `hypa.hypa_step`을 진행 구간 전체에 걸쳐 순차 리플레이(프로브 아닌 턴마다 1회 호출) — 요약 LLM 콜 발생, 내부 상태(`data["summaries"]`) 누적 | **유상태** — 1회 전체 리플레이 필요 | `benchmarks/eval/hypa.py`(`hypa_step`, `HypaSettings`), 상태 스냅샷은 `run2.py:462` `hypa_state` 패턴 승계 |
| **(c) 단순 turn-retrieval 베이스라인** | `windowing.token_trim`으로 자른 뒤, 잘린 구간에서 bigram top-k 발췌를 prepend | 무상태, 순수함수, LLM 0콜 | `benchmarks/eval/variants.py:32-55` `_bigrams`/`retrieve_turns` (현재 v1 하네스에 보존된 것을 그대로 가져다 씀) |
| **(d) Dreaming** | `SyncPath.process`/`record_response`로 진행 구간 전체를 1회 리플레이(assistant 텍스트는 얼린 원문 그대로 기록 — 나레이터 콜 없음), 주기적으로 `Dreamer.dream()` 트리거. 프로브 턴엔 `sp.process(probe question 포함 메시지)`가 **프로덕션과 동일한** 조립(청크+keyExcerpts+지식블록+마킹)을 그대로 반환 | **유상태** — 1회 리플레이, `JsonDirStorage` 디렉터리 자체가 캐시 | `dreaming/sync.py:110-190`(`SyncPath`), `dreaming/dreamer.py:333-360`(`Dreamer`), `dreaming/store.py`, `dreaming/storage.py` — `benchmarks/longmemeval/run_dreaming.py:70-90`이 이 패턴의 실물 선례 |
| **vanilla (대조)** | 무트림 — `frozen_history[:probe_turn]` 전체 | 무상태 | `windowing.py:49` `FULL_HISTORY` 승계, 예산 상한 의도적으로 없음(LITM 대조군) |

### 3.1 토큰 예산 상한 (동일 기준)

- `trim_budget = MAX_CONTEXT − preset_tokens − MAX_TOKENS − 50` (`run2.py:471-474` 공식 그대로, `config.MAX_CONTEXT=45000`, `config.MAX_TOKENS`는 env 기본 4000).
- (b) hypa의 memoryTokens는 `memoryTokensRatio × MAX_CONTEXT`(`hypa.py:53` 부근) — 같은 `MAX_CONTEXT`를 공유하면 이미 동일 풀 기준.
- (c) 리트리벌 베이스라인의 발췌 블록과 (d) dreaming의 지식블록은 **같은 상한**으로 맞출 것을 제안: `HOT_ZONE_CHAR_BUDGET=6000`자(`dreaming/assembly.py:24`)를 (c)에도 그대로 적용 — 현재 v1 `variants.py:20` `TRIM_WINDOW=8`(페어 수 기준)은 토큰 예산 개념이 없으므로, 이식 시 신규 shim이 필요하다(§7, §8-5).
- vanilla는 의도적으로 무상한 — 100턴 누적 시 `MAX_CONTEXT`를 실제로 넘을 수 있고, 이게 바로 LITM을 재는 대조군의 역할이다(§8-8에 리스크 명시).

### 3.2 공정성 핵심 규칙

프로브 턴에서 재생성된 assistant 텍스트는 **채점에만 쓰고 history에 반영하지 않는다**. 다음 턴부터는 다시 §2의 얼린 원문을 그대로 이어 붙인다. 이 규칙이 깨지면 첫 프로브 직후부터 4변형의 히스토리가 갈라져 "고정 건초더미"가 무의미해진다 — 구현 시 코드 주석과 리뷰 체크리스트에 반드시 명시해야 한다.

## 4. 프로브 주입

- 재질문 턴 = §2에서 얼린 9곳(T19/29/39/49/59/69/79/89/99) 그대로. "기존 40체크 재사용 여부"는 §2.1에서 밝혔듯 40체크가 실물로 존재한 적이 없어 재사용 대상이 아니다.
- 삽입 방식: `turns[i].user`(=`probes[].question`)를 프로브 발화로 그대로 채택 — 이미 정답 누출 게이트(`lucid.probe_leaks_value`, `lucid.py:138-144`)를 통과해 저장된 발화이므로 재검증 불필요(단, 원 런의 `totals.probe_leak_dropped=0`을 haystack 메타데이터에 같이 박제해 감사 가능하게 할 것 — 실측 확인: 이 파일의 `totals`에는 `probe_leak_*` 필드가 없어 원 런이 이 게이트 도입 이전 버전일 가능성이 있다. **haystack 채택 전 원 런의 게이트 통과 여부를 gates.py로 재검증 필요** — §8-10 추가).
- `distance_turns`/`in_window` 같은 참고 지표(`run2.py:326-353` `_record_probe`)는 변형마다 자기 알고리즘 기준으로 다시 계산해야 한다 — 트림 위치가 변형별로 다르므로 "이 프로브가 창 안/밖이었나"는 변형 종속값이다.

## 5. 채점

전부 기존 `benchmarks/eval/scoring.py`를 import로 재사용(신규 채점 로직 작성 안 함):

- **오라클**: `scoring.oracle_pass(reply, expected_value, wrong_value, char_name)` — 스탯바 헤더 제외(`_STATBAR` 정규식, `scoring.py:39`)와 캐릭터 이름 부분일치 가드(`char_name` 인자, `scoring.py:87-88`)가 이미 반영돼 있다. 실물 확인: 이 카드의 응답이 `[🖼|soyeon_winter_default|위지소연]` 형태의 선두 브래킷 태그로 시작하는데(원문 T39 응답 확인) `_STATBAR`(`^\s*\[[^\]]*\]\s*(?:-{2,}\s*)?`)가 이 패턴도 벗겨낸다 — 재구현 없이 그대로 맞는다.
- **judge**: `scoring.judge_pass(llm, ptype, fact_text, expected_value, question, reply, wrong_value)` — 유형별(recall/relation/update/false) 프롬프트가 이미 구현됨(`scoring.py:44-121`), reference-guided + 이진 판정 근거는 그대로 승계.
- **미스 원인 분해**: `scoring.decompose_miss(data_dir, session, fact)` — dreaming에만 적용(저장소가 존재하는 변형), 나머지는 `"-"` (`run2.py:322-325` 관례).
- **이중 오라클**: 오라클+judge 병행 유지, 파싱 실패는 `None`으로 분모에서 제외(`scoring.py:15-18` 설계 근거 그대로).
- **게이트**: `gates.evaluate(result)`(G1~G9)는 haystack 자체의 유효성(사실 나이 분포, 프로브 누출 등)을 haystack 채택 시점에 1회 검증하는 쪽과, 변형 실행마다 반복하는 쪽으로 나뉜다 — G1(dream_ran 등)은 dreaming 변형 실행 결과에 종속이라 매 실행 재평가가 맞고, G2(distance_turns 중앙값)는 haystack 메타데이터로 고정이라 1회 검증으로 충분하다. 이 구분을 §8-6으로 남긴다.

## 6. 응답 모델·반복·시드·비용

- **나레이터**: `config.MODEL`(현재 `deepseek/deepseek-v4-pro`) 그대로 승계 — haystack 원문도 같은 모델이 썼어야 문체 일관성이 성립한다(재사용안이면 이미 충족).
- **judge**: `config.JUDGE_MODEL`(`anthropic/claude-sonnet-4.5`) 그대로.
- **반복**: 기존 "3반복"은 Lucid 디렉터의 진행 구간 변동성을 평균 내려는 목적이었다(매 반복마다 서사가 달랐으므로). 고정 건초더미에서는 진행 구간이 결정론이라 반복의 의미가 "프로브 턴 나레이터 응답의 샘플링 변동성"으로 좁혀진다 — 3반복 유지를 제안하되, dreaming/hypa의 **인제스천 자체를 반복마다 다시 할지**는 §8-4의 철학적 결정에 달려 있다(1회 캐시 재사용이면 반복은 순수하게 나레이터 샘플링만 재는 것이 됨).
- **시드**: `transport.py` 확인 결과 시드 고정 로직 없음(OpenRouter 경유, 프로바이더 라우팅이 `provider: {sort: "throughput"}`로 매 요청 바뀔 수 있음 — `transport.py:79-85`) → 시드 고정 불가, 기존 관행대로 반복 횟수로 변동성을 흡수.
- **비용 추정** (실측 근거: `v2-fix-drm-r0-run0.json` totals — narrator `cost=0.3805`, `cost_director(lucid)=0.2405`, `cost_judge=0.1082`, 100턴/9프로브):
  - haystack 재사용(§2.1 안) → 진행 구간 재생성 비용 **$0**(이미 지불된 매몰비용, 신규라면 narrator+lucid ≈ $0.62 참고).
  - dreaming 인제스천(1회): `Dreamer.dream()` 호출 수 미지 — [회수축 플랜 실행 후기](../plans/2026-08-10-retrieval-axis.md)에 "실LLM 1콜 dream 사이클 ~6분" 기록은 있으나 **$비용 실측 없음** → 파일럿 1회 필수(§8-7).
  - hypa 인제스천(1회): 요약 호출 수 ≈ 창 밖 pair 수 비례, `cost_hypa` 필드(`run2.py:414-415`)로 추적 가능하나 정확한 금액은 실측 필요.
  - 프로브 턴 라이브 나레이터: 4변형 × 9프로브 × 3반복 = 108회, 턴당 ≈ $0.002~0.003(실측 turn0=$0.00174, turn40=$0.00308) → **약 $0.2~0.35**.
  - judge: 동 108회 × ≈$0.012/probe(실측 $0.1082/9) → **약 $1.3**.
  - **총 추정**: haystack 재사용 + 인제스천 실측 전 가정 시 **$2~5** — 기존 야간 본런(4변형×3반복 ≈ $12~20, [HANDOFF](../../HANDOFF-2026-08-10.md)§5-1)보다 저렴하나, 인제스천 비용이 빠진 잠정치다.

## 7. 구현 형태

`benchmarks/eval/` 수정 금지 — 신규 패키지를 밖에 둔다(`benchmarks/retrieval_lab.py`가 `eval/` 밖에 신규 파일을 둔 선례를 따름).

```
benchmarks/fixed_haystack/
  __init__.py
  haystack.py    # §2 동결 스키마 로더/저장 — v2 run JSON → FixedHaystack 변환, gates 1회 검증
  ingest.py      # (b)(d) 유상태 변형의 1회 리플레이 인제스천 + 상태 캐시 저장/재로드
  assemble.py    # 4+1변형별 probe-turn 컨텍스트 조립. 무상태(a/c/vanilla)는 순수함수,
                 # 유상태(b/d)는 ingest.py가 만든 캐시를 소비
  run.py         # CLI: 프로브 턴 라이브 나레이터 호출 + scoring 이중 채점 + 반복
  report.py      # 집계 — report2.aggregate/render 패턴 참고(스키마 다르면 신규 구현)
tests/
  test_fixed_haystack_haystack.py
  test_fixed_haystack_assemble.py   # 무상태 변형 3종은 LLM 0콜로 완전 유닛 테스트 가능
```

재사용은 전부 import: `benchmarks.eval.{config, windowing, matching, scoring, quality, fidelity, transport, preset2wire, gates}`, `benchmarks.eval.lucid`(DirFact/Ledger 스키마만), `benchmarks.eval.variants`(retrieve_turns), `dreaming.{sync, dreamer, store, storage, llm, retrieval, records, chunks}`.

데이터 경로: `dreaming_data/fixed_haystack/{haystack_id}/`(gitignore 승계). CLI 예시:

```bash
python3 -m benchmarks.fixed_haystack.ingest --haystack fix-drm-r0 --variant dreaming,hypa   # 1회
python3 -m benchmarks.fixed_haystack.run --haystack fix-drm-r0 --variants dreaming,vanilla,trim,retrieval --runs 3
```

## 8. 미결 질문 (임의로 정하지 않음)

1. **Haystack 소스**: 기존 `fix-drm-r0`(9프로브, 즉시 시작 가능) vs 신규 authored 대본(프로브 수·이벤트 밀도를 설계 목표에 맞춰 조정 가능하지만 라이브 생성 비용·시간 재발생). 유저 결정 필요.
2. **프로브 수 확장**: §2.1의 "40체크는 실물 없음" 발견을 어떻게 반영할지 — `probe_every`를 낮추거나 authored 확장 구간을 덧붙여 늘릴지, 9개로 시작해 반복 횟수로 통계력을 보완할지.
3. **원문 진행 구간의 출처 편향**: §2.1에서 "원출처 변형이 무엇이든 무관하다"고 가정했는데, 이 haystack은 원래 dreaming 나레이터가 dreaming 변형의 컨텍스트를 보고 쓴 글이다 — 나레이터가 (인지하지 못해도) 자기 응답 스타일에 어떤 식으로든 압축된 컨텍스트의 흔적을 남겼을 가능성을 완전히 배제할 수 있는지는 실험적으로 검증되지 않았다.
4. **인제스천 반복 여부**: dreaming/hypa의 LLM 기반 인제스천(비결정적, temperature>0)을 반복(runs)마다 다시 할지, 1회 캐시로 고정할지 — 전자는 "저장 결과의 강건성"까지 재지만 비용이 반복 횟수만큼 늘고, 후자는 "조립 알고리즘의 순수 성능"만 통제된 채 잰다. 설계 철학 결정 필요.
5. **리트리벌 베이스라인 예산 이식**: 현재 `variants.py:20` `TRIM_WINDOW=8`은 페어 수 기준이라 토큰/문자 예산 개념이 없다 — `windowing.token_trim`과 결합하는 조립 shim을 새로 짜야 한다(§3.1). `retrieve_turns` 자체(순수함수)는 그대로 재사용 가능.
6. **게이트 재평가 시점**: haystack 채택 시 1회만 검증할 게이트(G2 등, haystack 메타데이터로 고정)와 변형 실행마다 재평가할 게이트(G1 등, 실행 결과 종속)를 어떻게 나눌지.
7. **인제스천 비용 실측**: dreaming(Dreamer.dream 호출 수·비용)과 hypa(요약 호출 비용) 모두 $금액 실측 기록이 없다 — 파일럿 1회로 먼저 재고 총예산을 승인받아야 한다.
8. **vanilla 무상한 정책의 리스크**: 100턴 누적 시 `MAX_CONTEXT`를 실제로 넘어 나레이터 API가 자체 절단하거나 에러를 낼 수 있다 — 원 run2도 같은 리스크를 안고 있었으나(`FULL_HISTORY`) 고정 건초더미에서 매 프로브 턴마다 반복 노출되므로 사전 실측 필요.
9. **카드 파일 정합성**: [HANDOFF](../../HANDOFF-2026-08-10.md)의 카드 경로(`dreaming_data/eval/card-soyeon-v2.json`)가 haystack v0(`fix-drm-r0` 세션)를 만든 실제 카드와 파일명 버전이 일치하는지 미검증 — 불일치 시 조립된 wire가 원문 생성 당시와 달라진다.
10. **프로브 누출 게이트 소급 검증**: `fix-drm-r0-run0.json`의 `totals`에 `probe_leak_dropped` 필드가 없다(현재 `lucid.probe_leaks_value` 게이트 도입 이전 버전일 가능성) — haystack 채택 전 9개 프로브 발화를 `probe_leaks_value`로 재검증해 실제로 정답을 누출하지 않는지 확인 필요.

## 9. 데이터 위생 (오염 방지 — 2026-08-11 추가)

`dreaming_data/`에는 세션 16개가 혼재한다(2026-08-11 실사): 코드 버전
(keyExcerpts 이전/이후), 설정(`DREAMING_IDLE_SECONDS=10` 병리 vs 정상 배치),
카드·프리셋이 제각각인데 **어떤 조건에서 생성됐는지 기록이 전무**하다.
정상 조건 산물은 `pilot-rules-r0`(08-11, 새 규칙+IDLE=100)뿐이고,
`fix-drm-r0`·`night-drm-r0` 등은 1턴 꿈 병리 데이터다. 벤치가 이 디렉터리를
직접 읽으면 오염이 그대로 측정에 들어간다. 다음 4규칙은 구현 시 의무다.

1. **벤치는 세션 디렉터리를 직접 읽지 않는다.** 입력은 오직 §2.2 동결
   파일(`fixed_haystack/{id}.json`)뿐. 원본 세션은 동결 파일로 변환하는
   시점에 1회만 읽는다.
2. **동결 파일에 provenance 메타데이터 의무**: `source_session`, 생성 당시
   git commit, 생성 설정(`DREAMING_IDLE_SECONDS`·dream model·narrator model·
   preset/card 경로와 해시), gates 검증 결과, `frozen_at`. 메타데이터 없는
   haystack은 로더가 거부한다.
3. **벤치 산출물은 전용 네임스페이스**(`dreaming_data/fixed_haystack/{id}/runs/…`)
   에만 쓴다. 기존 세션 디렉터리에 쓰기 금지.
4. **병리 데이터의 용도 한정**: `fix-drm-r0` 재사용(§2.1)은 진행 구간의
   **원문 텍스트만** 얼리는 것이다. 그 세션의 `facts/`·`episodes/` 스토어는
   1턴 꿈 병리 산물이므로 **재사용 금지** — dreaming 변형의 인제스천은 반드시
   현행 코드로 새로 리플레이한다(§3-d와 일치).

구세션 정리는 삭제 대신 `dreaming_data/archive/`로 이동을 권고 (유저 승인 후).

---

## 핵심 설계 결정 5줄 요약

1. 진행 구간(비-프로브 턴)의 (user, assistant) 텍스트를 전부 얼리고, **프로브 턴에서만** 변형별 조립 규칙으로 라이브 나레이터를 호출해 채점한다 — 프로브 응답은 채점 후 버리고 다음 턴은 다시 원문으로 이어붙인다.
2. Haystack v0는 기존 `dreaming_data/eval/v2-fix-drm-r0-run0.json`(100턴, 프로브 9개, ledger 365개) 재사용을 제안 — "5유형 40체크"는 실물로 존재한 적 없어 재사용 대상이 아님을 확인.
3. 4+1변형 중 trim/retrieval/vanilla는 얼린 히스토리의 순수 함수(무상태), hypa/dreaming은 1회 순차 리플레이 인제스천이 필요(유상태, LLM 비용이 haystack당 1회로 상각).
4. 채점은 `benchmarks/eval/scoring.py`(오라클+judge 이중, 스탯바 헤더 제외·캐릭터 이름 가드 이미 반영)를 그대로 import 재사용 — 신규 채점 로직 작성 안 함.
5. 신규 코드는 `benchmarks/fixed_haystack/`(eval/ 밖)에 두고 전부 import 재사용, 인제스천 비용(dreaming/hypa) 실측 전에는 총 예산을 확정할 수 없음.

## 미결 질문 목록 (재게시)

§8의 10개 항목 — Haystack 소스, 프로브 수 확장, 원문 출처 편향, 인제스천 반복 여부, 리트리벌 예산 이식, 게이트 재평가 시점, 인제스천 비용 실측, vanilla 무상한 리스크, 카드 파일 정합성, 프로브 누출 게이트 소급 검증.
