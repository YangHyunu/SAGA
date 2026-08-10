# HANDOFF — Lucid (유저 시뮬레이터, 구 "디렉터") 개편 (2026-08-10)

> Track B 전용 새 세션 진입점. 먼저 [HANDOFF-2026-08-10.md](HANDOFF-2026-08-10.md)(코드 지도·실행법·제약) + [AGENTS.md](../AGENTS.md) + [CLAUDE.md](../CLAUDE.md)를 읽을 것.
> **이름**: Lucid(자각몽 — Dreaming 세계관에서 "꿈인 걸 알면서 안에서 걷는 존재" = 테스트임을 알고 통제하는 시뮬레이터). 유저 최종 확정 전이면 확인받고 시작.

## 1. 미션

eval v2의 유저 시뮬레이터(현 "디렉터")를 업계 표준 구조로 개편한다. 목표 포지션은 **"인간 유저 대체"가 아니라 "재현 가능한 프로브 전달자 + 실패 감지기"** — 프롬프트 전용 LLM의 인간 행동 재현율 실측 11.86%, 시뮬레이터 모델 교체만으로 점수 ±9%p 요동이 근거.

## 2. 입력 자료 (조사 보고서 — research/는 gitignore라 **원 워크트리에만 존재**)

원 워크트리: `/Users/yanghyeon-u/Desktop/RISU_ENE/.claude/worktrees/annyeong-3b2696/research/analysis/`
- `user-simulator-2026-08-10.md` (r1) — 표준 아키텍처 §1(목표 표현 6패턴·페르소나·상태·종료·플래너/생성기 분리), G1–G6 격차 진단
- `user-simulator-r2-2026-08-10.md` (r2) — 신규 논문 13편 표, 실전 사례(τ²-bench v1.0.1·LangSmith openevals·Cekura/Coval/Parloa·RP업계), 협조 편향 실측 4건, 심층 코드 4건

새 워크트리에서 작업 시 위 2파일을 새 워크트리 research/analysis/로 복사부터 할 것 (gitignore 유지).

## 3. 개편 항목 (우선순위순, 총 ~2.5–3일)

1. **프롬프트 3층 분리** (τ²-bench, ~1일): 행동규칙(.md) / 페르소나(렌) / 시나리오+프로브플랜을 별파일로. 결과 JSON에 각 층 해시 박제 (`prompt_set` 옆).
2. **정보 격리 + 점진 공개** (UserLM 샤딩, 1번에 포함): Lucid는 프로브 질문만 알고 정답을 모르게 — 날조·복창 구조적 차단. 턴당 신규 사실 1개만 공개.
3. **목표=상태** (Goal Alignment, ~0.5일): 정적 프롬프트 대신 턴마다 갱신되는 프로브플랜 체크리스트. 스케줄은 기존에 있음 — 달성 판정 → 상태 갱신만 추가.
4. **종료 다중화 + 저항 카운터** (NCUser·"Never Walk Away", ~0.5일): STOP / OOC보고 / 포기 구분. n회 회피 시 강제 전환.
5. **모델 고정 + 2모델 교차** (~0.5일): 결과 JSON에 `sim_model` 박제, 서브셋만 교차런. (DeepSeek V4 Flash가 Pro와 체감 무차이라는 유저 관찰 — 교차런 후보.)
6. **실패 감지 1급 지표화** (ChatChecker·RP업계): OOC/모순/반복 카운터를 judge 점수와 별도 리포팅. 기존 리롤 게이트(quality.py) 확장.
7. **협조 편향 명시** (비용 0): 리포트에 "점수 = 상한" 해석 명문화. 비협조 페르소나 축은 v2.

## 4. 현행 코드 (건드릴 표면)

- `benchmarks/eval/director.py` — LlmFn 타입만 있는 얇은 모듈 (개편 후 lucid.py로 리네임 후보)
- `benchmarks/eval/prompts.py` — 프롬프트 8종 + `active()`/`override_from()` — 3층 분리의 본체
- `benchmarks/eval/run2.py` — `_play_turn`/`pick_beat`가 디렉터 호출부. 패치 표면 규약은 run2.py:47-56 주석이 정본
- `benchmarks/eval/quality.py` — 리롤·루프 게이트 (6번 항목의 기반)
- 테스트: `tests/test_eval_v2.py` 663개 — 실캡처 바이트 테스트는 캡처 당시 토글 박제됨 (config.TOGGLES 바꿔도 안 깨짐)

## 5. 제약

- **본런(Track A) 결과와의 비교성**: 개편 브랜치는 야간 본런이 끝날 때까지 main/dreaming-spec에 머지 금지. 개편 후 스모크는 같은 세션명 규약(`v2-<session>-runN.json`)으로.
- 디렉터 프롬프트에 카드 지식 선취 금지 (NPC_NAME 당채련 유도만 예외 — config.py 주석 참조).
- API 키·dreaming_data·research/ 커밋 금지, NSFW 재현 금지, 결과 스핀 금지 (상세: HANDOFF-2026-08-10.md §6).
