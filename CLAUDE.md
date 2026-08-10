# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

> **Project context**: [AGENTS.md](AGENTS.md) — RisuAI 소스 지도(external/ 심링크), research/ 규약, arca.live 접근 플레이북. 에이전트는 작업 전 AGENTS.md를 먼저 볼 것.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 5. Plan Postmortems (플랜 실행 후기)

**플랜도 코드처럼 결함이 있다. 실행 중 발견하면 이탈을 공개하고 교정하되, 완료 후 플랜 문서에 후기로 기록한다.**

- 구현 중 플랜의 오류(잘못된 스케치, 불가능한 수치 목표, 틀린 인터페이스 목록)를 발견하면: 조용히 따르지 말고, 교정 + 이탈 사유를 리포트에 명시.
- 플랜 실행 완료 후, 플랜 문서 하단에 "실행 후기" 섹션으로 발견된 플랜 결함을 기록 — 다음 플랜 작성 시 재발 방지 자료.

사례: [2026-08-10 eval 리팩터 후기](docs/superpowers/plans/2026-08-10-eval-refactor.md) — 결함 4건 전부 구현자가 발견·교정 (별칭 import-시점 바인딩 → 오버라이드 무효, `transport.key` 패치 no-op, "<400줄" 산술 불가능, 재수출·의존 목록 오기). 핵심 교훈: **패치·오버라이드가 도달해야 하는 호출부는 `module.NAME` 점 접근** (import-시점 별칭은 스냅샷이라 무효).

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
