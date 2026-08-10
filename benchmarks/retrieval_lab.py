"""benchmarks/retrieval_lab.py — 검색 오프라인 측정 ($0, LLM 0콜).

fix-drm-r0 스토어에 대해 **실제 render_knowledge 경로**로 프로브 정답의
IN/OUT과 필요 예산(budget_needed)을 잰다. FINDINGS §4 어휘 실측의 방향성
근사 — 정확한 순위 재현이 아니라 예산 결정의 근거 산출이 목적이다.

게이트 해석 주의 (플랜 리뷰 실측):
- T39만 비모호 (value 매칭 fact 정확히 1개 + 최신순에서 잘리던 케이스)
- T29는 value '렌'이 fact 48개에 걸려 순위 무의미 → 진단용
- T99는 현행 최신순으로도 이미 IN → 랭킹 무관
- T19·49·69·79·89는 추출실패 — 스토어에 정답 없음 (Track A 몫)

실행: DREAMING_DATA_DIR=/path/to/dreaming_data python3 -m benchmarks.retrieval_lab
      (메인 체크아웃에서는 env 생략 가능 — 기본값 ./dreaming_data)
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

from dreaming.retrieval import scene_query
from dreaming.storage import JsonDirStorage
from dreaming.store import MemoryStore
from dreaming.sync import render_knowledge

DATA_ROOT = Path(os.environ.get("DREAMING_DATA_DIR", "dreaming_data"))
SESSION = "fix-drm-r0"
RUN = DATA_ROOT / "eval" / "v2-fix-drm-r0-run0.json"
BUDGET_CHARS = 6000            # assembly.HOT_ZONE_CHAR_BUDGET와 동일


def _render(store, query: str, budget: int) -> str:
    """Task 3 배선 전(query 파라미터 없음)에도 baseline 측정이 되게 분기."""
    if "query" in inspect.signature(render_knowledge).parameters:
        return render_knowledge(store, query=query, budget=budget)
    return render_knowledge(store, budget=budget)


def budget_needed(store, query: str, answer: str) -> int | None:
    """정답이 지식 블록에 들어가는 최소 예산 (이분 탐색, 실제 렌더 경로)."""
    lo, hi = 200, 30000
    if answer not in _render(store, query, hi):
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if answer in _render(store, query, mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def main() -> int:
    if not RUN.is_file():
        print(f"skip: 데이터 없음 ({RUN}) — DREAMING_DATA_DIR로 메인 체크아웃의 "
              "dreaming_data를 지정하라")
        return 0
    store = MemoryStore(JsonDirStorage(DATA_ROOT), SESSION)
    raws = sorted((row for _, row in store._storage.scan(f"{SESSION}/raw")),
                  key=lambda r: r["turn_number"])
    pad = raws[0]["turn_number"] - 1           # _BASELINE_PAD 좌표계 유도
    raw_by_rel = {r["turn_number"] - pad: r for r in raws}
    facts = [f for f in store.list_facts() if f.pinned or f.status == "confirmed"]
    print(f"facts(주입 대상)={len(facts)}  pad={pad}")

    probes = json.load(open(RUN))["probes"]
    for p in probes:
        prev = raw_by_rel.get(p["turn"] - 1)   # 프로브 직전 턴 (상대 좌표)
        msgs = ([{"role": "assistant", "content": prev["assistant_text"]}]
                if prev else []) + [{"role": "user", "content": p["question"]}]
        query = scene_query(msgs)
        text = _render(store, query, BUDGET_CHARS)
        matches = sum(1 for f in facts if p["value"] in f.claim)
        need = budget_needed(store, query, p["value"])
        status = ("∅스토어에없음" if need is None
                  else ("IN " if p["value"] in text else "OUT")
                  + f" need={need}")
        print(f"T{p['turn']:>3} [{p['ptype']:>7}] {status} "
              f"(value매칭 {matches}개) {p['question'][:36]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
