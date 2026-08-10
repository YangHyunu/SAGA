"""런 유효성 게이트 (LUCID-CONTRACT.md §6).

결과 JSON만 보고 그 런의 회수율 수치가 해석 가능한지 판정한다. 구 JSON(이
게이트들이 신설되기 전에 저장된 결과)에는 아래에서 읽는 키 대부분이 없다 —
없는 키는 KeyError가 아니라 **해당 게이트의 실패**로 처리한다. 그래서 구
JSON에 이 함수를 돌리면 그 자체가 "그 런이 뭘 증명 못 했는가"의 목록이 된다.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from benchmarks.eval.windowing import FULL_HISTORY

Gate = Tuple[str, str]


def _check_g1(variant: str, totals: Dict) -> Gate | None:
    # dreaming만 해당. A7: 사이클 횟수 자체는 디스크에서 셀 수 없다
    # (dreaming/dreamer.py:341이 cursor를 단일 문서로 덮어써 이전 사이클
    # 기록이 남지 않는다) — 그래서 "몇 번 돌았나" 대신 "돌았는가 + 압축까지
    # 갔는가"로 재정의한다. C6: 압축은 plan이 None이 아닐 때만 파일이
    # 생기므로(dreamer.py:345-346) 파일 부재가 "이 런은 압축 층을 시험하지
    # 못했다"는 모호함 없는 신호다 — 압축 버그가 살아있는 한 dreaming 런은
    # 계속 이 게이트에서 떨어지는 게 정상이다.
    if variant != "dreaming":
        return None
    dream_ran = totals.get("dream_ran")
    episodes_written = totals.get("episodes_written")
    compression_planned = totals.get("compression_planned")
    ok = (bool(dream_ran)
          and isinstance(episodes_written, int) and episodes_written >= 1
          and bool(compression_planned))
    if ok:
        return None
    return ("G1", f"꿈 사이클 미확인: dream_ran={dream_ran!r} "
                  f"episodes_written={episodes_written!r} "
                  f"compression_planned={compression_planned!r}")


def _check_g2(variant: str, probes: List[Dict]) -> Gate | None:
    dts = [p["distance_turns"] for p in probes if "distance_turns" in p]
    if not dts:
        return ("G2", "probes에 distance_turns 없음")
    median = statistics.median(dts)
    if median < 15:
        return ("G2", f"distance_turns 중앙값 {median} < 15")
    if variant in FULL_HISTORY:
        # dreaming/vanilla는 트림 없이 풀 히스토리를 보내 in_window가 구조상
        # 항상 True다 — "창밖 비율 ≥50%" 조건은 이 두 변형에 적용 불가.
        return None
    n = len(probes)
    out_n = sum(1 for p in probes if p.get("in_window") is False)
    if n == 0 or out_n / n < 0.5:
        return ("G2", f"창밖(in_window=False) 프로브 비율 {out_n}/{n} < 50%")
    return None


def _check_g3(totals: Dict) -> Gate | None:
    leak = totals.get("probe_leak_dropped")
    if leak == 0:
        return None
    return ("G3", f"probe_leak_dropped={leak!r} (0이어야 함)")


def _check_g4(totals: Dict) -> Gate | None:
    scheduled = totals.get("probes_scheduled")
    delivered = totals.get("probes")
    if not isinstance(scheduled, int) or scheduled <= 0:
        return ("G4", f"probes_scheduled 없음/0 (probes={delivered!r})")
    ratio = (delivered or 0) / scheduled
    if ratio >= 0.8:
        return None
    return ("G4", f"probes/probes_scheduled={delivered}/{scheduled}"
                  f"={ratio:.0%} < 80%")


def _check_g5(totals: Dict) -> Gate | None:
    truncated = totals.get("truncated")
    if truncated == 0:
        return None
    return ("G5", f"truncated={truncated!r} (0이어야 함)")


def _check_g6(totals: Dict) -> Gate | None:
    flawed = totals.get("flawed")
    aborted = totals.get("aborted")
    if flawed == 0 and aborted == "":
        return None
    return ("G6", f"flawed={flawed!r} aborted={aborted!r}")


def _check_g7(totals: Dict) -> Gate | None:
    unparsed = totals.get("judge_unparsed")
    if unparsed == 0:
        return None
    return ("G7", f"judge_unparsed={unparsed!r} (0이어야 함)")


def _check_g8(totals: Dict, prompt_hashes: Dict) -> Gate | None:
    lucid_model = totals.get("lucid_model")
    if lucid_model and prompt_hashes:
        return None
    return ("G8", f"lucid_model={lucid_model!r} "
                  f"prompt_hashes={'있음' if prompt_hashes else '없음'}")


def evaluate(result: Dict) -> Dict[str, List[Gate]]:
    """게이트 8개(G1~G8)를 판정하고 G9(judge-사람 일치율)는 항상 경고로 낸다.

    반환: {"failed": [(게이트id, 사유), ...], "warnings": [...]}.
    """
    variant = result.get("variant")
    totals = result.get("totals") or {}
    probes = result.get("probes") or []
    prompt_hashes = result.get("prompt_hashes") or {}

    checks = (_check_g1(variant, totals),
             _check_g2(variant, probes),
             _check_g3(totals),
             _check_g4(totals),
             _check_g5(totals),
             _check_g6(totals),
             _check_g7(totals),
             _check_g8(totals, prompt_hashes))
    failed = [g for g in checks if g is not None]
    # G9는 자동 판정 불가 — judge-사람 일치율 수동 감사가 아직 없다
    # (LUCID-CONTRACT.md §6 G9). 감사가 돌기 전까지 judge 기반 지표
    # (judge_pass 등) 전부 유보 상태임을 매 런 결과에 표기한다.
    warnings: List[Gate] = [
        ("G9", "judge-사람 일치율 미검증 — 수동 감사(LUCID-CONTRACT.md §6 G9) "
               "실시 전까지 judge 기반 지표는 유보 상태")]
    return {"failed": failed, "warnings": warnings}
