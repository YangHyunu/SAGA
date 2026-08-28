"""니들 생존 오라클 — 기억 시스템 공용 비교축. LLM 0콜.

건초더미 `ledger`의 팩트 `value` 문자열이 (a) 주입된 기억 블록 안에,
(b) 남아 있는 원문 창 안에 실제로 있는지를 부분문자열로만 판정한다.
채점 모델이 없으니 시스템 간 비교가 모델 편차로 오염되지 않는다.

**기저율 주의**: 짧은 값은 우연히 일치한다. 1차 지표는 `probed: true` 팩트만
쓴다(프로브와 1:1이라 의미가 명확). 전체 원장은 보조 지표로 따로 담는다.
"""

from __future__ import annotations

from typing import Any, Dict, List

# lucid.eligible과 같은 문턱 (haystack.MIN_VALUE_LEN)
MIN_VALUE_LEN = 2


def _eligible(ledger: List[Dict], turn: int, probed_only: bool) -> List[Dict]:
    """이 턴 시점에 '이미 일어난' 팩트만 — 미래 팩트는 물을 수 없다."""
    return [f for f in ledger
            if f["turn"] < turn
            and len(f.get("value", "")) >= MIN_VALUE_LEN
            and (f.get("probed") if probed_only else True)]


def _bucket(facts: List[Dict], memory: str, window: str) -> Dict[str, Any]:
    in_mem = {f["fid"] for f in facts if f["value"] in memory}
    in_win = {f["fid"] for f in facts if f["value"] in window}
    covered = in_mem | in_win
    return {"n_eligible": len(facts), "in_memory": len(in_mem),
            "in_window": len(in_win), "covered": len(covered),
            "lost": len(facts) - len(covered),
            "coverage": round(len(covered) / len(facts), 4) if facts else None,
            "memory_only": len(in_mem - in_win)}


def annotate(result: Dict[str, Any], ledger: List[Dict]) -> Dict[str, Any]:
    """result["turns"] 각 항목에 `needle` 필드를 붙여 제자리 갱신하고 돌려준다."""
    msgs = result.get("messages") or []
    for rec in result["turns"]:
        ksm = rec.get("kept_start_msg") or 0
        window = "".join(m["content"] for m in msgs[ksm:])
        memory = rec.get("memory_text") or ""
        rec["needle"] = {
            "probed": _bucket(_eligible(ledger, rec["turn"], True),
                              memory, window),
            "all": _bucket(_eligible(ledger, rec["turn"], False),
                           memory, window),
        }
    return result


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    """런 전체 요약 — 마지막 턴 커버리지와 전 구간 평균."""
    rows = [t["needle"]["probed"] for t in result["turns"] if t.get("needle")]
    scored = [r for r in rows if r["coverage"] is not None]
    if not scored:
        return {"turns_scored": 0}
    return {"turns_scored": len(scored),
            "coverage_mean": round(sum(r["coverage"] for r in scored)
                                   / len(scored), 4),
            "coverage_last": scored[-1]["coverage"],
            "memory_only_last": scored[-1]["memory_only"],
            "lost_last": scored[-1]["lost"]}


# ── 기존 관찰 산출물에 소급 적용 ─────────────────────────────────────────

def _main(argv=None) -> None:
    """python -m benchmarks.fixed_haystack.needle <inspect json...>

    이미 만들어 둔 인스펙터 JSON에 니들 축을 붙이고 HTML을 다시 렌더한다.
    LLM 0콜 — 원장과 저장된 memory_text/kept_start_msg만 읽는다.
    """
    import argparse
    import json
    import pathlib

    from benchmarks.fixed_haystack import haystack, inspect_hypa

    ap = argparse.ArgumentParser(description=_main.__doc__)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args(argv)

    cache: Dict[str, List[Dict]] = {}
    for path in a.paths:
        p = pathlib.Path(path)
        result = json.loads(p.read_text(encoding="utf-8"))
        hsid = result["meta"]["hsid"]
        if hsid not in cache:
            hs = haystack.load(f"dreaming_data/fixed_haystack/{hsid}.json")
            cache[hsid] = hs.ledger
        annotate(result, cache[hsid])
        result["meta"]["needle"] = summarize(result)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                     encoding="utf-8")
        p.with_suffix(".html").write_text(inspect_hypa.render_html(result),
                                          encoding="utf-8")
        s = result["meta"]["needle"]
        print(f"{p.name}: 커버리지 평균 {s.get('coverage_mean')} · "
              f"마지막 {s.get('coverage_last')} · "
              f"검색기여 {s.get('memory_only_last')}")


if __name__ == "__main__":
    _main()
