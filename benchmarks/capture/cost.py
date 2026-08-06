"""캡처한 RisuAI 요청으로 프리픽스 캐시 비용을 잰다 — 현재 vs 1안.

1안 = keyed(non-constant) 로어북 엔트리를 프리픽스에서 빼고 매 턴 재전송.

usage:
  python3 benchmarks/capture/cost.py --captures dreaming_data/captures \
      --card "/path/THE AMOROUS REALM Ⅱ.charx" --user "Ren Amamiya"

주의 — keyed 본문만 `replace("")` 하면 구분자 개행이 남아서 프리픽스가 계속
흔들린다. `.strip()` 매칭 + 빈 줄 정규화까지 해야 msg[0]이 전 턴 동일해진다.
그래서 실행 결과의 `msg[0] 길이 … 전 턴 동일` 줄을 **반드시 먼저 확인**할 것.
변동이 있으면 숫자는 못 믿는다.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import zipfile

CPT = 2.5  # 자 → 토큰 근사 (한글 혼합 기준)

# (캐시읽기, 미스입력, 캐시쓰기) $/1M — OpenRouter 실측 2026-08-05
PRICES = {
    "deepseek-v4-pro": (0.004, 0.435, 0.0),
    "haiku-4.5": (0.100, 1.000, 1.25),
}


def load_keyed(card_path: str, user: str) -> list[str]:
    with zipfile.ZipFile(card_path) as z:
        card = json.loads(z.read("card.json"))["data"]
    name = card["name"]
    return [
        (e.get("content") or "").replace("{{user}}", user).replace("{{char}}", name)
        for e in card["character_book"]["entries"]
        if not e.get("constant")
        and (e.get("content") or "").strip()
        and e.get("enabled") is not False
    ]


def wire(path: str) -> list[tuple[str, str]]:
    d = json.load(open(path))
    return [
        (
            m["role"],
            m["content"]
            if isinstance(m["content"], str)
            else "".join(x.get("text", "") for x in m["content"]),
        )
        for m in d["messages"]
    ]


def lcp(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def pull(s: str, keyed: list[str]) -> tuple[str, int]:
    """keyed 본문 + 앞뒤 공백을 통째로 들어내고 남은 빈 줄을 정규화."""
    n = 0
    for body in keyed:
        b = body.strip()
        if len(b) < 40 or b not in s:
            continue
        s = s.replace(b, "")
        n += len(b)
    return re.sub(r"\n{3,}", "\n\n", s), n


def evaluate(turns, keyed, fix: bool):
    prev, hit, miss, resent = None, 0, 0, 0
    sizes = []
    for t in turns:
        t = list(t)
        if fix:
            s, n = pull(t[0][1], keyed)
            t[0] = (t[0][0], s)
            resent += n
            sizes.append(len(s))
        f = " ".join(r + c for r, c in t)
        k = lcp(prev, f) if prev else 0
        hit += k
        miss += len(f) - k
        prev = f
    return hit / CPT, miss / CPT, resent / CPT, sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="dreaming_data/captures")
    ap.add_argument("--card", required=True)
    ap.add_argument("--user", required=True, help="RisuAI 페르소나 이름 (그대로)")
    a = ap.parse_args()

    keyed = load_keyed(a.card, a.user)
    files = sorted(glob.glob(os.path.join(a.captures, "req-*.json")))
    if not files:
        raise SystemExit(f"캡처 없음: {a.captures}")
    turns = [wire(f) for f in files]
    print(f"{len(turns)}턴 / keyed {len(keyed)}개 / user={a.user!r}")

    for label, fix in (
        ("현재 — keyed가 프리픽스 안", False),
        ("1안 — keyed만 프리픽스 밖", True),
    ):
        hit, miss, resent, sizes = evaluate(turns, keyed, fix)
        tot = hit + miss + resent
        print(f"\n=== {label} ===")
        if sizes:
            ok = "전 턴 동일 ✅" if len(set(sizes)) == 1 else "변동 있음 ❌ 숫자 신뢰 불가"
            print(f"  msg[0] 길이: {min(sizes):,} ~ {max(sizes):,} ({ok})")
        print(
            f"  캐시읽기 {hit:>9,.0f}t | 미스 {miss:>9,.0f}t | "
            f"keyed재전송 {resent:>8,.0f}t | 히트율 {hit / tot * 100:>4.1f}%"
        )
        for m, (rd, ms, wr) in PRICES.items():
            c = hit * rd / 1e6 + miss * (wr or ms) / 1e6 + resent * ms / 1e6
            print(f"    {m:<18} ${c:>6.3f}")


if __name__ == "__main__":
    main()
