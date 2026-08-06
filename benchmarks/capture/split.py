"""캡처된 msg[0] 안에서 constant 로어 vs keyed 로어의 실제 비중을 잰다.

1안(keyed만 프리픽스 밖) 비용을 계산하려면 매 턴 재전송할 keyed 분량(A)이 필요하다.

usage:
  python3 benchmarks/capture/split.py --captures dreaming_data/captures \
      --card "/path/THE AMOROUS REALM Ⅱ.charx" --user "Ren Amamiya"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile

CPT = 2.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="dreaming_data/captures")
    ap.add_argument("--card", required=True)
    ap.add_argument("--user", required=True)
    a = ap.parse_args()

    with zipfile.ZipFile(a.card) as z:
        card = json.loads(z.read("card.json"))["data"]

    def macro(t):
        return (t or "").replace("{{user}}", a.user).replace("{{char}}", card["name"])

    entries = [
        (e.get("name") or "?", bool(e.get("constant")), macro(e.get("content")))
        for e in card["character_book"]["entries"]
        if (e.get("content") or "").strip() and e.get("enabled") is not False
    ]

    def sys0(path):
        m = json.load(open(path))["messages"][0]["content"]
        return m if isinstance(m, str) else "".join(x.get("text", "") for x in m)

    print(
        f"{'턴':>3} {'const개':>7} {'const자':>9} {'keyed개':>7} "
        f"{'keyed자':>9} {'keyed토큰':>9} {'미매칭자':>9}"
    )
    rows = []
    for f in sorted(glob.glob(os.path.join(a.captures, "req-*.json"))):
        s = sys0(f)
        cn = ck = csz = ksz = 0
        for _name, is_const, body in entries:
            probe = body.strip()[:80]
            if len(probe) < 20 or probe not in s:
                continue
            if is_const:
                cn += 1
                csz += len(body)
            else:
                ck += 1
                ksz += len(body)
        rows.append((cn, csz, ck, ksz))
        print(
            f"{len(rows):>3} {cn:>7} {csz:>9,} {ck:>7} {ksz:>9,} "
            f"{ksz / CPT:>9,.0f} {len(s) - csz - ksz:>9,}"
        )

    avg_k = sum(r[3] for r in rows) / len(rows)
    print(f"\nkeyed 평균 {avg_k:,.0f}자 = {avg_k / CPT:,.0f}토큰/턴  ← 1안 재전송량(A)")
    print(f"const 평균 {sum(r[1] for r in rows) / len(rows):,.0f}자 (프리픽스 잔류)")


if __name__ == "__main__":
    main()
