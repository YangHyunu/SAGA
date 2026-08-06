"""corpus 캡처를 PairLedger에 재생해 원장 시간순 정합성을 검사한다.

usage: python3 -m benchmarks.capture.replay_ledger <corpus_dir> [<corpus_dir> ...]

corpus_dir 안의 req-*.json(진짜 RisuAI 와이어 캡처)을 이름순으로 SyncPath에
먹이고, 끝난 뒤 원장을 검사한다. 응답 텍스트는 캡처에 없으므로 다음 요청의
"마지막 user 직전 assistant"에서 합성한다 (마지막 요청은 기록 생략).

검사 항목 (corpus3→4 재생으로 실증된 붕괴의 재발 방지):
  1) 같은 유저 발화가 복수 턴 번호로 기록 (리롤 오판)
  2) 기록 턴 순서가 마지막 요청의 대화 순서와 역전
위반 시 exit 1. 코퍼스 데이터는 카드 저작물 포함 — 커밋 금지, 도구만 커밋.
"""
import json
import pathlib
import sys
import tempfile

from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath


def _load(dirs):
    reqs = []
    for d in dirs:
        for p in sorted(pathlib.Path(d).glob("req-*.json")):
            reqs.append((f"{pathlib.Path(d).name}/{p.stem}",
                         json.loads(p.read_text())["messages"]))
    return reqs


def _last_user_idx(msgs):
    return max((i for i, m in enumerate(msgs) if m["role"] == "user"),
               default=None)


def _reply_for(reqs, n):
    if n + 1 >= len(reqs):
        return None
    nxt = reqs[n + 1][1]
    li = _last_user_idx(nxt)
    for i in range(li - 1, -1, -1):
        if nxt[i]["role"] == "assistant":
            return nxt[i]["content"]
    return None


def main(argv):
    if not argv:
        print(__doc__)
        return 2
    reqs = _load(argv)
    storage = JsonDirStorage(pathlib.Path(tempfile.mkdtemp(prefix="replay-")))
    sp = SyncPath(storage, "cap")
    quarantined = 0
    for n, (name, msgs) in enumerate(reqs):
        _, v = sp.process(msgs)
        reply = _reply_for(reqs, n)
        if reply is not None:
            sp.record_response(v, msgs, reply)
        quarantined += bool(v.quarantine)
        print(f"{name}: {v.kind:12s} pos={v.position} offset={v.offset} "
              f"aligned={v.aligned} quarantine={v.quarantine}")

    rows = sorted(storage.scan("cap/raw"), key=lambda kv: kv[0])
    by_text = {}
    for _, r in rows:
        by_text.setdefault(r["user_text"], []).append(r["turn_number"])
    dups = {t[:24]: ns for t, ns in by_text.items() if len(ns) > 1}

    gt = [m["content"] for m in reqs[-1][1] if m["role"] == "user"]
    pos = {t: i for i, t in enumerate(gt)}
    seq = [r["user_text"] for _, r in
           sorted(rows, key=lambda kv: kv[1]["turn_number"])]
    known = [t for t in seq if t in pos]
    inversions = [(a[:16], b[:16]) for a, b in zip(known, known[1:])
                  if pos[a] > pos[b]]

    print(f"\nrecorded={len(rows)} quarantined={quarantined}")
    print("중복 턴 기록:", dups if dups else "없음")
    print("시간순 역전:", inversions if inversions else "없음")
    return 1 if (dups or inversions) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
