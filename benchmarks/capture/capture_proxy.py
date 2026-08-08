"""골든 픽스처 캡처 — RisuAI가 실제로 보낸 요청 원문을 통째로 떨군다.

RisuAI → :8788 (여기) → :8787 (dreaming) → 업스트림

dreaming이 손대기 **전**의 바디를 저장한다. 시뮬 출력과 바이트 diff 하는 게 목적.
브라우저에서 직접 부르므로 CORS 프리플라이트를 열어둔다 (로컬 전용).

웹판 RisuAI는 호스트명이 localhost/127.0.0.1/0.0.0.0이면 요청을 거부한다
(globalApi.svelte.ts:598,740 — 데스크톱/노드서버가 아니면 무조건). 목록에 없는
IPv6 루프백 `[::1]`로 붙으면 통과하고, 브라우저도 ::1은 루프백으로 인정해
HTTPS 페이지에서 mixed content로 막지 않는다. 그래서 기본 바인드가 ::1이다.

usage: python3 capture_proxy.py [--out DIR] [--forward URL] [--port 8788]
                                [--host ::1]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

OUT = pathlib.Path("captures")
FORWARD = "http://127.0.0.1:8787"
_n = 0


@app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def passthrough(path: str, request: Request):
    global _n
    body = await request.body()

    if request.method == "OPTIONS":
        return Response(status_code=204)

    if body and "chat/completions" in path:
        _n += 1
        stamp = f"{_n:03d}"
        try:
            parsed = json.loads(body)
        except Exception:                       # noqa: BLE001
            parsed = None
        (OUT / f"req-{stamp}.json").write_bytes(body)
        if parsed is not None:
            msgs = parsed.get("messages") or []
            meta = {
                "n": _n, "path": path, "model": parsed.get("model"),
                "stream": parsed.get("stream"),
                "n_messages": len(msgs),
                "roles": [m.get("role") for m in msgs],
                "sizes": [len(m.get("content") or "")
                          if isinstance(m.get("content"), str) else -1
                          for m in msgs],
                "top_keys": sorted(parsed.keys()),
                "headers": {k: v for k, v in request.headers.items()
                            if k.lower() not in ("authorization", "cookie",
                                                 "x-api-key")},
            }
            (OUT / f"meta-{stamp}.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=1))
            print(f"[capture {stamp}] msgs={len(msgs)} "
                  f"roles={meta['roles']}", flush=True)

    fwd_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length",
                                        "accept-encoding", "connection")}
    url = f"{FORWARD}/{path}"
    t0 = time.time()
    async with httpx.AsyncClient(timeout=600) as c:
        r = await c.request(request.method, url, content=body,
                            headers=fwd_headers,
                            params=dict(request.query_params))
    print(f"  -> {r.status_code} {time.time() - t0:.1f}s", flush=True)

    ct = r.headers.get("content-type", "")
    if "text/event-stream" in ct:
        return StreamingResponse(iter([r.content]), media_type=ct)
    try:
        parsed_resp = r.json()
    except Exception:                           # noqa: BLE001
        return Response(r.content, status_code=r.status_code,
                        media_type=ct or "application/octet-stream")
    # 프로바이더가 보고한 캐시 실측(usage)을 남긴다 — 오프라인 바이트 계산의 검산용
    if body and "chat/completions" in path and isinstance(parsed_resp, dict):
        u = parsed_resp.get("usage")
        if u:
            (OUT / f"usage-{_n:03d}.json").write_text(
                json.dumps(u, ensure_ascii=False, indent=1))
    return JSONResponse(parsed_resp, status_code=r.status_code)


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures")
    ap.add_argument("--forward", default=FORWARD)
    ap.add_argument("--port", type=int, default=8788)
    ap.add_argument("--host", default="::1")
    a = ap.parse_args()
    OUT = pathlib.Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    FORWARD = a.forward.rstrip("/")
    print(f"capture → {OUT.resolve()}  forward → {FORWARD}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
