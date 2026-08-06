"""계측기 영점 조정 — 프로바이더가 프리픽스 캐시를 하는지 + 보고하는지 확인한다.

캡처한 요청 하나를 **같은 바이트로 두 번** 쏘고 usage를 읽는다.
2번째의 cached_tokens가 0이 아니면 계측 가능하다는 뜻.

dreaming을 우회해 OpenRouter로 직접 간다 (dreaming_data 오염 방지).

usage:
  python3 benchmarks/capture/instrument.py --req dreaming_data/captures/req-001.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx

BASE = os.environ.get("DREAMING_UPSTREAM_BASE", "https://openrouter.ai/api/v1")

MODELS = [
    "deepseek/deepseek-v4-pro",
    "google/gemini-2.5-flash",
    "openai/gpt-5.1",
    "anthropic/claude-haiku-4.5",
]


def load_key() -> str:
    key = os.environ.get("DREAMING_UPSTREAM_KEY", "")
    if key:
        return key
    for line in open(".env"):
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("DREAMING_UPSTREAM_KEY 없음")


def usage_of(r: httpx.Response) -> dict:
    if r.status_code != 200:
        return {"error": f"{r.status_code} {r.text[:300]}"}
    d = r.json()
    u = d.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    return {
        "prompt": u.get("prompt_tokens"),
        "cached": det.get("cached_tokens", u.get("cached_tokens")),
        "write": u.get("cache_creation_input_tokens"),
        "raw_usage_keys": sorted(u.keys()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--req", default="dreaming_data/captures/req-001.json")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--gap", type=float, default=4.0, help="두 콜 사이 대기(초)")
    a = ap.parse_args()

    body = json.load(open(a.req))
    msgs = body["messages"]
    size = sum(len(m["content"]) if isinstance(m["content"], str) else 0 for m in msgs)
    print(f"{a.req} — {len(msgs)}메시지 / {size:,}자\n")

    key = load_key()
    hdr = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    for model in a.models:
        payload = {"model": model, "messages": msgs, "max_tokens": 16, "stream": False}
        print(f"=== {model} ===")
        with httpx.Client(timeout=300) as c:
            for i in (1, 2):
                t0 = time.time()
                r = c.post(f"{BASE}/chat/completions", headers=hdr, json=payload)
                u = usage_of(r)
                print(f"  콜{i} ({time.time() - t0:>5.1f}s) {u}")
                if "error" in u:
                    break
                if i == 1:
                    time.sleep(a.gap)
        print()

    # cache_control 관용 프로브 — 작은 페이로드로 비-Anthropic이 400을 내는지만 본다
    print("=== cache_control 관용 프로브 (소형 페이로드) ===")
    probe = [{"role": "user",
              "content": [{"type": "text", "text": "say ok",
                           "cache_control": {"type": "ephemeral"}}]}]
    with httpx.Client(timeout=120) as c:
        for model in a.models:
            r = c.post(f"{BASE}/chat/completions", headers=hdr,
                       json={"model": model, "messages": probe, "max_tokens": 8})
            verdict = "OK" if r.status_code == 200 else f"{r.status_code} {r.text[:160]}"
            print(f"  {model:<32} {verdict}")


if __name__ == "__main__":
    main()
