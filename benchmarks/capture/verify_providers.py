"""멀티 프로바이더 검증 — 캡처 코퍼스를 재생하며 프로바이더 보고 캐시 숫자로
1안(lore_shift) 효과를 검산한다.

arm 2개 × 프로바이더 N개:
  raw     — RisuAI 원문 그대로 (1안 없음)
  shifted — dreaming.lore_shift 적용 (1안, 실제 프록시와 동일 코드 경로)

각 arm은 코퍼스 앞 N턴을 순서대로 쏘고 (max_tokens=1) usage를 읽는다.
Anthropic 계열은 cache_control 없으면 캐시가 아예 없으므로 마킹을 주입한다
(첫 system + 마지막 assistant — dreaming.marking과 동일 위치).

usage:
  python3 benchmarks/capture/verify_providers.py --captures dreaming_data/corpus2 \
      --card "/path/카드.charx" --user "Ren Amamiya (雨宮 蓮)" \
      --turns 8 --providers deepseek gemini gpt haiku
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from dreaming.lore_shift import load_keyed, shift_keyed  # noqa: E402

PROVIDERS = {
    # name: (base_url, model, env_key, anthropic_marking)
    "deepseek": ("https://api.deepseek.com", "deepseek-v4-flash",
                 "DEEPSEEK_API_KEY", False),
    "gemini": ("https://openrouter.ai/api/v1", "google/gemini-2.5-flash",
               "DREAMING_UPSTREAM_KEY", False),
    "gpt": ("https://openrouter.ai/api/v1", "openai/gpt-5.1",
            "DREAMING_UPSTREAM_KEY", False),
    "haiku": ("https://openrouter.ai/api/v1", "anthropic/claude-haiku-4.5",
              "DREAMING_UPSTREAM_KEY", True),
}


def env(name: str) -> str:
    v = os.environ.get(name)
    if v:
        return v
    for line in open(".env"):
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"{name} 없음")


def mark_anthropic(msgs):
    """dreaming.marking과 동일 위치에 cache_control (OpenRouter part 형식)."""
    out = [dict(m) for m in msgs]
    cc = {"type": "ephemeral"}

    last_system = None
    for i, m in enumerate(out):
        if m.get("role") != "system":
            break
        last_system = i
    last_assistant = None
    for i, m in enumerate(out):
        if m.get("role") == "assistant":
            last_assistant = i
    for i in (last_system, last_assistant):
        if i is not None and isinstance(out[i].get("content"), str):
            out[i]["content"] = [{"type": "text", "text": out[i]["content"],
                                  "cache_control": cc}]
    return out


def cache_of(u: dict) -> tuple:
    """(cached, prompt) — 프로바이더별 usage 필드 통일."""
    det = u.get("prompt_tokens_details") or {}
    cached = u.get("prompt_cache_hit_tokens", det.get("cached_tokens", 0)) or 0
    return cached, u.get("prompt_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="dreaming_data/corpus2")
    ap.add_argument("--card", required=True)
    ap.add_argument("--user", required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--providers", nargs="*", default=["deepseek", "gemini"])
    ap.add_argument("--gap", type=float, default=3.0)
    a = ap.parse_args()

    keyed = load_keyed(a.card, a.user)
    files = sorted(glob.glob(os.path.join(a.captures, "req-*.json")))[:a.turns]
    if not files:
        raise SystemExit(f"캡처 없음: {a.captures}")
    turns_raw = [json.load(open(f))["messages"] for f in files]
    turns_shift = [shift_keyed(m, keyed)[0] for m in turns_raw]
    print(f"{len(files)}턴 × arm 2 × 프로바이더 {len(a.providers)}개")

    results = {}
    for prov in a.providers:
        base, model, key_env, needs_mark = PROVIDERS[prov]
        hdr = {"Authorization": f"Bearer {env(key_env)}",
               "Content-Type": "application/json"}
        for arm, turns in (("raw", turns_raw), ("shifted", turns_shift)):
            tot_cached = tot_prompt = 0
            rows = []
            with httpx.Client(timeout=300) as c:
                for i, msgs in enumerate(turns, 1):
                    wire = mark_anthropic(msgs) if needs_mark else msgs
                    r = c.post(f"{base}/chat/completions", headers=hdr, json={
                        "model": model, "messages": wire,
                        "max_tokens": 1, "stream": False})
                    if r.status_code != 200:
                        rows.append(f"    T{i} ERR {r.status_code} {r.text[:120]}")
                        break
                    u = r.json().get("usage") or {}
                    cached, prompt = cache_of(u)
                    tot_cached += cached
                    tot_prompt += prompt
                    rows.append(f"    T{i} prompt={prompt:>7,} cached={cached:>7,}"
                                f" ({cached / prompt * 100 if prompt else 0:4.0f}%)")
                    time.sleep(a.gap)
            rate = tot_cached / tot_prompt * 100 if tot_prompt else 0
            results[(prov, arm)] = (tot_cached, tot_prompt, rate)
            print(f"\n== {prov} / {arm} — 히트 {rate:.1f}% "
                  f"({tot_cached:,}/{tot_prompt:,}) ==")
            print("\n".join(rows))

    print("\n===== 요약 =====")
    print(f"{'프로바이더':<10} {'raw':>8} {'shifted':>8} {'개선':>7}")
    for prov in a.providers:
        r0 = results.get((prov, "raw"), (0, 0, 0))[2]
        r1 = results.get((prov, "shifted"), (0, 0, 0))[2]
        print(f"{prov:<10} {r0:>7.1f}% {r1:>7.1f}% {r1 - r0:>+6.1f}p")


if __name__ == "__main__":
    main()
