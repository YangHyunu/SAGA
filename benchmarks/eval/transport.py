"""HTTP·LLM 팩토리 (EVAL2). run2·hypa 공용 하위 계층 — 순환 임포트 절단용.

_key/_mk_llm/_call_upstream 등 원래 run2.py에 있던 전송 계층을 여기로 옮겼다.
hypa.py가 run2를 거치지 않고 이 모듈을 직접 참조해 run2↔hypa 순환을 끊는다.
run2.py는 하위호환을 위해 옛 이름(_key 등)으로 재노출한다.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import httpx

from benchmarks.eval import config
from benchmarks.eval.config import (JUDGE_MODEL, LUCID_MODEL, MAX_TOKENS,
                                    MODEL, PROXY, ROOT, UPSTREAM)
from benchmarks.eval.lucid import LlmFn

# hypa 요약 모델 — Lucid 축과 분리 (감사 R2: LUCID_MODEL이 두 축을 동시에 움직임)
SUMMARY_MODEL = os.environ.get("HYPA_SUMMARY_MODEL", config.LUCID_MODEL)


def key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("DREAMING_UPSTREAM_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    raise SystemExit("no DREAMING_UPSTREAM_KEY in .env")


def mk_llm(model: str, temperature: float) -> LlmFn:
    client = httpx.Client(base_url=UPSTREAM, timeout=120,
                          headers={"Authorization": f"Bearer {key()}"})

    def call(system: str, user: str) -> str:
        r = client.post("/chat/completions", json={
            "model": model, "max_tokens": 400, "temperature": temperature,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        r.raise_for_status()
        data = r.json()
        u = data.get("usage") or {}
        call.cost += u.get("cost") or 0.0      # 부대비용도 잰다 — 나레이터만
        call.calls += 1                        # 재면 총비용을 과소보고한다
        return data["choices"][0]["message"]["content"] or ""
    call.cost, call.calls = 0.0, 0
    return call


def make_judge_llm() -> LlmFn:
    return mk_llm(JUDGE_MODEL, 0.0)


def make_lucid_llm() -> LlmFn:
    return mk_llm(LUCID_MODEL, 0.7)


def call_upstream(variant: str, session: str, key: str,
                  msgs: List[Dict]) -> Dict:
    """일시 오류(5xx·타임아웃)는 재시도 — 한 번의 502가 100턴 런을 죽였다
    (night2-drm 실측: 프록시 업스트림 ReadTimeout → 502 → 즉사)."""
    for attempt in range(3):
        try:
            return call_upstream_once(variant, session, key, msgs)
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if (isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500):
                raise                          # 4xx는 우리 잘못 — 즉시 전파
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("unreachable")


def call_upstream_once(variant: str, session: str, key: str,
                       msgs: List[Dict]) -> Dict:
    t0 = time.time()
    body: Dict = {"model": MODEL, "max_tokens": MAX_TOKENS, "messages": msgs,
                  # 기본 라우팅은 프로바이더 룰렛 — 실측: 같은 11.8K 와이어가
                  # SiliconFlow 무한 행(하트비트 공백만 옴, read timeout 안
                  # 터짐), GMICloud 173s(reasoning 12K 폭주), CoreWeave 9s.
                  # throughput 정렬로 빠른 쪽 고정. 프록시는 body를 그대로
                  # 전달하므로 dreaming 경로에도 적용된다.
                  "provider": {"sort": "throughput"}}
    if config.REASONING_EFFORT:            # 미설정이면 필드 자체를 안 보낸다
        # 숫자면 추론 예산(토큰 상한), 아니면 effort 문자열. flash는
        # effort=max에서 추론이 1K~19K로 확률 폭주(동일 요청·동일
        # 프로바이더 실측) — 예산 캡만이 시간·비용을 묶는다.
        rv = config.REASONING_EFFORT
        body["reasoning"] = ({"max_tokens": int(rv)} if rv.isdigit()
                             else {"effort": rv})
    if variant == "dreaming":
        r = httpx.post(PROXY + "/v1/chat/completions", timeout=300,
                       headers={"x-dreaming-session-id": session}, json=body)
    else:
        r = httpx.post(UPSTREAM + "/chat/completions", timeout=300,
                       headers={"Authorization": f"Bearer {key}"},
                       json={**body, "usage": {"include": True}})
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    cached = det.get("cached_tokens", 0) or u.get("prompt_cache_hit_tokens", 0)
    choice = d["choices"][0]
    # content는 None일 수 있다 (프로바이더 필터 등) — 빈 응답은 리롤 게이트가
    # language_drift로 걷어내도록 ""로 강제한다 (실측: pilot80b가 None에 죽음)
    return {"reply": choice["message"]["content"] or "",
            "finish": choice.get("finish_reason", ""),
            "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "completion": u.get("completion_tokens", 0),
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}
