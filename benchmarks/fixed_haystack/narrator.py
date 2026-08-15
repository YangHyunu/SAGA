"""프로브 턴 나레이터 호출기 — 모델을 인자로 받는다 (구현 계약 §3).

`transport.call_upstream_once`를 재사용할 수 없다: 모델·응답 캡을 import 시점에
고정된 모듈 별칭(`from ...config import MODEL, MAX_TOKENS`)에서 읽어 런타임
오버라이드가 조용한 no-op이고, `benchmarks/eval/` 수정은 금지다
(2026-08-10 eval 리팩터 후기의 "import-시점 별칭은 스냅샷" 교훈과 같은 함정).
→ 재시도·응답 파싱 형태만 `transport.py:59-112`와 동일하게 두고 모델은 인자로 받는다.
"""

from __future__ import annotations

import time
from typing import Dict, List

import httpx

from benchmarks.eval.config import UPSTREAM

MAX_ATTEMPTS = 3
BACKOFF_SEC = 15      # attempt마다 배수 — 15s, 30s (transport와 동일)


def call(model: str, msgs: List[Dict], key: str, *, max_tokens: int,
         reasoning: str = "", timeout: float = 300) -> Dict:
    """반환 {reply, finish, prompt, cached, completion, cost, sec}.

    5xx·TransportError는 3회 재시도(15s·30s 백오프), 4xx는 즉시 전파 —
    한 번의 502가 런을 죽였던 night2-drm 실측 근거 그대로.
    """
    for attempt in range(MAX_ATTEMPTS):
        try:
            return call_once(model, msgs, key, max_tokens=max_tokens,
                             reasoning=reasoning, timeout=timeout)
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            if (isinstance(e, httpx.HTTPStatusError)
                    and e.response.status_code < 500):
                raise                          # 4xx는 우리 잘못 — 즉시 전파
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(BACKOFF_SEC * (attempt + 1))
    raise RuntimeError("unreachable")


def call_once(model: str, msgs: List[Dict], key: str, *, max_tokens: int,
              reasoning: str = "", timeout: float = 300) -> Dict:
    t0 = time.time()
    body: Dict = {"model": model, "max_tokens": max_tokens, "messages": msgs,
                  # 프로바이더 룰렛 방지 — transport.py:79-85 실측 근거 승계.
                  "provider": {"sort": "throughput"},
                  "usage": {"include": True}}
    if reasoning:
        # 숫자면 추론 예산(토큰 상한), 아니면 effort 문자열. 얼린 원문은
        # 예산 4000으로 생성됐다(계약 §0) — 프로브도 같은 값이어야 응답
        # 길이·비용이 재현된다.
        body["reasoning"] = ({"max_tokens": int(reasoning)}
                             if reasoning.isdigit() else {"effort": reasoning})
    r = httpx.post(UPSTREAM + "/chat/completions", timeout=timeout,
                   headers={"Authorization": f"Bearer {key}"}, json=body)
    r.raise_for_status()
    d = r.json()
    u = d.get("usage", {})
    det = u.get("prompt_tokens_details", {})
    cached = det.get("cached_tokens", 0) or u.get("prompt_cache_hit_tokens", 0)
    choice = d["choices"][0]
    # content는 None일 수 있다 (프로바이더 필터 등) — 빈 응답은 리롤 게이트가
    # language_drift로 걷어내도록 ""로 강제한다 (transport와 동일).
    return {"reply": choice["message"]["content"] or "",
            "finish": choice.get("finish_reason", ""),
            "prompt": u.get("prompt_tokens", 0), "cached": cached,
            "completion": u.get("completion_tokens", 0),
            "cost": u.get("cost", 0.0), "sec": round(time.time() - t0, 1)}
