"""dreaming/tracing.py — LangSmith 관측 (선택적, fail-open).

LANGSMITH_TRACING=true + langsmith 설치 시에만 활성. 드리밍 핵심 동작을
LangSmith 런으로 남긴다 — 턴 동기 경로(dreaming.turn: 쿼리·주입 블록·격리)와
꿈(dreaming.dream: 추출 리포트). rlog가 id만 남기는 것과 달리 여기엔 블록
전문이 실린다 — 유저 본인 계정으로의 관측이 목적이라서다.

전송은 데몬 스레드로 — 동기 경로 한복판에서 HTTP로 채팅을 지연시키지
않는다. 실패는 로그만 남기고 삼킨다 (스펙 §2.6 fail-open).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_client = None
_enabled: Optional[bool] = None


def _get():
    global _client, _enabled
    if _enabled is None:
        _enabled = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
        if _enabled:
            try:
                from langsmith import Client
                _client = Client()
            except Exception:
                logger.warning("[tracing] langsmith 비활성 (미설치/설정 오류)")
                _enabled = False
    return _client if _enabled else None


def log_run(name: str, inputs: Dict, outputs: Dict,
            metadata: Optional[Dict] = None) -> None:
    client = _get()
    if client is None:
        return

    def _post() -> None:
        try:
            client.create_run(
                name=name, run_type="chain", inputs=inputs, outputs=outputs,
                extra={"metadata": metadata or {}},
                # LANGSMITH_PROJECT(=SAGA 등 상위 프로젝트)와 섞이지 않게
                # 드리밍 전용 프로젝트를 쓴다 — 첫 전송 시 자동 생성된다.
                project_name=os.environ.get("DREAMING_LANGSMITH_PROJECT",
                                            "dreaming"),
            )
        except Exception:
            logger.exception("[tracing] log_run failed: %s", name)

    threading.Thread(target=_post, daemon=True).start()
