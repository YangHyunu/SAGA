"""dreaming/proxy.py — Phase 1 리버스 프록시 (스펙 §8).

RisuAI(OpenAI 호환 커스텀 URL) → 여기 → OpenAI 호환 업스트림(기본 OpenRouter).
동기 경로(SyncPath)로 주입·마킹하고, 응답 후 원장 기록·유휴 타이머·캐치업 드림.
fail-open: dreaming 오류는 절대 채팅을 막지 않는다 (스펙 §2.6).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
except ImportError:          # dotenv 없어도 동작 (환경변수 직접 export)
    load_dotenv = None

from dreaming.dreamer import Dreamer
from dreaming.idle import IdleWatcher
from dreaming.llm import LLMClient, OpenAICompatLLM
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath
from dreaming.upstream import OpenAIUpstream, to_wire

logger = logging.getLogger(__name__)

_SESSION_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class Settings(BaseModel):
    data_dir: str
    upstream_base_url: str
    upstream_api_key: str = ""
    idle_seconds: float = 300.0          # 캐시 TTL 5m와 동기 (스펙 §3.2)
    dream_base_url: str = ""
    dream_api_key: str = ""
    dream_model: str = ""                # 비면 Dreamer 비활성

    @classmethod
    def from_env(cls) -> "Settings":
        if load_dotenv is not None:
            # 기본 load_dotenv()는 호출 파일 기준 탐색이라 cwd를 명시한다.
            load_dotenv(Path.cwd() / ".env")   # 기존 환경변수는 안 덮음

        return cls(
            data_dir=os.environ.get("DREAMING_DATA_DIR", "./dreaming_data"),
            upstream_base_url=os.environ.get(
                "DREAMING_UPSTREAM_BASE", "https://openrouter.ai/api/v1"),
            upstream_api_key=os.environ.get("DREAMING_UPSTREAM_KEY", ""),
            idle_seconds=float(os.environ.get("DREAMING_IDLE_SECONDS", "300")),
            dream_base_url=os.environ.get("DREAMING_DREAM_BASE", ""),
            dream_api_key=os.environ.get("DREAMING_DREAM_KEY", ""),
            dream_model=os.environ.get("DREAMING_DREAM_MODEL", ""),
        )


def _session_of(request: Request, body: Dict) -> str:
    raw = request.headers.get("x-dreaming-session-id") or str(body.get("user") or "")
    s = _SESSION_SANITIZE_RE.sub("-", raw).strip("-")
    if not s or s in (".", ".."):
        return "default"
    return s


def _assistant_text(resp: Dict) -> str:
    try:
        content = resp["choices"][0]["message"]["content"]
        return content if isinstance(content, str) else ""
    except (KeyError, IndexError, TypeError):
        return ""


def create_app(settings: Settings, *,
               upstream=None,
               dream_llm: Optional[LLMClient] = None) -> FastAPI:
    storage = JsonDirStorage(Path(settings.data_dir))
    up = upstream or OpenAIUpstream(
        settings.upstream_base_url, settings.upstream_api_key)

    llm = dream_llm
    if llm is None and settings.dream_model:
        llm = OpenAICompatLLM(
            settings.dream_base_url or settings.upstream_base_url,
            settings.dream_api_key or settings.upstream_api_key,
            settings.dream_model)
    dreamer = Dreamer(storage, llm) if llm is not None else None

    async def _on_idle(session: str) -> None:
        if dreamer is not None:
            await dreamer.dream(session)

    watcher = IdleWatcher(settings.idle_seconds, _on_idle)
    syncpaths: Dict[str, SyncPath] = {}
    seen_sessions: set = set()

    app = FastAPI(title="Dreaming Proxy")
    app.state.storage = storage
    app.state.dreamer = dreamer
    app.state.watcher = watcher

    def _sync(session: str) -> SyncPath:
        if session not in syncpaths:
            syncpaths[session] = SyncPath(storage, session)
        return syncpaths[session]

    def _finish(session: str, verdict, original_messages: List[Dict],
                assistant_text: str) -> None:
        """응답 완료 후: 원장 기록 → 유휴 타이머. 전부 fail-open."""
        if verdict is not None and assistant_text:
            try:
                _sync(session).record_response(
                    verdict, original_messages, assistant_text)
            except Exception:
                logger.exception("[proxy] record failed: %s", session)
        watcher.touch(session)

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        original_messages = body.get("messages") or []
        session = _session_of(request, body)

        # 캐치업 드림 (스펙 §3.2): 이번 요청을 기록하기 *전에* backlog를 봐야
        # 신규 세션 오탐이 없다. 첫 요청은 즉시 통과, 꿈은 백그라운드.
        if (dreamer is not None and session not in seen_sessions
                and dreamer.has_backlog(session)):
            asyncio.create_task(dreamer.dream(session))
        seen_sessions.add(session)

        try:
            out, verdict = _sync(session).process(original_messages)
        except Exception:
            logger.exception("[proxy] sync path failed (fail-open): %s", session)
            out, verdict = original_messages, None

        payload = dict(body)
        payload["messages"] = to_wire(out)

        if body.get("stream"):
            async def relay():
                parts: List[str] = []
                buf = b""
                try:
                    async for chunk in up.stream(payload):
                        buf += chunk
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            s = line.decode("utf-8", "ignore").strip()
                            if not s.startswith("data:"):
                                continue
                            data = s[len("data:"):].strip()
                            if data == "[DONE]":
                                continue
                            try:
                                delta = (json.loads(data)["choices"][0]
                                         ["delta"].get("content"))
                            except (ValueError, KeyError, IndexError, TypeError):
                                delta = None
                            if delta:
                                parts.append(delta)
                        yield chunk
                finally:
                    _finish(session, verdict, original_messages, "".join(parts))
            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            resp = await up.complete(payload)
        except Exception:
            logger.exception("[proxy] upstream failed")
            return JSONResponse(status_code=502, content={
                "error": "upstream_error", "message": "upstream request failed"})
        _finish(session, verdict, original_messages, _assistant_text(resp))
        return JSONResponse(resp)

    return app
