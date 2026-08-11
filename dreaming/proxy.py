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

from dreaming import pressure
from dreaming.dreamer import Dreamer
from dreaming.idle import IdleWatcher
from dreaming.llm import LLMClient, OpenAICompatLLM
from dreaming.lore_shift import load_keyed
from dreaming.storage import JsonDirStorage
from dreaming.sync import SyncPath
from dreaming.upstream import OpenAIUpstream, to_wire

logger = logging.getLogger(__name__)

_SESSION_SANITIZE_RE = re.compile(r"[^A-Za-z0-9._-]+")

# 리포 루트 (dreaming/의 부모) — .env·데이터 기본 경로의 앵커.
# cwd 기준이면 실행 위치마다 달라져서 못 쓴다.
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseModel):
    data_dir: str
    upstream_base_url: str
    upstream_api_key: str = ""
    idle_seconds: float = 300.0          # 캐시 TTL 5m와 동기 (스펙 §3.2)
    pressure_chars: int = pressure.DEFAULT_THRESHOLD_CHARS   # §6.3, 0=비활성
    dream_base_url: str = ""
    dream_api_key: str = ""
    dream_model: str = ""                # 비면 Dreamer 비활성
    card_path: str = ""                  # .charx 경로 — 있으면 1안 활성 (스펙 §5)
    card_user: str = ""                  # RisuAI 페르소나 이름 ({{user}} 치환용)

    @classmethod
    def from_env(cls, root: Optional[Path] = None) -> "Settings":
        root = root or _ROOT
        if load_dotenv is not None:
            load_dotenv(root / ".env")   # 기존 환경변수는 안 덮음
        return cls(
            data_dir=os.environ.get("DREAMING_DATA_DIR",
                                    str(root / "dreaming_data")),
            upstream_base_url=os.environ.get(
                "DREAMING_UPSTREAM_BASE", "https://openrouter.ai/api/v1"),
            upstream_api_key=os.environ.get("DREAMING_UPSTREAM_KEY", ""),
            idle_seconds=float(os.environ.get("DREAMING_IDLE_SECONDS", "300")),
            pressure_chars=int(os.environ.get(
                "DREAMING_PRESSURE_CHARS",
                str(pressure.DEFAULT_THRESHOLD_CHARS))),
            dream_base_url=os.environ.get("DREAMING_DREAM_BASE", ""),
            dream_api_key=os.environ.get("DREAMING_DREAM_KEY", ""),
            dream_model=os.environ.get("DREAMING_DREAM_MODEL", ""),
            card_path=os.environ.get("DREAMING_CARD_PATH", ""),
            card_user=os.environ.get("DREAMING_CARD_USER", ""),
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

    # cache_control 마킹은 Anthropic 규약 — 직접 Anthropic이거나 OpenRouter
    # (프로바이더별 번역/제거를 대신 해줌)일 때만 보낸다. DeepSeek 본가처럼
    # 자동 프리픽스 캐싱인 곳엔 미지 필드라 보내지 않는다 (스펙 §6.3).
    mark_cache_enabled = any(
        h in settings.upstream_base_url for h in ("openrouter", "anthropic"))
    if not mark_cache_enabled:
        logger.info("[proxy] cache_control 마킹 비활성 — 업스트림 %s",
                    settings.upstream_base_url)

    keyed_lore: List[str] = []
    if settings.card_path and settings.card_user:
        try:
            keyed_lore = load_keyed(settings.card_path, settings.card_user)
            logger.info("[proxy] 1안 활성 — keyed 로어 %d개 (%s)",
                        len(keyed_lore), settings.card_path)
        except Exception:
            logger.exception("[proxy] card load failed (1안 비활성, fail-open)")

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
            syncpaths[session] = SyncPath(storage, session,
                                          keyed_lore=keyed_lore,
                                          mark_cache_enabled=mark_cache_enabled)
        return syncpaths[session]

    def _finish(session: str, verdict, original_messages: List[Dict],
                assistant_text: str, prompt_size: int = 0) -> None:
        """응답 완료 후: 원장 기록 → 유휴 타이머 → 임계 강제 꿈. 전부 fail-open."""
        if verdict is not None and assistant_text:
            try:
                _sync(session).record_response(
                    verdict, original_messages, assistant_text)
            except Exception:
                logger.exception("[proxy] record failed: %s", session)
        watcher.touch(session)
        # §6.3 폴백: 연속 채팅으로 유휴가 영원히 안 와도 임계 초과 시 강제 꿈.
        # 기록 후에 재야 방금 턴이 backlog에 포함된다. 다음 요청이 새 플랜을
        # 받아 캐시가 한 번 깨진다 — BOUNDARY_STEP 계단 + MIN_BACKLOG_TURNS가
        # 그 빈도를 묶어 턴당 상각을 지킨다 (dreaming/pressure.py).
        if dreamer is None or not (0 < settings.pressure_chars <= prompt_size):
            return                       # 크기 선-게이트 — backlog 스캔 절약
        try:
            backlog = len(dreamer.snapshot(session))
            if pressure.should_force(prompt_size, settings.pressure_chars,
                                     backlog):
                asyncio.create_task(dreamer.dream(session))
        except Exception:
            logger.exception("[proxy] forced dream failed: %s", session)

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

        # §6.3 압박 신호: 압축·주입이 끝난 실제 발신 프롬프트 크기 —
        # 압축이 못 줄인 잔여분이 곧 압박이다 (원문 누적량 아님).
        prompt_size = pressure.prompt_chars(out)

        payload = dict(body)
        payload["messages"] = to_wire(out)

        # 파라미터 번역: RisuAI의 thinking_tokens는 자체 발명 이름이라
        # DeepSeek 본가가 무시한다 → v4는 기본 thinking이라 0을 보내도
        # CoT가 content를 갉아먹는다 (실측: content가 빈 응답 재현).
        # 본가 공식 스위치로 번역한다.
        if "deepseek" in settings.upstream_base_url:
            tt = payload.pop("thinking_tokens", None)
            if tt is not None:
                payload["thinking"] = {
                    "type": "disabled" if tt == 0 else "enabled"}

        # 키 pass-through: RisuAI 키 필드의 진짜 키를 우선한다 (키 무보관 구조).
        # 더미 키(짧은 토큰)는 무시하고 .env 폴백 — RisuAI 기본값이 더미라서.
        auth = request.headers.get("authorization") or ""
        token = auth.removeprefix("Bearer ").strip()
        auth = auth if len(token) >= 20 else None

        if body.get("stream"):
            async def relay():
                parts: List[str] = []
                buf = b""
                try:
                    async for chunk in up.stream(payload, auth=auth):
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
                    _finish(session, verdict, original_messages, "".join(parts),
                            prompt_size=prompt_size)
            return StreamingResponse(relay(), media_type="text/event-stream")

        try:
            resp = await up.complete(payload, auth=auth)
        except Exception:
            logger.exception("[proxy] upstream failed")
            return JSONResponse(status_code=502, content={
                "error": "upstream_error", "message": "upstream request failed"})
        _finish(session, verdict, original_messages, _assistant_text(resp),
                prompt_size=prompt_size)
        return JSONResponse(resp)

    return app
