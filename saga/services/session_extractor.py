"""Session ID / scriptstate / gen params extraction from incoming requests."""
import json
import hashlib
import logging

from fastapi import HTTPException, Request

from saga.core import dependencies as deps
from saga.models import ChatCompletionRequest

logger = logging.getLogger(__name__)


def extract_session_id(request: ChatCompletionRequest, raw_request: Request) -> str | None:
    """Extract session ID with priority: header > user field > system hash.

    1. X-SAGA-Session-ID header (explicit)
    2. request.user field (OpenAI spec, configurable in RisuAI)
    3. System message hash fallback (legacy, backward-compatible)
    """
    header_id = raw_request.headers.get("x-saga-session-id", "").strip()
    if header_id:
        if not deps._SESSION_ID_RE.match(header_id):
            raise HTTPException(400, "Invalid session ID format")
        return header_id

    if request.user:
        user = request.user.strip()
        if not deps._SESSION_ID_RE.match(user):
            raise HTTPException(400, "Invalid user/session ID format")
        return user

    for msg in request.messages:
        if msg.role == "system":
            first_para = msg.get_text_content().split('\n\n')[0][:300]
            sid = hashlib.sha256(first_para.encode()).hexdigest()[:16]
            logger.debug(f"[Session] Hash input (first 150ch): {first_para[:150]!r}")
            logger.debug(f"[Session] Hash result: {sid}")
            return sid
    return None


def extract_scriptstate(raw_request: Request) -> dict | None:
    """Extract scriptstate from x-saga-scriptstate header (JSON-encoded dict).

    RisuAI plugin sends: x-saga-scriptstate: {"$hp":"85","$location":"dungeon"}
    The '$' prefix is stripped — RisuAI internally prefixes scriptstate keys with '$'.
    Returns None if header is absent or invalid.
    """
    raw = raw_request.headers.get("x-saga-scriptstate", "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            logger.warning(f"[Scriptstate] Expected dict, got {type(data).__name__}")
            return None
        cleaned = {}
        for k, v in data.items():
            key = k.lstrip("$") if isinstance(k, str) else str(k)
            cleaned[key] = v
        return cleaned if cleaned else None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"[Scriptstate] Invalid JSON in header: {e}")
        return None


def extract_gen_params(request: ChatCompletionRequest) -> dict:
    """Extract optional generation parameters from the request for LLM forwarding."""
    params = {}
    if request.top_p is not None:
        params["top_p"] = request.top_p
    if request.frequency_penalty is not None:
        params["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        params["presence_penalty"] = request.presence_penalty
    if request.stop is not None:
        params["stop"] = request.stop
    return params
