"""Bearer token authentication middleware for SAGA."""
import hmac

from fastapi import HTTPException, Request

from saga.core import dependencies as deps


async def verify_bearer(request: Request) -> None:
    """Verify API key from Authorization header if server.api_key is configured.

    Supports both 'Authorization: Bearer <key>' and 'Authorization: <key>' formats
    for compatibility with various OpenAI-compatible clients.
    """
    api_key = (deps.config.server.api_key if deps.config else None) or ""
    if not api_key:
        return  # Auth disabled

    auth_header = request.headers.get("authorization", "")
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Missing API key in Authorization header",
                              "type": "auth_error", "code": "missing_api_key"}},
        )

    # Support both "Bearer <key>" and raw "<key>"
    token = auth_header.removeprefix("Bearer ").removeprefix("bearer ").strip()
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Empty API key in Authorization header",
                              "type": "auth_error", "code": "empty_api_key"}},
        )

    if not hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8")):
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key",
                              "type": "auth_error", "code": "invalid_api_key"}},
        )
