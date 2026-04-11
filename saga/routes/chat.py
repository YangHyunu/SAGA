"""Chat completions route — OpenAI-compatible proxy endpoint."""
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from saga.middleware.auth import verify_bearer
from saga.models import ChatCompletionRequest
from saga.services.chat_handler import handle_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    _auth: None = Depends(verify_bearer),
):
    """OpenAI-compatible chat completions endpoint."""
    try:
        return await handle_chat(request, raw_request)
    except Exception as e:
        logger.error(f"[SAGA] Chat error: {e}", exc_info=True)
        return JSONResponse(
            status_code=502,
            content={"error": "upstream_error", "message": "LLM backend request failed"},
        )
