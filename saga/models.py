"""
saga/models.py — Pydantic request/response models for the SAGA RP proxy.

Covers:
- OpenAI-compatible chat completion request/response
- SSE streaming variants
- Session and status helpers
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core message types
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a conversation turn."""

    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None

    def get_text_content(self) -> str:
        """Extract plain text from content (handles multimodal arrays)."""
        if isinstance(self.content, str):
            return self.content
        return "\n".join(
            c.get("text", "") for c in self.content if c.get("type") == "text"
        )


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model_config = {"extra": "ignore"}

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# Streaming response (SSE)
# ---------------------------------------------------------------------------


class StreamDelta(BaseModel):
    """Incremental content delta for a streaming chunk."""

    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: StreamDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """A single SSE chunk in a streaming chat completion."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# ---------------------------------------------------------------------------
# Session and status helpers
# ---------------------------------------------------------------------------


class StatusResponse(BaseModel):
    status: str = "ok"
    active_sessions: int = 0
    version: str = "3.0.0"


