"""
saga/models.py — Pydantic request/response models for the SAGA RP proxy.

Covers:
- OpenAI-compatible chat completion request/response
- SSE streaming variants
- Session and status helpers
- StateBlockData for structured RP state extraction
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core message types
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    """A single message in a conversation turn."""

    role: str
    content: str
    name: Optional[str] = None
    cache_control: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None


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


class SessionInfo(BaseModel):
    id: str
    name: str
    turn_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class StatusResponse(BaseModel):
    status: str = "ok"
    active_sessions: int = 0
    version: str = "3.0.0"


# ---------------------------------------------------------------------------
# State block — structured RP state extracted from narrative
# ---------------------------------------------------------------------------


class RelationshipChange(BaseModel):
    """A single relationship delta between two entities."""

    # Using field aliases so the model accepts both attribute-style and
    # dict-style access without clashing with Python builtins.
    from_entity: str = Field(..., alias="from")
    to_entity: str = Field(..., alias="to")
    type: str
    delta: int

    model_config = {"populate_by_name": True}


class ItemTransfer(BaseModel):
    """An item transferred from one party to another."""

    item: str
    to: str


class StateBlockData(BaseModel):
    """Structured state block produced by the extraction agent after each turn.

    All fields default to empty/falsy values so partial extraction is safe.
    """

    location: str
    location_moved: bool = False
    hp_change: int = 0
    items_gained: List[str] = Field(default_factory=list)
    items_lost: List[str] = Field(default_factory=list)
    items_transferred: List[ItemTransfer] = Field(default_factory=list)
    npc_met: List[str] = Field(default_factory=list)
    npc_separated: List[str] = Field(default_factory=list)
    relationship_changes: List[RelationshipChange] = Field(default_factory=list)
    mood: str = ""
    event_trigger: Optional[str] = None
    notes: str = ""
