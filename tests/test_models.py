"""Unit tests for saga/models.py.

Covers:
- ChatMessage.get_text_content: string passthrough and multimodal array extraction
- ChatCompletionRequest: extra fields ignored, defaults, stop variants
- StateBlockData: partial construction safety
- RelationshipChange: alias-based construction
"""
import pytest
from saga.models import (
    ChatMessage,
    ChatCompletionRequest,
    StateBlockData,
    RelationshipChange,
    ItemTransfer,
    ChatCompletionResponse,
    Choice,
    Usage,
    StreamDelta,
    StreamChoice,
    ChatCompletionChunk,
)


# ─────────────────────────────────────────────────────────────
# ChatMessage.get_text_content
# ─────────────────────────────────────────────────────────────

class TestGetTextContent:
    def test_plain_string_returned_as_is(self):
        msg = ChatMessage(role="user", content="Hello world")
        assert msg.get_text_content() == "Hello world"

    def test_empty_string(self):
        msg = ChatMessage(role="user", content="")
        assert msg.get_text_content() == ""

    def test_multimodal_array_extracts_text_blocks(self):
        msg = ChatMessage(role="user", content=[
            {"type": "text", "text": "First part"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
            {"type": "text", "text": "Second part"},
        ])
        result = msg.get_text_content()
        assert "First part" in result
        assert "Second part" in result

    def test_multimodal_array_skips_non_text_blocks(self):
        msg = ChatMessage(role="user", content=[
            {"type": "image_url", "image_url": {"url": "https://x.com/img.png"}},
        ])
        result = msg.get_text_content()
        assert result == ""

    def test_multimodal_array_multiple_text_joined_with_newline(self):
        msg = ChatMessage(role="user", content=[
            {"type": "text", "text": "Line A"},
            {"type": "text", "text": "Line B"},
        ])
        assert msg.get_text_content() == "Line A\nLine B"

    def test_multimodal_array_missing_text_key_treated_as_empty(self):
        msg = ChatMessage(role="user", content=[
            {"type": "text"},  # no "text" key
        ])
        result = msg.get_text_content()
        assert result == ""

    def test_empty_content_array(self):
        msg = ChatMessage(role="user", content=[])
        assert msg.get_text_content() == ""

    def test_cache_control_field_accepted(self):
        msg = ChatMessage(
            role="assistant",
            content="Hello",
            cache_control={"type": "ephemeral"},
        )
        assert msg.cache_control == {"type": "ephemeral"}


# ─────────────────────────────────────────────────────────────
# ChatCompletionRequest
# ─────────────────────────────────────────────────────────────

class TestChatCompletionRequest:
    def _base_messages(self):
        return [{"role": "user", "content": "Hi"}]

    def test_extra_fields_ignored(self):
        """OpenAI clients send unknown fields; they must be silently ignored."""
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=self._base_messages(),
            n=1,                          # unknown field
            logprobs=False,               # unknown field
            response_format={"type": "text"},  # unknown field
        )
        assert req.model == "gpt-4"

    def test_stream_defaults_to_false(self):
        req = ChatCompletionRequest(model="gpt-4", messages=self._base_messages())
        assert req.stream is False

    def test_stop_as_string(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=self._base_messages(),
            stop="STOP",
        )
        assert req.stop == "STOP"

    def test_stop_as_list(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=self._base_messages(),
            stop=["END", "STOP"],
        )
        assert req.stop == ["END", "STOP"]

    def test_optional_fields_default_none(self):
        req = ChatCompletionRequest(model="gpt-4", messages=self._base_messages())
        assert req.temperature is None
        assert req.max_tokens is None
        assert req.top_p is None
        assert req.frequency_penalty is None
        assert req.presence_penalty is None
        assert req.stop is None
        assert req.user is None

    def test_user_field_accepted(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=self._base_messages(),
            user="session-123",
        )
        assert req.user == "session-123"

    def test_messages_parsed_as_chat_message_objects(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert req.messages[1].get_text_content() == "Hello"

    def test_multimodal_messages_accepted(self):
        req = ChatCompletionRequest(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "https://x.com/img.png"}},
                ],
            }],
        )
        assert req.messages[0].get_text_content() == "Describe this image"


# ─────────────────────────────────────────────────────────────
# StateBlockData
# ─────────────────────────────────────────────────────────────

class TestStateBlockData:
    def test_minimal_construction(self):
        """Only 'location' is required; all lists default to empty."""
        state = StateBlockData(location="던전")
        assert state.location == "던전"
        assert state.location_moved is False
        assert state.hp_change == 0
        assert state.items_gained == []
        assert state.items_lost == []
        assert state.items_transferred == []
        assert state.npc_met == []
        assert state.npc_separated == []
        assert state.relationship_changes == []
        assert state.mood == ""
        assert state.event_trigger is None
        assert state.notes == ""

    def test_full_construction(self):
        state = StateBlockData(
            location="마을",
            location_moved=True,
            hp_change=-10,
            items_gained=["검"],
            items_lost=["방패"],
            npc_met=["고블린"],
            mood="긴장",
            event_trigger="boss_appear",
            notes="보스 방 근처",
        )
        assert state.location_moved is True
        assert state.hp_change == -10
        assert state.event_trigger == "boss_appear"

    def test_event_trigger_none_by_default(self):
        state = StateBlockData(location="forest")
        assert state.event_trigger is None


# ─────────────────────────────────────────────────────────────
# RelationshipChange (alias-based field)
# ─────────────────────────────────────────────────────────────

class TestRelationshipChange:
    def test_construct_with_aliases(self):
        rc = RelationshipChange(**{"from": "Hero", "to": "Elf", "type": "trust", "delta": 10})
        assert rc.from_entity == "Hero"
        assert rc.to_entity == "Elf"
        assert rc.type == "trust"
        assert rc.delta == 10

    def test_construct_with_field_names(self):
        rc = RelationshipChange(from_entity="A", to_entity="B", type="ally", delta=5)
        assert rc.from_entity == "A"
        assert rc.to_entity == "B"

    def test_negative_delta(self):
        rc = RelationshipChange(**{"from": "A", "to": "B", "type": "hostile", "delta": -20})
        assert rc.delta == -20


# ─────────────────────────────────────────────────────────────
# ItemTransfer
# ─────────────────────────────────────────────────────────────

class TestItemTransfer:
    def test_basic(self):
        it = ItemTransfer(item="검", to="NPC")
        assert it.item == "검"
        assert it.to == "NPC"


# ─────────────────────────────────────────────────────────────
# ChatCompletionResponse structure
# ─────────────────────────────────────────────────────────────

class TestChatCompletionResponse:
    def test_default_id_generated(self):
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[Choice(
                message=ChatMessage(role="assistant", content="Hello"),
                finish_reason="stop",
            )],
        )
        assert resp.id.startswith("chatcmpl-")

    def test_usage_defaults_to_zero(self):
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[Choice(
                message=ChatMessage(role="assistant", content="Hi"),
            )],
        )
        assert resp.usage.prompt_tokens == 0
        assert resp.usage.completion_tokens == 0
        assert resp.usage.total_tokens == 0

    def test_model_dump_json_produces_valid_json(self):
        import json
        resp = ChatCompletionResponse(
            model="gpt-4",
            choices=[Choice(
                message=ChatMessage(role="assistant", content="Hello"),
            )],
        )
        data = json.loads(resp.model_dump_json())
        assert data["object"] == "chat.completion"
        assert "choices" in data


# ─────────────────────────────────────────────────────────────
# StreamDelta / StreamChoice / ChatCompletionChunk
# ─────────────────────────────────────────────────────────────

class TestStreamingModels:
    def test_stream_delta_content_only(self):
        delta = StreamDelta(content="Hello")
        assert delta.content == "Hello"
        assert delta.role is None

    def test_stream_choice_finish_reason_none_during_stream(self):
        choice = StreamChoice(delta=StreamDelta(content="chunk"), finish_reason=None)
        assert choice.finish_reason is None

    def test_chunk_model_dump_json_roundtrip(self):
        import json
        chunk = ChatCompletionChunk(
            model="saga-proxy",
            choices=[StreamChoice(delta=StreamDelta(content="hello"), finish_reason=None)],
        )
        data = json.loads(chunk.model_dump_json())
        assert data["object"] == "chat.completion.chunk"
        assert data["choices"][0]["delta"]["content"] == "hello"
