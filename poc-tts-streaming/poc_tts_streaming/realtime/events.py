"""Realtime API event vocabulary and client-event validation.

The single place GA event names live. Verified against the API reference on
2026-08-23. No aiortc, no torch.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from poc_tts_streaming.realtime.ids import new_id


class E:
    """Server event types."""
    ERROR = "error"
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    CONVERSATION_CREATED = "conversation.created"
    ITEM_ADDED = "conversation.item.added"
    ITEM_DONE = "conversation.item.done"
    ITEM_DELETED = "conversation.item.deleted"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    OUTPUT_ITEM_ADDED = "response.output_item.added"
    OUTPUT_ITEM_DONE = "response.output_item.done"
    CONTENT_PART_ADDED = "response.content_part.added"
    CONTENT_PART_DONE = "response.content_part.done"
    AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    AUDIO_DONE = "response.output_audio.done"
    OUTPUT_AUDIO_BUFFER_STARTED = "output_audio_buffer.started"
    OUTPUT_AUDIO_BUFFER_STOPPED = "output_audio_buffer.stopped"
    OUTPUT_AUDIO_BUFFER_CLEARED = "output_audio_buffer.cleared"


class EventError(Exception):
    """An error to report as an `error` event (or an HTTP error body)."""

    def __init__(self, code: str, message: str, *, param: str | None = None,
                 event_id: str | None = None, error_type: str = "invalid_request_error"):
        super().__init__(message)
        self.code, self.message, self.param = code, message, param
        self.event_id, self.error_type = event_id, error_type

    def as_dict(self) -> dict:
        return {"type": self.error_type, "code": self.code, "message": self.message,
                "param": self.param, "event_id": self.event_id}


# ---- client events ---------------------------------------------------------

class _Base(BaseModel):
    event_id: Optional[str] = None


class SessionUpdate(_Base):
    type: Literal["session.update"]
    session: dict = Field(default_factory=dict)


class ConversationItemCreate(_Base):
    type: Literal["conversation.item.create"]
    item: dict
    previous_item_id: Optional[str] = None


class ConversationItemDelete(_Base):
    type: Literal["conversation.item.delete"]
    item_id: str


class ResponseCreate(_Base):
    type: Literal["response.create"]
    response: dict = Field(default_factory=dict)


class ResponseCancel(_Base):
    type: Literal["response.cancel"]
    response_id: Optional[str] = None


class OutputAudioBufferClear(_Base):
    type: Literal["output_audio_buffer.clear"]


# Discriminated on "type" so an unknown/mismatched tag fails as a single
# union_tag_invalid/union_tag_not_found error instead of pydantic's default
# "smart" union mode, which validates against every member and reports one
# field error per member (always led by the first member's, regardless of
# which type was actually sent) -- indistinguishable from a real schema
# failure on a known type.
ClientEvent = Annotated[
    Union[
        SessionUpdate, ConversationItemCreate, ConversationItemDelete,
        ResponseCreate, ResponseCancel, OutputAudioBufferClear,
    ],
    Field(discriminator="type"),
]
_ADAPTER = TypeAdapter(ClientEvent)

# Real Realtime client events this TTS server deliberately does not implement.
UNSUPPORTED = frozenset({
    "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear",
    "conversation.item.truncate", "conversation.item.retrieve",
})


def parse_client_event(raw: str) -> ClientEvent:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EventError("invalid_json", f"invalid JSON: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise EventError("invalid_json", "event must be a JSON object")
    event_id = data.get("event_id") if isinstance(data.get("event_id"), str) else None
    type_ = data.get("type")
    if type_ in UNSUPPORTED:
        raise EventError("unsupported_event",
                         f"'{type_}' is not supported by this server (text-to-speech only)",
                         param="type", event_id=event_id)
    try:
        return _ADAPTER.validate_python(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = [str(p) for p in first.get("loc", ()) if p not in ("tagged-union",)]
        # A type that matches no model shows up as a union tag failure on "type".
        if not isinstance(type_, str) or first.get("type") in ("union_tag_invalid", "union_tag_not_found"):
            raise EventError("unknown_event", f"unknown event type {type_!r}",
                             param="type", event_id=event_id) from exc
        param = loc[-1] if loc else None
        code = "missing_required_parameter" if first.get("type") == "missing" else "invalid_value"
        raise EventError(code, first.get("msg", "invalid event"), param=param,
                         event_id=event_id) from exc


# ---- server events ---------------------------------------------------------

def server_event(type_: str, **fields) -> dict:
    return {"type": type_, "event_id": new_id("event"), **fields}


def error_event(err: EventError) -> dict:
    return server_event(E.ERROR, error=err.as_dict())
