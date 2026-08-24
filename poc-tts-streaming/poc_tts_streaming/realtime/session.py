"""OpenAI Realtime protocol state machine for a text-to-speech server.

Pure Python: no aiortc, no torch. Audio leaves through an AudioSink; events
leave through a `send` callable. This is the module a Rust port of the
protocol would mirror.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

import numpy as np

from poc_tts_streaming.realtime.events import (
    E, ConversationItemCreate, ConversationItemDelete, EventError, OutputAudioBufferClear,
    ResponseCancel, ResponseCreate, SessionUpdate, error_event, parse_client_event, server_event,
)
from poc_tts_streaming.realtime.ids import new_id

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000


# ---- knobs -------------------------------------------------------------------

_RANGES: dict[str, tuple[type, float, float]] = {
    "temperature": (float, 0.0, 2.0),
    "exaggeration": (float, 0.0, 2.0),
    "cfg_scale": (float, 0.0, 5.0),
    "num_steps": (int, 1, 32),
    "n_cfm_timesteps": (int, 1, 8),
    "chunk_size": (int, 50, 500),
}
_BOOLS = ("split_text", "split_on_clauses")


@dataclass(frozen=True)
class ChatterboxKnobs:
    temperature: float
    exaggeration: float
    cfg_scale: float
    num_steps: int
    n_cfm_timesteps: int
    chunk_size: int
    split_text: bool
    split_on_clauses: bool

    @classmethod
    def from_config(cls, generation_cfg: dict) -> "ChatterboxKnobs":
        g = generation_cfg
        return cls(
            temperature=float(g.get("temperature", 0.6)),
            exaggeration=float(g.get("exaggeration", 0.5)),
            cfg_scale=float(g.get("cfg_scale", 1.0)),
            num_steps=int(g.get("num_steps", 10)),
            n_cfm_timesteps=int(g.get("n_cfm_timesteps", 2)),
            chunk_size=int(g.get("chunk_size", 120)),
            split_text=bool(g.get("split_text", True)),
            split_on_clauses=bool(g.get("split_on_clauses", True)),
        )

    def merged(self, patch: dict, *, param_prefix: str) -> "ChatterboxKnobs":
        if not isinstance(patch, dict):
            raise EventError("invalid_value", "x_chatterbox must be an object", param=param_prefix)
        values = dataclasses.asdict(self)
        for key, raw in patch.items():
            param = f"{param_prefix}.{key}"
            if key in _BOOLS:
                if not isinstance(raw, bool):
                    raise EventError("invalid_value", f"{key} must be a boolean", param=param)
                values[key] = raw
            elif key in _RANGES:
                typ, lo, hi = _RANGES[key]
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    raise EventError("invalid_value", f"{key} must be a number", param=param)
                if not lo <= raw <= hi:
                    raise EventError("invalid_value", f"{key} must be between {lo} and {hi}", param=param)
                values[key] = typ(raw)
            else:
                raise EventError("unknown_parameter", f"unknown x_chatterbox parameter {key!r}", param=param)
        return ChatterboxKnobs(**values)

    def as_engine_kwargs(self) -> dict:
        return dataclasses.asdict(self)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


# ---- collaborators ---------------------------------------------------------

Synthesizer = Callable[[str, str, ChatterboxKnobs, threading.Event], Iterator[tuple[str, np.ndarray]]]


class AudioSink(Protocol):
    def push(self, pcm: np.ndarray) -> None: ...
    def flush(self) -> None: ...
    def clear(self) -> None: ...
    async def drained(self) -> None: ...


class SynthWorker:
    """One synthesis at a time per engine: a single-thread executor."""

    def __init__(self) -> None:
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="synth")

    def submit(self, fn: Callable[[], None]) -> concurrent.futures.Future:
        return self._pool.submit(fn)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


# ---- session -----------------------------------------------------------------

_DONE = object()


@dataclass
class _Response:
    id: str
    item_id: str
    cancel: threading.Event
    closed: bool = False
    status: str = "in_progress"
    transcript: str = ""
    started: bool = False
    error: dict | None = None
    metadata: dict | None = None


class RealtimeSession:
    def __init__(
        self, *, send: Callable[[dict], None], synthesizer: Synthesizer, sink: AudioSink,
        worker: SynthWorker, voices: Callable[[], list[str]], voice: str, knobs: ChatterboxKnobs,
        model: str = "chatterbox-flash", session_patch: dict | None = None,
    ) -> None:
        self.id = new_id("sess")
        self.conversation_id = new_id("conv")
        self._send, self._synthesizer, self._sink, self._worker = send, synthesizer, sink, worker
        self._voices, self._model = voices, model
        self._voice, self._knobs = voice, knobs
        self._instructions = ""
        self._items: list[dict] = []
        self._unspoken: list[str] = []
        self._active: _Response | None = None
        self._playout_token = 0
        if session_patch:
            self.apply_session_patch(session_patch)

    # ---- session object -----------------------------------------------------

    def session_object(self) -> dict:
        fmt = {"type": "audio/pcm", "rate": SAMPLE_RATE}
        return {
            "type": "realtime", "id": self.id, "object": "realtime.session", "model": self._model,
            "output_modalities": ["audio"], "instructions": self._instructions,
            "audio": {"input": {"format": fmt, "turn_detection": None},
                      "output": {"format": fmt, "voice": self._voice, "speed": 1.0}},
            "x_chatterbox": self._knobs.as_dict(),
        }

    def _check_voice(self, voice, param: str) -> str:
        if not isinstance(voice, str) or voice not in self._voices():
            raise EventError("invalid_value", f"unknown voice {voice!r}; use a reference clip filename",
                             param=param)
        return voice

    def apply_session_patch(self, patch: dict) -> None:
        """Validate the whole patch first, then apply -- an error leaves the
        session exactly as it was."""
        if not isinstance(patch, dict):
            raise EventError("invalid_value", "session must be an object", param="session")
        voice, knobs, instructions = self._voice, self._knobs, self._instructions
        out = patch.get("audio", {}).get("output", {}) if isinstance(patch.get("audio"), dict) else {}
        if "voice" in out:
            voice = self._check_voice(out["voice"], "session.audio.output.voice")
        if "x_chatterbox" in patch:
            knobs = self._knobs.merged(patch["x_chatterbox"], param_prefix="session.x_chatterbox")
        if "instructions" in patch:
            if not isinstance(patch["instructions"], str):
                raise EventError("invalid_value", "instructions must be a string", param="session.instructions")
            instructions = patch["instructions"]
        if "output_modalities" in patch and patch["output_modalities"] != ["audio"]:
            raise EventError("invalid_value", "this server only produces audio",
                             param="session.output_modalities")
        self._voice, self._knobs, self._instructions = voice, knobs, instructions

    # ---- lifecycle ------------------------------------------------------------

    async def open(self) -> None:
        self._send(server_event(E.SESSION_CREATED, session=self.session_object()))
        self._send(server_event(E.CONVERSATION_CREATED,
                                conversation={"id": self.conversation_id, "object": "realtime.conversation"}))

    async def close(self) -> None:
        if self._active is not None and not self._active.closed:
            self._active.cancel.set()
            self._active.closed = True
            self._active = None

    async def handle(self, raw: str) -> None:
        try:
            event = parse_client_event(raw)
        except EventError as err:
            self._send(error_event(err))
            return
        try:
            if isinstance(event, SessionUpdate):
                self.apply_session_patch(event.session)
                self._send(server_event(E.SESSION_UPDATED, session=self.session_object()))
            elif isinstance(event, ConversationItemCreate):
                self._on_item_create(event)
            elif isinstance(event, ConversationItemDelete):
                self._on_item_delete(event)
            elif isinstance(event, ResponseCreate):
                await self._on_response_create(event)
            elif isinstance(event, ResponseCancel):
                await self._on_response_cancel()
            elif isinstance(event, OutputAudioBufferClear):
                self._playout_token += 1
                self._sink.clear()
                self._send(server_event(E.OUTPUT_AUDIO_BUFFER_CLEARED,
                                        response_id=self._active.id if self._active else None))
        except EventError as err:
            err.event_id = err.event_id or event.event_id
            self._send(error_event(err))

    # ---- conversation items -----------------------------------------------------

    @staticmethod
    def _user_text(item: dict, param: str) -> str:
        if item.get("type") != "message":
            raise EventError("invalid_value", "only message items are supported", param=f"{param}.type")
        if item.get("role") != "user":
            raise EventError("invalid_value", "only user messages can be spoken", param=f"{param}.role")
        content = item.get("content")
        if not isinstance(content, list) or not content:
            raise EventError("missing_required_parameter", "content is required", param=f"{param}.content")
        texts = []
        for i, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "input_text" or not isinstance(part.get("text"), str):
                raise EventError("invalid_value", "content parts must be input_text",
                                 param=f"{param}.content[{i}].type")
            texts.append(part["text"])
        return " ".join(texts)

    def _on_item_create(self, event: ConversationItemCreate) -> None:
        text = self._user_text(event.item, "item")
        item = {"id": event.item.get("id") or new_id("item"), "object": "realtime.item", "type": "message",
                "status": "completed", "role": "user", "content": [{"type": "input_text", "text": text}]}
        previous = self._items[-1]["id"] if self._items else None
        self._items.append(item)
        self._unspoken.append(item["id"])
        self._send(server_event(E.ITEM_ADDED, item=item, previous_item_id=previous))
        self._send(server_event(E.ITEM_DONE, item=item, previous_item_id=previous))

    def _on_item_delete(self, event: ConversationItemDelete) -> None:
        before = len(self._items)
        self._items = [i for i in self._items if i["id"] != event.item_id]
        if len(self._items) == before:
            raise EventError("item_not_found", f"no item {event.item_id!r}", param="item_id")
        self._unspoken = [i for i in self._unspoken if i != event.item_id]
        self._send(server_event(E.ITEM_DELETED, item_id=event.item_id))

    # ---- responses -----------------------------------------------------------

    def _response_object(self, resp: _Response, output: list[dict]) -> dict:
        return {"id": resp.id, "object": "realtime.response", "status": resp.status,
                "status_details": ({"type": resp.status, "error": resp.error} if resp.error else None),
                "output": output, "conversation_id": self.conversation_id,
                "output_modalities": ["audio"], "metadata": resp.metadata,
                "usage": ({"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
                          if resp.status != "in_progress" else None)}

    def _assistant_item(self, resp: _Response, done: bool) -> dict:
        return {"id": resp.item_id, "object": "realtime.item", "type": "message",
                "status": "completed" if done else "in_progress", "role": "assistant",
                "content": [{"type": "audio", "transcript": resp.transcript}] if done else []}

    async def _on_response_create(self, event: ResponseCreate) -> None:
        if self._active is not None and not self._active.closed:
            raise EventError("conversation_already_has_active_response",
                             "a response is already in progress; cancel it first")
        spec = event.response if isinstance(event.response, dict) else {}
        if "input" in spec:
            if not isinstance(spec["input"], list):
                raise EventError("invalid_value", "response.input must be a list", param="response.input")
            text = " ".join(self._user_text(i, f"response.input[{n}]") for n, i in enumerate(spec["input"]))
            spoken_ids: list[str] = []
        else:
            by_id = {i["id"]: i for i in self._items}
            text = " ".join(by_id[i]["content"][0]["text"] for i in self._unspoken if i in by_id)
            spoken_ids = list(self._unspoken)
        if not text.strip():
            raise EventError("invalid_request_error", "nothing to speak: add a user message item first",
                             param="response.input")
        voice = self._voice
        out = spec.get("audio", {}).get("output", {}) if isinstance(spec.get("audio"), dict) else {}
        if "voice" in out:
            voice = self._check_voice(out["voice"], "response.audio.output.voice")
        knobs = self._knobs
        if "x_chatterbox" in spec:
            knobs = knobs.merged(spec["x_chatterbox"], param_prefix="response.x_chatterbox")

        resp = _Response(id=new_id("resp"), item_id=new_id("item"), cancel=threading.Event(),
                         metadata=spec.get("metadata"))
        self._active = resp
        self._unspoken = [i for i in self._unspoken if i not in spoken_ids]
        self._send(server_event(E.RESPONSE_CREATED, response=self._response_object(resp, [])))
        self._send(server_event(E.OUTPUT_ITEM_ADDED, response_id=resp.id, output_index=0,
                                item=self._assistant_item(resp, done=False)))
        self._send(server_event(E.CONTENT_PART_ADDED, response_id=resp.id, item_id=resp.item_id,
                                output_index=0, content_index=0, part={"type": "audio", "transcript": ""}))
        asyncio.ensure_future(self._run_response(resp, text, voice, knobs))

    async def _run_response(self, resp: _Response, text: str, voice: str, knobs: ChatterboxKnobs) -> None:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def produce() -> None:
            try:
                for chunk in self._synthesizer(text, voice, knobs, resp.cancel):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # noqa: BLE001 -- reported as response.failed
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        self._worker.submit(produce)
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            if resp.closed:
                continue  # cancelled: discard whatever the worker still produces
            if isinstance(item, Exception):
                logger.error("response %s failed: %s", resp.id, item)
                resp.error = {"type": "server_error", "code": "synthesis_failed", "message": str(item)}
                await self._finish(resp, "failed")
                continue
            chunk_text, pcm = item
            delta = chunk_text + " "
            resp.transcript += delta
            self._send(server_event(E.AUDIO_TRANSCRIPT_DELTA, response_id=resp.id, item_id=resp.item_id,
                                    output_index=0, content_index=0, delta=delta))
            self._sink.push(pcm)
            if not resp.started:
                resp.started = True
                self._send(server_event(E.OUTPUT_AUDIO_BUFFER_STARTED, response_id=resp.id))
        if not resp.closed:
            await self._finish(resp, "completed")

    async def _finish(self, resp: _Response, status: str) -> None:
        if resp.closed:
            return
        resp.closed, resp.status = True, status
        self._sink.flush()
        common = dict(response_id=resp.id, item_id=resp.item_id, output_index=0, content_index=0)
        self._send(server_event(E.AUDIO_TRANSCRIPT_DONE, transcript=resp.transcript, **common))
        self._send(server_event(E.AUDIO_DONE, **common))
        self._send(server_event(E.CONTENT_PART_DONE, part={"type": "audio", "transcript": resp.transcript},
                                **common))
        item = self._assistant_item(resp, done=True)
        self._items.append(item)
        self._send(server_event(E.OUTPUT_ITEM_DONE, response_id=resp.id, output_index=0, item=item))
        self._send(server_event(E.RESPONSE_DONE, response=self._response_object(resp, [item])))
        if self._active is resp:
            self._active = None
        if resp.started:
            self._playout_token += 1
            asyncio.ensure_future(self._after_playout(resp.id, self._playout_token))

    async def _after_playout(self, response_id: str, token: int) -> None:
        await self._sink.drained()
        if token == self._playout_token:
            self._send(server_event(E.OUTPUT_AUDIO_BUFFER_STOPPED, response_id=response_id))

    async def _on_response_cancel(self) -> None:
        resp = self._active
        if resp is None or resp.closed:
            raise EventError("response_cancel_not_active", "no active response to cancel")
        resp.cancel.set()
        await self._finish(resp, "cancelled")
