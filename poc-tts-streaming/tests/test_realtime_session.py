import asyncio
import json
import threading

import numpy as np
import pytest

from poc_tts_streaming.realtime.events import EventError
from poc_tts_streaming.realtime.session import (
    ChatterboxKnobs, ProducerError, RealtimeSession, SynthWorker, worker_stream,
)

KNOBS = ChatterboxKnobs.from_config({
    "temperature": 0.6, "exaggeration": 0.5, "cfg_scale": 1.0,
    "num_steps": 4, "n_cfm_timesteps": 1, "chunk_size": 120,
    "split_text": True, "split_on_clauses": True,
})


class FakeSink:
    def __init__(self):
        self.pushed, self.flushed, self.cleared = [], 0, 0
    def push(self, pcm): self.pushed.append(pcm)
    def flush(self): self.flushed += 1
    def clear(self): self.cleared += 1
    async def drained(self): return None


class FakeSynth:
    """Splits on '. ' and yields 100 samples per sentence; records calls."""
    def __init__(self, gate: threading.Event | None = None):
        self.calls, self.gate = [], gate
    def __call__(self, text, voice, knobs, cancel):
        self.calls.append((text, voice, knobs))
        for i, sentence in enumerate(s for s in text.split(". ") if s):
            if self.gate is not None and i > 0:
                self.gate.wait(2)
            if cancel.is_set():
                return
            yield sentence, np.full(100, 0.1, dtype=np.float32)


def make_session(synth=None, sink=None, voice="one-one.mp3"):
    sent = []
    worker = SynthWorker()
    session = RealtimeSession(
        send=sent.append, synthesizer=synth or FakeSynth(), sink=sink or FakeSink(),
        worker=worker, voices=lambda: ["one-one.mp3", "marvin.wav"], voice=voice, knobs=KNOBS,
    )
    return session, sent, worker


def types(sent):
    return [e["type"] for e in sent]


async def until(sent, type_, timeout=5):
    for _ in range(int(timeout * 100)):
        if any(e["type"] == type_ for e in sent):
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never saw {type_}; got {types(sent)}")


def run(coro):
    return asyncio.run(coro)


def test_open_sends_session_then_conversation_created():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        assert types(sent) == ["session.created", "conversation.created"]
        s = sent[0]["session"]
        assert s["type"] == "realtime" and s["object"] == "realtime.session"
        assert s["id"] == session.id and s["model"] == "chatterbox-flash"
        assert s["output_modalities"] == ["audio"]
        assert s["audio"]["output"]["voice"] == "one-one.mp3"
        assert s["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
        assert s["x_chatterbox"]["num_steps"] == 4
        assert sent[1]["conversation"]["object"] == "realtime.conversation"
    run(main())


def test_session_update_changes_voice_and_knobs():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update", "session": {
            "audio": {"output": {"voice": "marvin.wav"}},
            "x_chatterbox": {"num_steps": 8, "split_on_clauses": False}}}))
        assert types(sent)[-1] == "session.updated"
        s = sent[-1]["session"]
        assert s["audio"]["output"]["voice"] == "marvin.wav"
        assert s["x_chatterbox"]["num_steps"] == 8 and s["x_chatterbox"]["split_on_clauses"] is False
        assert s["x_chatterbox"]["temperature"] == 0.6, "untouched knobs keep their value"
    run(main())


def test_unknown_voice_is_an_error_and_session_is_unchanged():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update", "event_id": "evt_7",
                                         "session": {"audio": {"output": {"voice": "ghost.wav"}}}}))
        err = sent[-1]
        assert err["type"] == "error"
        assert err["error"]["code"] == "invalid_value"
        assert err["error"]["param"] == "session.audio.output.voice"
        assert err["error"]["event_id"] == "evt_7"
        assert session.session_object()["audio"]["output"]["voice"] == "one-one.mp3"
    run(main())


def test_out_of_range_knob_is_an_error():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "session.update",
                                         "session": {"x_chatterbox": {"num_steps": 99}}}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["param"] == "session.x_chatterbox.num_steps"
    run(main())


def test_item_create_emits_added_and_done():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there."}]}}))
        assert types(sent)[-2:] == ["conversation.item.added", "conversation.item.done"]
        item = sent[-1]["item"]
        assert item["id"].startswith("item_") and item["object"] == "realtime.item"
        assert item["role"] == "user" and item["status"] == "completed"
        assert item["content"] == [{"type": "input_text", "text": "Hello there."}]
        assert sent[-1]["previous_item_id"] is None
    run(main())


def test_assistant_items_are_rejected():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "x"}]}}))
        assert sent[-1]["type"] == "error" and sent[-1]["error"]["param"] == "item.role"
    run(main())


RESPONSE_SEQUENCE = [
    "response.created",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_audio_transcript.delta",   # "Hello there"
    "output_audio_buffer.started",
    "response.output_audio_transcript.delta",   # "General Kenobi."
    "response.output_audio_transcript.done",
    "response.output_audio.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.done",
    "output_audio_buffer.stopped",
]


def test_full_response_sequence_for_two_sentences():
    async def main():
        synth, sink = FakeSynth(), FakeSink()
        session, sent, _ = make_session(synth, sink)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "Hello there. General Kenobi."}]}}))
        n = len(sent)
        await session.handle(json.dumps({"type": "response.create", "event_id": "evt_r"}))
        await until(sent, "output_audio_buffer.stopped")
        assert types(sent)[n:] == RESPONSE_SEQUENCE
        deltas = [e["delta"] for e in sent if e["type"] == "response.output_audio_transcript.delta"]
        assert deltas == ["Hello there ", "General Kenobi. "]
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "completed" and done["id"].startswith("resp_")
        assert done["output"][0]["role"] == "assistant"
        assert done["output"][0]["content"][0] == {"type": "audio", "transcript": "Hello there General Kenobi. "}
        assert done["usage"]["output_tokens"] == 0
        assert len(sink.pushed) == 2 and sink.flushed == 1
        assert synth.calls[0][:2] == ("Hello there. General Kenobi.", "one-one.mp3")
        created = [e for e in sent if e["type"] == "response.created"][0]
        assert created["response"]["status"] == "in_progress"
        started = [e for e in sent if e["type"] == "output_audio_buffer.started"][0]
        assert started["response_id"] == done["id"]
    run(main())


def test_response_create_with_inline_input_speaks_that_text_only():
    async def main():
        synth = FakeSynth()
        session, sent, _ = make_session(synth)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "ignored."}]}}))
        await session.handle(json.dumps({"type": "response.create", "response": {"input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Spoken."}]}]}}))
        await until(sent, "response.done")
        assert synth.calls[0][0] == "Spoken."
    run(main())


def test_response_x_chatterbox_and_voice_override_for_one_response():
    async def main():
        synth = FakeSynth()
        session, sent, _ = make_session(synth)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}}))
        await session.handle(json.dumps({"type": "response.create", "response": {
            "audio": {"output": {"voice": "marvin.wav"}}, "x_chatterbox": {"num_steps": 2}}}))
        await until(sent, "response.done")
        text, voice, knobs = synth.calls[0]
        assert voice == "marvin.wav" and knobs.num_steps == 2
        assert session.session_object()["audio"]["output"]["voice"] == "one-one.mp3"
    run(main())


def test_nothing_to_speak_is_an_error_and_no_response():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle(json.dumps({"type": "response.create", "event_id": "evt_e"}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["event_id"] == "evt_e"
        assert "response.created" not in types(sent)
    run(main())


def test_second_response_while_active_is_an_error():
    async def main():
        gate = threading.Event()
        session, sent, _ = make_session(FakeSynth(gate))
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "One. Two."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "output_audio_buffer.started")
        await session.handle(json.dumps({"type": "response.create"}))
        assert sent[-1]["type"] == "error"
        assert sent[-1]["error"]["code"] == "conversation_already_has_active_response"
        gate.set()
        await until(sent, "response.done")
    run(main())


def test_cancel_closes_the_response_as_cancelled_and_discards_later_chunks():
    async def main():
        gate = threading.Event()
        sink = FakeSink()
        session, sent, _ = make_session(FakeSynth(gate), sink)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "One. Two."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "output_audio_buffer.started")
        await session.handle(json.dumps({"type": "response.cancel"}))
        await until(sent, "response.done")
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "cancelled"
        gate.set()
        await asyncio.sleep(0.1)
        assert len(sink.pushed) == 1, "the chunk finished after cancel must be discarded"
        assert types(sent).count("response.output_audio_transcript.delta") == 1
    run(main())


def test_output_audio_buffer_clear_clears_the_sink_and_reports():
    async def main():
        sink = FakeSink()
        session, sent, _ = make_session(sink=sink)
        await session.open()
        await session.handle(json.dumps({"type": "output_audio_buffer.clear"}))
        assert sink.cleared == 1 and sent[-1]["type"] == "output_audio_buffer.cleared"
    run(main())


def test_synthesizer_failure_marks_the_response_failed():
    async def main():
        def boom(text, voice, knobs, cancel):
            raise RuntimeError("ran out of VRAM during generation. VRAM 0.1 GB free")
            yield  # pragma: no cover
        session, sent, _ = make_session(boom)
        await session.open()
        await session.handle(json.dumps({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}}))
        await session.handle(json.dumps({"type": "response.create"}))
        await until(sent, "response.done")
        done = [e for e in sent if e["type"] == "response.done"][0]["response"]
        assert done["status"] == "failed"
        assert "VRAM" in done["status_details"]["error"]["message"]
    run(main())


def test_unsupported_and_unknown_events_produce_error_events():
    async def main():
        session, sent, _ = make_session()
        await session.open()
        await session.handle('{"type": "input_audio_buffer.append", "audio": "AAAA"}')
        assert sent[-1]["error"]["code"] == "unsupported_event"
        await session.handle('{bad')
        assert sent[-1]["error"]["code"] == "invalid_json"
    run(main())


def test_knobs_merge_validates_ranges():
    with pytest.raises(EventError) as exc:
        KNOBS.merged({"temperature": 5}, param_prefix="session.x_chatterbox")
    assert exc.value.param == "session.x_chatterbox.temperature"
    assert KNOBS.merged({"chunk_size": 200}, param_prefix="x").chunk_size == 200
    assert KNOBS.as_engine_kwargs()["cfg_scale"] == 1.0


async def until_count(sent, type_, n, timeout=5):
    for _ in range(int(timeout * 100)):
        if sum(1 for e in sent if e["type"] == type_) >= n:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"never saw {n}x {type_}; got {types(sent)}")


def test_stopped_is_sent_for_each_response_even_when_the_next_finishes_first():
    async def main():
        class SlowSink(FakeSink):
            def __init__(self):
                super().__init__()
                self.release = asyncio.Event()
            async def drained(self):
                await self.release.wait()
        sink = SlowSink()
        session, sent, _ = make_session(sink=sink)
        await session.open()
        for n, text in enumerate(("A.", "B."), start=1):
            await session.handle(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}}))
            await session.handle(json.dumps({"type": "response.create"}))
            await until_count(sent, "response.done", n)
        assert types(sent).count("output_audio_buffer.stopped") == 0
        sink.release.set()
        await until_count(sent, "output_audio_buffer.stopped", 2)
        done_ids = [e["response"]["id"] for e in sent if e["type"] == "response.done"]
        stopped_ids = [e["response_id"] for e in sent if e["type"] == "output_audio_buffer.stopped"]
        assert sorted(stopped_ids) == sorted(done_ids)
    run(main())


def test_send_failure_mid_response_marks_it_failed_and_frees_the_session():
    async def main():
        sent, state = [], {"raised": False}
        def flaky_send(ev):
            sent.append(ev)
            if ev["type"] == "response.output_audio_transcript.delta" and not state["raised"]:
                state["raised"] = True
                raise ConnectionError("channel closed")
        session = RealtimeSession(
            send=flaky_send, synthesizer=FakeSynth(), sink=FakeSink(), worker=SynthWorker(),
            voices=lambda: ["one-one.mp3"], voice="one-one.mp3", knobs=KNOBS)
        await session.open()
        for n in (1, 2):
            await session.handle(json.dumps({"type": "conversation.item.create", "item": {
                "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi."}]}}))
            await session.handle(json.dumps({"type": "response.create"}))
            await until_count(sent, "response.done", n)
        statuses = [e["response"]["status"] for e in sent if e["type"] == "response.done"]
        assert statuses == ["failed", "completed"]
        assert "error" not in types(sent), "the second response.create must not be rejected"
    run(main())


def test_worker_stream_ends_instead_of_hanging_when_the_job_is_cancelled_before_it_starts():
    async def main():
        worker = SynthWorker()
        release = threading.Event()
        worker.submit(lambda: release.wait(5))  # occupies the single worker thread

        async def consume():
            return [i async for i in worker_stream(worker, lambda: iter([1, 2, 3]))]

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.05)
        worker.shutdown()  # cancels the queued job before it ever runs
        assert await asyncio.wait_for(task, 2) == []
        release.set()
    run(main())


def test_worker_stream_yields_earlier_items_then_raises_producer_error():
    async def main():
        def gen():
            yield 1
            raise RuntimeError("boom")
        got = []
        with pytest.raises(ProducerError) as exc:
            async for i in worker_stream(SynthWorker(), gen):
                got.append(i)
        assert got == [1]
        assert str(exc.value.cause) == "boom"
    run(main())
