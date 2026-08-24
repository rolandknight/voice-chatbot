import asyncio

import numpy as np
import pytest

from poc_tts_streaming.track import PcmQueueTrack


def run(coro):
    return asyncio.run(coro)


async def _frames(track, n):
    return [await track.recv() for _ in range(n)]


def test_recv_returns_20ms_s16_mono_frames_with_advancing_pts():
    track = PcmQueueTrack(paced=False)
    track.push(np.full(1001, 0.5, dtype=np.float32))
    track.flush()
    frames = run(_frames(track, 4))
    assert [f.samples for f in frames] == [480, 480, 480, 480]
    assert [f.pts for f in frames] == [0, 480, 960, 1440]
    assert all(f.sample_rate == 24000 and f.format.name == "s16" and f.layout.name == "mono"
               for f in frames)
    first = np.frombuffer(bytes(frames[0].planes[0]), dtype=np.int16)
    assert first[0] == 16383
    fourth = np.frombuffer(bytes(frames[3].planes[0]), dtype=np.int16)
    assert not fourth.any(), "underrun must produce silence, never a stall"


def test_clear_drops_queued_audio():
    track = PcmQueueTrack(paced=False)
    track.push(np.ones(4800, dtype=np.float32))
    assert track.queued_frames == 10
    track.clear()
    assert track.queued_frames == 0
    frame = run(track.recv())
    assert not np.frombuffer(bytes(frame.planes[0]), dtype=np.int16).any()


def test_drained_resolves_when_queue_empties():
    async def main():
        track = PcmQueueTrack(paced=False)
        track.push(np.ones(960, dtype=np.float32))
        waiter = asyncio.ensure_future(track.drained())
        await asyncio.sleep(0)
        assert not waiter.done()
        await track.recv()
        await track.recv()
        await asyncio.wait_for(waiter, 1)
    run(main())


def test_drained_on_empty_track_returns_immediately():
    async def main():
        track = PcmQueueTrack(paced=False)
        await asyncio.wait_for(track.drained(), 1)
    run(main())


def test_recv_after_stop_raises_media_stream_error():
    from aiortc.mediastreams import MediaStreamError
    track = PcmQueueTrack(paced=False)
    track.stop()
    with pytest.raises(MediaStreamError):
        run(track.recv())
