import asyncio

import pytest

from poc_tts_streaming.realtime.events import EventError
from poc_tts_streaming.realtime.webrtc import CallRegistry


def test_create_validates_the_session_patch_before_touching_webrtc():
    def build_session(send, sink, session_patch=None):
        if session_patch and session_patch.get("bad"):
            raise EventError("invalid_value", "bad patch", param="session.bad")
        raise AssertionError("must not be called for a valid patch before the channel opens")

    registry = CallRegistry()
    with pytest.raises(EventError) as exc:
        asyncio.run(registry.create("v=0\r\noffer", build_session, session_patch={"bad": True}))
    assert exc.value.param == "session.bad"
    assert len(registry) == 0
