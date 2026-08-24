import json

import pytest

from poc_tts_streaming.realtime.events import (
    E, EventError, ResponseCreate, SessionUpdate, error_event, parse_client_event, server_event,
)
from poc_tts_streaming.realtime.ids import new_id


def test_new_id_has_prefix_and_is_unique():
    a, b = new_id("sess"), new_id("sess")
    assert a.startswith("sess_") and a != b and len(a) == len("sess_") + 16


def test_parse_session_update():
    ev = parse_client_event(json.dumps({
        "type": "session.update", "event_id": "evt_1",
        "session": {"audio": {"output": {"voice": "marvin.wav"}}}}))
    assert isinstance(ev, SessionUpdate)
    assert ev.event_id == "evt_1"
    assert ev.session["audio"]["output"]["voice"] == "marvin.wav"


def test_parse_response_create_defaults_to_empty_response():
    ev = parse_client_event('{"type": "response.create"}')
    assert isinstance(ev, ResponseCreate) and ev.response == {}


def test_bad_json_is_an_event_error():
    with pytest.raises(EventError) as exc:
        parse_client_event("{not json")
    assert exc.value.code == "invalid_json"


def test_unknown_type_is_an_event_error_with_the_event_id_echoed():
    with pytest.raises(EventError) as exc:
        parse_client_event('{"type": "nope.nothing", "event_id": "evt_9"}')
    assert exc.value.code == "unknown_event" and exc.value.event_id == "evt_9"


@pytest.mark.parametrize("t", [
    "input_audio_buffer.append", "input_audio_buffer.commit", "input_audio_buffer.clear",
    "conversation.item.truncate", "conversation.item.retrieve",
])
def test_known_but_unsupported_types_say_so(t):
    with pytest.raises(EventError) as exc:
        parse_client_event(json.dumps({"type": t}))
    assert exc.value.code == "unsupported_event"
    assert "not supported" in exc.value.message


def test_schema_failure_names_the_param():
    with pytest.raises(EventError) as exc:
        parse_client_event('{"type": "conversation.item.delete"}')
    assert exc.value.code == "missing_required_parameter"
    assert exc.value.param == "item_id"


@pytest.mark.parametrize("raw", ['{"type": ["a", "b"]}', '{"type": {"x": 1}}', '{"type": 7}'])
def test_non_string_type_is_unknown_event_not_a_crash(raw):
    with pytest.raises(EventError) as exc:
        parse_client_event(raw)
    assert exc.value.code == "unknown_event"


def test_server_event_adds_an_event_id():
    ev = server_event(E.SESSION_CREATED, session={"id": "sess_x"})
    assert ev["type"] == "session.created"
    assert ev["event_id"].startswith("event_")
    assert ev["session"] == {"id": "sess_x"}


def test_error_event_shape():
    ev = error_event(EventError("invalid_value", "bad voice", param="session.audio.output.voice",
                                event_id="evt_3"))
    assert ev["type"] == "error"
    assert ev["error"] == {
        "type": "invalid_request_error", "code": "invalid_value", "message": "bad voice",
        "param": "session.audio.output.voice", "event_id": "evt_3",
    }
