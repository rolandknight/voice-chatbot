"""Unit coverage for the FlowCat event-wire compatibility layer."""

from __future__ import annotations

from .flowcat_adapter import normalize_event


def test_normalize_user_transcript_from_rtf_payload() -> None:
    raw = {
        "type": "rtf-user-transcription",
        "payload": {"text": "What time is it?", "final": True},
    }

    event = normalize_event(raw, ts=123.5)

    assert event is not None
    assert event.kind == "transcript-user"
    assert event.text == "What time is it?"
    assert event.ts == 123.5
    assert event.raw is raw


def test_normalize_keeps_legacy_data_envelope_compatibility() -> None:
    raw = {
        "type": "rtf-bot-transcription",
        "data": {"text": "Ready.", "final": True},
    }

    event = normalize_event(raw, ts=7.0)

    assert event is not None
    assert event.kind == "transcript-bot"
    assert event.text == "Ready."
    assert event.ts == 7.0
