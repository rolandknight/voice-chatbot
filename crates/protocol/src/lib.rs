//! Messages the server sends the native client over the call's events
//! WebSocket, beyond FlowCat's own `rtf-*` frames. Every frame on that socket
//! is `{"type": <kind>, "payload": <object>}`; this crate owns the payloads
//! for the kinds voice-chatbot adds.

use serde::{Deserialize, Serialize};

/// `type` of the media-control frame. Audio the skills start (BBC radio,
/// on-demand shows, generated sound effects) plays on the client, next to
/// the call audio; the server only sends commands and tracks what it asked for.
pub const MEDIA_EVENT: &str = "media";

/// Payload of a [`MEDIA_EVENT`] frame.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum MediaCommand {
    /// Start streaming `url` (replaces whatever is playing). `title` is for logs/UI.
    ///
    /// `live` picks how the client ducks it while the assistant speaks: a live
    /// stream drops its gain and keeps decoding so it stays at the live edge,
    /// while a recorded one stops its decoder and resumes in place.
    Play {
        url: String,
        title: String,
        #[serde(default = "live_by_default")]
        live: bool,
    },
    /// Play a one-shot clip. With `after_speech`, the client waits for the
    /// assistant to finish speaking first (the tool reply is spoken before the
    /// clip), capped by [`AFTER_SPEECH_CAP_SECS`].
    PlayFile {
        url: String,
        after_speech: bool,
    },
    Stop,
    Pause,
    Resume,
}

/// Radio is the common case, and a gain duck never stalls a decoder.
fn live_by_default() -> bool {
    true
}

/// Longest the client waits for the assistant to go quiet before a
/// `PlayFile { after_speech: true }` plays anyway.
pub const AFTER_SPEECH_CAP_SECS: u64 = 20;

impl MediaCommand {
    pub fn to_payload(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("MediaCommand serializes")
    }

    pub fn from_payload(payload: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(payload.clone())
    }
}

/// `type` of the wake-state frame. In Listen mode the server publishes one on
/// every wake-word fire (with the head, its score and the persona it selected)
/// and one when the session window expires; mirrors the Pipecat ControlChannel
/// `wake` message.
pub const WAKE_EVENT: &str = "wake";

/// Payload of a [`WAKE_EVENT`] frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum WakeState {
    /// A wake word fired: `model` is the head file stem (`hey_marvin`),
    /// `persona` the voice it selected (absent when the head maps to none).
    Awake {
        model: String,
        score: f32,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        persona: Option<String>,
    },
    /// The session window elapsed; a wake word is needed again.
    Asleep,
}

impl WakeState {
    pub fn to_payload(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("WakeState serializes")
    }

    pub fn from_payload(payload: &serde_json::Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(payload.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wire_shape_is_action_tagged() {
        let cmd = MediaCommand::Play {
            url: "http://x/y.m3u8".into(),
            title: "BBC Radio 4".into(),
            live: true,
        };
        assert_eq!(
            cmd.to_payload(),
            json!({"action": "play", "url": "http://x/y.m3u8", "title": "BBC Radio 4", "live": true})
        );
        assert_eq!(MediaCommand::Stop.to_payload(), json!({"action": "stop"}));
        assert_eq!(
            MediaCommand::from_payload(
                &json!({"action": "play_file", "url": "u", "after_speech": true})
            )
            .unwrap(),
            MediaCommand::PlayFile {
                url: "u".into(),
                after_speech: true
            }
        );
        assert!(MediaCommand::from_payload(&json!({"action": "dance"})).is_err());
    }

    #[test]
    fn wake_state_is_state_tagged() {
        let awake = WakeState::Awake {
            model: "hey_marvin".into(),
            score: 0.875,
            persona: Some("marvin".into()),
        };
        assert_eq!(
            awake.to_payload(),
            json!({"state": "awake", "model": "hey_marvin", "score": 0.875, "persona": "marvin"})
        );
        assert_eq!(WakeState::Asleep.to_payload(), json!({"state": "asleep"}));
        assert_eq!(
            WakeState::from_payload(&json!({"state": "awake", "model": "hey_babel", "score": 0.5}))
                .unwrap(),
            WakeState::Awake {
                model: "hey_babel".into(),
                score: 0.5,
                persona: None
            }
        );
        assert!(WakeState::from_payload(&json!({"state": "dreaming"})).is_err());
    }

    #[test]
    fn play_carries_whether_the_stream_is_live() {
        let cmd = MediaCommand::Play {
            url: "http://example/x.m3u8".into(),
            title: "BBC Radio 4".into(),
            live: true,
        };
        let payload = cmd.to_payload();
        assert_eq!(payload["action"], "play");
        assert_eq!(payload["live"], true);
        assert_eq!(MediaCommand::from_payload(&payload).unwrap(), cmd);
    }

    #[test]
    fn a_play_from_an_older_server_is_treated_as_live() {
        // Radio is the common case, and ducking a live stream by gain is the
        // safe default: it never stalls a decoder that cannot be paused.
        let payload = serde_json::json!({
            "action": "play",
            "url": "http://example/x.m3u8",
            "title": "BBC Radio 4"
        });
        let MediaCommand::Play { live, .. } = MediaCommand::from_payload(&payload).unwrap() else {
            panic!("expected a Play");
        };
        assert!(live);
    }
}
