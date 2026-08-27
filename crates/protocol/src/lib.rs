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
    Play {
        url: String,
        title: String,
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn wire_shape_is_action_tagged() {
        let cmd = MediaCommand::Play {
            url: "http://x/y.m3u8".into(),
            title: "BBC Radio 4".into(),
        };
        assert_eq!(
            cmd.to_payload(),
            json!({"action": "play", "url": "http://x/y.m3u8", "title": "BBC Radio 4"})
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
}
