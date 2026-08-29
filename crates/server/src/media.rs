//! Per-call media control. Skills ask this to play/stop audio; it forwards
//! the command to the native client over the call's events WebSocket
//! (`voice_chatbot_protocol::MediaCommand`) and remembers what it asked for,
//! so "stop" and cross-stops (radio vs Spotify) work without asking the client.

use std::sync::Mutex;

use flowcat_server::events::CallEvents;
use voice_chatbot_protocol::{MediaCommand, MEDIA_EVENT};

/// What the client was last told to stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NowPlaying {
    pub title: String,
}

pub struct MediaController {
    events: CallEvents,
    playing: Mutex<Option<NowPlaying>>,
}

impl MediaController {
    pub fn new(events: CallEvents) -> Self {
        Self {
            events,
            playing: Mutex::new(None),
        }
    }

    fn send(&self, cmd: &MediaCommand) {
        tracing::info!(?cmd, "media command");
        self.events.publish(MEDIA_EVENT, cmd.to_payload());
    }

    /// Stream `url` on the client, replacing whatever was playing. `live`
    /// distinguishes a broadcast from a recorded programme; see
    /// [`voice_chatbot_protocol::MediaCommand::Play`].
    pub fn play_stream(&self, url: &str, title: &str, live: bool) {
        *self.playing.lock().unwrap() = Some(NowPlaying {
            title: title.to_string(),
        });
        self.send(&MediaCommand::Play {
            url: url.to_string(),
            title: title.to_string(),
            live,
        });
    }

    /// One-shot clip (sound effects); doesn't change the "now playing" stream.
    pub fn play_file(&self, url: &str, after_speech: bool) {
        self.send(&MediaCommand::PlayFile {
            url: url.to_string(),
            after_speech,
        });
    }

    /// Stop the stream. Returns what was playing, if anything.
    pub fn stop(&self) -> Option<NowPlaying> {
        let was = self.playing.lock().unwrap().take();
        if was.is_some() {
            self.send(&MediaCommand::Stop);
        }
        was
    }

    pub fn now_playing(&self) -> Option<NowPlaying> {
        self.playing.lock().unwrap().clone()
    }

    pub fn is_playing(&self) -> bool {
        self.playing.lock().unwrap().is_some()
    }
}
