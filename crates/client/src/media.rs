//! Client-side media playback: BBC radio/shows and sound effects the server's
//! skills ask for arrive as `{"type":"media"}` events, are decoded by `ffmpeg`
//! and are mixed into the call's own output stream.
//!
//! Playing through the call's stream rather than a second process is what puts
//! media on the speakerphone at all: while a call is up, CPAL holds that card
//! outright and nothing else can open it by any path.
//!
//! Ducking: while the assistant speaks (`rtf-bot-started-speaking` …
//! `rtf-bot-stopped-speaking`) a live stream drops to [`gain::DUCKED`] and
//! keeps decoding, so it stays at the live edge; a recorded one stops its
//! decoder and resumes in place. A `play_file` with `after_speech` waits for
//! the same boundary.

pub mod decoder;
pub mod duck;
pub mod gain;

use std::process::{Command, Stdio};
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use serde_json::Value;
use voice_chatbot_protocol::{MediaCommand, AFTER_SPEECH_CAP_SECS, MEDIA_EVENT};

use decoder::Decoder;
use duck::{Duck, Transport};
use gain::Gain;

/// What to report for a decoder that exited on its own. `None` for a clean
/// exit (the stream simply ended).
fn exit_line(title: &str, code: Option<i32>) -> Option<String> {
    match code {
        Some(0) => None,
        Some(code) => Some(format!(
            "[media: {title} stopped unexpectedly (ffmpeg exit {code})]"
        )),
        None => Some(format!(
            "[media: {title} stopped unexpectedly (ffmpeg killed by signal)]"
        )),
    }
}

pub struct MediaPlayer {
    /// Server base URL; relative media URLs (`/sfx/x.flac`) resolve against it.
    server_base: String,
    media_tx: SyncSender<Vec<i16>>,
    gain: Gain,
    sample_rate: u32,
    decoder: Option<Decoder>,
    duck: Duck,
    title: String,
    /// A `play_file { after_speech }` waiting for the assistant to finish.
    pending: Option<(String, Instant)>,
    exit_report: Option<String>,
}

impl MediaPlayer {
    pub fn new(
        media_tx: SyncSender<Vec<i16>>,
        gain: Gain,
        sample_rate: u32,
        server_base: &str,
    ) -> Self {
        Self {
            server_base: server_base.trim_end_matches('/').to_string(),
            media_tx,
            gain,
            sample_rate,
            decoder: None,
            duck: Duck::new(),
            title: String::new(),
            pending: None,
            exit_report: None,
        }
    }

    /// ffmpeg decodes every format this plays; without it nothing can play.
    pub fn is_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn resolve(&self, url: &str) -> String {
        if url.starts_with('/') {
            format!("{}{url}", self.server_base)
        } else {
            url.to_string()
        }
    }

    /// Push the state machine's current intent at the mixer and the decoder.
    fn apply_duck(&mut self) {
        self.gain.ramp_to(self.duck.gain());
        if let Some(decoder) = &self.decoder {
            decoder.set_running(self.duck.transport() == Transport::Running);
        }
    }

    /// Dispatch one events-WebSocket frame. Returns a line to print, if any.
    pub fn on_event(&mut self, kind: &str, payload: &Value) -> Option<String> {
        match kind {
            MEDIA_EVENT => match MediaCommand::from_payload(payload) {
                Ok(cmd) => self.apply(cmd),
                Err(error) => {
                    tracing::warn!(%error, "media: bad command");
                    None
                }
            },
            "rtf-bot-started-speaking" => {
                self.duck.set_bot_speaking(true);
                self.apply_duck();
                None
            }
            "rtf-bot-stopped-speaking" => {
                self.duck.set_bot_speaking(false);
                self.apply_duck();
                self.pending
                    .take()
                    .map(|(url, _)| self.play(&url, "sound effect", false))
            }
            _ => None,
        }
    }

    /// Time-based housekeeping (call every second or so).
    pub fn tick(&mut self) -> Option<String> {
        self.reap();
        if let Some(line) = self.exit_report.take() {
            return Some(line);
        }
        match &self.pending {
            Some((_, since)) if since.elapsed() >= Duration::from_secs(AFTER_SPEECH_CAP_SECS) => {
                tracing::warn!(
                    "media: timed out waiting for the assistant to finish; playing anyway"
                );
                let (url, _) = self.pending.take()?;
                Some(self.play(&url, "sound effect", false))
            }
            _ => None,
        }
    }

    fn apply(&mut self, cmd: MediaCommand) -> Option<String> {
        match cmd {
            MediaCommand::Play { url, title, live } => Some(self.play(&url, &title, live)),
            MediaCommand::PlayFile { url, after_speech } => {
                if after_speech && self.duck.is_playing_speech() {
                    self.pending = Some((url, Instant::now()));
                    None
                } else {
                    Some(self.play(&url, "sound effect", false))
                }
            }
            MediaCommand::Stop => self.stop().then(|| "[media stopped]".to_string()),
            MediaCommand::Pause => {
                self.duck.set_user_paused(true);
                self.apply_duck();
                None
            }
            MediaCommand::Resume => {
                self.duck.set_user_paused(false);
                self.apply_duck();
                None
            }
        }
    }

    fn play(&mut self, url: &str, title: &str, live: bool) -> String {
        self.stop();
        let url = self.resolve(url);
        // Start ducked when the reply is still being spoken: there is no
        // earlier level to fade from, and fading in from full is the overlap
        // being avoided.
        let jump = self.duck.start(live);
        if jump {
            self.gain.jump_to(self.duck.gain());
        } else {
            self.gain.ramp_to(self.duck.gain());
        }
        match Decoder::spawn(&url, self.sample_rate, live, self.media_tx.clone()) {
            Ok(decoder) => {
                decoder.set_running(self.duck.transport() == Transport::Running);
                self.decoder = Some(decoder);
                self.title = title.to_string();
                format!("[media: playing {title}]")
            }
            Err(error) => {
                tracing::warn!(%error, "media: failed to start ffmpeg (is it installed?)");
                self.duck.stop();
                self.gain.ramp_to(0.0);
                format!("[media: cannot play {title}: ffmpeg failed to start]")
            }
        }
    }

    /// Stop playback. True when something was playing.
    pub fn stop(&mut self) -> bool {
        self.pending = None;
        let was_playing = self.decoder.take().is_some();
        self.duck.stop();
        // The ramp to 0 covers whatever is already queued in the channel.
        self.gain.ramp_to(0.0);
        if was_playing {
            tracing::info!(title = %self.title, "media: stopped");
        }
        was_playing
    }

    /// Notice a decoder that ended on its own, and say so when it failed.
    fn reap(&mut self) {
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        let Some(code) = decoder.finished() else {
            return; // still playing
        };
        self.decoder = None;
        self.duck.stop();
        self.gain.ramp_to(0.0);
        if let Some(line) = exit_line(&self.title, code) {
            tracing::warn!(title = %self.title, ?code, "media: decoder exited on its own");
            self.exit_report = Some(line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::gain::{DUCKED, FULL};
    use serde_json::json;

    fn player() -> (MediaPlayer, std::sync::mpsc::Receiver<Vec<i16>>, Gain) {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let gain = Gain::new(FULL);
        let player = MediaPlayer::new(tx, gain.clone(), 48_000, "http://127.0.0.1:6210");
        (player, rx, gain)
    }

    #[test]
    fn a_relative_clip_url_resolves_against_the_server() {
        let (player, _rx, _gain) = player();
        assert_eq!(
            player.resolve("/sfx/woosh.flac"),
            "http://127.0.0.1:6210/sfx/woosh.flac"
        );
        assert_eq!(
            player.resolve("http://elsewhere/x.m3u8"),
            "http://elsewhere/x.m3u8"
        );
    }

    #[test]
    fn speaking_boundaries_move_the_shared_gain() {
        let (mut player, _rx, gain) = player();
        player.duck.start(true);
        player.apply_duck();
        assert_eq!(gain.target(), FULL);

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        assert_eq!(gain.target(), DUCKED);

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), FULL);
    }

    #[test]
    fn an_explicit_pause_is_not_undone_by_the_next_reply() {
        let (mut player, _rx, gain) = player();
        player.duck.start(true);
        player.on_event(MEDIA_EVENT, &json!({"action": "pause"}));
        assert_eq!(gain.target(), 0.0);

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), 0.0, "still paused");

        player.on_event(MEDIA_EVENT, &json!({"action": "resume"}));
        assert_eq!(gain.target(), FULL);
    }

    #[test]
    fn a_clip_waits_for_the_assistant_when_asked_to() {
        let (mut player, _rx, _gain) = player();
        player.on_event("rtf-bot-started-speaking", &Value::Null);
        let line = player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play_file", "url": "/sfx/x.flac", "after_speech": true}),
        );
        assert_eq!(line, None, "held until the assistant finishes");
        assert!(player.pending.is_some());
    }

    #[test]
    fn exit_line_is_quiet_about_a_clean_exit_and_loud_about_a_failure() {
        assert_eq!(exit_line("BBC Radio 4", Some(0)), None);
        assert_eq!(
            exit_line("BBC Radio 4", Some(1)).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (ffmpeg exit 1)]")
        );
        assert_eq!(
            exit_line("BBC Radio 4", None).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (ffmpeg killed by signal)]")
        );
    }
}
