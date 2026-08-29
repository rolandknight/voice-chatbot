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
use std::sync::mpsc::{Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use voice_chatbot_protocol::{MediaCommand, AFTER_SPEECH_CAP_SECS, MEDIA_EVENT};

use decoder::Decoder;
use duck::{Duck, Transport};
use gain::Gain;

/// What to report for a decoder that exited on its own. `None` for a clean
/// exit (the stream simply ended). A failure is worth surfacing because it is
/// otherwise silent: [`decoder::Decoder::spawn`] nulls ffmpeg's stderr and
/// [`MediaPlayer::play`] only checks that the *spawn* succeeded, so a stream
/// that ffmpeg immediately refused would otherwise read as a playing radio
/// that makes no sound.
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
    /// Spent buffers coming back from the mixer, handed to each decoder's
    /// feeder so it refills them instead of allocating.
    recycle: Arc<Mutex<Receiver<Vec<i16>>>>,
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
        recycle: Receiver<Vec<i16>>,
        gain: Gain,
        sample_rate: u32,
        server_base: &str,
    ) -> Self {
        Self {
            server_base: server_base.trim_end_matches('/').to_string(),
            media_tx,
            recycle: Arc::new(Mutex::new(recycle)),
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
        match Decoder::spawn(
            &url,
            self.sample_rate,
            live,
            self.media_tx.clone(),
            Arc::clone(&self.recycle),
        ) {
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
        self.gain.ramp_to(0.0);
        // Discard whatever is already queued in the mixer: a ramp to 0 only
        // covers gain, not the chunks themselves, and a station switch spawns
        // the next decoder right after this returns -- those queued chunks
        // are still the previous stream's audio and must not play out under
        // the new one.
        self.gain.flush();
        if was_playing {
            tracing::info!(title = %self.title, "media: stopped");
        }
        was_playing
    }

    /// Notice a decoder that ended on its own — once its last sample has
    /// actually reached the mixer — and say so when it failed.
    fn reap(&mut self) {
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        let Some(code) = decoder.finished() else {
            return; // still decoding
        };
        if !decoder.drained() {
            // ffmpeg exiting means it finished *writing*, not that playback
            // ended: the OS pipe (~0.68 s) and the channel (160 ms) still hold
            // audio nobody has heard, and dropping the decoder discards all of
            // it. Measured on a 3 s clip: ffmpeg exits with 0.725 s still in
            // flight. Wait for the feeder to reach EOF instead.
            return;
        }
        self.decoder = None;
        self.duck.stop();
        // No ramp to 0 here: the source has simply gone dry and the mixer's
        // silence path takes it from there. Fading would cut the very tail
        // this wait exists to preserve, and every `play`/`stop` sets the gain
        // itself, so nothing downstream depends on it.
        if let Some(line) = exit_line(&self.title, code) {
            tracing::warn!(title = %self.title, ?code, "media: decoder exited on its own");
            self.exit_report = Some(line);
        }
    }
}

impl Drop for MediaPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::gain::{DUCKED, FULL};
    use serde_json::json;

    fn player() -> (MediaPlayer, std::sync::mpsc::Receiver<Vec<i16>>, Gain) {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let (_recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(8);
        let gain = Gain::new(FULL);
        let player = MediaPlayer::new(
            tx,
            recycle_rx,
            gain.clone(),
            48_000,
            "http://127.0.0.1:6210",
        );
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

#[cfg(test)]
mod live_tests {
    //! Decodes ~3 s of BBC Radio 4 through the real ffmpeg:
    //! `cargo test -p voice-chatbot-client -- --ignored live`.
    use super::*;
    use serde_json::json;

    const RADIO_4: &str = "http://as-hls-ww-live.akamaized.net/pool_55057080/live/ww/bbc_radio_fourfm/bbc_radio_fourfm.isml/bbc_radio_fourfm-audio%3d96000.norewind.m3u8";

    #[test]
    #[ignore]
    fn live_radio_reaches_the_mixer_and_ducks_without_stopping() {
        assert!(MediaPlayer::is_available(), "ffmpeg not installed");
        let (tx, rx) = std::sync::mpsc::sync_channel(512);
        let (_recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(512);
        let gain = Gain::new(gain::FULL);
        let mut player = MediaPlayer::new(
            tx,
            recycle_rx,
            gain.clone(),
            48_000,
            "http://127.0.0.1:6210",
        );

        let line = player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play", "url": RADIO_4, "title": "BBC Radio 4", "live": true}),
        );
        assert_eq!(line.as_deref(), Some("[media: playing BBC Radio 4]"));

        std::thread::sleep(Duration::from_secs(3));
        let samples: Vec<i16> = rx.try_iter().flatten().collect();
        assert!(
            samples.len() > 48_000,
            "expected at least a second of audio, got {}",
            samples.len()
        );
        // Real programme audio, not silence and not clipping. Measured on
        // 2026-08-28: RMS -20.3 dBFS, peak 18083.
        let peak = samples
            .iter()
            .map(|s| i32::from(s.abs()))
            .max()
            .unwrap_or(0);
        assert!((1_000..32_767).contains(&peak), "peak {peak}");

        // A live stream ducks by gain and keeps decoding.
        player.on_event("rtf-bot-started-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::DUCKED);
        assert!(player.decoder.is_some(), "live radio must keep decoding");

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::FULL);

        assert!(player.stop());
    }

    /// The case the previous (subprocess-player) build got wrong: asked for
    /// mid-reply, radio came up at full volume over the assistant.
    #[test]
    #[ignore]
    fn live_radio_asked_for_mid_reply_starts_quiet() {
        assert!(MediaPlayer::is_available(), "ffmpeg not installed");
        let (tx, _rx) = std::sync::mpsc::sync_channel(512);
        let (_recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(512);
        let gain = Gain::new(gain::FULL);
        let mut player = MediaPlayer::new(
            tx,
            recycle_rx,
            gain.clone(),
            48_000,
            "http://127.0.0.1:6210",
        );

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play", "url": RADIO_4, "title": "BBC Radio 4", "live": true}),
        );
        assert_eq!(gain.target(), gain::DUCKED, "must start quiet");

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::FULL, "and come up when the reply ends");
        assert!(player.stop());
    }
}
