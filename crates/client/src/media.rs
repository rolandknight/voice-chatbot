//! Client-side media playback: BBC radio/shows and sound effects the server's
//! skills ask for arrive as `{"type":"media"}` events and play here through
//! `mpv`, on the same output device as the call (port of scripts/radio.py's
//! `RadioPlayer`, minus the ffplay fallback).
//!
//! Ducking: while the assistant speaks (`rtf-bot-started-speaking` …
//! `rtf-bot-stopped-speaking`) a stream is paused over mpv's JSON IPC and
//! resumed after, as the Python chatbot did. A `play_file` with
//! `after_speech` waits for that same boundary.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;
use voice_chatbot_protocol::{MediaCommand, AFTER_SPEECH_CAP_SECS, MEDIA_EVENT};

pub struct MediaPlayer {
    /// Server base URL; relative media URLs (`/sfx/x.flac`) resolve against it.
    server_base: String,
    ipc_path: PathBuf,
    /// mpv `--audio-device` matched to the call's output device, if any.
    device: Option<String>,
    child: Option<std::process::Child>,
    title: String,
    ducked: bool,
    bot_speaking: bool,
    /// A `play_file { after_speech }` waiting for the assistant to finish.
    pending: Option<(String, Instant)>,
}

impl MediaPlayer {
    /// `output_device_name` is the cpal name of the call's playback device; the
    /// matching mpv device is looked up once (`mpv --audio-device=help`).
    pub fn new(output_device_name: Option<&str>, server_base: &str) -> Self {
        let device = output_device_name.and_then(mpv_device_for);
        match &device {
            Some(d) => tracing::info!(device = %d, "media: mpv output device"),
            None => tracing::info!("media: mpv will use the system default output"),
        }
        Self {
            server_base: server_base.trim_end_matches('/').to_string(),
            ipc_path: std::env::temp_dir()
                .join(format!("voice-chatbot-mpv-{}.sock", std::process::id())),
            device,
            child: None,
            title: String::new(),
            ducked: false,
            bot_speaking: false,
            pending: None,
        }
    }

    pub fn is_available() -> bool {
        Command::new("mpv")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
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
                self.bot_speaking = true;
                if self.is_playing() && !self.ducked && self.ipc(&["set_property", "pause"], true) {
                    self.ducked = true;
                }
                None
            }
            "rtf-bot-stopped-speaking" => {
                self.bot_speaking = false;
                if self.ducked {
                    self.ducked = false;
                    self.ipc(&["set_property", "pause"], false);
                }
                self.pending
                    .take()
                    .map(|(url, _)| self.play(&url, "sound effect", false))
            }
            _ => None,
        }
    }

    /// Time-based housekeeping (call every second or so): reaps a finished
    /// mpv and plays an `after_speech` clip whose wait has hit the cap.
    pub fn tick(&mut self) -> Option<String> {
        self.reap();
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
            MediaCommand::Play { url, title } => Some(self.play(&url, &title, true)),
            MediaCommand::PlayFile { url, after_speech } => {
                if after_speech && self.bot_speaking {
                    self.pending = Some((url, Instant::now()));
                    None
                } else {
                    Some(self.play(&url, "sound effect", false))
                }
            }
            MediaCommand::Stop => self.stop().then(|| "[media stopped]".to_string()),
            MediaCommand::Pause => {
                self.ipc(&["set_property", "pause"], true);
                None
            }
            MediaCommand::Resume => {
                self.ipc(&["set_property", "pause"], false);
                None
            }
        }
    }

    fn is_playing(&mut self) -> bool {
        self.reap();
        self.child.is_some()
    }

    fn reap(&mut self) {
        if let Some(child) = self.child.as_mut() {
            if matches!(child.try_wait(), Ok(Some(_)) | Err(_)) {
                self.child = None;
                self.ducked = false;
            }
        }
    }

    fn play(&mut self, url: &str, title: &str, stream: bool) -> String {
        self.stop();
        let url = if url.starts_with('/') {
            format!("{}{url}", self.server_base)
        } else {
            url.to_string()
        };
        let url = url.as_str();
        let _ = std::fs::remove_file(&self.ipc_path);
        let mut cmd = Command::new("mpv");
        cmd.args(["--no-video", "--no-terminal", "--idle=no"])
            .arg(format!("--input-ipc-server={}", self.ipc_path.display()));
        if stream {
            cmd.args(["--cache=yes", "--demuxer-readahead-secs=4"]);
        }
        if let Some(d) = &self.device {
            cmd.arg(format!("--audio-device={d}"));
        }
        cmd.arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        match cmd.spawn() {
            Ok(child) => {
                self.child = Some(child);
                self.title = title.to_string();
                self.ducked = false;
                // Wait briefly for the IPC socket so the first pause doesn't race.
                let deadline = Instant::now() + Duration::from_secs(2);
                while Instant::now() < deadline && !self.ipc_path.exists() {
                    std::thread::sleep(Duration::from_millis(50));
                }
                if self.bot_speaking && stream && self.ipc(&["set_property", "pause"], true) {
                    self.ducked = true;
                }
                format!("[media: playing {title}]")
            }
            Err(error) => {
                tracing::warn!(%error, "media: failed to start mpv (is it installed?)");
                format!("[media: cannot play {title}: mpv failed to start]")
            }
        }
    }

    /// Kill the player. True when something was playing.
    pub fn stop(&mut self) -> bool {
        self.pending = None;
        let Some(mut child) = self.child.take() else {
            return false;
        };
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&self.ipc_path);
        tracing::info!(title = %self.title, "media: stopped");
        self.ducked = false;
        true
    }

    /// mpv JSON IPC: `{"command": [cmd..., value]}`. False if not connected.
    fn ipc(&self, command: &[&str], value: bool) -> bool {
        #[cfg(unix)]
        {
            use std::os::unix::net::UnixStream;
            if self.child.is_none() || !self.ipc_path.exists() {
                return false;
            }
            let mut args: Vec<Value> = command.iter().map(|c| Value::from(*c)).collect();
            args.push(Value::from(value));
            let line = serde_json::json!({ "command": args }).to_string() + "\n";
            match UnixStream::connect(&self.ipc_path) {
                Ok(mut s) => {
                    let _ = s.set_write_timeout(Some(Duration::from_millis(500)));
                    s.write_all(line.as_bytes()).is_ok()
                }
                Err(error) => {
                    tracing::debug!(%error, "media: mpv IPC send failed");
                    false
                }
            }
        }
        #[cfg(not(unix))]
        {
            let _ = (command, value);
            false
        }
    }
}

impl Drop for MediaPlayer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// `mpv --audio-device=help` prints lines like
/// `  'coreaudio/Jabra Speak2 40 UC' (Jabra Speak2 40 UC)`; pick the one whose
/// description contains the cpal device name (case-insensitive).
fn mpv_device_for(cpal_name: &str) -> Option<String> {
    let out = Command::new("mpv")
        .arg("--audio-device=help")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    parse_mpv_device_list(&String::from_utf8_lossy(&out.stdout), cpal_name)
}

fn parse_mpv_device_list(listing: &str, cpal_name: &str) -> Option<String> {
    let want = cpal_name.trim().to_lowercase();
    if want.is_empty() {
        return None;
    }
    listing.lines().find_map(|line| {
        let line = line.trim();
        let start = line.find('\'')? + 1;
        let end = start + line[start..].find('\'')?;
        let id = &line[start..end];
        let description = line[end + 1..]
            .trim()
            .trim_matches(|c| c == '(' || c == ')');
        (id != "auto"
            && (description.to_lowercase().contains(&want) || id.to_lowercase().contains(&want)))
        .then(|| id.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_mpv_device_matching_the_cpal_name() {
        let listing = "List of detected audio devices:\n  'auto' (Autoselect device)\n  'coreaudio/MacBook Pro Speakers' (MacBook Pro Speakers)\n  'coreaudio/Jabra Speak2 40 UC' (Jabra Speak2 40 UC)\n";
        assert_eq!(
            parse_mpv_device_list(listing, "Jabra Speak2 40 UC").as_deref(),
            Some("coreaudio/Jabra Speak2 40 UC")
        );
        assert_eq!(
            parse_mpv_device_list(listing, "jabra").as_deref(),
            Some("coreaudio/Jabra Speak2 40 UC")
        );
        assert_eq!(parse_mpv_device_list(listing, "Scarlett"), None);
        assert_eq!(parse_mpv_device_list(listing, ""), None);
    }
}

#[cfg(test)]
mod live_tests {
    //! Plays ~3 s of BBC Radio 4 through mpv: `cargo test -p voice-chatbot-client -- --ignored live`.
    use super::*;
    use serde_json::json;

    #[test]
    #[ignore]
    fn live_mpv_play_duck_stop() {
        assert!(MediaPlayer::is_available(), "mpv not installed");
        let mut p = MediaPlayer::new(None, "http://127.0.0.1:6210");
        let url = "http://as-hls-ww-live.akamaized.net/pool_55057080/live/ww/bbc_radio_fourfm/bbc_radio_fourfm.isml/bbc_radio_fourfm-audio%3d96000.norewind.m3u8";
        let line = p.on_event(
            MEDIA_EVENT,
            &json!({"action": "play", "url": url, "title": "BBC Radio 4"}),
        );
        assert_eq!(line.as_deref(), Some("[media: playing BBC Radio 4]"));
        std::thread::sleep(Duration::from_secs(2));
        assert!(p.is_playing(), "mpv exited early");
        assert!(p.ipc_path.exists(), "no IPC socket");
        p.on_event("rtf-bot-started-speaking", &Value::Null);
        assert!(p.ducked, "pause over IPC failed");
        std::thread::sleep(Duration::from_millis(500));
        p.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert!(!p.ducked);
        std::thread::sleep(Duration::from_millis(500));
        let stopped = p.on_event(MEDIA_EVENT, &json!({"action": "stop"}));
        assert_eq!(stopped.as_deref(), Some("[media stopped]"));
        assert!(!p.is_playing());
        assert!(!p.ipc_path.exists());
    }
}
