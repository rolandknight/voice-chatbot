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
    /// Set by `reap` when the player died on its own; drained by `tick` so the
    /// report survives a `reap` from `is_playing`.
    exit_report: Option<String>,
}

impl MediaPlayer {
    /// `output_device` is the cpal `(name, id)` of the call's playback device;
    /// the matching mpv device is looked up once (`mpv --audio-device=help`).
    /// A device the call holds exclusively is deliberately *not* targeted --
    /// see [`call_holds_device_exclusively`].
    pub fn new(output_device: Option<(&str, &str)>, server_base: &str) -> Self {
        let device = match output_device {
            Some((_, id)) if call_holds_device_exclusively(id) => {
                tracing::info!(
                    %id,
                    "media: the call holds this device exclusively; mpv will use the \
                     system default output instead"
                );
                None
            }
            Some((name, _)) => mpv_device_for(name),
            None => None,
        };
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
            exit_report: None,
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
        let Some(child) = self.child.as_mut() else {
            return;
        };
        let code = match child.try_wait() {
            Ok(None) => return, // still playing
            Ok(Some(status)) => status.code(),
            Err(_) => None, // cannot query it; treat it as gone
        };
        self.child = None;
        self.ducked = false;
        if let Some(line) = exit_line(&self.title, code) {
            tracing::warn!(title = %self.title, ?code, "media: player exited on its own");
            self.exit_report = Some(line);
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
///
/// Linux needs two allowances CoreAudio does not. The same card appears both
/// as the sound server's shared node and as a family of raw `alsa/` hardware
/// aliases, and the call already holds that hardware open through CPAL -- a
/// second open exits 2 -- so a shared node must win. The two layers also spell
/// the device differently: CPAL says `Jabra Speak2 40 UC, USB Audio` while
/// PipeWire says `Jabra Speak2 40 UC Analog Stereo`, and only the *hardware*
/// descriptions carry CPAL's spelling verbatim. Matching the model (the part
/// before CPAL's comma) is what lets the shared node match at all.
fn mpv_device_for(cpal_name: &str) -> Option<String> {
    let out = Command::new("mpv")
        .arg("--audio-device=help")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    parse_mpv_device_list(&String::from_utf8_lossy(&out.stdout), cpal_name)
}

/// True when the call's own CPAL stream owns the card outright, so no second
/// process can open it by any path.
///
/// ALSA's raw PCMs (`plughw`, `hw`, `front`) go straight at the hardware, and
/// [`crate::audio`] prefers `plughw` on purpose because the shared
/// `sysdefault`/dmix path underruns against a 20 ms period. The cost is that
/// while a call is up the sound server cannot open that card either, so
/// pointing mpv at the speakerphone -- even at the server's own node for it --
/// leaves mpv stalled rather than playing. `default`/`sysdefault` route
/// through the server and stay shareable, and non-ALSA hosts (CoreAudio) share
/// by design.
fn call_holds_device_exclusively(cpal_id: &str) -> bool {
    let Some(pcm) = cpal_id.strip_prefix("alsa:") else {
        return false;
    };
    matches!(
        pcm.split(':').next().unwrap_or_default(),
        "plughw" | "hw" | "front"
    )
}

/// What to report for a player that exited on its own. `None` for a clean
/// exit (the stream simply ended). A failure is worth surfacing because it is
/// otherwise silent: mpv's stderr goes to `Stdio::null()` and [`MediaPlayer::play`]
/// only checks that the *spawn* succeeded, so a refused audio device reads as a
/// playing radio that makes no sound.
fn exit_line(title: &str, code: Option<i32>) -> Option<String> {
    match code {
        Some(0) => None,
        Some(code) => Some(format!(
            "[media: {title} stopped unexpectedly (mpv exit {code})]"
        )),
        None => Some(format!(
            "[media: {title} stopped unexpectedly (mpv killed by signal)]"
        )),
    }
}

fn parse_mpv_device_list(listing: &str, cpal_name: &str) -> Option<String> {
    let want = cpal_name.trim().to_lowercase();
    if want.is_empty() {
        return None;
    }
    // CPAL names a Linux device "<model>, <interface>"; the sound server names
    // the same card "<model> <profile>". The model is the part they share.
    let model = want.split(',').next().unwrap_or(&want).trim();

    let mut hardware_fallback = None;
    for line in listing.lines() {
        let line = line.trim();
        let Some(start) = line.find('\'').map(|i| i + 1) else {
            continue;
        };
        let Some(end) = line[start..].find('\'').map(|i| start + i) else {
            continue;
        };
        let id = &line[start..end];
        if id == "auto" {
            continue;
        }
        let description = line[end + 1..]
            .trim()
            .trim_matches(|c| c == '(' || c == ')')
            .to_lowercase();
        let id_lower = id.to_lowercase();
        let matches = |needle: &str| {
            !needle.is_empty() && (description.contains(needle) || id_lower.contains(needle))
        };
        if !matches(&want) && !matches(model) {
            continue;
        }
        // A raw `alsa/` alias is the PCM the call itself holds; keep it only as
        // a last resort for hosts with no sound server running.
        if !id.starts_with("alsa/") {
            return Some(id.to_string());
        }
        hardware_fallback.get_or_insert_with(|| id.to_string());
    }
    hardware_fallback
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

    /// A real `mpv --audio-device=help` on Linux/PipeWire, trimmed to the
    /// entries that matter. Unlike CoreAudio, this list also carries **raw
    /// ALSA hardware** devices for the same card, and their descriptions are
    /// the ones that contain CPAL's Linux device name verbatim.
    const LINUX_PIPEWIRE_LISTING: &str = "\
List of detected audio devices:
  'auto' (Autoselect device)
  'pipewire' (Default (pipewire))
  'pipewire/alsa_output.pci-0000_00_1f.3.analog-stereo' (Built-in Audio Analog Stereo)
  'pipewire/alsa_output.usb-QTIL_Jabra_Speak2_40_UC_6CFBED16334C-00.analog-stereo' (Jabra Speak2 40 UC Analog Stereo)
  'pulse/alsa_output.usb-QTIL_Jabra_Speak2_40_UC_6CFBED16334C-00.analog-stereo' (Jabra Speak2 40 UC Analog Stereo)
  'alsa' (Default (alsa))
  'alsa/plughw:CARD=UC,DEV=0' (Jabra Speak2 40 UC, USB Audio/Hardware device with all software conversions)
  'alsa/sysdefault:CARD=UC' (Jabra Speak2 40 UC, USB Audio/Default Audio Device)
  'alsa/front:CARD=UC,DEV=0' (Jabra Speak2 40 UC, USB Audio/Front output / input)
  'alsa/dmix:CARD=UC,DEV=0' (Jabra Speak2 40 UC, USB Audio/Direct sample mixing device)
";

    /// `alias_rank` deliberately opens the speakerphone's raw `plughw` PCM,
    /// because the shared `sysdefault`/dmix path underruns. That owns the card
    /// outright: nothing else can open it, *not even through the sound
    /// server*. Pointing mpv at it does not fail fast, it hangs -- so the only
    /// device that can still make a sound is the system default, which mpv
    /// uses when given no `--audio-device` at all.
    #[test]
    fn declines_to_target_a_device_the_call_holds_exclusively() {
        for id in [
            "alsa:plughw:CARD=UC,DEV=0",
            "alsa:hw:CARD=1,DEV=0",
            "alsa:front:CARD=UC,DEV=0",
        ] {
            assert!(call_holds_device_exclusively(id), "{id} is exclusive");
        }
        // Routed through the sound server, so media can share it.
        for id in ["alsa:default", "alsa:sysdefault:CARD=UC"] {
            assert!(!call_holds_device_exclusively(id), "{id} is shareable");
        }
        // CoreAudio shares by design; the Mac must keep targeting the Jabra.
        assert!(!call_holds_device_exclusively(
            "coreaudio:Jabra Speak2 40 UC"
        ));
    }

    /// A player that dies on its own is otherwise invisible: mpv's stderr goes
    /// to `Stdio::null()`, and `play` only checks that the *spawn* succeeded.
    /// That is what let a failed device open look like a working radio.
    #[test]
    fn reports_a_player_that_died_but_stays_quiet_about_a_clean_exit() {
        assert_eq!(exit_line("BBC Radio 4", Some(0)), None);
        assert_eq!(
            exit_line("BBC Radio 4", Some(2)).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (mpv exit 2)]")
        );
        assert_eq!(
            exit_line("BBC Radio 4", None).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (mpv killed by signal)]")
        );
    }

    /// The call already holds the speakerphone's raw PCM through CPAL, so a
    /// second open of the same hardware fails (`mpv` exits 2). Only the sound
    /// server's shared node can be opened alongside the call, so the match
    /// must prefer it over any `alsa/` hardware alias -- even though CPAL's
    /// Linux name ("Jabra Speak2 40 UC, USB Audio") appears verbatim in the
    /// hardware descriptions and only partially in the shared one.
    #[test]
    fn prefers_the_shared_sound_server_device_over_raw_alsa_hardware() {
        let picked = parse_mpv_device_list(LINUX_PIPEWIRE_LISTING, "Jabra Speak2 40 UC, USB Audio")
            .expect("a device for the Jabra");
        assert!(
            !picked.starts_with("alsa/"),
            "picked the raw ALSA hardware device {picked:?}, which the call already holds open"
        );
        assert_eq!(
            picked,
            "pipewire/alsa_output.usb-QTIL_Jabra_Speak2_40_UC_6CFBED16334C-00.analog-stereo"
        );
    }
}

#[cfg(test)]
mod live_tests {
    //! Plays ~3 s of BBC Radio 4 through mpv: `cargo test -p voice-chatbot-client -- --ignored live`.
    use super::*;
    use serde_json::json;

    /// End-to-end device choice against the real `mpv --audio-device=help` on
    /// this host. The speakerphone the call opens (`plughw`) must not be
    /// targeted at all -- mpv stalls on it rather than failing -- while a
    /// device routed through the sound server must resolve to the server's own
    /// node, never to an `alsa/` alias of the same card.
    #[test]
    #[ignore]
    fn live_device_choice_skips_the_call_device_and_shares_the_rest() {
        assert!(MediaPlayer::is_available(), "mpv not installed");
        let name = "Jabra Speak2 40 UC, USB Audio";
        if mpv_device_for(name).is_none() {
            eprintln!("no Jabra on this host; nothing to check");
            return;
        }

        // What `alias_rank` actually opens for a call: not targeted.
        let held = MediaPlayer::new(
            Some((name, "alsa:plughw:CARD=UC,DEV=0")),
            "http://127.0.0.1",
        );
        assert_eq!(
            held.device, None,
            "targeted a device the call holds exclusively"
        );

        // Routed through the sound server: shareable, so target its node.
        let shared = MediaPlayer::new(Some((name, "alsa:default")), "http://127.0.0.1");
        let picked = shared.device.clone().expect("a device for the Jabra");
        println!("shared-device match: {picked}");
        assert!(
            !picked.starts_with("alsa/"),
            "picked the raw ALSA alias {picked:?}, which the call would hold open"
        );
    }

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
