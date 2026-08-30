//! openWakeWord in Rust, framework-free: a multi-head detector bank and the
//! Listen-mode state machine. Used by the server's FlowCat `WakeGate`
//! (browser clients — server-side wake) and by the native WebRTC client,
//! which detects on-device and tells the server which persona woke.
//!
//! - [`WakeBank`] — one melspectrogram → embedding frontend
//!   (`oww_rs::AudioFeaturesTract`) feeding N per-word heads (one per
//!   `hey_<persona>.onnx` in a directory). `feed()` raw 16 kHz PCM, get the
//!   best detection. Each head keeps openWakeWord's own 12-window smoothing,
//!   so heads never silence each other.
//! - [`GateCore`] — IDLE / AWAKE with a silence-based session window and a
//!   cross-head cooldown, pure so it is unit-tested without an audio stack.
//!
//! Persona convention: the head's file stem minus a leading `hey_`, with `_`
//! as `-` (`hey_one_one.onnx` → `one-one`), which must name a Qwen preset in
//! `voices/`. See docs/plans/wakeword-in-server.md.
//!
//! Chain shapes (openWakeWord v0.5.x): melspec `[1, N] → [1,1,5·(N/1280),32]`
//! (transformed `x/10 + 2`); embedding `[1,76,32,1] → [1,1,1,96]` over the
//! last 76 mel frames per 1280-sample step; head `[1,16,96] → [1,1]` sigmoid.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// openWakeWord's step size: 80 ms @ 16 kHz.
pub const CHUNK: usize = 1280;
/// Sample rate every detector input must be at.
pub const SAMPLE_RATE: u32 = 16_000;

/// Minimum gap between two fires across all heads: one breath can't be heard
/// twice, by the same head or by a near-miss neighbour.
pub const CROSS_HEAD_COOLDOWN: Duration = Duration::from_secs(2);

/// A speaking hold older than this is treated as stale. Start and stop edges
/// are paired at the source (a barge-in emits the stop), but a dropped events
/// socket or a killed TTS must not leave the session awake forever.
pub const MAX_SPEAKING_HOLD: Duration = Duration::from_secs(120);

// ===========================================================================
// Head discovery
// ===========================================================================

/// Persona a head file activates: the stem minus a leading `hey_`/`hey-`,
/// lowercased, `_` as `-` (the Qwen preset naming in `voices/`).
pub fn persona_for_head(stem: &str) -> String {
    let lower = stem.trim().to_ascii_lowercase();
    let body = lower
        .strip_prefix("hey_")
        .or_else(|| lower.strip_prefix("hey-"))
        .unwrap_or(&lower);
    body.replace('_', "-")
}

/// Resolve the head files from `WAKE_DIR` (every `*.onnx`, name-sorted)
/// or, when the directory is unset, the single `WAKE_MODEL`. Relative
/// paths resolve against `root`. Empty result = push mode. Each entry is
/// `(path, persona)`.
pub fn resolve_heads(
    root: &Path,
    dir: &str,
    single: &str,
) -> std::result::Result<Vec<(PathBuf, String)>, String> {
    let absolute = |p: &str| {
        let p = Path::new(p);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            root.join(p)
        }
    };
    let mut paths: Vec<PathBuf> = Vec::new();
    if !dir.trim().is_empty() {
        if !single.trim().is_empty() {
            tracing::warn!("WAKE_DIR and WAKE_MODEL are both set; using the directory");
        }
        let dir = absolute(dir.trim());
        let entries = std::fs::read_dir(&dir)
            .map_err(|e| format!("wake head directory {}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| e.to_string())?.path();
            if path.extension().and_then(|e| e.to_str()) == Some("onnx") {
                paths.push(path);
            }
        }
        if paths.is_empty() {
            return Err(format!(
                "wake head directory {} holds no .onnx head models",
                dir.display()
            ));
        }
        paths.sort();
    } else if !single.trim().is_empty() {
        paths.push(absolute(single.trim()));
    }
    let mut heads = Vec::new();
    for path in paths {
        std::fs::metadata(&path).map_err(|e| format!("wake head {}: {e}", path.display()))?;
        let stem = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .ok_or_else(|| format!("wake head {}: no file stem", path.display()))?;
        heads.push((path, persona_for_head(&stem)));
    }
    Ok(heads)
}

// ===========================================================================
// Detector bank
// ===========================================================================

/// A fire from one head.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub head: usize,
    pub probability: f32,
}

/// What the gate needs from a detector; [`WakeBank`] in production, a scripted
/// fake in tests.
pub trait WakeDetector: Send + 'static {
    /// Feed raw 16 kHz mono s16 samples; the strongest fire in this batch.
    fn feed(&mut self, samples: &[i16]) -> Option<Detection>;
    fn head_name(&self, head: usize) -> &str;
    fn head_persona(&self, head: usize) -> &str;
}

/// One per-word head and the persona it activates.
pub struct WakeHead {
    pub name: String,
    pub persona: String,
    model: oww_rs::oww::OwwModel,
}

/// Shared frontend + N heads, buffering arbitrary input into the 1280-sample
/// steps the chain expects.
pub struct WakeBank {
    frontend: oww_rs::oww::audio::AudioFeaturesTract,
    heads: Vec<WakeHead>,
    pcm: Vec<i16>,
    threshold: f32,
}

impl WakeBank {
    /// Load `(path, persona)` heads at one probability threshold.
    pub fn load(
        heads: &[(PathBuf, String)],
        threshold: f32,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        if heads.is_empty() {
            return Err("wake bank needs at least one head".into());
        }
        let mut loaded = Vec::with_capacity(heads.len());
        for (path, persona) in heads {
            let model = oww_rs::oww::OwwModel::head_from_path(path, threshold)
                .map_err(|e| format!("load wake head {}: {e}", path.display()))?;
            let name = path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "custom".to_string());
            loaded.push(WakeHead {
                name,
                persona: persona.clone(),
                model,
            });
        }
        Ok(Self {
            frontend: oww_rs::oww::audio::AudioFeaturesTract::create_default(),
            heads: loaded,
            pcm: Vec::new(),
            threshold,
        })
    }

    pub fn heads(&self) -> &[WakeHead] {
        &self.heads
    }
}

impl WakeDetector for WakeBank {
    fn feed(&mut self, samples: &[i16]) -> Option<Detection> {
        self.pcm.extend_from_slice(samples);
        let mut best: Option<Detection> = None;
        while self.pcm.len() >= CHUNK {
            let chunk: Vec<f32> = self.pcm.drain(..CHUNK).map(|s| s as f32).collect();
            // One melspec+embedding pass per step, shared by every head.
            let features = match self.frontend.get_audio_features(&chunk) {
                Ok(f) => f,
                Err(e) => {
                    tracing::warn!(error = %e, "wake frontend error; step skipped");
                    continue;
                }
            };
            for (i, head) in self.heads.iter_mut().enumerate() {
                let (fired, p) = head.model.detect(features.clone());
                // `p` is openWakeWord's smoothed score: 0 unless ≥2 of the last 12
                // windows crossed the threshold. Gate on it rather than on the
                // head's own `fired` flag — the head's 2 s refractory also
                // counts from load time, and the gate keeps its own cooldown.
                let _ = fired;
                if p >= self.threshold && p > best.map_or(0.0, |b| b.probability) {
                    best = Some(Detection {
                        head: i,
                        probability: p,
                    });
                }
            }
        }
        best
    }

    fn head_name(&self, head: usize) -> &str {
        &self.heads[head].name
    }

    fn head_persona(&self, head: usize) -> &str {
        &self.heads[head].persona
    }
}

// ===========================================================================
// Gate state machine (pure)
// ===========================================================================

/// What the gate must do in response to an input, beyond forwarding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Effect {
    /// A head fired. `open` = IDLE → AWAKE transition (replay pre-roll and
    /// synthesize the speaking edge); `false` = already awake, persona
    /// hand-over only.
    Wake {
        head: usize,
        probability: f32,
        open: bool,
    },
    /// The session window elapsed: back to IDLE.
    Sleep,
}

/// Whose speech is holding the session open. Each side is tracked separately
/// so the bot's reply and the caller's turn can overlap (barge-in).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speaker {
    User,
    Bot,
}

/// IDLE / AWAKE with a silence-based session window and a cross-head cooldown.
/// Time is passed in so tests drive it deterministically.
pub struct GateCore {
    awake: bool,
    last_voice: Instant,
    session_window: Duration,
    cooldown_until: Option<Instant>,
    /// When each side started speaking, while it still is: the session window
    /// is silence-based, so it must not run during speech. Indexed by
    /// [`Speaker`].
    speaking_since: [Option<Instant>; 2],
}

impl GateCore {
    pub fn new(session_window: Duration, now: Instant) -> Self {
        Self {
            awake: false,
            last_voice: now,
            session_window,
            cooldown_until: None,
            speaking_since: [None; 2],
        }
    }

    pub fn is_awake(&self) -> bool {
        self.awake
    }

    /// Is either side mid-utterance? A hold past [`MAX_SPEAKING_HOLD`] is
    /// stale (its stop edge was lost) and no longer counts.
    pub fn is_speaking(&self, now: Instant) -> bool {
        self.speaking_since
            .iter()
            .flatten()
            .any(|started| now.duration_since(*started) <= MAX_SPEAKING_HOLD)
    }

    /// Lazy session expiry; call first for every frame.
    pub fn tick(&mut self, now: Instant) -> Option<Effect> {
        if !self.awake {
            return None;
        }
        // Speech in progress suspends the window: the countdown to sleep runs
        // from the moment speaking stops, not from when it started.
        if self.is_speaking(now) {
            self.last_voice = now;
            return None;
        }
        if now.duration_since(self.last_voice) > self.session_window {
            self.awake = false;
            self.speaking_since = [None; 2];
            return Some(Effect::Sleep);
        }
        None
    }

    /// An audio step ran through the detector.
    pub fn on_audio(&mut self, fired: Option<Detection>, now: Instant) -> Option<Effect> {
        let d = fired?;
        if self.cooldown_until.is_some_and(|until| now < until) {
            return None;
        }
        self.cooldown_until = Some(now + CROSS_HEAD_COOLDOWN);
        let open = !self.awake;
        self.awake = true;
        self.last_voice = now;
        Some(Effect::Wake {
            head: d.head,
            probability: d.probability,
            open,
        })
    }

    /// Voice activity while awake (a transcription on the client) re-arms the
    /// session window.
    pub fn on_activity(&mut self, now: Instant) {
        if self.awake {
            self.last_voice = now;
        }
    }

    /// `who` started speaking (the VAD's rising edge, the bot's first TTS
    /// chunk): hold the session open until the matching
    /// [`GateCore::on_speaking_end`]. Ignored while idle, where no session
    /// exists to hold.
    pub fn on_speaking_start(&mut self, who: Speaker, now: Instant) {
        if self.awake {
            self.speaking_since[who as usize] = Some(now);
            self.last_voice = now;
        }
    }

    /// `who` stopped speaking: release the hold and start the session window
    /// from here.
    pub fn on_speaking_end(&mut self, who: Speaker, now: Instant) {
        self.speaking_since[who as usize] = None;
        self.on_activity(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persona_follows_the_file_stem() {
        assert_eq!(persona_for_head("hey_babel"), "babel");
        assert_eq!(persona_for_head("hey_marvin"), "marvin");
        assert_eq!(persona_for_head("hey_one_one"), "one-one");
        assert_eq!(persona_for_head("Hey-Jarvis"), "jarvis");
        assert_eq!(persona_for_head("babel"), "babel");
    }

    #[test]
    fn resolve_heads_reads_a_directory_or_a_single_file() {
        let tmp = std::env::temp_dir().join(format!("wake-heads-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();
        for name in ["hey_marvin.onnx", "hey_babel.onnx", "notes.txt"] {
            std::fs::write(tmp.join(name), b"x").unwrap();
        }
        let heads = resolve_heads(&tmp, tmp.to_str().unwrap(), "").unwrap();
        let personas: Vec<&str> = heads.iter().map(|(_, p)| p.as_str()).collect();
        assert_eq!(personas, ["babel", "marvin"], "name-sorted, .onnx only");

        let single = resolve_heads(&tmp, "", "hey_marvin.onnx").unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].0, tmp.join("hey_marvin.onnx"), "relative → root");
        assert_eq!(single[0].1, "marvin");

        assert!(resolve_heads(&tmp, "", "").unwrap().is_empty(), "push mode");
        let empty = tmp.join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        assert!(resolve_heads(&tmp, empty.to_str().unwrap(), "").is_err());
        assert!(resolve_heads(&tmp, "", "missing.onnx").is_err());
        let _ = std::fs::remove_dir_all(&tmp);
    }

    fn fire(head: usize) -> Option<Detection> {
        Some(Detection {
            head,
            probability: 0.9,
        })
    }

    #[test]
    fn gate_core_opens_hands_over_and_sleeps() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let mut core = GateCore::new(Duration::from_secs(15), t0);
        assert!(!core.is_awake());
        assert_eq!(core.on_audio(None, s(0.1)), None);

        // First fire opens the session.
        assert_eq!(
            core.on_audio(fire(0), s(1.0)),
            Some(Effect::Wake {
                head: 0,
                probability: 0.9,
                open: true
            })
        );
        assert!(core.is_awake());
        // Inside the cross-head cooldown a second fire is ignored, even from
        // another head.
        assert_eq!(core.on_audio(fire(1), s(2.0)), None);
        // After it, a different head hands the session over without reopening.
        assert_eq!(
            core.on_audio(fire(1), s(4.0)),
            Some(Effect::Wake {
                head: 1,
                probability: 0.9,
                open: false
            })
        );
        // Silence re-arms from the last voice activity, not from the fire.
        core.on_activity(s(10.0));
        assert_eq!(core.tick(s(24.0)), None);
        assert_eq!(core.tick(s(25.1)), Some(Effect::Sleep));
        assert!(!core.is_awake());
        assert_eq!(core.tick(s(30.0)), None, "sleep fires once");
        // Next fire opens a fresh session.
        assert_eq!(
            core.on_audio(fire(0), s(31.0)),
            Some(Effect::Wake {
                head: 0,
                probability: 0.9,
                open: true
            })
        );
    }

    #[test]
    fn the_session_window_runs_only_after_speaking_stops() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let mut core = GateCore::new(Duration::from_secs(15), t0);
        core.on_audio(fire(0), s(1.0));

        // A 40 s reply is four times the session window, and the gate must
        // not fall asleep in the middle of it.
        core.on_speaking_start(Speaker::Bot, s(2.0));
        assert_eq!(core.tick(s(20.0)), None);
        assert_eq!(core.tick(s(42.0)), None);
        assert!(core.is_awake());

        // The window starts at the stop edge, not at the start of the reply.
        core.on_speaking_end(Speaker::Bot, s(42.0));
        assert_eq!(core.tick(s(56.0)), None);
        assert_eq!(core.tick(s(57.1)), Some(Effect::Sleep));
    }

    #[test]
    fn overlapping_speakers_each_hold_the_session() {
        let t0 = Instant::now();
        let s = |secs: f32| t0 + Duration::from_secs_f32(secs);
        let mut core = GateCore::new(Duration::from_secs(15), t0);
        core.on_audio(fire(0), s(1.0));

        // Barge-in: the caller starts while the bot is still speaking, so the
        // bot's stop edge must not release the session.
        core.on_speaking_start(Speaker::Bot, s(2.0));
        core.on_speaking_start(Speaker::User, s(5.0));
        core.on_speaking_end(Speaker::Bot, s(6.0));
        assert_eq!(core.tick(s(40.0)), None, "the caller is still speaking");
        core.on_speaking_end(Speaker::User, s(40.0));
        assert_eq!(core.tick(s(55.1)), Some(Effect::Sleep));
    }

    #[test]
    fn a_lost_stop_edge_cannot_hold_the_session_forever() {
        let t0 = Instant::now();
        let s = |secs: u64| t0 + Duration::from_secs(secs);
        let mut core = GateCore::new(Duration::from_secs(15), t0);
        core.on_audio(fire(0), s(1));
        core.on_speaking_start(Speaker::Bot, s(2));
        // The stop edge never arrives (dropped events socket): the hold goes
        // stale and the window runs again.
        let stale = s(2) + MAX_SPEAKING_HOLD;
        assert_eq!(core.tick(stale), None);
        assert_eq!(
            core.tick(stale + Duration::from_secs(16)),
            Some(Effect::Sleep)
        );
        // A stale hold does not leak into the next session.
        core.on_audio(fire(0), stale + Duration::from_secs(20));
        assert!(!core.is_speaking(stale + Duration::from_secs(20)));
    }

    #[test]
    fn speaking_while_idle_does_not_hold_anything() {
        let t0 = Instant::now();
        let mut core = GateCore::new(Duration::from_secs(1), t0);
        core.on_speaking_start(Speaker::Bot, t0);
        assert!(!core.is_speaking(t0), "no session to hold open");
        assert_eq!(core.tick(t0 + Duration::from_secs(100)), None);
    }

    #[test]
    fn activity_while_idle_does_not_arm_anything() {
        let t0 = Instant::now();
        let mut core = GateCore::new(Duration::from_secs(1), t0);
        core.on_activity(t0 + Duration::from_secs(5));
        assert_eq!(core.tick(t0 + Duration::from_secs(100)), None);
    }

    fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf()
    }

    fn read_wav_s16(path: &Path) -> Vec<i16> {
        let bytes = std::fs::read(path).expect("read wav");
        bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect()
    }

    /// All three heads on the "Hey babel, what time is it?" fixture: babel
    /// fires, the others never do (the cross-talk check three heads need).
    /// Runs only when the trained heads and the fixture are present.
    #[test]
    fn bank_fires_only_the_babel_head_on_the_babel_fixture() {
        let root = repo_root();
        let dir = root.join("models/wakeword");
        let wav = root.join("fixtures/t13_wake.wav");
        if !dir.join("hey_babel.onnx").exists() || !wav.exists() {
            eprintln!("skipping: models/wakeword or fixtures/t13_wake.wav missing");
            return;
        }
        let heads = resolve_heads(&root, dir.to_str().unwrap(), "").unwrap();
        assert!(heads.len() >= 3, "expected the three trained heads");
        let mut bank = WakeBank::load(&heads, 0.3).expect("load heads");
        let babel = bank
            .heads()
            .iter()
            .position(|h| h.persona == "babel")
            .expect("babel head");
        let mut fired: Vec<Detection> = Vec::new();
        for chunk in read_wav_s16(&wav).chunks(1280) {
            if let Some(d) = bank.feed(chunk) {
                fired.push(d);
            }
        }
        assert!(!fired.is_empty(), "the babel fixture should trigger");
        for d in &fired {
            assert_eq!(
                d.head,
                babel,
                "{} fired on the babel fixture (p={:.3})",
                bank.head_name(d.head),
                d.probability
            );
        }
        let max_p = fired.iter().map(|d| d.probability).fold(0.0, f32::max);
        assert!(max_p > 0.4, "babel probability {max_p}");
    }
}
