//! On-device wake words for the native client.
//!
//! The server's Listen mode gate serves browser clients; this client detects
//! locally instead: captured audio runs through the shared
//! [`voice_chatbot_wake::WakeBank`] and is only forwarded to the call while a
//! wake session is open. On a fire the pre-roll (the wake word and the
//! command in the same breath) is replayed, and a `{"type":"wake"}` frame
//! goes to the server over the events socket so it selects that persona's
//! voice (`main.rs::apply_client_wake` on the server). The session is
//! re-armed by activity the server reports (transcriptions, bot turns) and
//! ends after `session_secs` of silence, which is reported as `asleep`.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use serde_json::json;
use tokio::sync::mpsc;
use voice_chatbot_protocol::{WakeState, WAKE_EVENT};
use voice_chatbot_wake::{Effect, GateCore, WakeBank, WakeDetector, SAMPLE_RATE};

use crate::resampler::StreamingResampler;

/// Wake heads and tuning, from the `call` command's `--wake-*` flags.
#[derive(Clone, Debug)]
pub struct WakeConfig {
    pub heads: Vec<(PathBuf, String)>,
    pub threshold: f32,
    pub session_secs: f32,
}

/// Latest conversation activity seen by the events task (a final
/// transcription, the bot starting or finishing a turn); the gate folds it
/// into its session window.
#[derive(Clone, Default)]
pub struct Activity(Arc<Mutex<Option<Instant>>>);

impl Activity {
    pub fn note(&self) {
        *self.0.lock().unwrap() = Some(Instant::now());
    }

    fn take(&self) -> Option<Instant> {
        self.0.lock().unwrap().take()
    }

    /// Event kinds that count as conversation activity.
    pub fn is_activity_event(kind: &str, payload: &serde_json::Value) -> bool {
        match kind {
            "rtf-user-transcription" => payload
                .get("final")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false),
            "rtf-bot-started-speaking" | "rtf-bot-stopped-speaking" => true,
            _ => false,
        }
    }
}

/// Client-side Listen-mode gate over captured PCM at `input_rate`.
pub struct ClientWakeGate {
    bank: WakeBank,
    core: GateCore,
    to_detector_rate: StreamingResampler,
    preroll: VecDeque<i16>,
    preroll_cap: usize,
    activity: Activity,
}

impl ClientWakeGate {
    pub fn new(cfg: &WakeConfig, input_rate: u32, activity: Activity) -> Result<Self> {
        let bank = WakeBank::load(&cfg.heads, cfg.threshold)
            .map_err(|e| anyhow::anyhow!("load wake heads: {e}"))?;
        Ok(Self {
            bank,
            core: GateCore::new(Duration::from_secs_f32(cfg.session_secs), Instant::now()),
            to_detector_rate: StreamingResampler::new(input_rate, SAMPLE_RATE)
                .context("create wake resampler")?,
            preroll: VecDeque::new(),
            preroll_cap: input_rate as usize / 2, // 0.5 s
            activity,
        })
    }

    pub fn is_awake(&self) -> bool {
        self.core.is_awake()
    }

    /// Run one captured block: the audio to forward to the call (empty while
    /// asleep; pre-roll + block on the opening fire) and a state change to
    /// report, if any.
    pub fn process(&mut self, pcm: &[i16], now: Instant) -> Result<(Vec<i16>, Option<WakeState>)> {
        if let Some(at) = self.activity.take() {
            self.core.on_activity(at);
        }
        let mut report = None;
        if let Some(Effect::Sleep) = self.core.tick(now) {
            report = Some(WakeState::Asleep);
        }
        let detector_pcm = self
            .to_detector_rate
            .process(pcm)
            .context("resample captured audio for the wake detector")?;
        // Always feed the detector: a starved detector re-fires its frozen
        // buffers the moment it resumes.
        let fired = self.bank.feed(&detector_pcm);
        if !self.core.is_awake() {
            for s in pcm {
                if self.preroll.len() == self.preroll_cap {
                    self.preroll.pop_front();
                }
                self.preroll.push_back(*s);
            }
        }
        let out = match self.core.on_audio(fired, now) {
            Some(Effect::Wake {
                head,
                probability,
                open,
            }) => {
                report = Some(WakeState::Awake {
                    model: self.bank.head_name(head).to_string(),
                    score: probability,
                    persona: Some(self.bank.head_persona(head).to_string()),
                });
                let mut out: Vec<i16> = if open {
                    self.preroll.drain(..).collect()
                } else {
                    Vec::new()
                };
                out.extend_from_slice(pcm);
                out
            }
            _ if self.core.is_awake() => pcm.to_vec(),
            _ => Vec::new(),
        };
        Ok((out, report))
    }
}

/// Wire frame for a state change, as the server expects it.
pub fn wake_frame(state: &WakeState) -> String {
    json!({ "type": WAKE_EVENT, "payload": state.to_payload() }).to_string()
}

/// Console line for a state change.
pub fn describe(state: &WakeState) -> String {
    match state {
        WakeState::Awake {
            model,
            score,
            persona,
        } => format!(
            "[wake: {} {score:.2}]",
            persona.as_deref().unwrap_or(model.as_str())
        ),
        WakeState::Asleep => "[asleep]".to_string(),
    }
}

/// Put the gate between the capture channel and the peer: returns the gated
/// capture channel. State changes go to `outbound` (the events socket) and
/// to stdout. Ends when the capture channel closes.
pub fn spawn(
    mut gate: ClientWakeGate,
    mut input: mpsc::Receiver<Vec<i16>>,
    outbound: mpsc::UnboundedSender<String>,
) -> mpsc::Receiver<Vec<i16>> {
    let (tx, rx) = mpsc::channel(64);
    tokio::spawn(async move {
        while let Some(pcm) = input.recv().await {
            let (out, report) = match gate.process(&pcm, Instant::now()) {
                Ok(r) => r,
                Err(error) => {
                    tracing::warn!(%error, "wake gate failed; passing audio through");
                    (pcm, None)
                }
            };
            if let Some(state) = report {
                println!("{}", describe(&state));
                let _ = outbound.send(wake_frame(&state));
            }
            if !out.is_empty() && tx.send(out).await.is_err() {
                break; // peer gone
            }
        }
    });
    rx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_events_are_finals_and_bot_turns() {
        assert!(Activity::is_activity_event(
            "rtf-user-transcription",
            &json!({"final": true})
        ));
        assert!(!Activity::is_activity_event(
            "rtf-user-transcription",
            &json!({"final": false})
        ));
        assert!(Activity::is_activity_event(
            "rtf-bot-stopped-speaking",
            &json!({})
        ));
        assert!(!Activity::is_activity_event("media", &json!({})));
    }

    #[test]
    fn wake_frame_and_description() {
        let awake = WakeState::Awake {
            model: "hey_marvin".into(),
            score: 0.875,
            persona: Some("marvin".into()),
        };
        assert_eq!(
            wake_frame(&awake),
            r#"{"payload":{"model":"hey_marvin","persona":"marvin","score":0.875,"state":"awake"},"type":"wake"}"#
        );
        assert_eq!(describe(&awake), "[wake: marvin 0.88]");
        assert_eq!(describe(&WakeState::Asleep), "[asleep]");
    }
}
