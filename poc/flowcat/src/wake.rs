//! openWakeWord in Rust: detector core + pipeline gate.
//!
//! Two layers, deliberately separated:
//!
//! - [`OpenWakeWord`] — the detector chain (melspectrogram → embedding →
//!   per-word head, all ONNX via `ort`), framework-free: `feed()` raw 16 kHz
//!   PCM, get detections. Reusable as-is in a future Rust satellite client
//!   (single-binary Pi client doing on-device wake + WebRTC).
//! - [`WakeGate`] — the FlowCat `FrameProcessor` wrapper implementing Babel's
//!   Listen mode: swallow audio/VAD edges until a wake word fires, then open a
//!   session window (pre-roll replay so the command tail isn't clipped),
//!   returning to IDLE after silence. Mirrors `wakeword_detector.py` in the
//!   Pipecat implementation.
//!
//! Chain shapes (openWakeWord v0.5.x): melspec `[1, N] → [1,1,5·(N/1280),32]`
//! (transformed `x/10 + 2`); embedding `[1,76,32,1] → [1,1,1,96]` over the
//! last 76 mel frames per 1280-sample step; head `[1,16,96] → [1,1]` sigmoid.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use flowcat_core::Result;

const CHUNK: usize = 1280; // 80 ms @ 16 kHz — openWakeWord's step size

/// Framework-free wake detector backed by the vendored `oww_rs` crate
/// (tract-based openWakeWord chain — validated at parity with the Python
/// reference implementation on our fixtures). Buffers arbitrary input into
/// the 1280-sample steps the chain expects.
pub struct OpenWakeWord {
    model: oww_rs::oww::OwwModel,
    pcm: Vec<i16>,
    threshold: f32,
}

impl OpenWakeWord {
    pub fn load(
        head_path: &str,
        threshold: f32,
    ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
        let model = oww_rs::oww::OwwModel::new_from_path(std::path::Path::new(head_path), threshold)?;
        Ok(Self {
            model,
            pcm: Vec::new(),
            threshold,
        })
    }

    /// Feed raw 16 kHz mono s16 samples; returns the max wake probability that
    /// crossed the threshold in this batch (None otherwise). The underlying
    /// model applies its own smoothing (12-window average) and a 2 s refractory
    /// period, so consecutive fires for one utterance are already suppressed.
    pub fn feed(&mut self, samples: &[i16]) -> Option<f32> {
        self.pcm.extend_from_slice(samples);
        let mut fired: Option<f32> = None;
        while self.pcm.len() >= CHUNK {
            let chunk: Vec<f32> = self.pcm.drain(..CHUNK).map(|s| s as f32).collect();
            let d = self.model.detection(chunk);
            if d.probability >= self.threshold && d.probability > fired.unwrap_or(0.0) {
                fired = Some(d.probability);
            }
        }
        fired
    }
}

enum GateState {
    Idle,
    Awake { last_voice: Instant },
}

/// Listen-mode wake gate (see module docs). Sits between the VAD and the
/// SpeechGate: IDLE swallows `InputAudio` and the VAD's user-speaking edges
/// (so no turn can start) while feeding the detector; on detection it replays
/// ~0.5 s of pre-roll, emits a synthetic `UserStartedSpeaking` (the VAD's own
/// rising edge was swallowed — without this the command in the same breath as
/// the wake word would be lost), and stays AWAKE until `session_window` of
/// silence.
pub struct WakeGate {
    detector: OpenWakeWord,
    state: GateState,
    preroll: VecDeque<i16>,
    preroll_cap: usize,
    session_window: Duration,
    cooldown_until: Instant,
    sample_rate: u32,
}

impl WakeGate {
    pub fn new(detector: OpenWakeWord, session_window_secs: f32) -> Self {
        Self {
            detector,
            state: GateState::Idle,
            preroll: VecDeque::new(),
            preroll_cap: 8000, // 0.5 s @16 kHz; rescaled in start()
            session_window: Duration::from_secs_f32(session_window_secs),
            cooldown_until: Instant::now(),
            sample_rate: 16_000,
        }
    }
}

#[async_trait]
impl FrameProcessor for WakeGate {
    fn name(&self) -> &str {
        "WakeGate"
    }

    async fn start(&mut self, _s: &ProcessorSetup, p: &StartParams) -> Result<()> {
        self.sample_rate = p.audio_in_sample_rate;
        self.preroll_cap = self.sample_rate as usize / 2;
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        // Lazy session expiry: any frame can retire an expired session.
        if let GateState::Awake { last_voice } = &self.state {
            if last_voice.elapsed() > self.session_window {
                tracing::info!("wake session expired → idle");
                self.state = GateState::Idle;
            }
        }
        match &env.frame {
            Frame::InputAudio(audio) => {
                // Always feed the detector (a starved detector re-fires its
                // frozen buffers the moment it resumes — RPi client lesson).
                let fired = self.detector.feed(&audio.pcm);
                match &mut self.state {
                    GateState::Idle => {
                        for s in &audio.pcm {
                            if self.preroll.len() == self.preroll_cap {
                                self.preroll.pop_front();
                            }
                            self.preroll.push_back(*s);
                        }
                        if let Some(p) = fired {
                            if Instant::now() >= self.cooldown_until {
                                tracing::info!(prob = p, "wake word detected → awake");
                                self.cooldown_until = Instant::now() + Duration::from_secs(2);
                                self.state = GateState::Awake {
                                    last_voice: Instant::now(),
                                };
                                link.push_down(Frame::UserStartedSpeaking).await;
                                let pre: Vec<i16> = self.preroll.drain(..).collect();
                                link.push_down(Frame::InputAudio(Arc::new(AudioFrame::mono(
                                    pre,
                                    self.sample_rate,
                                ))))
                                .await;
                                link.push(env.meta, env.frame, env.direction).await;
                            }
                        }
                        // Idle without detection: swallow the audio.
                    }
                    GateState::Awake { .. } => {
                        link.push(env.meta, env.frame, env.direction).await;
                    }
                }
            }
            Frame::UserStartedSpeaking | Frame::UserStoppedSpeaking => match &mut self.state {
                GateState::Idle => {} // swallow: no turns while idle
                GateState::Awake { last_voice } => {
                    if matches!(env.frame, Frame::UserStoppedSpeaking) {
                        *last_voice = Instant::now();
                    }
                    link.push(env.meta, env.frame, env.direction).await;
                }
            },
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Offline detector validation: feed a WAV (env POC_WAKE_TEST_WAV) through
    /// the chain and print the max probability. Run with --nocapture.
    #[test]
    fn wake_probability_on_fixture() {
        let wav = match std::env::var("POC_WAKE_TEST_WAV") {
            Ok(p) => p,
            Err(_) => return,
        };
        let poc = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let model = std::env::var("POC_WAKE_TEST_MODEL").unwrap_or_else(|_| {
            poc.parent().unwrap().join("models/wakeword/hey_babel.onnx").to_string_lossy().into_owned()
        });
        let mut det = OpenWakeWord::load(&model, 0.3).expect("load model");
        let bytes = std::fs::read(&wav).expect("read wav");
        let pcm: Vec<i16> = bytes[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let mut max_p = 0.0f32;
        for chunk in pcm.chunks(1280) {
            if let Some(p) = det.feed(chunk) {
                if p > max_p { max_p = p; }
            }
        }
        println!("max wake probability: {max_p:.4}");
        assert!(max_p > 0.4, "wake fixture should trigger (got {max_p})");
    }
}
