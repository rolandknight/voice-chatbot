//! Listen mode in the server pipeline: the FlowCat `FrameProcessor` around
//! [`voice_chatbot_wake::WakeBank`] / [`voice_chatbot_wake::GateCore`].
//! Used for browser clients (server-side wake); the native WebRTC client
//! detects on-device and reports over the events socket instead
//! (`main.rs::apply_client_wake`).
//!
//! Swallow audio/VAD edges until a wake word fires, then open a session
//! window (pre-roll replay so the command tail isn't clipped), select the
//! persona's voice on the call ([`CallState::set_voice`], what
//! `switch_persona` does), publish `wake` events to the client, and return to
//! IDLE after silence. Mirrors `wakeword_detector.py` in the Pipecat
//! implementation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::processor::{Envelope, FrameProcessor, Link, ProcessorSetup};
use flowcat_core::Result;
use flowcat_server::events::CallEvents;
use voice_chatbot_protocol::{WakeState, WAKE_EVENT};
use voice_chatbot_wake::{Effect, GateCore, WakeBank, WakeDetector};

use crate::skills::{CallState, LlmBackend};

// ===========================================================================
// FlowCat processor
// ===========================================================================

/// Listen-mode wake gate (see module docs). Sits between the VAD and the
/// SpeechGate: IDLE swallows `InputAudio` and the VAD's user-speaking edges
/// (so no turn can start) while feeding the detector; on detection it selects
/// the persona's voice, replays ~0.5 s of pre-roll, emits a synthetic
/// `UserStartedSpeaking` (the VAD's own rising edge was swallowed — without
/// this the command in the same breath as the wake word would be lost), and
/// stays AWAKE until `session_window` of silence. A different wake word while
/// awake hands the conversation to that persona without a new edge.
pub struct WakeGate<D: WakeDetector = WakeBank> {
    detector: D,
    core: GateCore,
    preroll: VecDeque<i16>,
    preroll_cap: usize,
    sample_rate: u32,
    state: Option<Arc<CallState>>,
    events: Option<CallEvents>,
}

impl<D: WakeDetector> WakeGate<D> {
    pub fn new(detector: D, session_window_secs: f32) -> Self {
        Self {
            detector,
            core: GateCore::new(Duration::from_secs_f32(session_window_secs), Instant::now()),
            preroll: VecDeque::new(),
            preroll_cap: 8000, // 0.5 s @16 kHz; rescaled in start()
            sample_rate: 16_000,
            state: None,
            events: None,
        }
    }

    /// The call's per-call flags: the fired head's persona becomes the voice,
    /// and `ask_claude`'s backend flip reverts on sleep.
    pub fn with_state(mut self, state: Arc<CallState>) -> Self {
        self.state = Some(state);
        self
    }

    /// Publish `wake` awake/asleep events on the call's events channel.
    pub fn with_events(mut self, events: CallEvents) -> Self {
        self.events = Some(events);
        self
    }

    fn publish(&self, state: WakeState) {
        if let Some(events) = &self.events {
            events.publish(WAKE_EVENT, state.to_payload());
        }
    }

    fn on_wake(&mut self, head: usize, probability: f32, open: bool) {
        let name = self.detector.head_name(head).to_string();
        let persona = self.detector.head_persona(head).to_string();
        tracing::info!(
            head = %name,
            prob = probability,
            persona = %persona,
            "wake word detected → {}",
            if open { "awake" } else { "persona hand-over" }
        );
        if let Some(state) = &self.state {
            state.set_voice(&persona);
        }
        self.publish(WakeState::Awake {
            model: name,
            score: probability,
            persona: Some(persona),
        });
    }

    fn on_sleep(&mut self) {
        tracing::info!("wake session expired → idle");
        if let Some(state) = &self.state {
            if state.backend() != LlmBackend::Local {
                tracing::info!("wake sleep: reverting the LLM backend to local");
                state.set_backend(LlmBackend::Local);
            }
        }
        self.publish(WakeState::Asleep);
    }
}

#[async_trait]
impl<D: WakeDetector> FrameProcessor for WakeGate<D> {
    fn name(&self) -> &str {
        "WakeGate"
    }

    async fn start(&mut self, _s: &ProcessorSetup, p: &StartParams) -> Result<()> {
        self.sample_rate = p.audio_in_sample_rate;
        self.preroll_cap = self.sample_rate as usize / 2;
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        let now = Instant::now();
        // Lazy session expiry: any frame can retire an expired session.
        if let Some(Effect::Sleep) = self.core.tick(now) {
            self.on_sleep();
        }
        match &env.frame {
            Frame::InputAudio(audio) => {
                // Always feed the detector (a starved detector re-fires its
                // frozen buffers the moment it resumes — RPi client lesson).
                let fired = self.detector.feed(&audio.pcm);
                if !self.core.is_awake() {
                    for s in &audio.pcm {
                        if self.preroll.len() == self.preroll_cap {
                            self.preroll.pop_front();
                        }
                        self.preroll.push_back(*s);
                    }
                }
                match self.core.on_audio(fired, now) {
                    Some(Effect::Wake {
                        head,
                        probability,
                        open,
                    }) => {
                        // Voice first: the reply's TTS must already be in the
                        // new voice when the turn runs.
                        self.on_wake(head, probability, open);
                        if open {
                            link.push_down(Frame::UserStartedSpeaking).await;
                            let pre: Vec<i16> = self.preroll.drain(..).collect();
                            link.push_down(Frame::InputAudio(Arc::new(AudioFrame::mono(
                                pre,
                                self.sample_rate,
                            ))))
                            .await;
                        }
                        link.push(env.meta, env.frame, env.direction).await;
                    }
                    _ if self.core.is_awake() => {
                        link.push(env.meta, env.frame, env.direction).await;
                    }
                    _ => {} // idle without detection: swallow the audio
                }
            }
            Frame::UserStartedSpeaking | Frame::UserStoppedSpeaking => {
                if self.core.is_awake() {
                    if matches!(env.frame, Frame::UserStoppedSpeaking) {
                        self.core.on_activity(now);
                    }
                    link.push(env.meta, env.frame, env.direction).await;
                }
                // idle: swallow — no turns while asleep
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}
