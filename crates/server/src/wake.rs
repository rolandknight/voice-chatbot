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

use flowcat_core::processor::frame::{AudioFrame, Frame, FrameMeta, StartParams};
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
            if open {
                state.arm_wake_grace();
            }
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

// ===========================================================================
// WakeGrace — one turn for "wake phrase … pause … command"
// ===========================================================================

/// A wake arm older than this is ignored: the wake event and the speech edge
/// it belongs to arrive within a second of each other (the native client
/// reports over the events socket while the audio rides WebRTC).
const WAKE_ARM_MAX_AGE: Duration = Duration::from_secs(5);

/// Sits after the wake gate (or alone, for the native client that detects
/// on-device) and ahead of the SpeechGate. With the default 0.2 s VAD stop, a
/// caller who pauses after the wake phrase — "Hey Marvin. … What time is it?"
/// — produces two finals: the bare wake phrase gets a reply of its own, and
/// the two turns then race each other (the double reply in docs/todo.md).
///
/// When [`CallState::arm_wake_grace`] has fired (the server gate, or the
/// client's `wake` event via `main.rs::apply_client_wake`), the FIRST
/// `UserStoppedSpeaking` is held for `grace`. Audio keeps flowing, so the
/// SpeechGate stays open and the STT keeps accumulating. Speech resuming inside
/// the window swallows both edges (one utterance); silence past the window
/// releases the held edge so a bare "Hey Marvin" still gets its answer. Later
/// turns in the session are untouched — no added latency there.
pub struct WakeGrace {
    state: Arc<CallState>,
    grace: Duration,
    /// The held end-of-speech edge and when it must be released.
    held: Option<(FrameMeta, Instant)>,
}

impl WakeGrace {
    pub fn new(state: Arc<CallState>, grace: Duration) -> Self {
        Self {
            state,
            grace,
            held: None,
        }
    }

    async fn release(&mut self, link: &Link, why: &str) {
        if let Some((meta, _)) = self.held.take() {
            tracing::debug!(why, "wake grace: releasing the held end-of-speech edge");
            link.push(
                meta,
                Frame::UserStoppedSpeaking,
                flowcat_core::processor::frame::Direction::Downstream,
            )
            .await;
        }
    }
}

#[async_trait]
impl FrameProcessor for WakeGrace {
    fn name(&self) -> &str {
        "WakeGrace"
    }

    /// Barge-in: nothing to hold on to.
    async fn on_interruption(&mut self) -> Result<()> {
        self.held = None;
        Ok(())
    }

    async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
        match &env.frame {
            Frame::InputAudio(_) => {
                // Lazy expiry on the audio clock (frames keep arriving while the
                // caller is silent), the same way the wake session expires.
                if matches!(self.held, Some((_, deadline)) if Instant::now() >= deadline) {
                    self.release(link, "silence past the grace window").await;
                }
                link.push(env.meta, env.frame, env.direction).await;
            }
            Frame::UserStoppedSpeaking => {
                if self.held.is_some() {
                    // Can't happen (a Started clears the hold) — never hold two.
                    self.release(link, "second edge while holding").await;
                }
                if self.state.take_wake_grace(WAKE_ARM_MAX_AGE) {
                    tracing::debug!(
                        grace_ms = self.grace.as_millis() as u64,
                        "wake grace: holding the wake phrase's end-of-speech edge"
                    );
                    self.held = Some((env.meta, Instant::now() + self.grace));
                } else {
                    link.push(env.meta, env.frame, env.direction).await;
                }
            }
            Frame::UserStartedSpeaking if self.held.is_some() => {
                // The command followed the wake phrase inside the window: merge
                // the two into one utterance by dropping both edges.
                tracing::debug!("wake grace: speech resumed, merging into one turn");
                self.held = None;
            }
            _ => link.push(env.meta, env.frame, env.direction).await,
        }
        Ok(())
    }
}

#[cfg(test)]
mod grace_tests {
    use super::*;
    use flowcat_core::pipeline::{Pipeline, PipelineTask, PipelineTaskParams};
    use std::sync::Mutex;

    /// Records the turn edges that make it past the grace stage.
    #[derive(Clone, Default)]
    struct EdgeCapture(Arc<Mutex<Vec<&'static str>>>);
    #[async_trait]
    impl FrameProcessor for EdgeCapture {
        fn name(&self) -> &str {
            "EdgeCapture"
        }
        async fn process_frame(&mut self, env: Envelope, link: &Link) -> Result<()> {
            if matches!(
                env.frame,
                Frame::UserStartedSpeaking | Frame::UserStoppedSpeaking
            ) {
                self.0.lock().unwrap().push(env.frame.name());
            }
            link.push(env.meta, env.frame, env.direction).await;
            Ok(())
        }
    }

    fn audio() -> Frame {
        Frame::InputAudio(Arc::new(AudioFrame::mono(vec![0i16; 320], 16_000)))
    }

    /// Feed `steps` through WakeGrace → EdgeCapture; a `None` step sleeps past
    /// the grace window.
    async fn run(
        state: Arc<CallState>,
        grace_ms: u64,
        steps: Vec<Option<Frame>>,
    ) -> Vec<&'static str> {
        let cap = EdgeCapture::default();
        let task = PipelineTask::new(
            Pipeline::new(vec![
                Box::new(WakeGrace::new(state, Duration::from_millis(grace_ms))),
                Box::new(cap.clone()),
            ]),
            PipelineTaskParams::default(),
            vec![],
        );
        // Drive the chain live (the grace deadline is wall-clock, so the sleep
        // step must happen while the stage is running, not before).
        let sender = task.queue_sender();
        let running = tokio::spawn(task.run());
        for step in steps {
            match step {
                Some(f) => sender.send(f).unwrap(),
                None => tokio::time::sleep(Duration::from_millis(grace_ms * 3)).await,
            }
        }
        sender.send(Frame::End { reason: None }).unwrap();
        tokio::time::timeout(Duration::from_secs(5), running)
            .await
            .expect("grace pipeline timed out")
            .expect("join")
            .expect("run ok");
        let out = cap.0.lock().unwrap().clone();
        out
    }

    #[tokio::test]
    async fn command_after_a_pause_merges_into_the_wake_turn() {
        let state = Arc::new(CallState::default());
        state.arm_wake_grace();
        let edges = run(
            state,
            50,
            vec![
                Some(Frame::UserStartedSpeaking), // "Hey Marvin"
                Some(audio()),
                Some(Frame::UserStoppedSpeaking), // pause → held
                Some(audio()),
                Some(Frame::UserStartedSpeaking), // "what time is it" → both edges dropped
                Some(audio()),
                Some(Frame::UserStoppedSpeaking), // the real end of the turn
            ],
        )
        .await;
        assert_eq!(edges, vec!["UserStartedSpeaking", "UserStoppedSpeaking"]);
    }

    #[tokio::test]
    async fn a_bare_wake_phrase_is_released_after_the_window() {
        let state = Arc::new(CallState::default());
        state.arm_wake_grace();
        let edges = run(
            state,
            50,
            vec![
                Some(Frame::UserStartedSpeaking),
                Some(audio()),
                Some(Frame::UserStoppedSpeaking), // held
                None,                             // silence past the window
                Some(audio()),                    // the audio clock releases it
                Some(Frame::UserStartedSpeaking), // a later, separate turn
                Some(Frame::UserStoppedSpeaking),
            ],
        )
        .await;
        assert_eq!(
            edges,
            vec![
                "UserStartedSpeaking",
                "UserStoppedSpeaking",
                "UserStartedSpeaking",
                "UserStoppedSpeaking"
            ]
        );
    }

    #[tokio::test]
    async fn without_a_wake_the_edges_pass_straight_through() {
        let edges = run(
            Arc::new(CallState::default()),
            50,
            vec![
                Some(Frame::UserStartedSpeaking),
                Some(Frame::UserStoppedSpeaking),
                Some(Frame::UserStartedSpeaking),
                Some(Frame::UserStoppedSpeaking),
            ],
        )
        .await;
        assert_eq!(edges.len(), 4);
    }

    #[test]
    fn a_stale_arm_is_ignored() {
        let state = CallState::default();
        assert!(!state.take_wake_grace(WAKE_ARM_MAX_AGE));
        state.arm_wake_grace();
        assert!(!state.take_wake_grace(Duration::ZERO), "older than max age");
        state.arm_wake_grace();
        assert!(state.take_wake_grace(WAKE_ARM_MAX_AGE));
        assert!(!state.take_wake_grace(WAKE_ARM_MAX_AGE), "consumed once");
    }
}
