//! Chatbot activity on the speakerphone's LED ring (docs/specs/jabra-led.md).
//!
//! Three inputs the client already has — wake state, turn events, the
//! server's turn-mute — fold into one [`Phase`], rendered as the standard
//! HID telephony LEDs: off-hook (solid green) while awake, +mute (solid
//! red) when the mic is gated. The Ring usage (flashing green) is left
//! unused — on the Speak2 40 it double-beeps — so Thinking shows solid
//! green like Listening. Asleep is dark; a bot speaking outranks asleep so
//! out-of-session audio (timer alarms) lights the ring while it plays.
//! The same [`Indication`] feeds every [`LedSink`] a session has — the ring,
//! and the WS2812 strip with its own vocabulary (ADR-0008). Dropping every
//! [`LedController`] clone puts them all to rest (asleep), except a sink
//! whose write already failed: a gone device cannot be cleared. [`Phase::Offline`]
//! is the one phase the tracker never produces: the call loop sets it on the
//! strip between sessions, when there is no server to have a phase with.

pub mod hid;
pub mod strip;

use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use voice_chatbot_protocol::WakeState;

/// What the LEDs show. On the ring, Listening, Thinking and Speaking all
/// render solid green; the strip tells them apart (led/strip.rs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Armed, waiting for the wake word (wake mode only).
    Asleep,
    /// Awake: audio streams to the server.
    Listening,
    /// The user's turn is in; the server is working on it.
    Thinking,
    /// The bot's audio is playing.
    Speaking,
    /// No server: unreachable, or the connection dropped. Between sessions
    /// only, set by the call loop rather than the tracker.
    Offline,
}

/// A phase plus the server's turn-mute overlay: everything one LED write needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Indication {
    pub phase: Phase,
    pub muted: bool,
}

impl Indication {
    /// The telephony LED usages to set: (off-hook, ring, mute).
    pub fn bits(self) -> (bool, bool, bool) {
        let off_hook = matches!(
            self.phase,
            Phase::Listening | Phase::Thinking | Phase::Speaking
        );
        // The Ring usage is audible on the Speak2 40 (a double beep on assert),
        // so it is never set; Thinking renders solid green, same as Listening.
        let ring = false;
        let mute = self.muted && off_hook;
        (off_hook, ring, mute)
    }
}

/// Something that shows an [`Indication`]: the ring writes its three LED
/// bits, the strip picks an animation. Split from the devices so the driver
/// task is testable without hardware.
pub trait LedSink: Send {
    fn set(&mut self, indication: Indication) -> anyhow::Result<()>;
}

/// Where the current turn stands, from the events socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Quiet,
    Thinking,
    Speaking,
}

/// Clonable handle the events and wake tasks feed. A driver task owns the
/// sinks; state changes coalesce through a watch channel, so a burst of
/// events costs at most one write per sink per settled state. Dropping every
/// clone clears the sinks and ends the driver (its JoinHandle resolves after
/// the clear, so a session can bound its teardown). A sink whose write fails
/// is dropped for the session, since a gone device cannot be cleared either.
#[derive(Clone)]
pub struct LedController(Arc<Mutex<Shared>>);

struct Shared {
    tracker: PhaseTracker,
    seen: watch::Sender<Indication>,
}

impl LedController {
    pub fn start(
        sinks: Vec<Box<dyn LedSink>>,
        awake_at_start: bool,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        let tracker = PhaseTracker::new(awake_at_start);
        let (seen, changes) = watch::channel(tracker.indication());
        let driver = tokio::spawn(drive(sinks, changes));
        (Self(Arc::new(Mutex::new(Shared { tracker, seen }))), driver)
    }

    /// Feed one raw events-WebSocket text frame (same parse-it-yourself
    /// contract as `note_activity` and `dispatch_media` in events.rs).
    pub fn on_event(&self, input: &str) {
        let Ok(message) = serde_json::from_str::<Value>(input) else {
            return;
        };
        let Some(kind) = message.get("type").and_then(Value::as_str) else {
            return;
        };
        let payload = message.get("payload").cloned().unwrap_or(Value::Null);
        let mut shared = self.0.lock().unwrap();
        if let Some(indication) = shared.tracker.on_event(kind, &payload) {
            let _ = shared.seen.send(indication);
        }
    }

    /// Feed a locally detected wake state change (wake::spawn).
    pub fn on_wake(&self, state: &WakeState) {
        let mut shared = self.0.lock().unwrap();
        if let Some(indication) = shared.tracker.on_wake(state) {
            let _ = shared.seen.send(indication);
        }
    }
}

/// Write every settled state change to every sink; on channel close (all
/// handles gone, i.e. the session ended) leave them dark. The writes are
/// small syscalls against device nodes, so they run on the blocking pool.
async fn drive(mut sinks: Vec<Box<dyn LedSink>>, mut changes: watch::Receiver<Indication>) {
    let mut shown: Option<Indication> = None;
    loop {
        let wanted = *changes.borrow_and_update();
        if shown != Some(wanted) {
            sinks = tokio::task::spawn_blocking(move || write_all(sinks, wanted))
                .await
                .expect("led sink panicked");
            if sinks.is_empty() {
                return;
            }
            shown = Some(wanted);
        }
        if changes.changed().await.is_err() {
            break;
        }
    }
    // Asleep is the resting state: dark on the ring, the idle sweep on the
    // strip (whose owner shows Offline instead if the server went away).
    let rest = Indication {
        phase: Phase::Asleep,
        muted: false,
    };
    // The sinks are dropped on the blocking pool too, in case one blocks.
    let _ = tokio::task::spawn_blocking(move || drop(write_all(sinks, rest))).await;
}

/// Show `wanted` on every sink, keeping the ones that succeeded. Unplugging
/// the speakerphone ends the audio session too, and the next session
/// re-opens the device; a failed sink just goes dark until then.
fn write_all(sinks: Vec<Box<dyn LedSink>>, wanted: Indication) -> Vec<Box<dyn LedSink>> {
    sinks
        .into_iter()
        .filter_map(|mut sink| match sink.set(wanted) {
            Ok(()) => Some(sink),
            Err(error) => {
                tracing::debug!(%error, "led write failed; no more writes to that device this session");
                None
            }
        })
        .collect()
}

/// Folds wake state, turn events and turn-mute into the shown [`Indication`].
#[derive(Debug)]
pub struct PhaseTracker {
    awake: bool,
    turn: Turn,
    muted: bool,
}

impl PhaseTracker {
    /// Push mode has no wake gate and starts awake; wake mode starts asleep.
    pub fn new(awake: bool) -> Self {
        Self {
            awake,
            turn: Turn::Quiet,
            muted: false,
        }
    }

    pub fn indication(&self) -> Indication {
        let phase = if self.turn == Turn::Speaking {
            Phase::Speaking
        } else if !self.awake {
            Phase::Asleep
        } else if self.turn == Turn::Thinking {
            Phase::Thinking
        } else {
            Phase::Listening
        };
        Indication {
            phase,
            muted: self.muted,
        }
    }

    fn set_awake(&mut self, awake: bool) {
        self.awake = awake;
        if !awake {
            // The mute belongs to a turn; a closed session has no turn.
            self.muted = false;
        }
    }

    /// Apply one events-WebSocket frame; the new indication if it changed.
    pub fn on_event(&mut self, kind: &str, payload: &Value) -> Option<Indication> {
        let before = self.indication();
        match kind {
            "rtf-user-transcription" => {
                let done = payload
                    .get("final")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                self.turn = if done { Turn::Thinking } else { Turn::Quiet };
            }
            "rtf-function-call-start" => self.turn = Turn::Thinking,
            "rtf-bot-started-speaking" => self.turn = Turn::Speaking,
            "rtf-bot-stopped-speaking" => self.turn = Turn::Quiet,
            "rtf-user-mute-started" => self.muted = true,
            "rtf-user-mute-stopped" => self.muted = false,
            voice_chatbot_protocol::WAKE_EVENT => {
                if let Ok(state) = WakeState::from_payload(payload) {
                    self.set_awake(matches!(state, WakeState::Awake { .. }));
                }
            }
            _ => {}
        }
        let after = self.indication();
        (after != before).then_some(after)
    }

    /// Apply a locally detected wake change; the new indication if it changed.
    pub fn on_wake(&mut self, state: &WakeState) -> Option<Indication> {
        let before = self.indication();
        self.set_awake(matches!(state, WakeState::Awake { .. }));
        let after = self.indication();
        (after != before).then_some(after)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn bits_render_each_phase() {
        let show = |phase, muted| Indication { phase, muted }.bits();
        assert_eq!(show(Phase::Asleep, false), (false, false, false));
        assert_eq!(show(Phase::Listening, false), (true, false, false));
        // Thinking renders solid green, same as listening: the Ring usage is
        // audible on the Speak2 40 (a double beep), so it is never asserted.
        assert_eq!(show(Phase::Thinking, false), (true, false, false));
        assert_eq!(show(Phase::Thinking, true), (true, false, true));
        assert_eq!(show(Phase::Speaking, false), (true, false, false));
        assert_eq!(show(Phase::Speaking, true), (true, false, true));
        assert_eq!(
            show(Phase::Asleep, true),
            (false, false, false),
            "mute never shows on a dark ring"
        );
        assert_eq!(
            show(Phase::Offline, false),
            (false, false, false),
            "no server: the ring is dark"
        );
    }

    #[test]
    fn wake_session_lights_and_darkens_the_ring() {
        let mut tracker = PhaseTracker::new(false);
        assert_eq!(tracker.indication().phase, Phase::Asleep);
        let lit = tracker
            .on_wake(&WakeState::Awake {
                model: "hey_marvin".into(),
                score: 0.9,
                persona: None,
            })
            .expect("asleep -> awake is a change");
        assert_eq!(lit.phase, Phase::Listening);
        let dark = tracker.on_wake(&WakeState::Asleep).expect("a change");
        assert_eq!(dark.phase, Phase::Asleep);
    }

    #[test]
    fn a_turn_flows_listening_thinking_speaking_listening() {
        let mut tracker = PhaseTracker::new(true);
        assert!(
            tracker
                .on_event(
                    "rtf-user-transcription",
                    &json!({"text": "hi", "final": false})
                )
                .is_none(),
            "partials mean the user is still talking: keep listening"
        );
        let thinking = tracker
            .on_event(
                "rtf-user-transcription",
                &json!({"text": "hi", "final": true}),
            )
            .unwrap();
        assert_eq!(thinking.phase, Phase::Thinking);
        assert!(
            tracker
                .on_event("rtf-function-call-start", &json!({}))
                .is_none(),
            "a running tool is still thinking"
        );
        let speaking = tracker
            .on_event("rtf-bot-started-speaking", &json!({}))
            .unwrap();
        assert_eq!(speaking.phase, Phase::Speaking);
        let back = tracker
            .on_event("rtf-bot-stopped-speaking", &json!({}))
            .unwrap();
        assert_eq!(back.phase, Phase::Listening);
    }

    #[test]
    fn turn_mute_overlays_red_until_lifted() {
        let mut tracker = PhaseTracker::new(true);
        let muted = tracker
            .on_event("rtf-user-mute-started", &json!({}))
            .unwrap();
        assert!(muted.muted);
        assert_eq!(
            muted.phase,
            Phase::Listening,
            "mute is an overlay, not a phase"
        );
        let lifted = tracker
            .on_event("rtf-user-mute-stopped", &json!({}))
            .unwrap();
        assert!(!lifted.muted);
    }

    #[test]
    fn falling_asleep_drops_a_stale_mute() {
        let mut tracker = PhaseTracker::new(true);
        tracker.on_event("rtf-user-mute-started", &json!({}));
        tracker.on_wake(&WakeState::Asleep);
        assert!(!tracker.indication().muted);
    }

    #[test]
    fn alarm_while_asleep_shows_speaking_then_dark() {
        let mut tracker = PhaseTracker::new(false);
        let alarm = tracker
            .on_event("rtf-bot-started-speaking", &json!({}))
            .unwrap();
        assert_eq!(alarm.phase, Phase::Speaking);
        let done = tracker
            .on_event("rtf-bot-stopped-speaking", &json!({}))
            .unwrap();
        assert_eq!(done.phase, Phase::Asleep, "back to dark, not to listening");
    }

    #[test]
    fn wake_frames_on_the_events_socket_work_too() {
        let mut tracker = PhaseTracker::new(false);
        let lit = tracker
            .on_event(
                "wake",
                &json!({"state": "awake", "model": "hey_marvin", "score": 0.9}),
            )
            .unwrap();
        assert_eq!(lit.phase, Phase::Listening);
    }

    #[test]
    fn unknown_events_change_nothing() {
        let mut tracker = PhaseTracker::new(true);
        assert!(tracker
            .on_event("rtf-bot-text", &json!({"text": "hi"}))
            .is_none());
        assert!(tracker.on_event("media", &json!({})).is_none());
    }

    /// Records what it is asked to show, tagged with its id; can be told to
    /// fail from a given write on, like a device that was unplugged.
    struct RecordingSink {
        id: usize,
        sent: std::sync::mpsc::Sender<(usize, Indication)>,
        fail_from_write: Option<usize>,
        writes: usize,
    }

    impl RecordingSink {
        fn new(id: usize, sent: &std::sync::mpsc::Sender<(usize, Indication)>) -> Self {
            Self {
                id,
                sent: sent.clone(),
                fail_from_write: None,
                writes: 0,
            }
        }
    }

    impl LedSink for RecordingSink {
        fn set(&mut self, indication: Indication) -> anyhow::Result<()> {
            self.writes += 1;
            if self.fail_from_write.is_some_and(|n| self.writes >= n) {
                anyhow::bail!("device gone");
            }
            self.sent.send((self.id, indication)).unwrap();
            Ok(())
        }
    }

    const DARK: Indication = Indication {
        phase: Phase::Asleep,
        muted: false,
    };
    const SPEAKING: Indication = Indication {
        phase: Phase::Speaking,
        muted: false,
    };

    #[tokio::test(flavor = "multi_thread")]
    async fn controller_writes_changes_and_clears_on_drop() {
        let timeout = std::time::Duration::from_secs(5);
        let (sent, written) = std::sync::mpsc::channel();
        let sink: Box<dyn LedSink> = Box::new(RecordingSink::new(0, &sent));
        let (led, done) = LedController::start(vec![sink], false);
        assert_eq!(
            written.recv_timeout(timeout).unwrap(),
            (0, DARK),
            "the starting state is written, clearing a crashed predecessor's leds"
        );
        led.on_event(r#"{"type":"rtf-bot-started-speaking","payload":{}}"#);
        assert_eq!(written.recv_timeout(timeout).unwrap(), (0, SPEAKING));
        led.on_event(r#"{"type":"rtf-bot-text","payload":{"text":"hi"}}"#);
        led.on_event("not json at all");
        drop(led);
        assert_eq!(
            written.recv_timeout(timeout).unwrap(),
            (0, DARK),
            "dropping the last handle darkens the leds"
        );
        done.await.unwrap();
        assert!(
            written.try_recv().is_err(),
            "no write happened for the no-change events"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn controller_fans_out_to_every_sink_and_drops_one_that_fails() {
        let timeout = std::time::Duration::from_secs(5);
        let (sent, written) = std::sync::mpsc::channel();
        let ring = RecordingSink::new(0, &sent);
        let mut strip = RecordingSink::new(1, &sent);
        strip.fail_from_write = Some(2);
        let sinks: Vec<Box<dyn LedSink>> = vec![Box::new(ring), Box::new(strip)];
        let (led, done) = LedController::start(sinks, true);
        let mut first = [
            written.recv_timeout(timeout).unwrap(),
            written.recv_timeout(timeout).unwrap(),
        ];
        first.sort_by_key(|(id, _)| *id);
        let listening = Indication {
            phase: Phase::Listening,
            muted: false,
        };
        assert_eq!(
            first,
            [(0, listening), (1, listening)],
            "both sinks get the start state"
        );
        led.on_event(r#"{"type":"rtf-bot-started-speaking","payload":{}}"#);
        assert_eq!(
            written.recv_timeout(timeout).unwrap(),
            (0, SPEAKING),
            "the ring still gets the change after the strip failed"
        );
        led.on_event(r#"{"type":"rtf-bot-stopped-speaking","payload":{}}"#);
        assert_eq!(written.recv_timeout(timeout).unwrap(), (0, listening));
        drop(led);
        assert_eq!(written.recv_timeout(timeout).unwrap(), (0, DARK));
        done.await.unwrap();
        assert!(
            written.try_recv().is_err(),
            "the failed sink is never written again, not even the clear"
        );
    }
}
