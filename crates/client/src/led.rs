//! Chatbot activity on the speakerphone's LED ring (docs/specs/jabra-led.md).
//!
//! Three inputs the client already has — wake state, turn events, the
//! server's turn-mute — fold into one [`Phase`], rendered as the standard
//! HID telephony LEDs: off-hook (solid green), +ring (flashing green),
//! +mute (solid red). Asleep is dark; a bot speaking outranks asleep so
//! out-of-session audio (timer alarms) lights the ring while it plays.
//! Dropping every [`LedController`] clone clears the ring, unless a write
//! already failed — a gone device cannot be cleared.

pub mod hid;

use serde_json::Value;
use std::sync::{Arc, Mutex};
use tokio::sync::watch;
use voice_chatbot_protocol::WakeState;

/// What the ring shows. Speaking and Listening render the same today (solid
/// green); they stay distinct because the derivation differs and a later
/// device may render them differently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Asleep,
    Listening,
    Thinking,
    Speaking,
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
        let off_hook = self.phase != Phase::Asleep;
        let ring = self.phase == Phase::Thinking;
        let mute = self.muted && off_hook;
        (off_hook, ring, mute)
    }
}

/// One write of the three LED bits to whatever renders them. Split from the
/// device so the driver task is testable without hardware.
pub trait LedSink: Send {
    fn set(&mut self, off_hook: bool, ring: bool, mute: bool) -> anyhow::Result<()>;
}

/// Where the current turn stands, from the events socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Quiet,
    Thinking,
    Speaking,
}

/// Clonable handle the events and wake tasks feed. A driver task owns the
/// sink; state changes coalesce through a watch channel, so a burst of
/// events costs at most one write per settled state. Dropping every clone
/// clears the ring and ends the driver (its JoinHandle resolves after the
/// clear, so a session can bound its teardown) — unless a write already
/// failed, since a gone device cannot be cleared.
#[derive(Clone)]
pub struct LedController(Arc<Mutex<Shared>>);

struct Shared {
    tracker: PhaseTracker,
    seen: watch::Sender<Indication>,
}

impl LedController {
    pub fn start(
        sink: Box<dyn LedSink>,
        awake_at_start: bool,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        let tracker = PhaseTracker::new(awake_at_start);
        let (seen, changes) = watch::channel(tracker.indication());
        let driver = tokio::spawn(drive(sink, changes));
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

/// Write every settled state change; on channel close (all handles gone,
/// i.e. the session ended) leave the ring dark. hidraw writes are small but
/// still syscalls against a device node, so they run on the blocking pool.
async fn drive(mut sink: Box<dyn LedSink>, mut changes: watch::Receiver<Indication>) {
    let mut shown: Option<Indication> = None;
    loop {
        let wanted = *changes.borrow_and_update();
        if shown != Some(wanted) {
            let (off_hook, ring, mute) = wanted.bits();
            let (returned, result) = tokio::task::spawn_blocking(move || {
                let result = sink.set(off_hook, ring, mute);
                (sink, result)
            })
            .await
            .expect("led sink panicked");
            sink = returned;
            if let Err(error) = result {
                // Unplugging the speakerphone ends the audio session too;
                // the next session re-opens the device. Just go dark.
                tracing::debug!(%error, "led write failed; no leds for this session");
                return;
            }
            shown = Some(wanted);
        }
        if changes.changed().await.is_err() {
            break;
        }
    }
    let _ = tokio::task::spawn_blocking(move || sink.set(false, false, false)).await;
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
        assert_eq!(show(Phase::Thinking, false), (true, true, false));
        assert_eq!(show(Phase::Speaking, false), (true, false, false));
        assert_eq!(show(Phase::Speaking, true), (true, false, true));
        assert_eq!(
            show(Phase::Asleep, true),
            (false, false, false),
            "mute never shows on a dark ring"
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

    struct RecordingSink(std::sync::mpsc::Sender<(bool, bool, bool)>);

    impl LedSink for RecordingSink {
        fn set(&mut self, off_hook: bool, ring: bool, mute: bool) -> anyhow::Result<()> {
            self.0.send((off_hook, ring, mute)).unwrap();
            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn controller_writes_changes_and_clears_on_drop() {
        let timeout = std::time::Duration::from_secs(5);
        let (sink_tx, written) = std::sync::mpsc::channel();
        let (led, done) = LedController::start(Box::new(RecordingSink(sink_tx)), false);
        assert_eq!(
            written.recv_timeout(timeout).unwrap(),
            (false, false, false),
            "the starting state is written, clearing a crashed predecessor's leds"
        );
        led.on_event(r#"{"type":"rtf-bot-started-speaking","payload":{}}"#);
        assert_eq!(written.recv_timeout(timeout).unwrap(), (true, false, false));
        led.on_event(r#"{"type":"rtf-bot-text","payload":{"text":"hi"}}"#);
        led.on_event("not json at all");
        drop(led);
        assert_eq!(
            written.recv_timeout(timeout).unwrap(),
            (false, false, false),
            "dropping the last handle darkens the ring"
        );
        done.await.unwrap();
        assert!(
            written.try_recv().is_err(),
            "no write happened for the no-change events"
        );
    }
}
