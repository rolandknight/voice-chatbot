//! Chatbot activity on the speakerphone's LED ring (docs/specs/jabra-led.md).
//!
//! Three inputs the client already has — wake state, turn events, the
//! server's turn-mute — fold into one [`Phase`], rendered as the standard
//! HID telephony LEDs: off-hook (solid green), +ring (flashing green),
//! +mute (solid red). Asleep is dark; a bot speaking outranks asleep so
//! out-of-session audio (timer alarms) lights the ring while it plays.

pub mod hid;

use serde_json::Value;
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

/// Where the current turn stands, from the events socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Turn {
    Quiet,
    Thinking,
    Speaking,
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
}
