//! What the media source should be doing right now.
//!
//! Two independent concerns the previous player conflated into one `pause`
//! property: how loud media is (a gain the mixer ramps) and whether the
//! decoder is running at all (transport). Keeping them apart is what lets a
//! live stream duck without losing its place at the live edge, and what stops
//! a deliberate pause being undone by the end of the assistant's next reply.

use crate::media::gain::{DUCKED, FULL};

/// Whether the decoder should be pulling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Transport {
    Running,
    Stopped,
}

/// The media source's current intent.
pub struct Duck {
    playing: bool,
    live: bool,
    bot_speaking: bool,
    user_paused: bool,
}

impl Duck {
    pub fn new() -> Self {
        Self {
            playing: false,
            live: false,
            bot_speaking: false,
            user_paused: false,
        }
    }

    /// Begin a stream. Returns true when it must start *at* the ducked gain
    /// rather than ramp down to it — the common case, since radio is asked for
    /// by voice and the reply is still being spoken when the stream opens.
    pub fn start(&mut self, live: bool) -> bool {
        self.playing = true;
        self.live = live;
        self.user_paused = false;
        self.bot_speaking && live
    }

    pub fn stop(&mut self) {
        self.playing = false;
        self.user_paused = false;
    }

    pub fn set_bot_speaking(&mut self, speaking: bool) {
        self.bot_speaking = speaking;
    }

    pub fn set_user_paused(&mut self, paused: bool) {
        self.user_paused = paused;
    }

    pub fn is_playing(&self) -> bool {
        self.playing
    }

    /// Whether the assistant is mid-reply, for `after_speech` clips.
    pub fn is_playing_speech(&self) -> bool {
        self.bot_speaking
    }

    pub fn gain(&self) -> f32 {
        if !self.playing || self.user_paused {
            return 0.0;
        }
        match (self.bot_speaking, self.live) {
            (true, true) => DUCKED,
            (true, false) => 0.0,
            (false, _) => FULL,
        }
    }

    pub fn transport(&self) -> Transport {
        if !self.playing || self.user_paused {
            return Transport::Stopped;
        }
        // A live stream must keep consuming to stay at the live edge; a
        // recorded one stops so it resumes exactly where it left off.
        match (self.bot_speaking, self.live) {
            (true, false) => Transport::Stopped,
            _ => Transport::Running,
        }
    }
}

impl Default for Duck {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::gain::{DUCKED, FULL};

    #[test]
    fn nothing_playing_is_silent_and_stopped() {
        let duck = Duck::new();
        assert!(!duck.is_playing());
        assert_eq!(duck.gain(), 0.0);
        assert_eq!(duck.transport(), Transport::Stopped);
    }

    #[test]
    fn a_live_stream_ducks_by_gain_and_keeps_decoding() {
        let mut duck = Duck::new();
        duck.start(true);
        assert_eq!(duck.gain(), FULL);

        duck.set_bot_speaking(true);
        assert_eq!(duck.gain(), DUCKED, "live radio stays audible underneath");
        assert_eq!(
            duck.transport(),
            Transport::Running,
            "and must keep decoding to stay at the live edge"
        );

        duck.set_bot_speaking(false);
        assert_eq!(duck.gain(), FULL);
        assert_eq!(duck.transport(), Transport::Running);
    }

    #[test]
    fn a_recorded_stream_stops_its_decoder_so_it_resumes_in_place() {
        let mut duck = Duck::new();
        duck.start(false);

        duck.set_bot_speaking(true);
        assert_eq!(duck.gain(), 0.0);
        assert_eq!(duck.transport(), Transport::Stopped);

        duck.set_bot_speaking(false);
        assert_eq!(duck.gain(), FULL);
        assert_eq!(duck.transport(), Transport::Running);
    }

    #[test]
    fn a_stream_started_mid_reply_begins_ducked_without_a_ramp() {
        let mut duck = Duck::new();
        duck.set_bot_speaking(true);
        let jump = duck.start(true);
        assert!(jump, "there is no earlier level to fade from");
        assert_eq!(duck.gain(), DUCKED);
        assert_eq!(duck.transport(), Transport::Running);

        // It comes up only when the reply ends.
        duck.set_bot_speaking(false);
        assert_eq!(duck.gain(), FULL);
    }

    #[test]
    fn a_stream_started_in_silence_does_not_ask_for_a_jump() {
        let mut duck = Duck::new();
        assert!(!duck.start(true));
        assert_eq!(duck.gain(), FULL);
    }

    /// The bug the previous player had: one `pause` property served both concerns,
    /// so the next `rtf-bot-stopped-speaking` resumed a deliberate pause.
    #[test]
    fn an_explicit_pause_survives_the_assistant_speaking_afterwards() {
        let mut duck = Duck::new();
        duck.start(true);
        duck.set_user_paused(true);
        assert_eq!(duck.transport(), Transport::Stopped);

        duck.set_bot_speaking(true);
        duck.set_bot_speaking(false);
        assert_eq!(duck.gain(), 0.0, "still paused");
        assert_eq!(duck.transport(), Transport::Stopped);

        duck.set_user_paused(false);
        assert_eq!(duck.gain(), FULL);
        assert_eq!(duck.transport(), Transport::Running);
    }

    /// Stopping clears `playing`, and the stream started afterwards plays: a
    /// pause the user asked for before the stop does not carry into it.
    #[test]
    fn a_fresh_start_after_a_pause_is_playable() {
        let mut duck = Duck::new();
        duck.start(true);
        duck.set_user_paused(true);
        duck.stop();
        assert!(!duck.is_playing());

        // A fresh stream is not still paused.
        duck.start(true);
        assert_eq!(duck.transport(), Transport::Running);
        assert_eq!(duck.gain(), FULL);
    }

    /// A recorded show asked for mid-reply: silent and stopped until the
    /// assistant finishes, and it needs no jump because it was never audible.
    #[test]
    fn a_recorded_stream_started_mid_reply_waits_silently() {
        let mut duck = Duck::new();
        duck.set_bot_speaking(true);
        assert!(
            !duck.start(false),
            "nothing to jump from; it is silent anyway"
        );
        assert_eq!(duck.gain(), 0.0);
        assert_eq!(duck.transport(), Transport::Stopped);

        duck.set_bot_speaking(false);
        assert_eq!(duck.gain(), FULL);
        assert_eq!(duck.transport(), Transport::Running);
    }
}
