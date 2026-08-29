//! The media source's shared control state, between the control thread and
//! the CPAL output callback: its level, plus two one-shot requests (jump,
//! flush).
//!
//! The callback may not lock, so the target is an `f32` stored as bits in an
//! `AtomicU32` and the ramp is walked one step per sample. A *jump* is the
//! start-ducked case: a stream opening while the assistant already speaks has
//! no earlier level to fade from, so it begins at the target outright. A
//! *flush* is the source-switch case: whatever is already queued in the
//! mixer belongs to the previous stream and must be discarded, not played out
//! under the new one's gain.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Linear gain for −18 dB: what a live stream ducks to while the assistant speaks.
pub const DUCKED: f32 = 0.126;
/// Unducked playback.
pub const FULL: f32 = 1.0;

/// How long a full-scale (0 → 1) ramp takes. Short enough not to read as a
/// delay, long enough to avoid a zipper-noise step discontinuity.
const RAMP: Duration = Duration::from_millis(80);

/// Per-sample gain increment for `rate`.
pub fn step_for(sample_rate: u32) -> f32 {
    1.0 / (RAMP.as_secs_f32() * sample_rate as f32)
}

/// Move `current` one `step` toward `target`, never overshooting.
pub fn advance(current: f32, target: f32, step: f32) -> f32 {
    if current < target {
        (current + step).min(target)
    } else {
        (current - step).max(target)
    }
}

/// A gain target shared by the control thread and the audio callback.
#[derive(Clone)]
pub struct Gain {
    target: Arc<AtomicU32>,
    jump: Arc<AtomicBool>,
    flush: Arc<AtomicBool>,
}

impl Gain {
    pub fn new(value: f32) -> Self {
        Self {
            target: Arc::new(AtomicU32::new(value.to_bits())),
            jump: Arc::new(AtomicBool::new(false)),
            flush: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Fade to `value` over the ramp.
    pub fn ramp_to(&self, value: f32) {
        self.target.store(value.to_bits(), Ordering::Relaxed);
    }

    /// Be at `value` on the next sample, with no fade.
    pub fn jump_to(&self, value: f32) {
        self.target.store(value.to_bits(), Ordering::Relaxed);
        self.jump.store(true, Ordering::Release);
    }

    pub fn target(&self) -> f32 {
        f32::from_bits(self.target.load(Ordering::Relaxed))
    }

    /// True once per [`Self::jump_to`]; clears the request.
    pub fn take_jump(&self) -> bool {
        self.jump.swap(false, Ordering::Acquire)
    }

    /// Discard whatever media is already queued in the mixer. Used when the
    /// source changes: the buffered chunks belong to the previous stream and
    /// must not be heard under the new one.
    pub fn flush(&self) {
        self.flush.store(true, Ordering::Release);
    }

    /// True once per [`Self::flush`]; clears the request.
    pub fn take_flush(&self) -> bool {
        self.flush.swap(false, Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_walks_toward_the_target_and_stops_dead_on_it() {
        // Rising: never overshoots.
        assert_eq!(advance(0.0, 1.0, 0.25), 0.25);
        assert_eq!(advance(0.9, 1.0, 0.25), 1.0);
        // Falling: never undershoots.
        assert_eq!(advance(1.0, 0.126, 0.25), 0.75);
        assert_eq!(advance(0.2, 0.126, 0.25), 0.126);
        // Already there: stays.
        assert_eq!(advance(0.5, 0.5, 0.25), 0.5);
    }

    #[test]
    fn a_full_scale_ramp_takes_eighty_milliseconds_of_samples() {
        // An exact sample count would be brittle: `step_for` is not exactly
        // representable in f32, so accumulating it lands a hair either side of
        // the target. The property that matters is the duration.
        for rate in [16_000, 44_100, 48_000] {
            let step = step_for(rate);
            let mut current = 0.0;
            let mut samples = 0;
            while current < 1.0 {
                current = advance(current, 1.0, step);
                samples += 1;
            }
            let ms = samples as f32 / rate as f32 * 1000.0;
            assert!(
                (79.0..=81.0).contains(&ms),
                "a full-scale ramp at {rate} Hz took {ms} ms ({samples} samples)"
            );
        }
    }

    #[test]
    fn ramp_to_sets_a_target_without_asking_for_a_jump() {
        let gain = Gain::new(FULL);
        gain.ramp_to(DUCKED);
        assert_eq!(gain.target(), DUCKED);
        assert!(!gain.take_jump());
    }

    #[test]
    fn jump_to_requests_a_jump_exactly_once() {
        let gain = Gain::new(FULL);
        gain.jump_to(DUCKED);
        assert_eq!(gain.target(), DUCKED);
        assert!(gain.take_jump(), "the jump must be seen once");
        assert!(!gain.take_jump(), "and never twice");
    }

    #[test]
    fn a_clone_shares_one_state_so_the_callback_sees_control_thread_writes() {
        let control = Gain::new(FULL);
        let callback = control.clone();
        control.ramp_to(DUCKED);
        assert_eq!(callback.target(), DUCKED);
    }
}
