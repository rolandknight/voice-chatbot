# Embedded Media Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decode BBC radio, shows and sound effects with `ffmpeg` inside the client and mix them into the CPAL output stream the call already owns, so media plays on the call's speakerphone and ducks by gain instead of an mpv IPC pause.

**Architecture:** `ffmpeg` becomes a decoder, never a player: it is handed no audio device and writes raw `s16le` mono PCM at the device's rate to stdout. A feeder thread pumps that into a second bounded channel read by the CPAL output callback, which sums it with the call's TTS and applies a ramped gain. The hardware callback is the only clock, so nothing drifts; OS pipe backpressure paces the decoder and gives "pause" for free.

**Tech Stack:** Rust 2021, `cpal` 0.18.2 (pinned), `std::sync::mpsc::SyncSender`, `std::process` + a `std::thread` feeder, `ffmpeg` as a system binary. No new crate dependencies.

**Spec:** `docs/superpowers/specs/2026-08-28-embedded-media-player-design.md`

## Global Constraints

- Audio crossing the `audio` module's channel boundary is **mono `i16`** this phase (`crates/client/src/audio.rs:1-6`). Stereo is explicitly deferred; do not widen `output_tx`.
- The CPAL output callback must not allocate, block, or lock. `try_recv` and atomics only.
- Duck level is **−18 dB = 0.126 linear**. Ramp is **80 ms** at the device rate.
- Live streams duck by gain and keep decoding; recorded streams stop the decoder and resume in place.
- A stream that starts while the assistant is speaking **starts at** the ducked gain with no ramp.
- ffmpeg is invoked as `ffmpeg -hide_banner -loglevel error -i <url> -f s16le -acodec pcm_s16le -ac 1 -ar <device_rate> -`; live adds `-live_start_index -1 -reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5`.
- No `mpv` may remain anywhere in `crates/` when the plan is complete.
- Every task ends green: `cargo fmt --all -- --check`, `cargo clippy -p voice-chatbot-client --all-targets` (0 diagnostics), `cargo test -p voice-chatbot-client -p voice-chatbot-protocol`.

---

### Task 1: Ramped, shareable gain

**Files:**
- Create: `crates/client/src/media/gain.rs`
- Modify: `crates/client/src/media.rs` (add `pub mod gain;` at the top of the file, after the `use` block)
- Test: same file, `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: nothing.
- Produces: `pub struct Gain` with `Gain::new(f32) -> Gain`, `Gain::clone(&self) -> Gain`, `Gain::ramp_to(&self, f32)`, `Gain::jump_to(&self, f32)`, `Gain::target(&self) -> f32`, `Gain::take_jump(&self) -> bool`. Free functions `pub fn advance(current: f32, target: f32, step: f32) -> f32` and `pub fn step_for(sample_rate: u32) -> f32`. Constants `pub const DUCKED: f32 = 0.126;` and `pub const FULL: f32 = 1.0;`.

- [ ] **Step 1: Write the failing test**

Create `crates/client/src/media/gain.rs` containing only this test module:

```rust
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
        let rate = 48_000;
        let step = step_for(rate);
        let mut current = 0.0;
        let mut samples = 0;
        while current < 1.0 {
            current = advance(current, 1.0, step);
            samples += 1;
        }
        // 80 ms at 48 kHz.
        assert_eq!(samples, 3_840);
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib media::gain`
Expected: FAIL — compile errors, `cannot find function advance`, `cannot find type Gain`.

- [ ] **Step 3: Write minimal implementation**

Put this **above** the test module in `crates/client/src/media/gain.rs`:

```rust
//! The media source's playback gain, shared between the control thread and the
//! CPAL output callback.
//!
//! The callback may not lock, so the target is an `f32` stored as bits in an
//! `AtomicU32` and the ramp is walked one step per sample. A *jump* is the
//! start-ducked case: a stream opening while the assistant already speaks has
//! no earlier level to fade from, so it begins at the target outright.

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;

/// Linear gain for −18 dB: what a live stream ducks to while the assistant speaks.
pub const DUCKED: f32 = 0.126;
/// Unducked playback.
pub const FULL: f32 = 1.0;

/// How long a full-scale (0 → 1) ramp takes. Short enough not to read as a
/// delay, long enough to avoid a zipper-noise step discontinuity.
const RAMP: Duration = Duration::from_millis(80);

use std::time::Duration;

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
}

impl Gain {
    pub fn new(value: f32) -> Self {
        Self {
            target: Arc::new(AtomicU32::new(value.to_bits())),
            jump: Arc::new(AtomicBool::new(false)),
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
}
```

Then add to `crates/client/src/media.rs`, immediately after its `use` statements:

```rust
pub mod gain;
```

It must be `pub`, not `mod` or `pub(crate)`: Task 2 puts a `Gain` on a `pub`
field of `pub struct AudioIo`, and a crate-private type there trips the
`private_interfaces` lint.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-client --lib media::gain`
Expected: PASS, 5 tests.

Then `cargo fmt --all` and `cargo clippy -p voice-chatbot-client --all-targets` — expect 0 diagnostics. Move the `use std::time::Duration;` up with the other imports if clippy or fmt complains about import ordering.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/media/gain.rs crates/client/src/media.rs
git commit -m "feat(client): ramped gain shared with the audio callback"
```

---

### Task 2: Mix a second source into the output callback

**Files:**
- Modify: `crates/client/src/audio.rs` — replace `struct OutputQueue` and its impl (currently at `audio.rs:817-848`), change `build_output_stream` (`audio.rs:754`) and `build_output_stream_typed` (`audio.rs:779`), `AudioDevices::open_with_capacities` (`audio.rs:203`), `struct AudioIo` (`audio.rs:246`), `struct AudioIoParts` (`audio.rs:285`), `AudioIo::into_parts` (`audio.rs:305`)
- Test: `crates/client/src/audio.rs`, existing `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: `crate::media::gain::{Gain, advance, step_for, FULL}` from Task 1.
- Produces: `AudioIo.media_tx: SyncSender<Vec<i16>>`, `AudioIo.media_gain: Gain`, and the same two fields on `AudioIoParts`. `build_output_stream` now returns `Result<(cpal::Stream, SyncSender<Vec<i16>>, SyncSender<Vec<i16>>, Gain)>` as `(stream, voice_tx, media_tx, gain)`.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/client/src/audio.rs`:

```rust
    fn mixer_of(voice: Vec<i16>, media: Vec<i16>, gain_value: f32) -> OutputMixer {
        let (voice_tx, voice_rx) = mpsc::sync_channel(4);
        let (media_tx, media_rx) = mpsc::sync_channel(4);
        if !voice.is_empty() {
            voice_tx.try_send(voice).expect("queue voice");
        }
        if !media.is_empty() {
            media_tx.try_send(media).expect("queue media");
        }
        drop(voice_tx);
        drop(media_tx);
        let gain = crate::media::gain::Gain::new(gain_value);
        // A step of 1.0 settles the ramp on the first sample, so these tests
        // assert mixing rather than ramp timing.
        OutputMixer::new(voice_rx, media_rx, gain, 1.0)
    }

    #[test]
    fn mixer_is_silent_only_when_both_sources_are_dry() {
        let mut empty = mixer_of(vec![], vec![], 1.0);
        assert_eq!(empty.next_sample(), None);

        let mut voice_only = mixer_of(vec![100], vec![], 1.0);
        assert_eq!(voice_only.next_sample(), Some(100));
        assert_eq!(voice_only.next_sample(), None);

        let mut media_only = mixer_of(vec![], vec![100], 1.0);
        assert_eq!(media_only.next_sample(), Some(100));
        assert_eq!(media_only.next_sample(), None);
    }

    #[test]
    fn mixer_sums_both_sources_and_scales_only_the_media_one() {
        let mut full = mixer_of(vec![1000], vec![1000], 1.0);
        assert_eq!(full.next_sample(), Some(2000));

        // The voice is untouched by the media gain.
        let mut ducked = mixer_of(vec![1000], vec![1000], 0.5);
        assert_eq!(ducked.next_sample(), Some(1500));

        // Fully ducked media still keeps the stream alive.
        let mut silent = mixer_of(vec![], vec![1000], 0.0);
        assert_eq!(silent.next_sample(), Some(0));
    }

    #[test]
    fn mixer_saturates_instead_of_wrapping() {
        let mut hot = mixer_of(vec![30000], vec![30000], 1.0);
        assert_eq!(hot.next_sample(), Some(i16::MAX));

        let mut cold = mixer_of(vec![-30000], vec![-30000], 1.0);
        assert_eq!(cold.next_sample(), Some(i16::MIN));
    }

    #[test]
    fn mixer_ramps_the_media_gain_and_a_jump_skips_the_ramp() {
        let (voice_tx, voice_rx) = mpsc::sync_channel(4);
        let (media_tx, media_rx) = mpsc::sync_channel(4);
        media_tx.try_send(vec![1000; 4]).expect("queue media");
        drop(voice_tx);
        drop(media_tx);
        let gain = crate::media::gain::Gain::new(0.0);
        let mut mixer = OutputMixer::new(voice_rx, media_rx, gain.clone(), 0.25);

        // Ramping up from 0: the first sample is one step in, not the target.
        gain.ramp_to(1.0);
        assert_eq!(mixer.next_sample(), Some(250));
        assert_eq!(mixer.next_sample(), Some(500));

        // A jump lands on the target immediately.
        gain.jump_to(0.0);
        assert_eq!(mixer.next_sample(), Some(0));
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib audio::tests::mixer`
Expected: FAIL — `cannot find type OutputMixer in this scope`.

- [ ] **Step 3: Write minimal implementation**

In `crates/client/src/audio.rs`, replace the whole `struct OutputQueue { .. }` + `impl OutputQueue { .. }` block (`audio.rs:817-848`) with:

```rust
/// One producer feeding the output callback: a queue plus a read cursor.
struct Source {
    receiver: Receiver<Vec<i16>>,
    current: Vec<i16>,
    offset: usize,
}

impl Source {
    fn new(receiver: Receiver<Vec<i16>>) -> Self {
        Self {
            receiver,
            current: Vec::new(),
            offset: 0,
        }
    }

    fn next_sample(&mut self) -> Option<i16> {
        loop {
            if let Some(&sample) = self.current.get(self.offset) {
                self.offset += 1;
                return Some(sample);
            }
            match self.receiver.try_recv() {
                Ok(chunk) => {
                    self.current = chunk;
                    self.offset = 0;
                }
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return None,
            }
        }
    }
}

/// Sums the call's voice with the media player, scaling media by a ramped
/// gain. The hardware callback pulls this, so the ramp advances on the audio
/// clock and needs no timer.
struct OutputMixer {
    voice: Source,
    media: Source,
    gain: crate::media::gain::Gain,
    current_gain: f32,
    step: f32,
}

impl OutputMixer {
    fn new(
        voice: Receiver<Vec<i16>>,
        media: Receiver<Vec<i16>>,
        gain: crate::media::gain::Gain,
        step: f32,
    ) -> Self {
        let current_gain = gain.target();
        Self {
            voice: Source::new(voice),
            media: Source::new(media),
            gain,
            current_gain,
            step,
        }
    }

    fn next_sample(&mut self) -> Option<i16> {
        let voice = self.voice.next_sample();
        let media = self.media.next_sample();

        // Advance the ramp every sample, so a gap in the media queue cannot
        // strand the gain mid-fade.
        let target = self.gain.target();
        self.current_gain = if self.gain.take_jump() {
            target
        } else {
            crate::media::gain::advance(self.current_gain, target, self.step)
        };

        if voice.is_none() && media.is_none() {
            return None;
        }
        let media = f32::from(media.unwrap_or(0)) * self.current_gain;
        let mixed = i32::from(voice.unwrap_or(0)) + media as i32;
        Some(mixed.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16)
    }
}
```

In `build_output_stream_typed` (`audio.rs:779`), change the signature's return type and the channel setup. Replace:

```rust
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let mut queued_audio = OutputQueue::new(receiver);
```

with:

```rust
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let (media_sender, media_receiver) = mpsc::sync_channel(capacity);
        let gain = crate::media::gain::Gain::new(crate::media::gain::FULL);
        let mut queued_audio = OutputMixer::new(
            receiver,
            media_receiver,
            gain.clone(),
            crate::media::gain::step_for(config.sample_rate),
        );
```

and its `Ok(stream) => return Ok((stream, sender)),` becomes
`Ok(stream) => return Ok((stream, sender, media_sender, gain)),`.

Change both function signatures from
`Result<(cpal::Stream, SyncSender<Vec<i16>>)>` to
`Result<(cpal::Stream, SyncSender<Vec<i16>>, SyncSender<Vec<i16>>, crate::media::gain::Gain)>`,
and in `build_output_stream` (`audio.rs:754`) every `build_output_stream_typed::<T>(..)` arm already forwards its return value unchanged, so no arm bodies need editing.

In `open_with_capacities` (`audio.rs:203`) change:

```rust
        let (output_stream, output_tx) = build_output_stream(output, output_capacity)?;
```

to:

```rust
        let (output_stream, output_tx, media_tx, media_gain) =
            build_output_stream(output, output_capacity)?;
```

and add `media_tx,` and `media_gain,` to the `AudioIo { .. }` literal.

Add to `struct AudioIo` (`audio.rs:246`), after `output_tx`:

```rust
    /// Mono `i16` media chunks, summed with `output_tx` under a ramped gain.
    pub media_tx: SyncSender<Vec<i16>>,
    /// Ramped gain applied to `media_tx` only.
    pub media_gain: crate::media::gain::Gain,
```

Add the same two fields to `struct AudioIoParts` (`audio.rs:285`) and forward them in `AudioIo::into_parts` (`audio.rs:305`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-client --lib audio::`
Expected: PASS. The pre-existing `audio::tests` must all still pass.

`main.rs:266` destructures `AudioIoParts` with a field list; add `media_tx,` and `media_gain,` there or it will not compile. Prefix both with `_` for now (`media_tx: _media_tx,`) since Task 5 wires them.

Then `cargo fmt --all`, `cargo clippy -p voice-chatbot-client --all-targets` — 0 diagnostics.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/audio.rs crates/client/src/media.rs crates/client/src/main.rs
git commit -m "feat(client): mix a gained media source into the output callback"
```

---

### Task 3: Carry `live` on the protocol and set it at both server call sites

**Files:**
- Modify: `crates/protocol/src/lib.rs:18-21`
- Modify: `crates/server/src/media.rs:36-44` (`play_stream`)
- Modify: `crates/server/src/skills/radio.rs:190`
- Modify: `crates/server/src/skills/shows.rs:384`
- Test: `crates/protocol/src/lib.rs`, existing `#[cfg(test)] mod tests`

**Interfaces:**
- Consumes: nothing.
- Produces: `MediaCommand::Play { url: String, title: String, live: bool }`. `MediaController::play_stream(&self, url: &str, title: &str, live: bool)`.

- [ ] **Step 1: Write the failing test**

Add to `#[cfg(test)] mod tests` in `crates/protocol/src/lib.rs`:

```rust
    #[test]
    fn play_carries_whether_the_stream_is_live() {
        let cmd = MediaCommand::Play {
            url: "http://example/x.m3u8".into(),
            title: "BBC Radio 4".into(),
            live: true,
        };
        let payload = cmd.to_payload();
        assert_eq!(payload["action"], "play");
        assert_eq!(payload["live"], true);
        assert_eq!(MediaCommand::from_payload(&payload).unwrap(), cmd);
    }

    #[test]
    fn a_play_from_an_older_server_is_treated_as_live() {
        // Radio is the common case, and ducking a live stream by gain is the
        // safe default: it never stalls a decoder that cannot be paused.
        let payload = serde_json::json!({
            "action": "play",
            "url": "http://example/x.m3u8",
            "title": "BBC Radio 4"
        });
        let MediaCommand::Play { live, .. } = MediaCommand::from_payload(&payload).unwrap() else {
            panic!("expected a Play");
        };
        assert!(live);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-protocol`
Expected: FAIL — `struct variant MediaCommand::Play has no field named live`.

- [ ] **Step 3: Write minimal implementation**

In `crates/protocol/src/lib.rs`, change the `Play` variant (`lib.rs:18-21`) to:

```rust
    /// Start streaming `url` (replaces whatever is playing). `title` is for logs/UI.
    ///
    /// `live` picks how the client ducks it while the assistant speaks: a live
    /// stream drops its gain and keeps decoding so it stays at the live edge,
    /// while a recorded one stops its decoder and resumes in place.
    Play {
        url: String,
        title: String,
        #[serde(default = "live_by_default")]
        live: bool,
    },
```

and add, just below the enum:

```rust
/// Radio is the common case, and a gain duck never stalls a decoder.
fn live_by_default() -> bool {
    true
}
```

In `crates/server/src/media.rs`, change `play_stream` (`media.rs:36`) to:

```rust
    /// Stream `url` on the client, replacing whatever was playing. `live`
    /// distinguishes a broadcast from a recorded programme; see
    /// [`voice_chatbot_protocol::MediaCommand::Play`].
    pub fn play_stream(&self, url: &str, title: &str, live: bool) {
        *self.playing.lock().unwrap() = Some(NowPlaying {
            title: title.to_string(),
        });
        self.send(&MediaCommand::Play {
            url: url.to_string(),
            title: title.to_string(),
            live,
        });
    }
```

In `crates/server/src/skills/radio.rs:190`:

```rust
        media.play_stream(&station.url, station.display, true);
```

In `crates/server/src/skills/shows.rs:384`:

```rust
        media.play_stream(&episode.url, &episode.display, false);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-protocol`
Expected: PASS.

Then `cargo check -p voice-chatbot-server` — the client will not compile yet because `media.rs` still matches `Play { url, title }`; add `live: _` to that pattern at `crates/client/src/media.rs:134` to keep the tree building. Task 5 uses it properly.

Run `cargo fmt --all` and `cargo clippy -p voice-chatbot-client --all-targets`.

- [ ] **Step 5: Commit**

```bash
git add crates/protocol/src/lib.rs crates/server/src/media.rs \
        crates/server/src/skills/radio.rs crates/server/src/skills/shows.rs \
        crates/client/src/media.rs
git commit -m "feat(protocol): carry whether a Play stream is live"
```

---

### Task 4: The duck / pause state machine

**Files:**
- Create: `crates/client/src/media/duck.rs`
- Modify: `crates/client/src/media.rs` (add `mod duck;` next to `pub mod gain;`)
- Test: same file

**Interfaces:**
- Consumes: `crate::media::gain::{DUCKED, FULL}` from Task 1.
- Produces: `pub enum Transport { Running, Stopped }`, `pub struct Duck` with `Duck::new() -> Duck`, `Duck::start(&mut self, live: bool) -> bool` (returns whether it starts ducked, i.e. needs a jump), `Duck::stop(&mut self)`, `Duck::set_bot_speaking(&mut self, bool)`, `Duck::set_user_paused(&mut self, bool)`, `Duck::gain(&self) -> f32`, `Duck::transport(&self) -> Transport`, `Duck::is_playing(&self) -> bool`. Task 6 adds one more accessor to this file, `Duck::is_playing_speech(&self) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `crates/client/src/media/duck.rs` containing only this test module:

```rust
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

    /// The bug the mpv build had: one `pause` property served both concerns,
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

    #[test]
    fn stopping_clears_play_state_including_a_user_pause() {
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
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib media::duck`
Expected: FAIL — `cannot find type Duck in this scope`.

- [ ] **Step 3: Write minimal implementation**

Put this above the tests in `crates/client/src/media/duck.rs`:

```rust
//! What the media source should be doing right now.
//!
//! Two independent concerns the mpv build conflated into one `pause`
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
```

Add `mod duck;` beside `pub mod gain;` in `crates/client/src/media.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-client --lib media::duck`
Expected: PASS, 7 tests. Then `cargo fmt --all` and `cargo clippy -p voice-chatbot-client --all-targets`.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/media/duck.rs crates/client/src/media.rs
git commit -m "feat(client): separate media ducking from transport"
```

---

### Task 5: The ffmpeg decoder and its feeder thread

**Files:**
- Create: `crates/client/src/media/decoder.rs`
- Modify: `crates/client/src/media.rs` (add `mod decoder;`)
- Test: same file

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pub struct Decoder` with `Decoder::spawn(url: &str, sample_rate: u32, live: bool, tx: SyncSender<Vec<i16>>) -> std::io::Result<Decoder>`, `Decoder::set_running(&self, running: bool)`, `Decoder::finished(&mut self) -> Option<Option<i32>>` (outer `None` = still running, inner `None` = killed by signal), `Decoder::command_args(url: &str, sample_rate: u32, live: bool) -> Vec<String>`. `Drop` kills the child and joins the feeder.

- [ ] **Step 1: Write the failing test**

Create `crates/client/src/media/decoder.rs` with only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_live_stream_starts_at_the_live_edge_and_reconnects() {
        let args = Decoder::command_args("http://example/x.m3u8", 48_000, true);
        let joined = args.join(" ");
        // ffmpeg's HLS default is -3 segments, ~19 s behind the live edge on
        // BBC's 6.4 s segments.
        assert!(joined.contains("-live_start_index -1"), "{joined}");
        assert!(joined.contains("-reconnect 1"), "{joined}");
        assert!(joined.contains("-reconnect_streamed 1"), "{joined}");
        assert!(joined.contains("-reconnect_delay_max 5"), "{joined}");
    }

    #[test]
    fn a_recorded_stream_gets_no_live_flags() {
        let joined = Decoder::command_args("http://example/x.m4a", 48_000, false).join(" ");
        assert!(!joined.contains("-live_start_index"), "{joined}");
        assert!(!joined.contains("-reconnect"), "{joined}");
    }

    #[test]
    fn always_decodes_to_mono_s16le_at_the_device_rate() {
        let args = Decoder::command_args("http://example/x.m3u8", 44_100, true);
        let joined = args.join(" ");
        assert!(joined.contains("-f s16le"), "{joined}");
        assert!(joined.contains("-acodec pcm_s16le"), "{joined}");
        assert!(joined.contains("-ac 1"), "{joined}");
        assert!(joined.contains("-ar 44100"), "{joined}");
        // ffmpeg must write to stdout, never to an audio device.
        assert_eq!(args.last().map(String::as_str), Some("-"));
        assert!(!joined.contains("-audio_device"), "{joined}");
    }

    #[test]
    fn twenty_millisecond_chunks_are_one_opus_frame_of_samples() {
        assert_eq!(chunk_samples(48_000), 960);
        assert_eq!(chunk_samples(44_100), 882);
        assert_eq!(chunk_samples(16_000), 320);
    }

    /// Uses the real ffmpeg to decode a generated tone, proving the argument
    /// list and the feeder agree with the binary we ship against.
    #[test]
    #[ignore]
    fn live_decodes_a_generated_tone_into_the_channel() {
        let (tx, rx) = std::sync::mpsc::sync_channel(256);
        // lavfi is ffmpeg's built-in generator: 0.5 s of 440 Hz.
        let mut decoder = Decoder::spawn("sine=frequency=440:duration=0.5", 48_000, false, tx)
            .expect("spawn ffmpeg");
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let samples: usize = rx.try_iter().map(|chunk| chunk.len()).sum();
        // 0.5 s at 48 kHz, allowing for the last partial chunk.
        assert!(
            (23_000..=24_100).contains(&samples),
            "got {samples} samples"
        );
        assert_eq!(decoder.finished(), Some(Some(0)), "ffmpeg should exit clean");
    }
}
```

Note: the ignored test passes a lavfi source; `command_args` must therefore accept `-f lavfi` when the url has no scheme. Rather than special-case it, the ignored test builds the decoder via `Decoder::spawn`, and `spawn` prepends `-f lavfi` when `url` contains no `://`. Add that to the implementation below.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib media::decoder`
Expected: FAIL — `cannot find type Decoder`, `cannot find function chunk_samples`.

- [ ] **Step 3: Write minimal implementation**

Put above the tests in `crates/client/src/media/decoder.rs`:

```rust
//! `ffmpeg` as a *decoder*, never a player.
//!
//! It is handed no audio device: it writes raw `s16le` mono PCM at the output
//! device's rate to stdout, and a feeder thread pumps that into the mixer's
//! channel. Two consequences worth knowing:
//!
//! * ffmpeg drains an HLS playlist at network speed (measured `speed=401x` on
//!   BBC Radio 4), so it does **not** pace itself. The OS pipe plus the bounded
//!   channel are the jitter buffer, and pipe backpressure is what paces it.
//! * Because of that, "pause" costs nothing: stop reading, the pipe fills,
//!   ffmpeg blocks on write, and the decoder stalls exactly in place.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

/// Samples in one 20 ms chunk — the granularity the call already runs on.
fn chunk_samples(sample_rate: u32) -> usize {
    sample_rate as usize / 50
}

pub struct Decoder {
    child: Child,
    running: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    feeder: Option<JoinHandle<()>>,
}

impl Decoder {
    /// The exact argument list, so it can be asserted without spawning.
    pub fn command_args(url: &str, sample_rate: u32, live: bool) -> Vec<String> {
        let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if live {
            for arg in [
                "-live_start_index",
                "-1",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
            ] {
                args.push(arg.to_string());
            }
        }
        if !url.contains("://") {
            // A generated source (used by the live test), not a URL.
            args.push("-f".into());
            args.push("lavfi".into());
        }
        args.push("-i".into());
        args.push(url.to_string());
        for arg in ["-f", "s16le", "-acodec", "pcm_s16le", "-ac", "1", "-ar"] {
            args.push(arg.to_string());
        }
        args.push(sample_rate.to_string());
        args.push("-".into());
        args
    }

    pub fn spawn(
        url: &str,
        sample_rate: u32,
        live: bool,
        tx: SyncSender<Vec<i16>>,
    ) -> std::io::Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(Self::command_args(url, sample_rate, live))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("stdout is piped");

        let running = Arc::new(AtomicBool::new(true));
        let stopping = Arc::new(AtomicBool::new(false));
        let feeder = {
            let running = Arc::clone(&running);
            let stopping = Arc::clone(&stopping);
            let chunk = chunk_samples(sample_rate);
            std::thread::spawn(move || {
                let mut bytes = vec![0u8; chunk * 2];
                loop {
                    if stopping.load(Ordering::Relaxed) {
                        return;
                    }
                    if !running.load(Ordering::Relaxed) {
                        // Not reading is the pause: the pipe fills and ffmpeg
                        // blocks on write, holding its place exactly.
                        std::thread::sleep(Duration::from_millis(20));
                        continue;
                    }
                    match stdout.read_exact(&mut bytes) {
                        Ok(()) => {
                            let samples = bytes
                                .chunks_exact(2)
                                .map(|pair| i16::from_le_bytes([pair[0], pair[1]]))
                                .collect();
                            // Blocking send is the backpressure that paces ffmpeg.
                            if tx.send(samples).is_err() {
                                return; // mixer gone
                            }
                        }
                        Err(_) => return, // EOF or the stream died
                    }
                }
            })
        };

        Ok(Self {
            child,
            running,
            stopping,
            feeder: Some(feeder),
        })
    }

    /// Whether the feeder pulls. False stalls ffmpeg in place.
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// `None` while it still runs; `Some(code)` once it has exited, where the
    /// inner `None` means it was killed by a signal.
    pub fn finished(&mut self) -> Option<Option<i32>> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            Ok(None) => None,
            Err(_) => Some(None),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::Relaxed);
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(feeder) = self.feeder.take() {
            let _ = feeder.join();
        }
    }
}
```

Add `mod decoder;` beside the other two in `crates/client/src/media.rs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-client --lib media::decoder`
Expected: PASS, 4 tests (1 ignored).

Then the real-ffmpeg check:
Run: `cargo test -p voice-chatbot-client --lib media::decoder -- --ignored --nocapture`
Expected: PASS. If ffmpeg is missing, install it (`sudo apt install ffmpeg`) — it is now a hard requirement.

`cargo fmt --all`, `cargo clippy -p voice-chatbot-client --all-targets`.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/media/decoder.rs crates/client/src/media.rs
git commit -m "feat(client): decode media with ffmpeg into a PCM channel"
```

---

### Task 6: Rewrite `MediaPlayer` on the mixer and delete mpv

**Files:**
- Modify: `crates/client/src/media.rs` — replace everything except the `mod` lines
- Modify: `crates/client/src/media/duck.rs` — add `Duck::is_playing_speech`
- Modify: `crates/client/src/main.rs:266` (destructure), `main.rs:308-315` (construction)
- Test: `crates/client/src/media.rs`

**Interfaces:**
- Consumes: `Decoder` (Task 5), `Duck`/`Transport` (Task 4), `Gain` + `DUCKED`/`FULL` (Task 1), `AudioIoParts.media_tx` / `.media_gain` (Task 2), `MediaCommand::Play { live }` (Task 3).
- Produces: `MediaPlayer::new(media_tx: SyncSender<Vec<i16>>, gain: Gain, sample_rate: u32, server_base: &str) -> MediaPlayer`, `MediaPlayer::is_available() -> bool`, `MediaPlayer::on_event(&mut self, kind: &str, payload: &Value) -> Option<String>`, `MediaPlayer::tick(&mut self) -> Option<String>`, `MediaPlayer::stop(&mut self) -> bool`.

- [ ] **Step 1: Write the failing test**

Replace the `#[cfg(test)] mod tests` in `crates/client/src/media.rs` with:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::gain::{DUCKED, FULL};
    use serde_json::json;

    fn player() -> (MediaPlayer, std::sync::mpsc::Receiver<Vec<i16>>, Gain) {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let gain = Gain::new(FULL);
        let player = MediaPlayer::new(tx, gain.clone(), 48_000, "http://127.0.0.1:6210");
        (player, rx, gain)
    }

    #[test]
    fn a_relative_clip_url_resolves_against_the_server() {
        let (player, _rx, _gain) = player();
        assert_eq!(
            player.resolve("/sfx/woosh.flac"),
            "http://127.0.0.1:6210/sfx/woosh.flac"
        );
        assert_eq!(
            player.resolve("http://elsewhere/x.m3u8"),
            "http://elsewhere/x.m3u8"
        );
    }

    #[test]
    fn speaking_boundaries_move_the_shared_gain() {
        let (mut player, _rx, gain) = player();
        player.duck.start(true);
        player.apply_duck();
        assert_eq!(gain.target(), FULL);

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        assert_eq!(gain.target(), DUCKED);

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), FULL);
    }

    #[test]
    fn an_explicit_pause_is_not_undone_by_the_next_reply() {
        let (mut player, _rx, gain) = player();
        player.duck.start(true);
        player.on_event(MEDIA_EVENT, &json!({"action": "pause"}));
        assert_eq!(gain.target(), 0.0);

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), 0.0, "still paused");

        player.on_event(MEDIA_EVENT, &json!({"action": "resume"}));
        assert_eq!(gain.target(), FULL);
    }

    #[test]
    fn a_clip_waits_for_the_assistant_when_asked_to() {
        let (mut player, _rx, _gain) = player();
        player.on_event("rtf-bot-started-speaking", &Value::Null);
        let line = player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play_file", "url": "/sfx/x.flac", "after_speech": true}),
        );
        assert_eq!(line, None, "held until the assistant finishes");
        assert!(player.pending.is_some());
    }

    #[test]
    fn exit_line_is_quiet_about_a_clean_exit_and_loud_about_a_failure() {
        assert_eq!(exit_line("BBC Radio 4", Some(0)), None);
        assert_eq!(
            exit_line("BBC Radio 4", Some(1)).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (ffmpeg exit 1)]")
        );
        assert_eq!(
            exit_line("BBC Radio 4", None).as_deref(),
            Some("[media: BBC Radio 4 stopped unexpectedly (ffmpeg killed by signal)]")
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib media::tests`
Expected: FAIL — `MediaPlayer::new` takes the wrong arguments; no field `duck`; no method `resolve`/`apply_duck`.

- [ ] **Step 3: Write minimal implementation**

Replace all of `crates/client/src/media.rs` above its test module with:

```rust
//! Client-side media playback: BBC radio/shows and sound effects the server's
//! skills ask for arrive as `{"type":"media"}` events, are decoded by `ffmpeg`
//! and are mixed into the call's own output stream.
//!
//! Playing through the call's stream rather than a second process is what puts
//! media on the speakerphone at all: while a call is up, CPAL holds that card
//! outright and nothing else can open it by any path.
//!
//! Ducking: while the assistant speaks (`rtf-bot-started-speaking` …
//! `rtf-bot-stopped-speaking`) a live stream drops to [`gain::DUCKED`] and
//! keeps decoding, so it stays at the live edge; a recorded one stops its
//! decoder and resumes in place. A `play_file` with `after_speech` waits for
//! the same boundary.

pub mod gain;
mod decoder;
mod duck;

use std::process::{Command, Stdio};
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use serde_json::Value;
use voice_chatbot_protocol::{MediaCommand, AFTER_SPEECH_CAP_SECS, MEDIA_EVENT};

use decoder::Decoder;
use duck::{Duck, Transport};
use gain::Gain;

/// What to report for a decoder that exited on its own. `None` for a clean
/// exit (the stream simply ended).
fn exit_line(title: &str, code: Option<i32>) -> Option<String> {
    match code {
        Some(0) => None,
        Some(code) => Some(format!(
            "[media: {title} stopped unexpectedly (ffmpeg exit {code})]"
        )),
        None => Some(format!(
            "[media: {title} stopped unexpectedly (ffmpeg killed by signal)]"
        )),
    }
}

pub struct MediaPlayer {
    /// Server base URL; relative media URLs (`/sfx/x.flac`) resolve against it.
    server_base: String,
    media_tx: SyncSender<Vec<i16>>,
    gain: Gain,
    sample_rate: u32,
    decoder: Option<Decoder>,
    duck: Duck,
    title: String,
    /// A `play_file { after_speech }` waiting for the assistant to finish.
    pending: Option<(String, Instant)>,
    exit_report: Option<String>,
}

impl MediaPlayer {
    pub fn new(
        media_tx: SyncSender<Vec<i16>>,
        gain: Gain,
        sample_rate: u32,
        server_base: &str,
    ) -> Self {
        Self {
            server_base: server_base.trim_end_matches('/').to_string(),
            media_tx,
            gain,
            sample_rate,
            decoder: None,
            duck: Duck::new(),
            title: String::new(),
            pending: None,
            exit_report: None,
        }
    }

    /// ffmpeg decodes every format this plays; without it nothing can play.
    pub fn is_available() -> bool {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn resolve(&self, url: &str) -> String {
        if url.starts_with('/') {
            format!("{}{url}", self.server_base)
        } else {
            url.to_string()
        }
    }

    /// Push the state machine's current intent at the mixer and the decoder.
    fn apply_duck(&mut self) {
        self.gain.ramp_to(self.duck.gain());
        if let Some(decoder) = &self.decoder {
            decoder.set_running(self.duck.transport() == Transport::Running);
        }
    }

    /// Dispatch one events-WebSocket frame. Returns a line to print, if any.
    pub fn on_event(&mut self, kind: &str, payload: &Value) -> Option<String> {
        match kind {
            MEDIA_EVENT => match MediaCommand::from_payload(payload) {
                Ok(cmd) => self.apply(cmd),
                Err(error) => {
                    tracing::warn!(%error, "media: bad command");
                    None
                }
            },
            "rtf-bot-started-speaking" => {
                self.duck.set_bot_speaking(true);
                self.apply_duck();
                None
            }
            "rtf-bot-stopped-speaking" => {
                self.duck.set_bot_speaking(false);
                self.apply_duck();
                self.pending
                    .take()
                    .map(|(url, _)| self.play(&url, "sound effect", false))
            }
            _ => None,
        }
    }

    /// Time-based housekeeping (call every second or so).
    pub fn tick(&mut self) -> Option<String> {
        self.reap();
        if let Some(line) = self.exit_report.take() {
            return Some(line);
        }
        match &self.pending {
            Some((_, since)) if since.elapsed() >= Duration::from_secs(AFTER_SPEECH_CAP_SECS) => {
                tracing::warn!(
                    "media: timed out waiting for the assistant to finish; playing anyway"
                );
                let (url, _) = self.pending.take()?;
                Some(self.play(&url, "sound effect", false))
            }
            _ => None,
        }
    }

    fn apply(&mut self, cmd: MediaCommand) -> Option<String> {
        match cmd {
            MediaCommand::Play { url, title, live } => Some(self.play(&url, &title, live)),
            MediaCommand::PlayFile { url, after_speech } => {
                if after_speech && self.duck.is_playing_speech() {
                    self.pending = Some((url, Instant::now()));
                    None
                } else {
                    Some(self.play(&url, "sound effect", false))
                }
            }
            MediaCommand::Stop => self.stop().then(|| "[media stopped]".to_string()),
            MediaCommand::Pause => {
                self.duck.set_user_paused(true);
                self.apply_duck();
                None
            }
            MediaCommand::Resume => {
                self.duck.set_user_paused(false);
                self.apply_duck();
                None
            }
        }
    }

    fn play(&mut self, url: &str, title: &str, live: bool) -> String {
        self.stop();
        let url = self.resolve(url);
        // Start ducked when the reply is still being spoken: there is no
        // earlier level to fade from, and fading in from full is the overlap
        // being avoided.
        let jump = self.duck.start(live);
        if jump {
            self.gain.jump_to(self.duck.gain());
        } else {
            self.gain.ramp_to(self.duck.gain());
        }
        match Decoder::spawn(&url, self.sample_rate, live, self.media_tx.clone()) {
            Ok(decoder) => {
                decoder.set_running(self.duck.transport() == Transport::Running);
                self.decoder = Some(decoder);
                self.title = title.to_string();
                format!("[media: playing {title}]")
            }
            Err(error) => {
                tracing::warn!(%error, "media: failed to start ffmpeg (is it installed?)");
                self.duck.stop();
                self.gain.ramp_to(0.0);
                format!("[media: cannot play {title}: ffmpeg failed to start]")
            }
        }
    }

    /// Stop playback. True when something was playing.
    pub fn stop(&mut self) -> bool {
        self.pending = None;
        let was_playing = self.decoder.take().is_some();
        self.duck.stop();
        // The ramp to 0 covers whatever is already queued in the channel.
        self.gain.ramp_to(0.0);
        if was_playing {
            tracing::info!(title = %self.title, "media: stopped");
        }
        was_playing
    }

    /// Notice a decoder that ended on its own, and say so when it failed.
    fn reap(&mut self) {
        let Some(decoder) = self.decoder.as_mut() else {
            return;
        };
        let Some(code) = decoder.finished() else {
            return; // still playing
        };
        self.decoder = None;
        self.duck.stop();
        self.gain.ramp_to(0.0);
        if let Some(line) = exit_line(&self.title, code) {
            tracing::warn!(title = %self.title, ?code, "media: decoder exited on its own");
            self.exit_report = Some(line);
        }
    }
}
```

Add to `crates/client/src/media/duck.rs`, inside `impl Duck`:

```rust
    /// Whether the assistant is mid-reply, for `after_speech` clips.
    pub fn is_playing_speech(&self) -> bool {
        self.bot_speaking
    }
```

In `crates/client/src/main.rs`, change the `AudioIoParts` destructure (`main.rs:259-267`) to bind `media_tx,` and `media_gain,` for real, and replace the media construction (`main.rs:308-315`) with:

```rust
    // Radio, shows and sound effects mix into the call's own output stream, so
    // they play on the call's device. Without ffmpeg the call still works;
    // media commands are logged and dropped.
    let media = if MediaPlayer::is_available() {
        Some(MediaPlayer::new(
            media_tx,
            media_gain,
            output_rate,
            server_url,
        ))
    } else {
        if describe_devices {
            tracing::warn!(
                "ffmpeg not found; BBC radio, shows and sound effects will not play \
                 (brew/apt install ffmpeg)"
            );
        }
        None
    };
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p voice-chatbot-client --lib media::`
Expected: PASS.

Confirm mpv is gone: `! grep -ri "mpv" crates/ --include=*.rs` must print nothing and exit 1 when grep finds nothing (use `grep -ri mpv crates/ --include=*.rs; echo "exit=$?"` and expect `exit=1`).

Run the whole suite: `cargo test -p voice-chatbot-client -p voice-chatbot-protocol`, then `cargo check -p voice-chatbot-server`, `cargo fmt --all`, `cargo clippy -p voice-chatbot-client --all-targets`.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/media.rs crates/client/src/media/duck.rs crates/client/src/main.rs
git commit -m "feat(client): play media through the call's own output stream"
```

---

### Task 7: Live verification and documentation

**Files:**
- Modify: `crates/client/src/media.rs` (add a `#[cfg(test)] mod live_tests`)
- Modify: `README.md:9-10`, `README.md:24`, `README.md:182-183`, `README.md:196-198`, `README.md:249-255`
- Modify: `crates/client/README.md:129-132`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Write the failing test**

Add to the end of `crates/client/src/media.rs`:

```rust
#[cfg(test)]
mod live_tests {
    //! Decodes ~3 s of BBC Radio 4 through the real ffmpeg:
    //! `cargo test -p voice-chatbot-client -- --ignored live`.
    use super::*;
    use serde_json::json;

    const RADIO_4: &str = "http://as-hls-ww-live.akamaized.net/pool_55057080/live/ww/bbc_radio_fourfm/bbc_radio_fourfm.isml/bbc_radio_fourfm-audio%3d96000.norewind.m3u8";

    #[test]
    #[ignore]
    fn live_radio_reaches_the_mixer_and_ducks_without_stopping() {
        assert!(MediaPlayer::is_available(), "ffmpeg not installed");
        let (tx, rx) = std::sync::mpsc::sync_channel(512);
        let gain = Gain::new(gain::FULL);
        let mut player = MediaPlayer::new(tx, gain.clone(), 48_000, "http://127.0.0.1:6210");

        let line = player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play", "url": RADIO_4, "title": "BBC Radio 4", "live": true}),
        );
        assert_eq!(line.as_deref(), Some("[media: playing BBC Radio 4]"));

        std::thread::sleep(Duration::from_secs(3));
        let samples: Vec<i16> = rx.try_iter().flatten().collect();
        assert!(
            samples.len() > 48_000,
            "expected at least a second of audio, got {}",
            samples.len()
        );
        // Real programme audio, not silence and not clipping. Measured on
        // 2026-08-28: RMS -20.3 dBFS, peak 18083.
        let peak = samples.iter().map(|s| i32::from(s.abs())).max().unwrap_or(0);
        assert!((1_000..32_767).contains(&peak), "peak {peak}");

        // A live stream ducks by gain and keeps decoding.
        player.on_event("rtf-bot-started-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::DUCKED);
        assert!(player.decoder.is_some(), "live radio must keep decoding");

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::FULL);

        assert!(player.stop());
    }

    /// The case the mpv build got wrong: asked for mid-reply, radio came up at
    /// full volume over the assistant.
    #[test]
    #[ignore]
    fn live_radio_asked_for_mid_reply_starts_quiet() {
        assert!(MediaPlayer::is_available(), "ffmpeg not installed");
        let (tx, _rx) = std::sync::mpsc::sync_channel(512);
        let gain = Gain::new(gain::FULL);
        let mut player = MediaPlayer::new(tx, gain.clone(), 48_000, "http://127.0.0.1:6210");

        player.on_event("rtf-bot-started-speaking", &Value::Null);
        player.on_event(
            MEDIA_EVENT,
            &json!({"action": "play", "url": RADIO_4, "title": "BBC Radio 4", "live": true}),
        );
        assert_eq!(gain.target(), gain::DUCKED, "must start quiet");

        player.on_event("rtf-bot-stopped-speaking", &Value::Null);
        assert_eq!(gain.target(), gain::FULL, "and come up when the reply ends");
        assert!(player.stop());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p voice-chatbot-client --lib media::live_tests -- --ignored --nocapture`
Expected: these should PASS if Tasks 1-6 are correct. If either fails, the failure is a real defect in the earlier tasks — fix it there, not here. A `peak` of 0 means no audio reached the mixer; a `gain.target()` of `FULL` in the second test means the start-ducked path is not wired.

- [ ] **Step 3: Update the documentation**

The routing caveat is now false and must go. In `README.md`, replace the sentence at line 24 ("While a call holds the speakerphone the rest of the desktop cannot route to it, so `mpv` media plays on the system default sink instead.") with:

```markdown
  Media (radio, shows, sound effects) is decoded by `ffmpeg` and mixed into the
  call's own output stream, so it plays on the call's device — including the
  speakerphone, which a second process could not open while the call holds it.
```

At `README.md:9-10`, replace the `mpv` install note with `ffmpeg` (`brew install ffmpeg` / `apt install ffmpeg`). Do the same at `README.md:182-183`, `README.md:196-198` and `README.md:249-255`, replacing every description of mpv's JSON-IPC pause with the gain duck: live radio drops to −18 dB over 80 ms and keeps decoding; recorded shows stop their decoder and resume in place.

In `crates/client/README.md`, replace lines 129-132 ("Either way the call owns the speakerphone … while a call is up.") with:

```markdown
Either way the call owns the speakerphone for its duration: a sound server
cannot route other desktop audio to a card that CPAL holds directly. Media is
therefore not a second process at all — `ffmpeg` decodes it to raw PCM and the
output callback sums it with the call's voice, so it reaches the same device.
```

- [ ] **Step 4: Verify everything is green**

```bash
cargo fmt --all -- --check
cargo clippy -p voice-chatbot-client --all-targets
cargo test -p voice-chatbot-client -p voice-chatbot-protocol
cargo test -p voice-chatbot-client -- --ignored live
grep -ri mpv crates/ README.md docs/prd docs/adr --include=*.rs --include=*.md; echo "exit=$?"
```

Expected: fmt clean, 0 clippy diagnostics, all tests pass, and the final `grep` exits 1 apart from historical references in `docs/adr/` and `docs/research/`, which are records of past decisions and must NOT be rewritten.

- [ ] **Step 5: Commit**

```bash
git add crates/client/src/media.rs README.md crates/client/README.md
git commit -m "docs: media plays on the call's device, decoded by ffmpeg"
```

---

## Manual verification on real hardware

Automated tests cannot prove the speakerphone actually makes the sound. After Task 7, with the server running:

1. Start a call: `make call`.
2. Ask for radio ("play BBC Radio 4"). Expect it out of the **Jabra**, not the built-in speakers, and expect it to **start quiet** under the spoken reply, coming up as the reply ends.
3. Speak again while it plays. Expect it to drop to a murmur and return, with no gap or rebuffer.
4. Ask for a recorded show, interrupt it, and confirm it resumes exactly where it left off.
5. Say "pause the radio", then speak again. Expect it to stay paused.

If −18 dB or 80 ms feels wrong on the hardware, tune `DUCKED` and `RAMP` in `crates/client/src/media/gain.rs`; they are starting values, not measured ones.
