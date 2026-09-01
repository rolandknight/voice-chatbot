# jabra-led — drive the Speak2 40's LED ring from the client

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Date:** 2026-09-01
**Status:** Not started.
**Spec:** `docs/specs/jabra-led.md` — read it first; it carries the device research (HID
telephony usages, what the ring can show, kernel plumbing) that this plan assumes.

**Goal:** The client shows chatbot activity (asleep / listening / thinking / speaking /
mic-gated) on the Jabra speakerphone's LED ring by writing standard HID telephony
output reports, degrading silently to today's behavior when no such device is present.

**Architecture:** One new module pair. `crates/client/src/led.rs` holds a pure phase
state machine (`PhaseTracker`), a clonable `LedController` handle, and a driver task
that owns the device behind a `tokio::sync::watch` channel (writes coalesce; dropping
every handle clears the ring). `crates/client/src/led/hid.rs` finds the Jabra telephony
HID interface and maps its report descriptor to the three LED bits — no hardcoded
report bytes. The controller is wired in as a third consumer in `events::run` (beside
`Activity::note` and `MediaPlayer::on_event`) plus a wake-state hook in `wake::spawn`,
and constructed per session in `run_session` following the `MediaPlayer::is_available`
warn-once-and-degrade pattern.

**Tech stack:** Rust 2021, tokio. Two new deps in `crates/client`:
`hidapi = { version = "2.6", default-features = false, features = ["linux-native-basic-udev", "macos-shared-device"] }`
(hidraw backend in Rust — no libudev/libusb linkage, so no `Cross.toml` change) and
`hidreport = "0.6"` (report-descriptor parser).

## Global constraints

- Branch: `jabra-led` off `main` (isolated worktree per superpowers:using-git-worktrees).
- Every task ends green on `make check` (fmt, clippy `-D warnings`, tests). On this
  machine a pyo3-ffi build failure means `crates/qwen-tts/.venv` is missing — that is
  environment, not your diff; create the venv per the Makefile, don't "fix" the code.
- **No test may need hardware or network.** Hardware verification is Task 8, manual.
- Env vars use bare names (`LED`), matching `SERVER_URL` / `INPUT_DEVICE`.
- The Pi cross-build must stay green with **no `Cross.toml` edits** (verified in
  Task 3). If it breaks there, stop and reconsider the dependency features rather than
  adding apt packages ad hoc.
- Commit style: `feat(led): …` / `feat(deploy): …` one commit per task, matching the
  repo's `type(scope): lowercase summary` convention.

## Facts the implementer will need

**HID (from the spec, condensed):**
- Jabra vendor ID `0x0b0e`. The telephony collection is usage page `0x0B`; the LEDs
  are **output** report usages from the LED page `0x08`: Off-Hook `0x17`, Ring `0x18`,
  Mute `0x09`. Off-hook = solid green ring, +ring = flashing green, +mute = solid red.
- hidapi's `HidDevice::write` requires the **report ID as the first byte** of the
  buffer. Devices without numbered reports would use `0x00`, but every known Jabra
  telephony collection numbers its reports — `map_leds` skips unnumbered output
  reports and Task 8's probe would expose a counterexample.
- Clearing an LED means **rewriting its whole report with that bit zero** — HID output
  reports set every field they carry, so all mapped reports are written on every
  change.

**Crate APIs (verified against docs.rs on 2026-09-01):**
- `hidapi` 2.6.7: `HidApi::new()`, `device_list()` → `DeviceInfo` (`vendor_id()`,
  `path()`, `product_string()`), `open_path()`, `HidDevice::write(&[u8])`,
  `HidDevice::get_report_descriptor(&mut buf)` with `MAX_REPORT_DESCRIPTOR_SIZE`.
- `hidreport` 0.6: `ReportDescriptor::try_from(&[u8])`, `output_reports()`,
  `find_output_report(&[u8])`; reports implement the `Report` trait (`report_id()`,
  `fields()`, and a byte-size accessor — the code below says `size_in_bytes()`; if the
  compile disagrees, check docs.rs/hidreport/0.6.0 for the exact name and adjust
  mechanically). `Field::Variable(var)` carries `var.usage`
  (`usage_page`/`usage_id` newtypes, convert with `u16::from`), `var.bits`
  (`RangeInclusive<usize>`), and `var.extract(&report_bytes)` for reading a value
  back. `ReportId` converts with `u8::from`.
- **Bit-index convention risk:** the code assumes `var.bits` counts from the start of
  the full report buffer *including* the report-ID byte (first data bit = 8). Task 3's
  round-trip test (compose → `find_output_report` → `extract`) is the arbiter: if it
  fails with bits off by 8, the fix is one `+ 8` in `map_leds`, nowhere else.
- The design assumes `hidapi::HidDevice: Send` (the driver moves the sink onto the
  blocking pool). Current hidapi implements it; if the compile disagrees, the fallback
  is a dedicated `std::thread` driver fed by a `std::sync::mpsc` channel instead of
  `spawn_blocking` — same `LedSink` trait, same tests.

**Repo:**
- `events::run` (`crates/client/src/events.rs:14`) is the single fan-out for server
  frames; `note_activity` and `dispatch_media` each re-parse the JSON text — the LED
  hook does the same (consistency over micro-optimization).
- Wake state changes are *local* (`wake::spawn`, `crates/client/src/wake.rs:191`);
  `"wake"` frames can also arrive on the events socket (`events.rs:143` renders them).
  The tracker handles both; double delivery is harmless (idempotent).
- `main.rs` test `parses_call_device_selectors_and_global_log_filter` destructures
  `Command::Call` **exhaustively** — adding the `led` field breaks it; Task 6 updates it.
- `deploy/rpi/README.md` has uncommitted local edits (unrelated trim-boot work). Edit
  it additively; never revert what's already in the working tree.
- Deploy to the Pi: `make deploy-pi PI_HOST=<host>`; it rsyncs `deploy/rpi/` wholesale,
  so a new rules file rides along with no Makefile change.

---

### Task 1: `led.rs` — phases, indications, and the tracker

**Files:**
- Create: `crates/client/src/led.rs`
- Modify: `crates/client/src/lib.rs` (add `pub mod led;` between `events` and `media`)

**Interfaces:**
- Produces: `led::Phase { Asleep, Listening, Thinking, Speaking }`,
  `led::Indication { phase: Phase, muted: bool }` with `bits() -> (bool, bool, bool)`
  (off-hook, ring, mute), `led::PhaseTracker` with `new(awake: bool)`,
  `indication() -> Indication`,
  `on_event(&mut self, kind: &str, payload: &Value) -> Option<Indication>`,
  `on_wake(&mut self, state: &WakeState) -> Option<Indication>` (both return `Some`
  only on change).

- [ ] **Step 1: Write the failing tests**

Create `crates/client/src/led.rs`:

```rust
//! Chatbot activity on the speakerphone's LED ring (docs/specs/jabra-led.md).
//!
//! Three inputs the client already has — wake state, turn events, the
//! server's turn-mute — fold into one [`Phase`], rendered as the standard
//! HID telephony LEDs: off-hook (solid green), +ring (flashing green),
//! +mute (solid red). Asleep is dark; a bot speaking outranks asleep so
//! out-of-session audio (timer alarms) lights the ring while it plays.

use serde_json::Value;
use voice_chatbot_protocol::WakeState;

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
                .on_event("rtf-user-transcription", &json!({"text": "hi", "final": false}))
                .is_none(),
            "partials mean the user is still talking: keep listening"
        );
        let thinking = tracker
            .on_event("rtf-user-transcription", &json!({"text": "hi", "final": true}))
            .unwrap();
        assert_eq!(thinking.phase, Phase::Thinking);
        assert!(
            tracker.on_event("rtf-function-call-start", &json!({})).is_none(),
            "a running tool is still thinking"
        );
        let speaking = tracker.on_event("rtf-bot-started-speaking", &json!({})).unwrap();
        assert_eq!(speaking.phase, Phase::Speaking);
        let back = tracker.on_event("rtf-bot-stopped-speaking", &json!({})).unwrap();
        assert_eq!(back.phase, Phase::Listening);
    }

    #[test]
    fn turn_mute_overlays_red_until_lifted() {
        let mut tracker = PhaseTracker::new(true);
        let muted = tracker.on_event("rtf-user-mute-started", &json!({})).unwrap();
        assert!(muted.muted);
        assert_eq!(muted.phase, Phase::Listening, "mute is an overlay, not a phase");
        let lifted = tracker.on_event("rtf-user-mute-stopped", &json!({})).unwrap();
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
        let alarm = tracker.on_event("rtf-bot-started-speaking", &json!({})).unwrap();
        assert_eq!(alarm.phase, Phase::Speaking);
        let done = tracker.on_event("rtf-bot-stopped-speaking", &json!({})).unwrap();
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
        assert!(tracker.on_event("rtf-bot-text", &json!({"text": "hi"})).is_none());
        assert!(tracker.on_event("media", &json!({})).is_none());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-client led::`
Expected: compile error — `Phase`, `Indication`, `PhaseTracker` not found.

- [ ] **Step 3: Implement**

Above the test module in `led.rs`:

```rust
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
                let done = payload.get("final").and_then(Value::as_bool).unwrap_or(false);
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
```

In `crates/client/src/lib.rs`, add `pub mod led;` in alphabetical position (after
`events`, before `media`).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-client led::`
Expected: all 8 tests PASS.

- [ ] **Step 5: `make check`, then commit**

```bash
git add crates/client/src/led.rs crates/client/src/lib.rs
git commit -m "feat(led): phase tracker for speakerphone LED indication"
```

---

### Task 2: `led/hid.rs` — LED bits from a report descriptor

**Files:**
- Create: `crates/client/src/led/hid.rs`
- Modify: `crates/client/src/led.rs` (add `pub mod hid;` at the top, after the doc comment)
- Modify: `crates/client/Cargo.toml` (add `hidreport = "0.6"`)

**Interfaces:**
- Consumes: nothing from Task 1 (pure descriptor work).
- Produces: `led::hid::LedBit { report_id: u8, report_len: usize, bit: usize }`,
  `led::hid::LedMap { off_hook, ring, mute: Option<LedBit> }` (`Default`, `PartialEq`),
  `led::hid::map_leds(descriptor: &[u8]) -> anyhow::Result<LedMap>`, and the usage
  constants `USAGE_OFF_HOOK`, `USAGE_RING`, `USAGE_MUTE` (`u16`, `pub(crate)`).

- [ ] **Step 1: Add the dependency**

In `crates/client/Cargo.toml`, dependencies section (alphabetical, after `futures`):

```toml
hidreport = "0.6"
```

- [ ] **Step 2: Write the failing tests**

Create `crates/client/src/led/hid.rs`:

```rust
//! Finding a Jabra telephony HID interface and the output-report bits that
//! drive its LEDs. Layouts differ across Jabra models, so report IDs and
//! bit positions come from the interface's own report descriptor — never
//! from hardcoded bytes (docs/specs/jabra-led.md, R3).

use anyhow::Result;
use hidreport::{Field, Report, ReportDescriptor};

/// GN Audio (Jabra) USB vendor ID.
pub(crate) const JABRA_VENDOR_ID: u16 = 0x0b0e;
/// HID LED usage page, and the telephony LEDs on it (HID Usage Tables §11).
const LED_PAGE: u16 = 0x08;
pub(crate) const USAGE_MUTE: u16 = 0x09;
pub(crate) const USAGE_OFF_HOOK: u16 = 0x17;
pub(crate) const USAGE_RING: u16 = 0x18;

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal telephony collection shaped like a Jabra's: one input
    /// report (hook switch + mute button, which the mapper must ignore) and
    /// one output report with the three LEDs. Synthetic — the real Speak2 40
    /// descriptor is archived by the hardware-validation task.
    pub(super) const SYNTHETIC_TELEPHONY_DESCRIPTOR: &[u8] = &[
        0x05, 0x0B, // Usage Page (Telephony)
        0x09, 0x05, // Usage (Headset)
        0xA1, 0x01, // Collection (Application)
        0x85, 0x01, //   Report ID (1)
        0x0B, 0x20, 0x00, 0x0B, 0x00, //   Usage (Telephony: Hook Switch)
        0x0B, 0x2F, 0x00, 0x0B, 0x00, //   Usage (Telephony: Phone Mute)
        0x75, 0x01, //   Report Size (1)
        0x95, 0x02, //   Report Count (2)
        0x81, 0x02, //   Input (Data,Var,Abs)
        0x75, 0x06, //   Report Size (6)
        0x95, 0x01, //   Report Count (1)
        0x81, 0x01, //   Input (Const) — padding
        0x85, 0x02, //   Report ID (2)
        0x05, 0x08, //   Usage Page (LED)
        0x09, 0x17, //   Usage (Off-Hook)
        0x09, 0x09, //   Usage (Mute)
        0x09, 0x18, //   Usage (Ring)
        0x75, 0x01, //   Report Size (1)
        0x95, 0x03, //   Report Count (3)
        0x91, 0x02, //   Output (Data,Var,Abs)
        0x75, 0x05, //   Report Size (5)
        0x95, 0x01, //   Report Count (1)
        0x91, 0x01, //   Output (Const) — padding
        0xC0, // End Collection
    ];

    #[test]
    fn maps_the_three_leds_from_a_telephony_descriptor() {
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let off_hook = map.off_hook.expect("off-hook led");
        let mute = map.mute.expect("mute led");
        let ring = map.ring.expect("ring led");
        assert_eq!(off_hook.report_id, 2);
        assert_eq!(off_hook.report_len, 2, "report id byte + one data byte");
        // Descriptor order is off-hook, mute, ring: consecutive bits. Absolute
        // offsets follow hidreport's convention; relative order is what matters.
        assert_eq!(mute.bit, off_hook.bit + 1);
        assert_eq!(ring.bit, off_hook.bit + 2);
    }

    #[test]
    fn a_descriptor_without_led_outputs_maps_to_nothing() {
        // Consumer-control page, input only — like a Jabra's volume interface.
        const NO_LEDS: &[u8] = &[
            0x05, 0x0C, // Usage Page (Consumer)
            0x09, 0x01, // Usage (Consumer Control)
            0xA1, 0x01, // Collection (Application)
            0x85, 0x03, //   Report ID (3)
            0x09, 0xE9, //   Usage (Volume Up)
            0x09, 0xEA, //   Usage (Volume Down)
            0x75, 0x01, 0x95, 0x02, //   1 bit x 2
            0x81, 0x02, //   Input (Data,Var,Abs)
            0x75, 0x06, 0x95, 0x01, 0x81, 0x01, // padding
            0xC0, // End Collection
        ];
        assert_eq!(map_leds(NO_LEDS).unwrap(), LedMap::default());
    }
}
```

Note the input usages use the long form (`0x0B …`) purely so the fixture is explicit
about pages; if `hidreport` rejects it, the two-byte local form (`0x05 0x0B` page +
`0x09 0x20` usage) is equivalent — the mapper never looks at input reports either way.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-client led::hid`
Expected: compile error — `LedBit`, `LedMap`, `map_leds` not found.

- [ ] **Step 4: Implement**

Above the test module:

```rust
/// One LED bit inside an output report: which report, how many bytes one
/// write of it is (report ID included), and the bit's index counted from
/// the start of that buffer (hidreport's convention).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LedBit {
    pub report_id: u8,
    pub report_len: usize,
    pub bit: usize,
}

/// Where the three telephony LEDs live on one interface, if they exist.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LedMap {
    pub off_hook: Option<LedBit>,
    pub ring: Option<LedBit>,
    pub mute: Option<LedBit>,
}

/// Pull the three LED bits out of a descriptor's numbered output reports.
pub fn map_leds(descriptor: &[u8]) -> Result<LedMap> {
    let parsed = ReportDescriptor::try_from(descriptor)
        .map_err(|error| anyhow::anyhow!("parse report descriptor: {error:?}"))?;
    let mut map = LedMap::default();
    for report in parsed.output_reports() {
        // Jabra telephony collections number their reports; hidapi needs the
        // ID as the write's first byte, so unnumbered ones are skipped.
        let Some(report_id) = report.report_id() else {
            continue;
        };
        let report_id = u8::from(report_id);
        let report_len = report.size_in_bytes();
        for field in report.fields() {
            let Field::Variable(var) = field else { continue };
            if u16::from(var.usage.usage_page) != LED_PAGE {
                continue;
            }
            let led = LedBit {
                report_id,
                report_len,
                bit: *var.bits.start(),
            };
            match u16::from(var.usage.usage_id) {
                USAGE_OFF_HOOK => map.off_hook.get_or_insert(led),
                USAGE_RING => map.ring.get_or_insert(led),
                USAGE_MUTE => map.mute.get_or_insert(led),
                _ => continue,
            };
        }
    }
    Ok(map)
}
```

Add `pub mod hid;` at the top of `crates/client/src/led.rs` (right under the module
doc comment).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-client led::hid`
Expected: both tests PASS. If the compile disputes a `hidreport` name
(`size_in_bytes`, field access), fix against docs.rs/hidreport/0.6.0 — the shapes are
right, the names may drift.

- [ ] **Step 6: `make check`, then commit**

```bash
git add crates/client/src/led.rs crates/client/src/led/hid.rs crates/client/Cargo.toml Cargo.lock
git commit -m "feat(led): map telephony LED bits from a HID report descriptor"
```

---

### Task 3: `led/hid.rs` — compose writes, open the device, prove the cross-build

**Files:**
- Modify: `crates/client/src/led/hid.rs`
- Modify: `crates/client/src/led.rs` (add the `LedSink` trait)
- Modify: `crates/client/Cargo.toml` (add `hidapi`)

**Interfaces:**
- Consumes: `LedMap` / `LedBit` / `map_leds` from Task 2.
- Produces: `led::LedSink` trait
  (`fn set(&mut self, off_hook: bool, ring: bool, mute: bool) -> anyhow::Result<()>`,
  supertrait `Send`); `led::hid::compose(&LedMap, bool, bool, bool) -> Vec<Vec<u8>>`;
  `led::hid::TelephonyLeds` (implements `LedSink`) with `describe() -> String`;
  `led::hid::open() -> anyhow::Result<TelephonyLeds>`.

- [ ] **Step 1: Add the dependency**

```toml
hidapi = { version = "2.6", default-features = false, features = ["linux-native-basic-udev", "macos-shared-device"] }
```

`linux-native-basic-udev` is the hidraw backend written in Rust with sysfs-based
enumeration — no libudev or C hidapi to link, which is what keeps `Cross.toml`
untouched. `macos-shared-device` keeps macOS opens non-exclusive.

- [ ] **Step 2: Verify the Pi cross-build immediately**

Run: `make client-build-pi`
Expected: builds clean with no `Cross.toml` changes. This is the plan's riskiest
assumption; surface a failure now, before code depends on it. (Per Global
constraints: if it fails, stop and revisit the feature choice — e.g. plain
`linux-native` — rather than adding apt packages.)

- [ ] **Step 3: Write the failing tests**

Append to the test module in `led/hid.rs`:

```rust
    #[test]
    fn composed_reports_round_trip_through_the_descriptor() {
        let parsed = ReportDescriptor::try_from(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let buffers = compose(&map, true, false, true);
        assert_eq!(buffers.len(), 1, "all three leds live in one report");
        let buffer = &buffers[0];
        assert_eq!(buffer[0], 2, "hidapi wants the report id first");
        assert_eq!(buffer.len(), 2);
        // Read the buffer back through hidreport itself: pins the bit-index
        // convention without hardcoding it.
        let report = parsed.find_output_report(buffer).expect("report 2");
        let mut lit = std::collections::HashMap::new();
        for field in report.fields() {
            if let Field::Variable(var) = field {
                let value: u32 = var.extract(buffer).unwrap().into();
                lit.insert(u16::from(var.usage.usage_id), value);
            }
        }
        assert_eq!(lit[&USAGE_OFF_HOOK], 1);
        assert_eq!(lit[&USAGE_RING], 0);
        assert_eq!(lit[&USAGE_MUTE], 1);
    }

    #[test]
    fn clearing_writes_the_report_as_zeros() {
        let map = map_leds(SYNTHETIC_TELEPHONY_DESCRIPTOR).unwrap();
        let buffers = compose(&map, false, false, false);
        assert_eq!(buffers.len(), 1, "a mapped report is written even all-clear");
        assert_eq!(buffers[0][0], 2);
        assert!(buffers[0][1..].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn an_empty_map_composes_no_writes() {
        assert!(compose(&LedMap::default(), true, true, true).is_empty());
    }
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-client led::hid`
Expected: compile error — `compose` not found.

- [ ] **Step 5: Implement**

In `crates/client/src/led.rs`, after `Indication`:

```rust
/// One write of the three LED bits to whatever renders them. Split from the
/// device so the driver task is testable without hardware.
pub trait LedSink: Send {
    fn set(&mut self, off_hook: bool, ring: bool, mute: bool) -> anyhow::Result<()>;
}
```

In `led/hid.rs`, extend the imports and add:

```rust
use std::ffi::CStr;

use anyhow::{bail, Context};
use hidapi::{HidApi, HidDevice, MAX_REPORT_DESCRIPTOR_SIZE};

use crate::led::LedSink;

/// Build the output-report buffers (report ID first, as hidapi wants) that
/// show exactly the given bits. Every report carrying a mapped LED is
/// emitted, so clearing a bit rewrites its report with that bit zero.
pub fn compose(map: &LedMap, off_hook: bool, ring: bool, mute: bool) -> Vec<Vec<u8>> {
    let mut buffers: Vec<Vec<u8>> = Vec::new();
    for (led, on) in [(map.off_hook, off_hook), (map.ring, ring), (map.mute, mute)] {
        let Some(led) = led else { continue };
        let buffer = match buffers.iter_mut().find(|buffer| buffer[0] == led.report_id) {
            Some(buffer) => buffer,
            None => {
                buffers.push(vec![0u8; led.report_len]);
                let buffer = buffers.last_mut().unwrap();
                buffer[0] = led.report_id;
                buffer
            }
        };
        if on {
            buffer[led.bit / 8] |= 1 << (led.bit % 8);
        }
    }
    buffers
}

/// A Jabra telephony HID interface with its LED layout.
pub struct TelephonyLeds {
    device: HidDevice,
    map: LedMap,
    product: String,
}

impl TelephonyLeds {
    /// One console-worthy line: what was opened and where its LEDs sit.
    pub fn describe(&self) -> String {
        format!("{} ({:?})", self.product, self.map)
    }
}

impl LedSink for TelephonyLeds {
    fn set(&mut self, off_hook: bool, ring: bool, mute: bool) -> Result<()> {
        for buffer in compose(&self.map, off_hook, ring, mute) {
            self.device
                .write(&buffer)
                .with_context(|| format!("write led report {:#04x}", buffer[0]))?;
        }
        Ok(())
    }
}

/// The linux-native backend may not implement `get_report_descriptor`; the
/// descriptor is also a plain sysfs file, keyed by the hidraw node name.
#[cfg(target_os = "linux")]
fn descriptor_from_sysfs(path: &CStr) -> Option<Vec<u8>> {
    let node = std::path::Path::new(path.to_str().ok()?).file_name()?.to_str()?;
    std::fs::read(format!("/sys/class/hidraw/{node}/device/report_descriptor")).ok()
}

#[cfg(not(target_os = "linux"))]
fn descriptor_from_sysfs(_path: &CStr) -> Option<Vec<u8>> {
    None
}

/// Find the first Jabra HID interface whose descriptor offers an Off-Hook
/// LED output — that is the telephony collection. A Jabra presents several
/// HID interfaces (consumer volume, vendor pages); content, not interface
/// number, is what identifies the right one.
pub fn open() -> Result<TelephonyLeds> {
    let api = HidApi::new().context("initialize hidapi")?;
    let mut tried = Vec::new();
    for info in api.device_list().filter(|d| d.vendor_id() == JABRA_VENDOR_ID) {
        let path = info.path();
        let shown = path.to_string_lossy().into_owned();
        let device = match api.open_path(path) {
            Ok(device) => device,
            Err(error) => {
                tried.push(format!("{shown}: open: {error}"));
                continue;
            }
        };
        let mut buffer = [0u8; MAX_REPORT_DESCRIPTOR_SIZE];
        let descriptor = match device.get_report_descriptor(&mut buffer) {
            Ok(length) => buffer[..length].to_vec(),
            Err(_) => match descriptor_from_sysfs(path) {
                Some(descriptor) => descriptor,
                None => {
                    tried.push(format!("{shown}: no report descriptor"));
                    continue;
                }
            },
        };
        let map = match map_leds(&descriptor) {
            Ok(map) => map,
            Err(error) => {
                tried.push(format!("{shown}: {error}"));
                continue;
            }
        };
        if map.off_hook.is_none() {
            tried.push(format!("{shown}: no off-hook led"));
            continue;
        }
        let product = info
            .product_string()
            .unwrap_or("Jabra")
            .trim()
            .to_string();
        return Ok(TelephonyLeds { device, map, product });
    }
    if tried.is_empty() {
        bail!("no Jabra usb hid device present (vendor {JABRA_VENDOR_ID:#06x})");
    }
    bail!("no Jabra telephony led interface: {}", tried.join("; "))
}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-client led::hid`
Expected: all 5 hid tests PASS. If `composed_reports_round_trip_through_the_descriptor`
fails with the LEDs found at the wrong offsets, hidreport counts bits without the
report-ID byte: add `+ 8` where `LedBit::bit` is set in `map_leds` (and only there),
re-run.

- [ ] **Step 7: `make check`, then commit**

```bash
git add crates/client/src/led.rs crates/client/src/led/hid.rs crates/client/Cargo.toml Cargo.lock
git commit -m "feat(led): open a Jabra telephony interface and write LED reports"
```

---

### Task 4: `LedController` — coalescing driver that clears on drop

**Files:**
- Modify: `crates/client/src/led.rs`

**Interfaces:**
- Consumes: `PhaseTracker`, `Indication`, `LedSink` (Tasks 1, 3).
- Produces: `led::LedController` (`Clone`) with
  `start(sink: Box<dyn LedSink>, awake_at_start: bool) -> (LedController, tokio::task::JoinHandle<()>)`,
  `on_event(&self, input: &str)` (raw events-WebSocket text frame),
  `on_wake(&self, state: &WakeState)`. The `JoinHandle` resolves after the ring is
  cleared, once every `LedController` clone is dropped.

- [ ] **Step 1: Write the failing test**

Append to the test module in `led.rs`:

```rust
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p voice-chatbot-client led::tests::controller`
Expected: compile error — `LedController` not found.

- [ ] **Step 3: Implement**

In `led.rs` (new imports at the top: `use std::sync::{Arc, Mutex};` and
`use tokio::sync::watch;`):

```rust
/// Clonable handle the events and wake tasks feed. A driver task owns the
/// sink; state changes coalesce through a watch channel, so a burst of
/// events costs at most one write per settled state. Dropping every clone
/// clears the ring and ends the driver (its JoinHandle resolves after the
/// clear, so a session can bound its teardown).
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-client led::`
Expected: all led tests PASS, including the new controller test.

- [ ] **Step 5: `make check`, then commit**

```bash
git add crates/client/src/led.rs
git commit -m "feat(led): controller task that coalesces writes and clears on drop"
```

---

### Task 5: wire the controller into the events and wake tasks

**Files:**
- Modify: `crates/client/src/events.rs:14-20` (signature) and `:56-67` (hook)
- Modify: `crates/client/src/wake.rs:191-216` (`spawn`)

**Interfaces:**
- Consumes: `LedController::{on_event, on_wake}` (Task 4).
- Produces: `events::run(url, shutdown, media, outbound, activity, led: Option<LedController>)`;
  `wake::spawn(gate, input, outbound, led: Option<LedController>) -> mpsc::Receiver<Vec<i16>>`.
  Task 6's `run_session` calls both with these exact arities.

- [ ] **Step 1: Extend `events::run`**

Add the parameter after `activity`:

```rust
pub async fn run(
    url: Url,
    mut shutdown: watch::Receiver<bool>,
    mut media: Option<MediaPlayer>,
    mut outbound: tokio::sync::mpsc::UnboundedReceiver<String>,
    activity: crate::wake::Activity,
    led: Option<crate::led::LedController>,
) {
```

In the `Message::Text` arm, directly after `note_activity(&activity, &text);`:

```rust
                        if let Some(led) = &led {
                            led.on_event(&text);
                        }
```

- [ ] **Step 2: Extend `wake::spawn`**

Add the parameter and the hook beside the existing state-change print:

```rust
pub fn spawn(
    mut gate: ClientWakeGate,
    mut input: mpsc::Receiver<Vec<i16>>,
    outbound: mpsc::UnboundedSender<String>,
    led: Option<crate::led::LedController>,
) -> mpsc::Receiver<Vec<i16>> {
```

and inside the loop, where `report` is handled:

```rust
            if let Some(state) = report {
                println!("{}", describe(&state));
                if let Some(led) = &led {
                    led.on_wake(&state);
                }
                let _ = outbound.send(wake_frame(&state));
            }
```

- [ ] **Step 3: Verify it fails to compile at the call sites, then patch them minimally**

Run: `cargo check -p voice-chatbot-client`
Expected: errors in `main.rs` at the `events::run` spawn and the `wake::spawn` call —
proof the signatures changed. Patch both call sites with `None` **for now** (Task 6
replaces them with the real controller):

- `main.rs` events spawn: append `None,` after `activity.clone(),`
- `main.rs` wake branch: `voice_chatbot_client::wake::spawn(gate, input_rx, outbound_tx, None)`

- [ ] **Step 4: `make check`, then commit**

Existing `events::` and `wake::` unit tests cover `render`/`signal_for`/`Activity`,
none of which changed; they must all still pass.

```bash
git add crates/client/src/events.rs crates/client/src/wake.rs crates/client/src/main.rs
git commit -m "feat(led): feed the controller from the events and wake tasks"
```

---

### Task 6: CLI — `--led` mode, session integration, `led-test`

**Files:**
- Modify: `crates/client/src/main.rs` (Cli, `run_call`, `run_session`, tests)

**Interfaces:**
- Consumes: `led::hid::open`, `TelephonyLeds::describe`, `LedController::start`
  (exact shapes from Tasks 3–4), the Task 5 call-site arities.
- Produces: `--led auto|off` (env `LED`, default auto) on `call`; the `led-test`
  subcommand.

- [ ] **Step 1: Write the failing Cli tests**

In the `main.rs` test module:

```rust
    #[test]
    fn parses_led_mode() {
        let cli = Cli::try_parse_from(["client", "call", "--led", "off"]).unwrap();
        match cli.command {
            Command::Call { led, .. } => assert_eq!(led, LedMode::Off),
            _ => panic!("expected call"),
        }
        let cli = Cli::try_parse_from(["client", "call"]).unwrap();
        match cli.command {
            Command::Call { led, .. } => assert_eq!(led, LedMode::Auto, "auto by default"),
            _ => panic!("expected call"),
        }
        assert!(matches!(
            Cli::try_parse_from(["client", "led-test"]).unwrap().command,
            Command::LedTest
        ));
    }
```

Also extend `parses_call_device_selectors_and_global_log_filter`: its `Command::Call`
destructure is exhaustive, so add `led,` to the pattern and
`assert_eq!(led, LedMode::Auto);` to its body.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-client --bin voice-chatbot-client`
Expected: compile error — `LedMode` / `led` field / `LedTest` not found.

- [ ] **Step 3: Implement the Cli surface**

```rust
/// Whether to drive a speakerphone's LED ring (docs/specs/jabra-led.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum LedMode {
    /// Drive a Jabra's telephony LEDs when one is plugged in.
    Auto,
    /// Never touch the LEDs.
    Off,
}
```

New variant on `Command`:

```rust
    /// Probe the speakerphone's LEDs: open the Jabra telephony interface and
    /// cycle off -> listening -> thinking -> muted -> off.
    LedTest,
```

New arg on `Call`, after `wake_session_secs`:

```rust
        /// Show chatbot activity on the speakerphone's LED ring.
        #[arg(long, env = "LED", value_enum, default_value_t = LedMode::Auto)]
        led: LedMode,
```

Thread it through: destructure `led` in `main`'s `Command::Call` arm, pass it to
`run_call(..., led)`, add `led_mode: LedMode` params to `run_call` and `run_session`
(both already allow `too_many_arguments` or take few enough).

`main`'s match gains:

```rust
        Command::LedTest => led_test()?,
```

and at file scope:

```rust
/// Hardware probe for docs/specs/jabra-led.md: what does each LED state
/// look (and sound) like on the attached device?
fn led_test() -> Result<()> {
    use voice_chatbot_client::led::LedSink;
    let mut leds = voice_chatbot_client::led::hid::open()?;
    println!("driving {}", leds.describe());
    let steps: [(&str, (bool, bool, bool)); 5] = [
        ("off (asleep)", (false, false, false)),
        ("listening: off-hook -- expect solid green", (true, false, false)),
        (
            "thinking: off-hook + ring -- expect flashing green, and LISTEN: this must be silent",
            (true, true, false),
        ),
        ("muted: off-hook + mute -- expect solid red", (true, false, true)),
        ("off again", (false, false, false)),
    ];
    for (what, (off_hook, ring, mute)) in steps {
        println!("{what}");
        leds.set(off_hook, ring, mute)?;
        std::thread::sleep(Duration::from_secs(3));
    }
    Ok(())
}
```

- [ ] **Step 4: Construct the controller in `run_session`**

Replace Task 5's placeholder `None`s. After the `media` block (`main.rs:345-361`),
following its warn-once shape:

```rust
    // Chatbot activity on the speakerphone's LED ring. A missing device or
    // missing hidraw permissions degrade to running dark, like ffmpeg above.
    let (led, led_done) = match led_mode {
        LedMode::Off => (None, None),
        LedMode::Auto => match voice_chatbot_client::led::hid::open() {
            Ok(leds) => {
                if describe_devices {
                    eprintln!("leds:   {}", leds.describe());
                }
                let (controller, done) =
                    voice_chatbot_client::led::LedController::start(Box::new(leds), wake.is_none());
                (Some(controller), Some(done))
            }
            Err(error) => {
                if describe_devices {
                    tracing::info!(%error, "no speakerphone leds; running without");
                }
                (None, None)
            }
        },
    };
```

Pass `led.clone()` as the last argument of the `events::run` spawn and of
`wake::spawn` (the push-mode branch touches only `events::run`). Then in the teardown,
after the existing `event_task` timeout block:

```rust
    // The led driver darkens the ring once every handle is gone (the events
    // and wake tasks hold the other clones); bounded so a wedged device
    // write cannot stall teardown or leave the ring lit into the next state.
    drop(led);
    if let Some(done) = led_done {
        let _ = tokio::time::timeout(Duration::from_secs(1), done).await;
    }
```

(The 2 s `RECONNECT_DELAY` between sessions is what keeps this clear from racing the
next session's opening write.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-client`
Expected: all client tests PASS, including `parses_led_mode` and the updated
exhaustive destructure test.

- [ ] **Step 6: `make check`, run the binary once, then commit**

Sanity on this machine (no Jabra attached): `cargo run -p voice-chatbot-client -- led-test`
Expected: exits with `no Jabra usb hid device present (vendor 0x0b0e)` — the
degradation message, not a panic.

```bash
git add crates/client/src/main.rs
git commit -m "feat(led): --led mode, session wiring, and a led-test probe"
```

---

### Task 7: deploy — udev rule, installer, config docs

**Files:**
- Create: `deploy/rpi/99-voice-chatbot-jabra.rules`
- Modify: `deploy/rpi/install.sh` (after the unit install / `systemctl daemon-reload`,
  before the smoke test)
- Modify: `deploy/rpi/env.example`
- Modify: `deploy/rpi/README.md` — **additively**; it has unrelated uncommitted edits
  in the working tree, do not revert them

**Interfaces:**
- Consumes: nothing from the Rust tasks; ships alongside them.
- Produces: hidraw access for the service user on the Pi.

- [ ] **Step 1: Write the udev rule**

`deploy/rpi/99-voice-chatbot-jabra.rules`:

```
# Jabra (GN Audio) HID interfaces: the client drives the speakerphone's
# telephony LEDs over hidraw (docs/specs/jabra-led.md), and hidraw nodes are
# root-only by default. The service user is already in the audio group for
# /dev/snd, so the same group carries the speakerphone's other device node.
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="0b0e", MODE="0660", GROUP="audio"
```

- [ ] **Step 2: Install it from `install.sh`**

After the `systemctl daemon-reload` line:

```bash
# LED control needs the Jabra's hidraw node (see the rules file). Reload and
# retrigger so an already-plugged speakerphone gets the group without a reboot.
install -m 0644 "$SRC_DIR/99-voice-chatbot-jabra.rules" /etc/udev/rules.d/99-voice-chatbot-jabra.rules
udevadm control --reload-rules
udevadm trigger --subsystem-match=hidraw
```

- [ ] **Step 3: Document the knob in `env.example`**

After the `NO_WAKE` block:

```
# Chatbot activity on the speakerphone's LED ring (Jabra telephony HID):
# dark when asleep, green when listening, flashing green while thinking, red
# while the mic is gated. auto drives it when a Jabra is present; off never
# touches the LEDs.
#LED=auto
```

- [ ] **Step 4: README section**

Append to `deploy/rpi/README.md` (leaving the existing text and any pending local
edits untouched), matching its tone:

```markdown
## Speakerphone LEDs

The client shows chatbot activity on the Jabra's LED ring: dark when asleep,
solid green when listening, flashing green while thinking, red while the mic
is gated. This rides the speakerphone's standard telephony HID interface
(docs/specs/jabra-led.md), not the audio path, and needs `/dev/hidraw*`
access: install.sh ships a udev rule opening Jabra hidraw nodes to the
`audio` group the service already runs in. `LED=off` in `.env` disables it;
`voice-chatbot-client led-test` (with the service stopped) cycles the states
for a look. Running the client by hand on a dev machine needs the same udev
rule, or a `TAG+="uaccess"` variant for desktop logins.
```

- [ ] **Step 5: Shellcheck and commit**

Run: `shellcheck deploy/rpi/install.sh` (matching however the repo lints shell — if
shellcheck isn't installed, `bash -n deploy/rpi/install.sh` at minimum). `make check`
for the Rust side stays green untouched.

```bash
git add deploy/rpi/99-voice-chatbot-jabra.rules deploy/rpi/install.sh deploy/rpi/env.example deploy/rpi/README.md
git commit -m "feat(deploy): grant and document Jabra hidraw access for LEDs"
```

Careful with `deploy/rpi/README.md`: stage only your hunks if the trim-boot edits are
still uncommitted (`git add -p deploy/rpi/README.md`).

---

### Task 8: hardware validation on the Pi (manual)

**Files:**
- Create: `docs/research/jabra-speak2-40-hid.md` (findings + descriptor dump)
- Possibly modify: `crates/client/src/led.rs` (only if the ring is audible — see below)
- Modify: `docs/specs/jabra-led.md` (resolve the Open questions section)

This task needs the Speak2 40 plugged into the Pi (or any Linux box with the branch
deployed). Everything is observation; the one contingent code change is pre-planned.

- [ ] **Step 1: Deploy the branch**

```bash
make client-build-pi
make deploy-pi PI_HOST=<pi-host>
```

- [ ] **Step 2: Archive the real descriptor**

```bash
ssh <pi-host> 'for h in /sys/class/hidraw/hidraw*; do echo "== $h"; cat $h/device/uevent | grep -E "HID_NAME|HID_ID"; done'
ssh <pi-host> 'xxd /sys/class/hidraw/hidrawN/device/report_descriptor'   # N = the Jabra telephony one; try each 0B0E node
```

Save the dump and the interface inventory into `docs/research/jabra-speak2-40-hid.md`
with a sentence on which node carried the LEDs and what `led-test` printed as the
mapped layout.

- [ ] **Step 3: Run the probe**

```bash
ssh <pi-host> 'sudo systemctl stop voice-chatbot-client'
ssh <pi-host> 'cd /opt/voice-chatbot && ./voice-chatbot-client led-test'
```

Checklist to record in the research note:
- off → ring dark; listening → solid green; thinking → flashing green; muted → solid
  red; off again → dark.
- **Silence check:** the flashing-green step makes no sound. If the device plays its
  ringtone, apply the pre-planned fallback: in `Indication::bits`, change the ring
  line to `let ring = false;` with a comment naming the audible-ring finding, update
  `bits_render_each_phase` (Thinking → `(true, false, false)`), note it in the spec,
  commit as `fix(led): thinking falls back to solid green; the ring usage is audible`.
- Permission check: `led-test` ran as the normal user (udev rule working), not root.

- [ ] **Step 4: Live conversation check**

```bash
ssh <pi-host> 'sudo systemctl start voice-chatbot-client'
```

- Wake it; watch asleep → green → (speak a command) → flashing green → reply.
- Turn-mute red appears during the reply if the pipeline's turn-mute strategy is
  active (it may legitimately never show otherwise — record which).
- **Audio regression check (spec Open question 2):** conversation sounds identical to
  `LED=off` (set it in `/opt/voice-chatbot/.env`, restart, compare) — no call-start
  chime, no level/beamforming change. The capture-stall history
  (`crates/client/src/audio.rs` doc comment) makes "the mic still works" the headline
  assertion here.
- `sudo systemctl stop voice-chatbot-client` → ring goes dark within a second.

- [ ] **Step 5: Resolve the spec and commit**

Update `docs/specs/jabra-led.md`: mark the three Open questions answered with what was
observed, flip **Status** to reflect it. Commit research note + spec update:

```bash
git add docs/research/jabra-speak2-40-hid.md docs/specs/jabra-led.md
git commit -m "docs(led): Speak2 40 hardware findings; resolve spec open questions"
```

---

## Self-review (against the spec)

- **R1 phases/mapping** → Task 1 (`Phase`, `Indication::bits`, priority order tested).
- **R2 event derivation** → Task 1 (`on_event`/`on_wake`, all seven inputs tested,
  including socket-borne `wake` frames); Task 5 delivers the frames.
- **R3 discovery/degradation** → Task 3 (`open()` by descriptor content, never
  hardcoded bytes), Task 6 (warn-once, run dark; verified by the no-device
  `led-test` run).
- **R4 lifecycle** → Task 4 (initial write clears stale state; coalescing; clear on
  drop; dormant-on-write-error), Task 6 (bounded teardown await), Task 8 step 4
  (observed dark-on-stop).
- **R5 config** → Task 6 (`--led`/`LED`), Task 7 (env.example).
- **R6 probe** → Task 6 (`led-test`), exercised in Task 8.
- **R7 deploy** → Task 7 (rule, installer, README), verified unprivileged in Task 8.
- **R8 deps/cross-build** → Task 2/3 (pinned deps; `make client-build-pi` gate before
  any code depends on hidapi).
- **Open questions** → Task 8, with the audible-ring fallback pre-planned as a
  one-line change plus test update, not a redesign.

Known deliberate gaps, matching the spec's non-goals: no input-report reading, no
device selector, no server-commanded output. Type names were cross-checked across
tasks: `LedBit`/`LedMap`/`map_leds`/`compose`/`TelephonyLeds`/`describe`/`open`
(Tasks 2–3 ↔ 6), `LedSink::set(bool, bool, bool)` (Tasks 3 ↔ 4 ↔ 6),
`LedController::start -> (Self, JoinHandle<()>)` and `on_event(&str)`/`on_wake(&WakeState)`
(Tasks 4 ↔ 5 ↔ 6).
