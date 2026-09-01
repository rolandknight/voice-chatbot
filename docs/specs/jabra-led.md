# jabra-led — chatbot activity on the speakerphone's LED ring

**Date:** 2026-09-01
**Status:** Proposed. Protocol facts below are researched from public sources; the two
items under [Open questions](#open-questions-verify-on-hardware) need one probe session
with the device before the implementation is called done.
**Implementation plan:** `docs/plans/jabra-led.md`

## Goal

The Jabra Speak2 40 sitting on the table shows what the chatbot is doing — asleep,
listening, thinking, or mic-gated — on its LED ring, driven by the native client. Today
the only activity indication is stdout lines (`[awake: marvin 0.87]`,
`[assistant speaking]`), which a headless Pi satellite never shows anyone.

This delivers the client-side half of the PRD's status-indication idea
(`docs/prd/prd.md` RTC-2 `state: listening/thinking/speaking/idle`, SENS-2 LEDs) using
events the client already receives — no protocol or server change.

## Device research

### Control path: standard USB HID telephony

Every Jabra Speak exposes a **USB HID Telephony collection** (usage page `0x0B`) for
softphone call control, separate from its USB audio interfaces. The host drives the
LEDs by writing plain HID **output reports** carrying usages from the LED page
(`0x08`). Jabra's own WebHID demo does exactly this
(`device.sendReport(reportId, bits)`):

| Usage (page `0x08`) | ID | Speak2 40 ring shows |
|---|---|---|
| Off-Hook | `0x17` | solid green ("in call") |
| Ring | `0x18` | flashing green ("incoming call") |
| Mute | `0x09` | solid red (with off-hook set) |
| Hold / Microphone / On-Line | `0x20` / `0x21` / `0x2A` | present on some models |
| Telephony Ringer (page `0x0B`) | `0x9E` | audible ringtone — separate from the Ring LED |

The Speak2 40 user manual (§5.4 "Status LED ring") confirms the ring's full vocabulary:
solid green = in call, flashing green = incoming call, solid red = muted, off =
standby. Pink states (firmware update, factory reset) are device-internal. So the host
can produce **four distinguishable states**: off, solid green, flashing green, solid
red.

### Linux plumbing

- Mainline ships `hid-jabra` (authored by Jabra), binding all VID `0x0B0E` HID
  interfaces; it exists to keep the GN vendor usages out of the input layer. The
  telephony collection stays readable/writable from userspace via `/dev/hidraw*`
  **alongside** the kernel driver — no interface claiming or detaching (unlike libusb).
- The HID interface is disjoint from the USB audio interface, so LED traffic cannot
  disturb ALSA capture/playback.
- `/dev/hidraw*` is `root:root 0600` on Pi OS by default → a udev rule is required
  (see requirements).
- Kernel-list archaeology (2021 "USB HID headset features" series): Jabra devices only
  emit **mute-button** input events while off-hook, and only when the host echoes the
  mute LED state back. Irrelevant for output-only LED driving; load-bearing for the
  deferred button-input phase.

### Rejected alternatives

- **libjabra / Jabra Linux SDK** (what `aarnaud/jabra-busylight` uses): closed-source,
  x86-oriented; wrong for an aarch64 Pi and a Rust client.
- **GN vendor HID protocol** (usage page `0xFF30`): only needed for multicolor
  busylights on Evolve headsets; the Speak2 40 has no such hardware.
- **Teams-button purple LED**: driven by the Teams vendor protocol; out of scope.

## Requirements

### R1 — Phase model and LED mapping

The client derives a single activity phase from three orthogonal inputs it already
has — `awake` (wake gate state; push mode counts as always awake), `activity`
(quiet / thinking / bot speaking, from turn events), `muted` (server turn-mute) — and
renders it:

| Phase | Derivation (first match wins) | LED usages set | Ring shows |
|---|---|---|---|
| Speaking | bot is speaking | off-hook | solid green |
| Asleep | not awake | none | off |
| Thinking | user turn ended / tool running, no bot speech yet | off-hook + ring | flashing green |
| Listening | otherwise | off-hook | solid green |

`muted` is an overlay: whenever set (and the phase is not Asleep), the Mute usage is
also set → solid red. Turn-mute typically spans the bot's reply, so in that
configuration Speaking shows red ("not listening to you"), which is the device's native
meaning. Speaking outranks Asleep so out-of-session audio (timer alarms) lights the
ring while it plays, then goes dark again.

### R2 — Event derivation (no protocol change)

| Input | Source | Effect |
|---|---|---|
| `WakeState::Awake` / `Asleep` | local wake gate (`wake::spawn`), and `"wake"` frames on the events socket | `awake` := true / false (asleep also clears `muted`) |
| `rtf-user-transcription`, `final: true` | events socket | activity := Thinking |
| `rtf-user-transcription`, `final: false` | events socket | activity := Quiet (user is talking) |
| `rtf-function-call-start` | events socket | activity := Thinking |
| `rtf-bot-started-speaking` | events socket | activity := Speaking |
| `rtf-bot-stopped-speaking` | events socket | activity := Quiet |
| `rtf-user-mute-started` / `-stopped` | events socket (forwarded by the server today, ignored by the client today) | `muted` := true / false |

### R3 — Discovery and graceful degradation

Enumerate HID interfaces with vendor ID `0x0B0E`, parse each report descriptor, and
drive the first interface offering an Off-Hook LED output usage. Layouts vary across
Jabra models, so report IDs and bit positions come from the descriptor, never from
hardcoded bytes. No Jabra present, permission denied, or `LED=off` → log once at
session start and run exactly as today. Mirrors `MediaPlayer::is_available`.

### R4 — Lifecycle

- LED state is written on session start (also clearing stale LEDs from a crashed
  predecessor) and on every phase change; unchanged phases produce no writes.
- On session end (Ctrl-C, connection lost, shutdown) the ring is cleared to dark.
- Write failure mid-session (device unplugged) is logged at debug and LED driving goes
  dormant; unplugging the Jabra kills the audio session anyway, and the reconnect's new
  session re-opens the device.

### R5 — Configuration

`--led auto|off` on `call`, env `LED` (bare name, matching `SERVER_URL` /
`INPUT_DEVICE` convention), default `auto`. Documented in `deploy/rpi/env.example`.

### R6 — Probe subcommand

`voice-chatbot-client led-test`: opens the device, prints what it found (product
string, report layout), cycles off → listening → thinking → muted → off at 3 s per
step (slower than a real transition, for observability). This is both the
hardware-validation harness and a field diagnostic.

### R7 — Deploy (Raspberry Pi)

A udev rule shipped in `deploy/rpi/` and installed by `install.sh` grants the hidraw
nodes of VID `0x0B0E` to the `audio` group (`MODE="0660", GROUP="audio"`) — the
service and its user are already in `audio`, so no unit or group-membership change.
The rule rides to the Pi via the existing `rsync deploy/rpi/` in `make deploy-pi`.
Dev machines running the client by hand need the same rule (or a `TAG+="uaccess"`
variant); documented in `deploy/rpi/README.md`.

### R8 — Dependencies and cross-build

- `hidapi` (pinned `2.6`, `default-features = false`, features
  `linux-native-basic-udev` + `macos-shared-device`): hidraw backend implemented in
  Rust with no libudev/libusb linkage, so `Cross.toml`'s pre-build package list should
  need no additions; `macos-shared-device` keeps macOS opens non-exclusive. The Pi
  cross-build (`make client-build-pi`) must be verified green the moment the dependency
  lands — the `audiopus_sys` note in `Cross.toml` shows how silently this class of
  thing can rot.
- `hidreport` (pinned `0.6`) parses report descriptors; HID descriptor parsing has
  enough corners (global item stacks, push/pop) that the maintained parser is the right
  primitive over a hand-rolled one.

### Non-goals (this iteration)

- Reading telephony **input** reports (physical mute button gating capture, hook
  buttons as wake/sleep). Deferred; needs the off-hook/echo quirk handled.
- Teams-button LED, vendor GN protocol, multicolor anything.
- Server-commanded outputs (SENS-2 proper) — this is client-derived only.
- A device selector for multiple simultaneous Jabras (first match wins).

## Open questions (verify on hardware)

1. **Ring silence.** The audible ringer is a separate usage (`0x9E`), so setting only
   the Ring LED usage should flash silently — confirmed nowhere in public sources.
   If the device rings audibly, Thinking falls back to off-hook only (solid green,
   same as Listening).
2. **Off-hook side effects on audio.** Holding off-hook puts the device in "in call"
   state; verify capture/playback behave identically (no call-start chime, no DSP
   change) with LEDs driven vs. `LED=off`.
3. **Actual descriptor layout** of the Speak2 40's telephony collection (report IDs,
   which optional usages exist). Dump `/sys/class/hidraw/hidrawN/device/report_descriptor`
   on the Pi and archive it in `docs/research/`.

## Sources

- [Jabra telephony WebHID demo](https://github.com/pehandersen-jabra/telephony-webhid-demo) — usage map and output-report mechanics
- [Speak2 40 user manual, §5.4 Status LED ring](https://www.jabra.com/_/media/Jabra_VXi_Product-Documentation/Jabra-Speak2-40/User-Manuals/RevA/Jabra-Speak2-40_User-Manual_EN_English_RevA.pdf)
- [LKML: "Add support for common USB HID headset features"](https://lkml.iu.edu/hypermail/linux/kernel/2107.0/01639.html) and the [mute-LED sync patch](https://lkml.iu.edu/hypermail/linux/kernel/2107.0/01638.html)
- [aarnaud/jabra-busylight](https://github.com/aarnaud/jabra-busylight) — the libjabra route (rejected)
- [Jabra for Developers: Linux](https://developer.jabra.com/sdks-and-tools/linux)
- `drivers/hid/hid-jabra.c` in mainline Linux (present on Pi OS and desktop kernels)
