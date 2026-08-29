# Embedded media player — decode BBC radio, shows and sound effects inside the client and mix them into the call's own output stream

**Date:** 2026-08-28 · **Branch:** `embedded-media-player` · **Crates:** `client`, `protocol`, `server`

## Goal

Delete the `mpv` subprocess. Decode media with `ffmpeg` into raw PCM, mix it
into the CPAL output stream the call already owns, and duck it with a gain ramp
instead of a JSON-IPC pause.

Three things fall out of that, in priority order:

1. **Media plays on the call's device.** Today a call holds the speakerphone
   directly through CPAL, so radio comes out of the *system default sink*
   instead (`crates/client/README.md:130`, `README.md:24`). Mixing into the
   call's own stream is the only fix that does not involve giving up exclusive
   access.

   Measured 2026-08-28, and stronger than the READMEs imply: while the call
   holds `plughw:CARD=UC,DEV=0` (what `alias_rank`, `audio.rs:571`, opens on
   purpose), **no external process can reach that card by any path**. mpv aimed
   at the raw alias exits 2; mpv aimed at PipeWire's *own node* for the same
   card stalls indefinitely (killed at an 8 s timeout, having played nothing),
   because PipeWire cannot open the card either. So this is not a routing
   preference that a better `--audio-device` could fix -- an out-of-process
   player is structurally incapable of playing on the call's device. See commit
   `c28af33`, which makes the default-sink fallback deliberate rather than
   accidental.
2. **Ducking stops rebuffering live radio.** A pause/resume on a live HLS
   stream either goes stale or re-seeks to the live edge. A gain ramp keeps the
   decoder at the live edge and costs nothing.
3. **Explicit pause stops being silently undone.** See "Latent bug" below.

Non-goals: stereo output (deferred, §8), auto-restart after a mid-stream
ffmpeg death (deferred, §8), any change to capture, Opus, or the wake path.

## Why ffmpeg and not a Rust decoder

Measured on the Pop!_OS box, 2026-08-28, against BBC Radio 4:

```
http://as-hls-ww-live.akamaized.net/pool_55057080/live/ww/bbc_radio_fourfm/\
bbc_radio_fourfm.isml/bbc_radio_fourfm-audio%3d96000.norewind.m3u8
```

| Observation | Value |
| --- | --- |
| Container | MPEG-TS segments (`.ts`) |
| Codec | **HE-AAC** (`[15][0][0][0]`), 48000 Hz, stereo, fltp |
| Playlist window | 5 × `#EXTINF:6.4` ≈ 32 s, `#EXT-X-TARGETDURATION:6` |
| `-f s16le -ac 1 -ar 48000 -t 3` | 288000 bytes = 144000 samples = **3.000 s exactly** |
| Decoded level | peak 18083, RMS 3154.7 (**−20.3 dBFS**) — real programme audio |
| Drain rate | **`speed=401x`** |

Symphonia has **no MPEG-TS demuxer** and only partial HE-AAC/SBR support, so a
pure-Rust path would have to demux TS and handle SBR before playing a single
second of Radio 4. `libmpv` would keep the format handling but still opens its
own ALSA/PulseAudio stream, which leaves goal 1 unfixed. ffmpeg is the only
option that gives us the PCM and lets the client own the device.

`speed=401x` is the load-bearing measurement for §3: ffmpeg drains the
playlist backlog at network speed and does **not** self-pace.

## 1. Architecture

ffmpeg is a decoder, never a player — it is not given an audio device.

```
BBC HLS ──> ffmpeg ──stdout(s16le mono @ device_rate)──> feeder thread ──┐
                                                                         │ media_rx
                                                    OutputMixer.next_sample()
                                                                         │ peer_rx
server TTS ──> peer::run ────────────────────────────────────────────────┘
                                    │
                            fill_output_buffer ──> CPAL ──> Jabra
```

The hardware callback is the only clock. It already pulls sample-by-sample and
is already non-blocking (`audio.rs:817-848`), so mixing there needs no timer
and cannot drift against the device.

`MediaPlayer` is constructed inside `run_session` (`main.rs:308`), so media
only ever plays while a call is up and the CPAL output stream is guaranteed to
exist. There is no "media outside a call" case to design for.

## 2. The mixer — `crates/client/src/audio.rs`

`OutputQueue` (`audio.rs:817`) becomes `OutputMixer`, holding two independent
sources, each a `Receiver<Vec<i16>>` with its own cursor:

- `next_sample()` returns `Some` when **either** source has data, summing with
  saturation. It returns `None` only when both are dry, so
  `fill_output_buffer`'s silence-fill path (`audio.rs:864-869`) is unchanged.
- The media source carries a gain, stored as an `Arc<AtomicU32>` holding f32
  bits and cloned into the callback closure, so `MediaPlayer` retargets it from
  the control thread without a lock. The callback interpolates
  toward the target by a fixed per-sample step: an **80 ms ramp** at the device
  rate, which is short enough to be inaudible as a delay and long enough to
  avoid zipper noise.
- `try_recv` only. No allocation, no blocking, no locks in the callback — the
  module contract at `audio.rs:1-6` continues to hold.

`build_output_stream` (`audio.rs:754`) creates and returns both senders; the
existing "only the successful attempt's sender may escape" comment
(`audio.rs:751-753`) applies unchanged to the pair. `AudioIo` (`audio.rs:246`)
and `AudioIoParts` (`audio.rs:285`) expose both. `peer::run` keeps the TTS
sender (`main.rs:363`); `MediaPlayer` takes the other.

Gain and summation are per-source and channel-agnostic, so widening to stereo
later changes the chunk type, not this logic.

## 3. Pacing, buffering and backpressure

Backpressure is the mechanism, not an afterthought. The OS pipe (~64 KB ≈
0.68 s of mono 48 kHz `i16`) plus the bounded channel is the jitter buffer.
When the feeder stops reading, the pipe fills, ffmpeg blocks on write, and the
decoder stalls **exactly in place**. That is "true pause" with no pause logic.

Because ffmpeg drains at `401x` rather than self-pacing, playback settles a
fixed pipe-plus-channel distance behind the live edge and stays there — the
gain-duck keeps consuming in real time, so ducking does not accumulate latency.

Live streams pass `-live_start_index -1` to start at the live edge; ffmpeg's
HLS default of `-3` would start ~19 s back (3 × 6.4 s).

## 4. Duck and pause are separate concerns

**Latent bug being fixed.** Today ducking (`media.rs:76-88`) and explicit
`MediaCommand::Pause` (`media.rs:124`) both write mpv's single `pause`
property, so a deliberate "pause the radio" is silently resumed by the next
`rtf-bot-stopped-speaking`. Owning the mixer separates them.

| Event | Gain target | Decoder |
| --- | --- | --- |
| **`Play` starts while `bot_speaking`** | **starts at −18 dB, no ramp** | starts immediately |
| bot speaking, **live** | −18 dB | keeps running — stays at the live edge |
| bot speaking, **recorded** | 0 | feeder stops → ffmpeg blocks → resumes in place |
| explicit `Pause` | 0 | feeder stops, sets `user_paused` |
| `rtf-bot-stopped-speaking` | restore **unless `user_paused`** | restart feeder unless `user_paused` |
| `Stop` | 0 | kill ffmpeg, stop feeder |

`Stop` cannot drain the channel — a `std::sync::mpsc` has no sender-side
drain, and the receiver lives in the CPAL callback. It does not need to: the
ramp to 0 silences the ≤ 8 chunks (160 ms at a 20 ms period) already queued
while they play out harmlessly. Every transition is audibly governed by the
ramp, never by how much is in flight.

One mechanism for every audible transition (the ramp); one flag for whether the
decoder keeps running. Ducking a recorded show fades rather than cutting, and
loses no content: the samples already in the pipe are still there on resume.

**Starting ducked is the common case, not an edge case.** Radio is asked for by
voice, so the assistant is almost always still speaking the tool reply
("Playing BBC Radio 4") at the moment the stream opens — observed on the mpv
build, where radio comes up at full volume over the assistant. A stream that
starts while `bot_speaking` therefore begins **at** the ducked gain rather than
ramping down to it: there is no earlier level to fade from, and a fade-in from
full would be the very overlap being avoided. The first `rtf-bot-stopped-speaking`
ramps it up to full, using the same 80 ms ramp as every other transition.

The existing mpv build already attempts this (`media.rs:181`, pausing when
`bot_speaking`), so the requirement is not new — but a pause is the wrong
instrument for a live stream, and it is what this design replaces with a gain.

## 5. Protocol — one field

`crates/protocol/src/lib.rs:18`:

```rust
Play {
    url: String,
    title: String,
    /// Live streams duck by gain and stay at the live edge; recorded ones
    /// pause the decoder and resume in place.
    #[serde(default = "default_live")]
    live: bool,
},
```

Defaulting to `true` keeps an older server working. `Eq` still derives.

Call sites: `crates/server/src/skills/radio.rs:190` sets `live: true`;
`crates/server/src/skills/shows.rs:384` sets `live: false`. `play_stream`
(`crates/server/src/media.rs:36`) takes the flag through.

Shows sharing the `Play` path with live radio is the whole reason this field
exists — a uniform gain-duck would talk over a recorded programme without it.

## 6. The engine — `crates/client/src/media.rs`

```
ffmpeg -hide_banner -loglevel error -i <url>
       -f s16le -acodec pcm_s16le -ac 1 -ar <device_rate> -
```

plus, for `live: true`:

```
-live_start_index -1
-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5
```

ffmpeg performs the sample-rate conversion to the device rate, so `rubato` /
`StreamingResampler` is **not** on this path. The device rate is already known
(`main.rs:363` passes `output_rate`).

A feeder thread reads 20 ms chunks (`device_rate / 50` samples) from stdout and
pushes them to the media sender. EOF means the stream ended: clear state, emit
the status line. A non-zero exit is logged at `warn` **with the exit status**,
so a dead stream is diagnosable.

Deleted outright: the IPC socket and `ipc()` (`media.rs:207`), the 2 s
socket-existence race (`media.rs:176-179`), `reap()` (`media.rs:140`), and
`mpv_device_for` / `parse_mpv_device_list` (`media.rs:246-276`).
`is_available()` (`media.rs:55`) checks `ffmpeg -version`. The `after_speech`
pending logic (`media.rs:97-110`, `media.rs:115-122`) is unchanged.

The startup warning at `main.rs:312` becomes an ffmpeg one
(`brew install ffmpeg` / `apt install ffmpeg`).

## 7. Testing

Ducking becomes testable with no subprocess and no audio device, which it is
not today:

- **Unit (mixer):** summation; saturation at ±32767; ramp reaches target in
  80 ms worth of samples; both sources dry → `None`; one dry → `Some`.
- **Unit (state machine):** every row of §4, including *explicit pause survives
  `rtf-bot-stopped-speaking`* as a regression test for the latent bug.
- **Integration:** a generated WAV decoded through the real ffmpeg path,
  asserting sample-exact PCM out.
- **Live (`--ignored live`, following the existing convention at
  `media.rs:296`):** BBC Radio 4 for 3 s, asserting RMS lands in a sane band.
  Anchored on measured values: RMS −20.3 dBFS, peak 18083. Assert
  −35 < RMS dBFS < −10 and peak > 1000 rather than exact figures.

## 8. Deferred

| Item | Why not now |
| --- | --- |
| Stereo output boundary | `audio.rs:1-6` makes mono a module-wide contract; widening touches `output_tx`, `OutputMixer`, `fill_output_buffer` and `peer.rs`. Mono this phase, stereo a follow-up. The Jabra reference speaker is mono, so nothing regresses today. |
| Auto-restart after ffmpeg dies | Matches current behaviour (a dead `mpv` just stops). `-reconnect` covers transient blips; a supervisor is a separate change. Exit status is logged so it is diagnosable. |
| Per-source metering | No consumer yet. |

## 9. Files touched

| File | Change |
| --- | --- |
| `crates/client/src/audio.rs` | `OutputQueue` → `OutputMixer`; second source; atomic gain + ramp; two senders out of `build_output_stream`, `AudioIo`, `AudioIoParts` |
| `crates/client/src/media.rs` | Rewrite around ffmpeg + mixer handle; drop IPC, reap, device matching |
| `crates/client/src/main.rs` | Wire the media sender; ffmpeg availability warning |
| `crates/protocol/src/lib.rs` | `live: bool` on `MediaCommand::Play` |
| `crates/server/src/media.rs` | `play_stream` carries `live` |
| `crates/server/src/skills/radio.rs` | `live: true` |
| `crates/server/src/skills/shows.rs` | `live: false` |
| `README.md`, `crates/client/README.md` | Drop the default-sink routing caveat; mpv → ffmpeg in install notes |

## 10. Success criteria

1. BBC Radio 4 plays **out of the Jabra during a call**, not the default sink.
2. The assistant speaking fades radio to −18 dB over 80 ms and back, with no
   rebuffer and no drift from the live edge.
3. Radio asked for mid-sentence **starts** quiet under the assistant's reply
   and comes up only when it finishes — never in parallel at full volume.
4. A recorded show pauses on the assistant's speech and resumes in place.
5. "Pause the radio" survives the assistant speaking afterwards.
6. No `mpv` anywhere in `crates/`.
