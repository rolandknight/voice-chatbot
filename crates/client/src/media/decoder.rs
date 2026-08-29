//! `ffmpeg` as a *decoder*, never a player.
//!
//! It is handed no audio device: it writes raw `s16le` mono PCM at the output
//! device's rate to stdout, and a feeder thread pumps that into the mixer's
//! channel. Two consequences worth knowing:
//!
//! * ffmpeg drains an HLS playlist at network speed (measured `speed=401x` on
//!   BBC Radio 4), so it does **not** pace itself. The OS pipe plus the bounded
//!   channel are the jitter buffer: the feeder only reads the next chunk of
//!   stdout once the current one has landed in the channel, and that's what
//!   paces ffmpeg.
//! * Because of that, "pause" costs nothing: stop reading, the pipe fills,
//!   ffmpeg blocks on write, and the decoder stalls exactly in place.

use std::io::{BufRead, BufReader, Read};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Samples in one 20 ms chunk — the granularity the call already runs on.
fn chunk_samples(sample_rate: u32) -> usize {
    sample_rate as usize / 50
}

/// How much of ffmpeg's stderr to keep. `-loglevel error` limits it to a few
/// short lines; the cap only stops a pathological stream growing it forever.
const STDERR_CAP: usize = 512;

/// Whether the input is an HLS playlist, the only demuxer that knows
/// `-live_start_index`. Anything else refuses to open when handed it, so the
/// container decides this option — never the caller's `live` flag, which is
/// about ducking and defaults to `true` for a server too old to send it.
fn is_hls(url: &str) -> bool {
    url.split('?').next().unwrap_or(url).ends_with(".m3u8")
}

/// Whether ffmpeg will accept the `-reconnect*` options for this input: they
/// belong to the http protocol, and any other input refuses to open at all
/// ("Option reconnect not found", exit 8) rather than ignoring them.
fn is_http(url: &str) -> bool {
    url.starts_with("http://") || url.starts_with("https://")
}

/// Read `stream` to EOF, keeping the first [`STDERR_CAP`] bytes of it in
/// `sink` as one space-joined line.
///
/// Reading continues after the cap is reached and is deliberately not
/// abandoned on a poisoned lock: this is draining a pipe, and whatever stops
/// reading it blocks ffmpeg once the ~64 KB kernel buffer fills.
fn drain_stderr(stream: impl BufRead, sink: &Mutex<String>) {
    let mut full = false;
    for line in stream.lines().map_while(Result::ok) {
        if full {
            continue;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(mut sink) = sink.lock() {
            if !sink.is_empty() {
                sink.push(' ');
            }
            sink.push_str(line);
            full = sink.len() >= STDERR_CAP;
        }
    }
}

pub struct Decoder {
    child: Child,
    running: Arc<AtomicBool>,
    stopping: Arc<AtomicBool>,
    feeder: Option<JoinHandle<()>>,
    /// What ffmpeg said on stderr, collapsed to one line. Filled as the lines
    /// arrive rather than at EOF, so it is already there when a failed decoder
    /// is reaped.
    stderr: Arc<Mutex<String>>,
    stderr_reader: Option<JoinHandle<()>>,
}

impl Decoder {
    /// The exact argument list, so it can be asserted without spawning.
    pub fn command_args(url: &str, sample_rate: u32, live: bool) -> Vec<String> {
        let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if live && is_hls(url) {
            // ffmpeg's HLS default is -3 segments; -1 is the live edge itself.
            for arg in ["-live_start_index", "-1"] {
                args.push(arg.to_string());
            }
        }
        if is_http(url) {
            // A recorded programme needs this as much as a live stream does:
            // the feeder paces ffmpeg to realtime, so one HTTP connection is
            // held open for the whole episode and the CDN will eventually
            // drop it.
            for arg in [
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

    /// `recycle` carries spent buffers back from the mixer. The feeder refills
    /// those rather than allocating: the audio callback must not free a buffer,
    /// so it returns them here instead, and this is the other half of that.
    ///
    /// It is shared rather than owned because each `Decoder` moves it into its
    /// feeder thread and the next one needs it back. The mutex is uncontended
    /// in practice -- `Drop` joins the outgoing feeder before a new one spawns,
    /// so only one holder ever exists -- and it is never taken on the audio
    /// thread, which only ever *sends* down the return channel.
    pub fn spawn(
        url: &str,
        sample_rate: u32,
        live: bool,
        tx: SyncSender<Vec<i16>>,
        recycle: Arc<Mutex<Receiver<Vec<i16>>>>,
    ) -> std::io::Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args(Self::command_args(url, sample_rate, live))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            // Captured, never discarded: an unreadable stream is otherwise a
            // bare exit code, and every cause of it looks identical.
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("stdout is piped");

        let stderr = Arc::new(Mutex::new(String::new()));
        let stderr_reader = child.stderr.take().map(|pipe| {
            let sink = Arc::clone(&stderr);
            std::thread::spawn(move || drain_stderr(BufReader::new(pipe), &sink))
        });

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
                            // Refill a buffer the mixer has finished with. Only
                            // the first few chunks allocate; after that the same
                            // buffers cycle round, so neither thread touches the
                            // allocator in steady state.
                            let mut samples = recycle
                                .lock()
                                .ok()
                                .and_then(|spare| spare.try_recv().ok())
                                .unwrap_or_default();
                            samples.clear();
                            samples.extend(
                                bytes
                                    .chunks_exact(2)
                                    .map(|pair| i16::from_le_bytes([pair[0], pair[1]])),
                            );
                            // Backpressure is what paces ffmpeg: no more of stdout
                            // is read until this chunk lands. A blocking `send`
                            // would do that too, but uncancellably — a feeder
                            // parked in it never sees `stopping`, and `Drop`'s
                            // join would hang with it.
                            let mut pending = samples;
                            loop {
                                if stopping.load(Ordering::Relaxed) {
                                    return;
                                }
                                match tx.try_send(pending) {
                                    Ok(()) => break,
                                    Err(TrySendError::Full(returned)) => {
                                        pending = returned;
                                        std::thread::sleep(Duration::from_millis(5));
                                    }
                                    Err(TrySendError::Disconnected(_)) => return, // mixer gone
                                }
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
            stderr,
            stderr_reader,
        })
    }

    /// Whether the feeder pulls. False stalls ffmpeg in place.
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// True once the feeder has drained stdout to EOF: the last decoded sample
    /// is in the channel, not still in flight in the pipe.
    ///
    /// ffmpeg exiting is NOT the end of playback — it means only that ffmpeg
    /// finished *writing*, while the pipe and channel still hold audio nobody
    /// has heard. A paused feeder never finishes, so a ducked recorded show is
    /// never mistaken for one that ended.
    pub fn drained(&self) -> bool {
        self.feeder
            .as_ref()
            .is_some_and(|feeder| feeder.is_finished())
    }

    /// What ffmpeg wrote to stderr, as one line ("" when it said nothing).
    pub fn stderr_tail(&self) -> String {
        self.stderr.lock().map(|s| s.clone()).unwrap_or_default()
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
        // The write end went with the child, so the reader is already at EOF.
        if let Some(reader) = self.stderr_reader.take() {
            let _ = reader.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A return path with nothing on it: the feeder allocates every buffer, as
    /// it does before the mixer has handed any back. Tests that are not about
    /// recycling use this.
    fn no_recycling() -> Arc<Mutex<Receiver<Vec<i16>>>> {
        let (_tx, rx) = std::sync::mpsc::sync_channel(1);
        Arc::new(Mutex::new(rx))
    }

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
    fn a_recorded_stream_gets_no_live_edge_flag_but_still_reconnects() {
        let joined = Decoder::command_args("http://example/x.m4a", 48_000, false).join(" ");
        // Seeking to the live edge is meaningless for a recorded programme...
        assert!(!joined.contains("-live_start_index"), "{joined}");
        // ...but a 75-minute omnibus is read at realtime pace over one HTTP
        // connection, so it needs the same reconnect cover a live stream gets.
        assert!(joined.contains("-reconnect 1"), "{joined}");
        assert!(joined.contains("-reconnect_streamed 1"), "{joined}");
        assert!(joined.contains("-reconnect_delay_max 5"), "{joined}");
    }

    /// `-live_start_index` belongs to the *HLS* demuxer; an mp3 refuses to open
    /// at all when handed it ("Option live_start_index not found", exit 8).
    ///
    /// `live` is a statement about ducking, not about the container, and it
    /// arrives as `true` by default from a server too old to send the field —
    /// so it must never be the thing that decides a demuxer option.
    #[test]
    fn a_live_flagged_non_hls_url_still_gets_no_hls_only_option() {
        let joined = Decoder::command_args("http://bbc/omnibus.mp3", 48_000, true).join(" ");
        assert!(!joined.contains("-live_start_index"), "{joined}");
        assert!(joined.contains("-reconnect 1"), "{joined}");
    }

    /// `-reconnect` is an option of ffmpeg's *http* protocol. Handing it to any
    /// other input makes ffmpeg refuse to start at all ("Option reconnect not
    /// found", exit 8), which would take the sound effects down with it.
    #[test]
    fn a_non_http_source_gets_no_reconnect_flags() {
        let joined =
            Decoder::command_args("sine=frequency=440:duration=5", 48_000, false).join(" ");
        assert!(!joined.contains("-reconnect"), "{joined}");
        assert!(joined.contains("-f lavfi"), "{joined}");
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

    /// Capping what we *keep* must never cap what we *read*: ffmpeg's stderr
    /// is a pipe, and an unread one fills at ~64 KB and blocks the process
    /// mid-stream — turning a diagnostic into an outage.
    #[test]
    fn draining_stderr_reads_past_the_cap_so_ffmpeg_never_blocks() {
        let noise = "Error opening input: something went wrong\n".repeat(500);
        assert!(
            noise.len() > STDERR_CAP * 4,
            "the fixture must exceed the cap"
        );
        let mut stream = std::io::Cursor::new(noise.clone());
        let sink = Mutex::new(String::new());

        drain_stderr(&mut stream, &sink);

        assert_eq!(
            stream.position() as usize,
            noise.len(),
            "stopped reading early; the pipe would fill and block ffmpeg"
        );
        let kept = sink.lock().unwrap();
        assert!(kept.contains("Error opening input"), "kept nothing useful");
        assert!(
            kept.len() < STDERR_CAP * 2,
            "kept {} bytes, unbounded",
            kept.len()
        );
    }

    #[test]
    fn twenty_millisecond_chunks_are_one_opus_frame_of_samples() {
        assert_eq!(chunk_samples(48_000), 960);
        assert_eq!(chunk_samples(44_100), 882);
        assert_eq!(chunk_samples(16_000), 320);
    }

    /// The reason a stream would not play must survive to the caller. Exit
    /// status alone cannot carry it: ffmpeg answers 8 for a CDN 403, a DNS
    /// failure and a rejected option alike.
    #[test]
    #[ignore]
    fn live_a_refused_input_keeps_ffmpegs_reason() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(8);
        let mut decoder = Decoder::spawn(
            "sine=frequency=NOTANUMBER",
            48_000,
            false,
            tx,
            no_recycling(),
        )
        .expect("spawn ffmpeg");
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        while decoder.finished().is_none() && std::time::Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(
            matches!(decoder.finished(), Some(Some(code)) if code != 0),
            "ffmpeg should refuse it, got {:?}",
            decoder.finished()
        );
        // ffmpeg's own words, not a paraphrase: this is the whole payload.
        let reason = decoder.stderr_tail();
        assert!(
            reason.contains("Unable to parse option value"),
            "no reason captured, got {reason:?}"
        );
        // Several stderr lines collapsed into one, for a one-line report.
        assert!(!reason.contains('\n'), "must be one line, got {reason:?}");
    }

    /// Uses the real ffmpeg to decode a generated tone, proving the argument
    /// list and the feeder agree with the binary we ship against.
    #[test]
    #[ignore]
    fn live_decodes_a_generated_tone_into_the_channel() {
        let (tx, rx) = std::sync::mpsc::sync_channel(256);
        // lavfi is ffmpeg's built-in generator: 0.5 s of 440 Hz.
        let mut decoder = Decoder::spawn(
            "sine=frequency=440:duration=0.5",
            48_000,
            false,
            tx,
            no_recycling(),
        )
        .expect("spawn ffmpeg");
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let samples: usize = rx.try_iter().map(|chunk| chunk.len()).sum();
        // 0.5 s at 48 kHz, allowing for the last partial chunk.
        assert!(
            (23_000..=24_100).contains(&samples),
            "got {samples} samples"
        );
        assert_eq!(
            decoder.finished(),
            Some(Some(0)),
            "ffmpeg should exit clean"
        );
    }

    /// The mixer hands spent buffers back so the callback never has to free
    /// one. That only pays off if the feeder actually refills them instead of
    /// allocating, so assert the very allocation we supplied comes back out.
    #[test]
    #[ignore]
    fn live_the_feeder_refills_a_recycled_buffer_instead_of_allocating() {
        let (tx, rx) = std::sync::mpsc::sync_channel(8);
        let (recycle_tx, recycle_rx) = std::sync::mpsc::sync_channel(8);
        // A buffer already the right size, as the mixer would return.
        let spare = vec![0i16; 960];
        let spare_alloc = spare.as_ptr();
        recycle_tx
            .try_send(spare)
            .expect("offer a buffer for reuse");

        let _decoder = Decoder::spawn(
            "sine=frequency=440:duration=5",
            48_000,
            false,
            tx,
            Arc::new(Mutex::new(recycle_rx)),
        )
        .expect("spawn ffmpeg");
        let chunk = rx
            .recv_timeout(Duration::from_secs(5))
            .expect("a decoded chunk");
        assert_eq!(
            chunk.as_ptr(),
            spare_alloc,
            "the feeder allocated a fresh buffer instead of refilling the recycled one"
        );
        assert_eq!(chunk.len(), 960, "a full 20 ms chunk");
    }

    /// Pausing is "stop reading stdout": the pipe fills, ffmpeg blocks on
    /// write, and the decoder holds its place. Nothing else asserts this, and
    /// inverting `set_running` passes every other test in the crate.
    #[test]
    #[ignore]
    fn live_pausing_stops_the_flow_and_resuming_restarts_it() {
        let (tx, rx) = std::sync::mpsc::sync_channel(64);
        let decoder = Decoder::spawn(
            "sine=frequency=440:duration=30",
            48_000,
            false,
            tx,
            no_recycling(),
        )
        .expect("spawn ffmpeg");
        std::thread::sleep(Duration::from_millis(400));
        let flowing: usize = rx.try_iter().count();
        assert!(flowing > 0, "nothing decoded before the pause");

        decoder.set_running(false);
        // Let the in-flight chunk land, then drain so the count starts at zero.
        std::thread::sleep(Duration::from_millis(200));
        let _ = rx.try_iter().count();
        std::thread::sleep(Duration::from_millis(400));
        let while_paused: usize = rx.try_iter().count();
        assert_eq!(while_paused, 0, "the feeder kept pulling while paused");

        decoder.set_running(true);
        std::thread::sleep(Duration::from_millis(400));
        let after_resume: usize = rx.try_iter().count();
        assert!(after_resume > 0, "the feeder did not resume");
    }

    /// A feeder parked on a full channel must still shut down. The Receiver is
    /// alive but never drained, so a blocking `send` would never observe
    /// `stopping` and `Drop`'s join would hang forever.
    #[test]
    #[ignore]
    fn live_drop_returns_promptly_while_the_channel_is_full() {
        let (tx, _rx) = std::sync::mpsc::sync_channel(1);
        let decoder = Decoder::spawn(
            "sine=frequency=440:duration=30",
            48_000,
            false,
            tx,
            no_recycling(),
        )
        .expect("spawn ffmpeg");
        // Let it fill the one slot and park in the retry loop.
        std::thread::sleep(Duration::from_millis(500));
        let start = std::time::Instant::now();
        drop(decoder);
        let took = start.elapsed();
        assert!(
            took < Duration::from_secs(2),
            "Drop blocked for {took:?}; the feeder never observed `stopping`"
        );
    }
}
