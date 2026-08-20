//! Local Moonshine Streaming STT through Moonshine Voice's stable C ABI.
//!
//! The model is loaded once into [`MoonshineEngine`]. Every WebRTC call owns a
//! native stream and an OS worker thread, so ONNX inference never blocks a Tokio
//! runtime thread. The surrounding FlowCat `SpeechGate` remains authoritative
//! for endpointing: native updates are always interim, and `flush()` stops the
//! native stream, obtains exactly one final transcript, and starts a fresh
//! session for the next VAD turn.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;
use flowcat_core::{FlowcatError, Result};
use tokio::sync::oneshot;

const MOONSHINE_HEADER_VERSION: i32 = 30_000;
const MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING: u32 = 5;
const MOONSHINE_FLAG_FORCE_UPDATE: u32 = 1;
const DEFAULT_UPDATE_INTERVAL_MS: u64 = 250;

#[repr(C)]
struct MoonshineOption {
    name: *const c_char,
    value: *const c_char,
}

/// Only pointer identity matters to this integration; these arrays are owned
/// by the native transcriber and are never dereferenced here.
#[repr(C)]
struct NativeTranscriptLine {
    text: *const c_char,
    audio_data: *const f32,
    audio_data_count: usize,
    start_time: f32,
    duration: f32,
    id: u64,
    is_complete: i8,
    is_updated: i8,
    is_new: i8,
    has_text_changed: i8,
    have_speakers_changed: i8,
    speaker_spans: *const c_void,
    speaker_span_count: u64,
    last_transcription_latency_ms: u32,
    words: *const c_void,
    word_count: u64,
}

#[repr(C)]
struct NativeTranscript {
    lines: *const NativeTranscriptLine,
    line_count: u64,
}

extern "C" {
    fn moonshine_get_version() -> i32;
    fn moonshine_error_to_string(error: i32) -> *const c_char;
    fn moonshine_load_transcriber_from_files(
        path: *const c_char,
        model_arch: u32,
        options: *const MoonshineOption,
        options_count: u64,
        moonshine_version: i32,
    ) -> i32;
    fn moonshine_free_transcriber(transcriber_handle: i32);
    fn moonshine_create_stream(transcriber_handle: i32, flags: u32) -> i32;
    fn moonshine_free_stream(transcriber_handle: i32, stream_handle: i32) -> i32;
    fn moonshine_start_stream(transcriber_handle: i32, stream_handle: i32) -> i32;
    fn moonshine_stop_stream(transcriber_handle: i32, stream_handle: i32) -> i32;
    fn moonshine_transcribe_add_audio_to_stream(
        transcriber_handle: i32,
        stream_handle: i32,
        new_audio_data: *const f32,
        audio_length: u64,
        sample_rate: i32,
        flags: u32,
    ) -> i32;
    fn moonshine_transcribe_stream(
        transcriber_handle: i32,
        stream_handle: i32,
        flags: u32,
        out_transcript: *mut *mut NativeTranscript,
    ) -> i32;
}

/// A process-wide, preloaded Moonshine model.
///
/// Moonshine transcript pointers remain valid only until the next call on the
/// transcriber, even when that call addresses a different stream. The mutex is
/// therefore deliberately broader than a per-call stream lock: it is held from
/// `moonshine_transcribe_stream` through copying every returned string.
pub struct MoonshineEngine {
    handle: i32,
    native_lock: Mutex<()>,
}

impl MoonshineEngine {
    fn lock_native(&self) -> Result<MutexGuard<'_, ()>> {
        self.native_lock
            .lock()
            .map_err(|_| FlowcatError::Other("Moonshine native lock was poisoned".into()))
    }

    fn lock_native_for_drop(&self) -> MutexGuard<'_, ()> {
        self.native_lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Shared model weights used to create one stream per call.
pub type SharedMoonshineEngine = Arc<MoonshineEngine>;

/// Load the quantized English Medium Streaming model (architecture 5).
///
/// `keyterms` is an optional comma-separated contextual-bias list. Match the
/// desired spelling and capitalization; an empty value disables biasing.
pub fn load_engine(model_path: &str, keyterms: Option<&str>) -> Result<SharedMoonshineEngine> {
    if !Path::new(model_path).is_dir() {
        return Err(FlowcatError::Other(format!(
            "Moonshine model directory missing: {model_path}"
        )));
    }
    let path = CString::new(model_path)
        .map_err(|_| FlowcatError::Other("Moonshine model path contains a NUL byte".into()))?;

    // Explicitly retain streaming partial decoding even if an upstream default
    // changes. The CString storage must live through the native load call.
    let mut option_storage = vec![(
        CString::new("decode_incomplete_lines").unwrap(),
        CString::new("true").unwrap(),
    )];
    if let Some(keyterms) = keyterms.filter(|value| !value.trim().is_empty()) {
        option_storage.push((
            CString::new("keyterms").unwrap(),
            CString::new(keyterms)
                .map_err(|_| FlowcatError::Other("Moonshine keyterms contain a NUL byte".into()))?,
        ));
    }
    let options: Vec<MoonshineOption> = option_storage
        .iter()
        .map(|(name, value)| MoonshineOption {
            name: name.as_ptr(),
            value: value.as_ptr(),
        })
        .collect();

    let handle = unsafe {
        moonshine_load_transcriber_from_files(
            path.as_ptr(),
            MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
            options.as_ptr(),
            options.len() as u64,
            MOONSHINE_HEADER_VERSION,
        )
    };
    if handle < 0 {
        return Err(native_error("load transcriber", handle));
    }
    let library_version = unsafe { moonshine_get_version() };
    tracing::info!(
        library_version,
        model_arch = MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
        %model_path,
        "Moonshine model preloaded"
    );
    Ok(Arc::new(MoonshineEngine {
        handle,
        native_lock: Mutex::new(()),
    }))
}

impl Drop for MoonshineEngine {
    fn drop(&mut self) {
        let _native = self.lock_native_for_drop();
        unsafe { moonshine_free_transcriber(self.handle) };
    }
}

/// Per-call streaming service backed by a dedicated native stream and worker.
pub struct MoonshineStt {
    engine: SharedMoonshineEngine,
    update_interval_ms: u64,
    commands: Option<mpsc::Sender<WorkerCommand>>,
    updates: Option<mpsc::Receiver<Result<Vec<Frame>>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl MoonshineStt {
    pub fn from_engine(engine: SharedMoonshineEngine) -> Self {
        Self {
            engine,
            update_interval_ms: DEFAULT_UPDATE_INTERVAL_MS,
            commands: None,
            updates: None,
            worker: None,
        }
    }

    /// Set the transcript polling floor. Moonshine internally caches analyses
    /// until roughly 200 ms of new audio, so values below 200 ms only add calls.
    pub fn with_update_interval_ms(mut self, update_interval_ms: u64) -> Self {
        self.update_interval_ms = update_interval_ms.max(200);
        self
    }

    async fn request(
        &mut self,
        make_command: impl FnOnce(oneshot::Sender<Result<Vec<Frame>>>) -> WorkerCommand,
    ) -> Result<Vec<Frame>> {
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("Moonshine STT is not started".into()))?;
        let (reply_tx, reply_rx) = oneshot::channel();
        commands
            .send(make_command(reply_tx))
            .map_err(|_| FlowcatError::Other("Moonshine STT worker stopped".into()))?;
        reply_rx
            .await
            .map_err(|_| FlowcatError::Other("Moonshine STT worker dropped its reply".into()))?
    }

    /// Drain without waiting and keep only the newest interim. At the normal
    /// four-updates-per-second rate this channel stays tiny; coalescing also
    /// prevents a temporarily busy pipeline from replaying stale hypotheses.
    fn drain_updates(&mut self) -> Result<Vec<Frame>> {
        let updates = self
            .updates
            .as_mut()
            .ok_or_else(|| FlowcatError::Other("Moonshine STT is not started".into()))?;
        let mut latest = None;
        let mut error = None;
        loop {
            match updates.try_recv() {
                Ok(Ok(frames)) => latest = Some(frames),
                Ok(Err(update_error)) => {
                    if error.is_none() {
                        error = Some(update_error);
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if error.is_none() {
                        error = Some(FlowcatError::Other(
                            "Moonshine STT update channel disconnected".into(),
                        ));
                    }
                    break;
                }
            }
        }
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(latest.unwrap_or_default())
        }
    }
}

#[async_trait]
impl SttService for MoonshineStt {
    fn name(&self) -> &str {
        "babel-moonshine"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.commands.is_some() {
            return Ok(());
        }

        let (commands_tx, commands_rx) = mpsc::channel();
        let (updates_tx, updates_rx) = mpsc::channel();
        let (started_tx, started_rx) = oneshot::channel();
        let engine = Arc::clone(&self.engine);
        let update_interval_ms = self.update_interval_ms;
        let worker = std::thread::Builder::new()
            .name("moonshine-stt".into())
            .spawn(move || {
                let mut stream = match NativeStream::create(engine, update_interval_ms) {
                    Ok(stream) => {
                        let _ = started_tx.send(Ok(()));
                        stream
                    }
                    Err(error) => {
                        let _ = started_tx.send(Err(error));
                        return;
                    }
                };
                let mut pending = None;
                'worker: loop {
                    let command = match pending.take() {
                        Some(command) => command,
                        None => match commands_rx.recv() {
                            Ok(command) => command,
                            Err(_) => break,
                        },
                    };
                    match command {
                        WorkerCommand::Audio {
                            samples,
                            sample_rate,
                        } => {
                            let mut feed_failed = false;
                            if let Err(error) = stream.feed_audio(&samples, sample_rate) {
                                feed_failed = true;
                                let _ = updates_tx.send(Err(error));
                            }

                            // Native partial decoding can take longer than the
                            // nominal 250 ms update interval. Drain all audio
                            // already waiting and feed it cheaply, then decode
                            // at most once for the whole batch. A FIFO barrier
                            // takes precedence: flush/reset immediately instead
                            // of spending another decode on a stale interim.
                            loop {
                                match commands_rx.try_recv() {
                                    Ok(WorkerCommand::Audio {
                                        samples,
                                        sample_rate,
                                    }) => {
                                        if let Err(error) = stream.feed_audio(&samples, sample_rate)
                                        {
                                            feed_failed = true;
                                            let _ = updates_tx.send(Err(error));
                                        }
                                    }
                                    Ok(barrier) => {
                                        pending = Some(barrier);
                                        break;
                                    }
                                    Err(mpsc::TryRecvError::Empty) => break,
                                    Err(mpsc::TryRecvError::Disconnected) => break 'worker,
                                }
                            }

                            if pending.is_none() && !feed_failed {
                                match stream.maybe_update() {
                                    Ok(frames) if !frames.is_empty() => {
                                        let _ = updates_tx.send(Ok(frames));
                                    }
                                    Ok(_) => {}
                                    Err(error) => {
                                        let _ = updates_tx.send(Err(error));
                                    }
                                }
                            }
                        }
                        WorkerCommand::Flush { reply } => {
                            // Commands have one producer and one FIFO receiver,
                            // so every previously enqueued audio tail is consumed
                            // before this direct finalization reply is sent.
                            let _ = reply.send(stream.flush());
                        }
                        WorkerCommand::SetMuted { muted, reply } => {
                            let _ = reply.send(stream.set_muted(muted).map(|()| Vec::new()));
                        }
                        WorkerCommand::Shutdown => break,
                    }
                }
            })
            .map_err(FlowcatError::Io)?;

        match started_rx.await {
            Ok(Ok(())) => {
                self.commands = Some(commands_tx);
                self.updates = Some(updates_rx);
                self.worker = Some(worker);
                tracing::info!(
                    update_interval_ms,
                    "call-local Moonshine stream initialized"
                );
                Ok(())
            }
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(_) => {
                let _ = worker.join();
                Err(FlowcatError::Other(
                    "Moonshine STT worker exited during startup".into(),
                ))
            }
        }
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if audio.is_empty() {
            return self.drain_updates();
        }
        let sample_rate = i32::try_from(audio.sample_rate).map_err(|_| {
            FlowcatError::Other(format!(
                "Moonshine input sample rate is too large: {}",
                audio.sample_rate
            ))
        })?;
        let samples = mono_f32(&audio);
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("Moonshine STT is not started".into()))?;
        // std::sync::mpsc is unbounded: this only enqueues and never waits for
        // native decoding, leaving SttProcessor free to feed the VAD audio tail.
        commands
            .send(WorkerCommand::Audio {
                samples,
                sample_rate,
            })
            .map_err(|_| FlowcatError::Other("Moonshine STT worker stopped".into()))?;
        self.drain_updates()
    }

    async fn flush(&mut self) -> Result<Vec<Frame>> {
        let final_result = self.request(|reply| WorkerCommand::Flush { reply }).await;
        // The reply is sent after all earlier Audio commands, so every interim
        // produced for this turn is already available to drain without waiting.
        let interim_result = self.drain_updates();
        match final_result {
            Err(error) => Err(error),
            Ok(final_frames) => {
                // A stale background-interim error must never suppress a
                // successful authoritative final from the ordered flush.
                let mut frames = match interim_result {
                    Ok(frames) => frames,
                    Err(error) => {
                        tracing::warn!(%error, "discarding stale Moonshine interim error after successful final");
                        Vec::new()
                    }
                };
                frames.extend(final_frames);
                Ok(frames)
            }
        }
    }

    async fn set_muted(&mut self, muted: bool) {
        if self.commands.is_none() {
            return;
        }
        let reset_result = self
            .request(|reply| WorkerCommand::SetMuted { muted, reply })
            .await;
        // The reset reply is FIFO too. Discard all hypotheses from before the
        // mute boundary so they cannot appear on a later audio callback.
        let reset_error = reset_result.err();
        let update_error = self.drain_updates().err();
        if let Some(error) = reset_error.or(update_error) {
            tracing::warn!(%error, muted, "Moonshine mute reset failed");
        }
    }
}

impl Drop for MoonshineStt {
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(WorkerCommand::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

enum WorkerCommand {
    Audio {
        samples: Vec<f32>,
        sample_rate: i32,
    },
    Flush {
        reply: oneshot::Sender<Result<Vec<Frame>>>,
    },
    SetMuted {
        muted: bool,
        reply: oneshot::Sender<Result<Vec<Frame>>>,
    },
    Shutdown,
}

struct NativeStream {
    engine: SharedMoonshineEngine,
    handle: i32,
    update_interval_secs: f64,
    audio_since_update_secs: f64,
    last_interim: String,
    muted: bool,
    started: bool,
}

impl NativeStream {
    fn create(engine: SharedMoonshineEngine, update_interval_ms: u64) -> Result<Self> {
        let native = engine.lock_native()?;
        let handle = unsafe { moonshine_create_stream(engine.handle, 0) };
        if handle < 0 {
            return Err(native_error("create stream", handle));
        }
        let start_result = unsafe { moonshine_start_stream(engine.handle, handle) };
        if start_result != 0 {
            let free_result = unsafe { moonshine_free_stream(engine.handle, handle) };
            if free_result != 0 {
                tracing::warn!(error = %native_error("free unstarted stream", free_result));
            }
            return Err(native_error("start stream", start_result));
        }
        drop(native);
        Ok(Self {
            engine,
            handle,
            update_interval_secs: update_interval_ms as f64 / 1_000.0,
            audio_since_update_secs: 0.0,
            last_interim: String::new(),
            muted: false,
            started: true,
        })
    }

    fn feed_audio(&mut self, samples: &[f32], sample_rate: i32) -> Result<()> {
        if self.muted || samples.is_empty() {
            return Ok(());
        }
        if sample_rate <= 0 {
            return Err(FlowcatError::Other(format!(
                "invalid Moonshine input sample rate: {sample_rate}"
            )));
        }
        let audio_length = u64::try_from(samples.len())
            .map_err(|_| FlowcatError::Other("Moonshine audio chunk is too large".into()))?;
        {
            let _native = self.engine.lock_native()?;
            check_native("add audio", unsafe {
                moonshine_transcribe_add_audio_to_stream(
                    self.engine.handle,
                    self.handle,
                    samples.as_ptr(),
                    audio_length,
                    sample_rate,
                    0,
                )
            })?;
        }
        self.audio_since_update_secs += samples.len() as f64 / sample_rate as f64;
        Ok(())
    }

    fn maybe_update(&mut self) -> Result<Vec<Frame>> {
        if self.muted || self.audio_since_update_secs < self.update_interval_secs {
            return Ok(Vec::new());
        }
        self.audio_since_update_secs = 0.0;

        let (text, latency_ms) = self.transcript(0)?;
        tracing::debug!(latency_ms, text = %text, "Moonshine interim decoded");
        Ok(make_interim_frame(text, &mut self.last_interim)
            .into_iter()
            .collect())
    }

    fn flush(&mut self) -> Result<Vec<Frame>> {
        if self.muted {
            self.restart()?;
            return Ok(Vec::new());
        }

        {
            let _native = self.engine.lock_native()?;
            check_native("stop stream", unsafe {
                moonshine_stop_stream(self.engine.handle, self.handle)
            })?;
        }
        self.started = false;
        // Force one final decode after stopping even when less than Moonshine's
        // normal internal update window remains buffered.
        let decoded = self.transcript(MOONSHINE_FLAG_FORCE_UPDATE);
        // A start begins a fresh transcript document, so reconnects and later
        // turns reuse resident model weights without inheriting old text.
        let restart_result = self.start_fresh();
        let (text, latency_ms) = decoded?;
        restart_result?;
        tracing::info!(latency_ms, text = %text, "Moonshine utterance finalized");
        Ok(make_final_frame(text).into_iter().collect())
    }

    fn set_muted(&mut self, muted: bool) -> Result<()> {
        if self.muted == muted {
            return Ok(());
        }
        // Discard any in-flight hypothesis at a mute boundary. Starting again
        // is cheap and guarantees stale text cannot leak into the next turn.
        self.restart()?;
        self.muted = muted;
        Ok(())
    }

    fn restart(&mut self) -> Result<()> {
        if self.started {
            {
                let _native = self.engine.lock_native()?;
                check_native("stop stream", unsafe {
                    moonshine_stop_stream(self.engine.handle, self.handle)
                })?;
            }
            self.started = false;
        }
        self.start_fresh()
    }

    fn start_fresh(&mut self) -> Result<()> {
        {
            let _native = self.engine.lock_native()?;
            check_native("start stream", unsafe {
                moonshine_start_stream(self.engine.handle, self.handle)
            })?;
        }
        self.started = true;
        self.audio_since_update_secs = 0.0;
        self.last_interim.clear();
        Ok(())
    }

    /// Copy the native transcript while holding the engine-wide lock. The C
    /// ABI invalidates these pointers on the next call to this transcriber, not
    /// merely the next call to this particular stream.
    fn transcript(&mut self, flags: u32) -> Result<(String, u32)> {
        let _native = self.engine.lock_native()?;
        let mut transcript: *mut NativeTranscript = std::ptr::null_mut();
        check_native("transcribe stream", unsafe {
            moonshine_transcribe_stream(self.engine.handle, self.handle, flags, &mut transcript)
        })?;
        if transcript.is_null() {
            return Ok((String::new(), 0));
        }
        let transcript = unsafe { &*transcript };
        if transcript.lines.is_null() || transcript.line_count == 0 {
            return Ok((String::new(), 0));
        }
        let line_count = usize::try_from(transcript.line_count).map_err(|_| {
            FlowcatError::Other("Moonshine returned too many transcript lines".into())
        })?;
        let lines = unsafe { std::slice::from_raw_parts(transcript.lines, line_count) };
        let mut parts = Vec::with_capacity(lines.len());
        let mut latency_ms = 0;
        for line in lines {
            latency_ms = latency_ms.max(line.last_transcription_latency_ms);
            if line.text.is_null() {
                continue;
            }
            let text = unsafe { CStr::from_ptr(line.text) }.to_string_lossy();
            let text = text.trim();
            if !text.is_empty() {
                parts.push(text.to_owned());
            }
        }
        Ok((parts.join(" "), latency_ms))
    }
}

impl Drop for NativeStream {
    fn drop(&mut self) {
        let _native = self.engine.lock_native_for_drop();
        if self.started {
            let result = unsafe { moonshine_stop_stream(self.engine.handle, self.handle) };
            if result != 0 {
                tracing::warn!(error = %native_error("stop stream", result));
            }
        }
        let result = unsafe { moonshine_free_stream(self.engine.handle, self.handle) };
        if result != 0 {
            tracing::warn!(error = %native_error("free stream", result));
        }
    }
}

fn mono_f32(audio: &AudioFrame) -> Vec<f32> {
    let channels = usize::from(audio.num_channels.max(1));
    if channels == 1 {
        return audio
            .pcm
            .iter()
            .map(|sample| *sample as f32 / 32_768.0)
            .collect();
    }
    audio
        .pcm
        .chunks(channels)
        .map(|frame| {
            let sum: i64 = frame.iter().map(|sample| i64::from(*sample)).sum();
            (sum as f32 / frame.len() as f32) / 32_768.0
        })
        .collect()
}

fn make_interim_frame(text: String, last_interim: &mut String) -> Option<Frame> {
    if !is_spoken(&text) || text == *last_interim {
        return None;
    }
    last_interim.clone_from(&text);
    Some(Frame::InterimTranscription {
        text,
        user_id: Arc::from("user"),
        language: None,
    })
}

fn make_final_frame(text: String) -> Option<Frame> {
    is_spoken(&text).then(|| Frame::Transcription {
        text,
        user_id: Arc::from("user"),
        language: None,
        final_: true,
    })
}

fn is_spoken(text: &str) -> bool {
    text.chars().any(char::is_alphanumeric)
}

fn check_native(operation: &str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(native_error(operation, code))
    }
}

fn native_error(operation: &str, code: i32) -> FlowcatError {
    let message = unsafe {
        let ptr = moonshine_error_to_string(code);
        if ptr.is_null() {
            "unknown native error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };
    FlowcatError::Other(format!("Moonshine {operation}: {message} ({code})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_transcript_layout_matches_the_v3_lp64_abi() {
        if std::mem::size_of::<usize>() == 8 {
            assert_eq!(std::mem::size_of::<NativeTranscriptLine>(), 88);
            assert_eq!(std::mem::size_of::<NativeTranscript>(), 16);
        }
    }

    #[test]
    fn stereo_is_downmixed_before_native_stt() {
        let audio = AudioFrame {
            pcm: vec![32_767, -32_768, 16_384, 16_384],
            sample_rate: 16_000,
            num_channels: 2,
        };
        let samples = mono_f32(&audio);
        assert_eq!(samples.len(), 2);
        assert!(samples[0].abs() < 0.0001);
        assert!((samples[1] - 0.5).abs() < 0.0001);
    }

    #[test]
    fn interim_updates_are_deduplicated_until_text_changes() {
        let mut last = String::new();
        assert!(matches!(
            make_interim_frame("one two".into(), &mut last),
            Some(Frame::InterimTranscription { text, .. }) if text == "one two"
        ));
        assert!(make_interim_frame("one two".into(), &mut last).is_none());
        assert!(matches!(
            make_interim_frame("one two three".into(), &mut last),
            Some(Frame::InterimTranscription { text, .. }) if text == "one two three"
        ));
        assert!(make_interim_frame("...".into(), &mut last).is_none());
    }

    #[test]
    fn final_is_emitted_once_even_when_it_matches_the_last_interim() {
        let mut last = String::new();
        let interim = make_interim_frame("one two three four".into(), &mut last);
        assert!(interim.is_some());

        let frames: Vec<_> = make_final_frame("one two three four".into())
            .into_iter()
            .collect();
        assert_eq!(frames.len(), 1);
        assert!(matches!(
            &frames[0],
            Frame::Transcription { text, final_: true, .. } if text == "one two three four"
        ));
        assert!(make_final_frame("...".into()).is_none());
    }
}
