//! In-process NVIDIA Nemotron streaming STT through NeMo-Speech.cpp's stable C
//! ABI (`include/nemo_speech/asr.h`, `libnemo_speech_asr_c`), replacing the
//! WebSocket sidecar (`nemotron.rs`) the same way Qwen3-TTS runs in-process.
//!
//! One recognizer (model weights, Metal/CPU backend) is loaded per process;
//! every WebRTC call owns a native stream driven from its own OS worker thread,
//! so decoding never blocks a Tokio thread. FlowCat's `SpeechGate` stays
//! authoritative for turn boundaries: audio is pushed while the gate is open,
//! and `flush()` (the VAD falling edge) forces an end-of-utterance and returns
//! exactly one final transcript; the stream then continues for the next turn.
//! Native interims are display-only (`InterimTranscription`).

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;
use flowcat_core::{FlowcatError, Result};
use tokio::sync::oneshot;

// ---- C ABI (asr.h, v1: append-only structs led by `size`) ------------------

type Status = i32;
const OK: Status = 0;

#[repr(C)]
struct BackendConfig {
    size: usize,
    gpu: i32, // -1 = CPU, else device index (0 = Metal on Apple Silicon)
}

#[repr(C)]
struct ModelConfig {
    size: usize,
    path: *const c_char,
    name: *const c_char,
}

#[repr(C)]
struct StreamingConfig {
    size: usize,
    chunk_size: f32,
    ctc_left_padding: f32,
    ctc_right_padding: f32,
    rnnt_right_context: i32, // encoder frames; -1 = model default
}

#[repr(C)]
struct EndpointingConfig {
    size: usize,
    enable: bool,
    vad_based: bool,
    stop_history_eou_ms: i32,
}

#[repr(C)]
struct BatchingConfig {
    size: usize,
    enable: bool,
    max_batch_size: i32,
    max_queue_delay_us: i32,
    max_queue_depth: i32,
    ingress_cohort_delay_us: i32,
    state_arena_slots: i32,
}

#[repr(C)]
struct RecognizerConfig {
    size: usize,
    backend: *const BackendConfig,
    model: *const ModelConfig,
    streaming: *const StreamingConfig,
    decoder: *const c_void,
    vad: *const c_void,
    endpointing: *const EndpointingConfig,
    postproc: *const c_void,
    diar: *const c_void,
    batching: *const BatchingConfig,
}

#[repr(C)]
struct SpeechContext {
    size: usize,
    phrases: *const *const c_char,
    phrase_count: usize,
    boost: f32,
}

#[repr(C)]
struct RecognitionOptions {
    size: usize,
    request_id: *const c_char,
    language_code: *const c_char,
    interim_results: bool,
    enable_word_time_offsets: bool,
    enable_automatic_punctuation: bool,
    verbatim_transcripts: bool,
    profanity_filter: bool,
    stop_history_eou_ms: i32,
    speech_contexts: *const SpeechContext,
    speech_context_count: usize,
    max_alternatives: i32,
    enable_speaker_diarization: bool,
    max_speaker_count: i32,
}

#[repr(C)]
struct Recognizer {
    _private: [u8; 0],
}
#[repr(C)]
struct NativeStream {
    _private: [u8; 0],
}
#[repr(C)]
struct NativeResult {
    _private: [u8; 0],
}

extern "C" {
    fn nemo_speech_asr_recognition_options_default() -> RecognitionOptions;
    fn nemo_speech_asr_create(cfg: *const RecognizerConfig, out: *mut *mut Recognizer) -> Status;
    fn nemo_speech_asr_destroy(recognizer: *mut Recognizer);
    fn nemo_speech_asr_streaming_recognize(
        recognizer: *mut Recognizer,
        options: *const RecognitionOptions,
        out: *mut *mut NativeStream,
    ) -> Status;
    fn nemo_speech_asr_stream_push_f32(
        stream: *mut NativeStream,
        samples: *const f32,
        n_samples: usize,
        sample_rate: i32,
    ) -> Status;
    fn nemo_speech_asr_stream_force_endpoint(stream: *mut NativeStream) -> Status;
    fn nemo_speech_asr_stream_finish(stream: *mut NativeStream) -> Status;
    fn nemo_speech_asr_stream_next(
        stream: *mut NativeStream,
        out: *mut *mut NativeResult,
    ) -> Status;
    fn nemo_speech_asr_stream_close(stream: *mut NativeStream);
    fn nemo_speech_asr_result_is_final(result: *const NativeResult) -> bool;
    fn nemo_speech_asr_result_audio_processed(result: *const NativeResult) -> f32;
    fn nemo_speech_asr_result_transcript(result: *const NativeResult, alt: usize) -> *const c_char;
    fn nemo_speech_asr_result_destroy(result: *mut NativeResult);
    fn nemo_speech_asr_last_error() -> *const c_char;
    fn nemo_speech_asr_version() -> *const c_char;
}

fn last_error() -> String {
    let p = unsafe { nemo_speech_asr_last_error() };
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn check(operation: &str, status: Status) -> Result<()> {
    if status == OK {
        Ok(())
    } else {
        Err(FlowcatError::Other(format!(
            "Nemotron native {operation} failed (status {status}): {}",
            last_error()
        )))
    }
}

pub fn library_version() -> String {
    let p = unsafe { nemo_speech_asr_version() };
    if p.is_null() {
        return "?".into();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

// ---- Engine (process-wide recognizer) ----------------------------------------

/// Which accelerator to give the recognizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu(i32),
}

impl Device {
    /// `auto` | `metal` | `cpu` | `cuda:N` (the sidecar's vocabulary).
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cpu" => Ok(Self::Cpu),
            "auto" | "metal" => Ok(Self::Gpu(0)),
            other => other
                .strip_prefix("cuda:")
                .and_then(|n| n.parse::<i32>().ok())
                .map(Self::Gpu)
                .ok_or_else(|| {
                    FlowcatError::Other(format!(
                        "invalid POC_NEMOTRON_DEVICE {value:?} (expected auto, metal, cpu, or cuda:N)"
                    ))
                }),
        }
    }

    fn gpu_index(self) -> i32 {
        match self {
            Self::Cpu => -1,
            Self::Gpu(i) => i,
        }
    }
}

/// The loaded model. The C library supports independent streams from multiple
/// threads (its batching layer exists for that), so streams do not serialize on
/// the engine; the mutex only guards stream creation/close.
pub struct NemotronEngine {
    recognizer: *mut Recognizer,
    lifecycle: Mutex<()>,
    pub model_path: String,
    pub device: Device,
    pub right_context: i32,
}

// The recognizer handle is designed for concurrent use from multiple threads.
unsafe impl Send for NemotronEngine {}
unsafe impl Sync for NemotronEngine {}

pub type SharedNemotronEngine = Arc<NemotronEngine>;

/// Load the Q8 streaming model once. `right_context` follows the sidecar
/// (`--asr.streaming.rnnt_right_context`, 6 = 560 ms window; -1 = model default).
pub fn load_engine(
    model_path: &str,
    device: Device,
    right_context: i32,
) -> Result<SharedNemotronEngine> {
    if !Path::new(model_path).is_file() {
        return Err(FlowcatError::Other(format!(
            "Nemotron model missing: {model_path} (run ./scripts/setup_nemotron.sh)"
        )));
    }
    let path = CString::new(model_path)
        .map_err(|_| FlowcatError::Other("Nemotron model path contains a NUL byte".into()))?;
    let backend = BackendConfig {
        size: std::mem::size_of::<BackendConfig>(),
        gpu: device.gpu_index(),
    };
    let model = ModelConfig {
        size: std::mem::size_of::<ModelConfig>(),
        path: path.as_ptr(),
        name: std::ptr::null(),
    };
    // The CTC buffered-window fields must be valid even for this RNNT model
    // (the library validates the whole struct): NeMo's usual 1.6 s geometry.
    let streaming = StreamingConfig {
        size: std::mem::size_of::<StreamingConfig>(),
        chunk_size: 1.6,
        ctc_left_padding: 1.6,
        ctc_right_padding: 1.6,
        rnnt_right_context: right_context,
    };
    // FlowCat owns endpointing (SpeechGate + VAD); the model never ends turns.
    let endpointing = EndpointingConfig {
        size: std::mem::size_of::<EndpointingConfig>(),
        enable: false,
        vad_based: false,
        stop_history_eou_ms: 0,
    };
    // Single-stream laptop profile, as the sidecar was run (batching disabled).
    let batching = BatchingConfig {
        size: std::mem::size_of::<BatchingConfig>(),
        enable: false,
        max_batch_size: 0,
        max_queue_delay_us: 0,
        max_queue_depth: 0,
        ingress_cohort_delay_us: 0,
        state_arena_slots: 0,
    };
    let cfg = RecognizerConfig {
        size: std::mem::size_of::<RecognizerConfig>(),
        backend: &backend,
        model: &model,
        streaming: &streaming,
        decoder: std::ptr::null(),
        vad: std::ptr::null(),
        endpointing: &endpointing,
        postproc: std::ptr::null(),
        diar: std::ptr::null(),
        batching: &batching,
    };
    let started = Instant::now();
    let mut recognizer: *mut Recognizer = std::ptr::null_mut();
    check("create recognizer", unsafe {
        nemo_speech_asr_create(&cfg, &mut recognizer)
    })?;
    if recognizer.is_null() {
        return Err(FlowcatError::Other(
            "Nemotron native create returned no recognizer".into(),
        ));
    }
    tracing::info!(
        library = %library_version(), ?device, right_context, %model_path,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "Nemotron model preloaded (in-process)"
    );
    Ok(Arc::new(NemotronEngine {
        recognizer,
        lifecycle: Mutex::new(()),
        model_path: model_path.to_string(),
        device,
        right_context,
    }))
}

impl Drop for NemotronEngine {
    fn drop(&mut self) {
        let _guard = self.lifecycle.lock().unwrap_or_else(|p| p.into_inner());
        unsafe { nemo_speech_asr_destroy(self.recognizer) };
    }
}

// ---- Per-call native stream (worker thread) ----------------------------------

/// One decoded result, copied out of the native handle.
#[derive(Debug, Clone, PartialEq)]
pub struct Decoded {
    pub text: String,
    pub is_final: bool,
    pub audio_processed: f32,
}

struct Stream {
    engine: SharedNemotronEngine,
    handle: *mut NativeStream,
    /// Owned storage for the speech-context phrases passed at creation.
    _contexts: Vec<CString>,
    _context_ptrs: Vec<*const c_char>,
    muted: bool,
    /// Running hypothesis of the current utterance: non-final results are
    /// treated as the full hypothesis so far when they extend the previous one,
    /// else as deltas to append (both shapes observed from RNNT decoders).
    hypothesis: String,
    last_interim: String,
}

impl Stream {
    fn create(engine: SharedNemotronEngine, speech_contexts: &[String]) -> Result<Self> {
        let contexts: Vec<CString> = speech_contexts
            .iter()
            .filter(|s| !s.trim().is_empty())
            .map(|s| CString::new(s.trim()).unwrap_or_default())
            .collect();
        let context_ptrs: Vec<*const c_char> = contexts.iter().map(|c| c.as_ptr()).collect();
        let context = SpeechContext {
            size: std::mem::size_of::<SpeechContext>(),
            phrases: context_ptrs.as_ptr(),
            phrase_count: context_ptrs.len(),
            boost: 4.0,
        };
        let language = CString::new("en-US").unwrap();
        let mut options = unsafe { nemo_speech_asr_recognition_options_default() };
        options.language_code = language.as_ptr();
        options.interim_results = true;
        options.enable_automatic_punctuation = true;
        options.enable_word_time_offsets = false;
        if !context_ptrs.is_empty() {
            options.speech_contexts = &context;
            options.speech_context_count = 1;
        }
        let mut handle: *mut NativeStream = std::ptr::null_mut();
        {
            let _guard = engine
                .lifecycle
                .lock()
                .map_err(|_| FlowcatError::Other("Nemotron lifecycle lock poisoned".into()))?;
            check("streaming_recognize", unsafe {
                nemo_speech_asr_streaming_recognize(engine.recognizer, &options, &mut handle)
            })?;
        }
        if handle.is_null() {
            return Err(FlowcatError::Other(
                "Nemotron native returned no stream".into(),
            ));
        }
        Ok(Self {
            engine,
            handle,
            _contexts: contexts,
            _context_ptrs: context_ptrs,
            muted: false,
            hypothesis: String::new(),
            last_interim: String::new(),
        })
    }

    fn push(&mut self, samples: &[f32], sample_rate: i32) -> Result<()> {
        if self.muted || samples.is_empty() {
            return Ok(());
        }
        check("push audio", unsafe {
            nemo_speech_asr_stream_push_f32(
                self.handle,
                samples.as_ptr(),
                samples.len(),
                sample_rate,
            )
        })
    }

    /// Pull every result currently available (decoding happens inside `next`).
    fn drain(&mut self) -> Result<Vec<Decoded>> {
        let mut out = Vec::new();
        loop {
            let mut result: *mut NativeResult = std::ptr::null_mut();
            check("stream next", unsafe {
                nemo_speech_asr_stream_next(self.handle, &mut result)
            })?;
            if result.is_null() {
                break;
            }
            let decoded = unsafe {
                let text_ptr = nemo_speech_asr_result_transcript(result, 0);
                let text = if text_ptr.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(text_ptr)
                        .to_string_lossy()
                        .trim()
                        .to_string()
                };
                let d = Decoded {
                    text,
                    is_final: nemo_speech_asr_result_is_final(result),
                    audio_processed: nemo_speech_asr_result_audio_processed(result),
                };
                nemo_speech_asr_result_destroy(result);
                d
            };
            out.push(decoded);
        }
        Ok(out)
    }

    /// Fold a non-final result into the running hypothesis (see field doc).
    fn absorb_interim(&mut self, text: &str) {
        absorb(&mut self.hypothesis, text);
    }

    /// Interim frames for anything decoded so far this turn.
    fn interims(&mut self) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(Vec::new());
        }
        let mut frames = Vec::new();
        for d in self.drain()? {
            if d.is_final {
                // A final without our force_endpoint (model-side EOU is off, so
                // this should not happen); keep it as hypothesis text.
                self.absorb_interim(&d.text);
            } else {
                self.absorb_interim(&d.text);
            }
        }
        if self.hypothesis != self.last_interim {
            self.last_interim = self.hypothesis.clone();
            if let Some(text) = spoken_text(&self.hypothesis) {
                frames.push(interim_frame(text));
            }
        }
        Ok(frames)
    }

    /// VAD falling edge: force an end-of-utterance and collect the final.
    fn flush(&mut self) -> Result<Vec<Frame>> {
        if self.muted {
            self.hypothesis.clear();
            self.last_interim.clear();
            return Ok(Vec::new());
        }
        let started = Instant::now();
        check("force endpoint", unsafe {
            nemo_speech_asr_stream_force_endpoint(self.handle)
        })?;
        let mut final_text: Option<String> = None;
        // `next()` drives decoding; the forced EOU is honoured on it. Poll
        // briefly: the decoder may need a few calls to emit the final.
        let deadline = Instant::now() + Duration::from_millis(1_500);
        loop {
            let results = self.drain()?;
            for d in results {
                if d.is_final {
                    final_text = Some(d.text);
                } else {
                    self.absorb_interim(&d.text);
                }
            }
            if final_text.is_some() || Instant::now() > deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        let text = match final_text {
            Some(t) if !t.trim().is_empty() => t,
            _ => self.hypothesis.clone(),
        };
        self.hypothesis.clear();
        self.last_interim.clear();
        let latency_ms = started.elapsed().as_millis() as u64;
        match spoken_text(&text) {
            Some(text) => {
                tracing::info!(latency_ms, text = %text, "Nemotron utterance finalized (in-process)");
                Ok(vec![final_frame(text)])
            }
            None => {
                tracing::debug!(latency_ms, "Nemotron flush produced no speech");
                Ok(Vec::new())
            }
        }
    }

    fn set_muted(&mut self, muted: bool) -> Result<()> {
        if self.muted == muted {
            return Ok(());
        }
        // Discard anything in flight at the boundary so stale text never leaks
        // into the next turn.
        check("force endpoint (mute)", unsafe {
            nemo_speech_asr_stream_force_endpoint(self.handle)
        })?;
        let _ = self.drain()?;
        self.hypothesis.clear();
        self.last_interim.clear();
        self.muted = muted;
        Ok(())
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let _guard = self
            .engine
            .lifecycle
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        unsafe {
            let _ = nemo_speech_asr_stream_finish(self.handle);
            nemo_speech_asr_stream_close(self.handle);
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

/// Per-call service: a native stream on its own worker thread.
pub struct NemotronNativeStt {
    engine: SharedNemotronEngine,
    speech_contexts: Vec<String>,
    commands: Option<mpsc::Sender<WorkerCommand>>,
    updates: Option<mpsc::Receiver<Result<Vec<Frame>>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl NemotronNativeStt {
    pub fn from_engine(engine: SharedNemotronEngine, speech_contexts: Vec<String>) -> Self {
        Self {
            engine,
            speech_contexts,
            commands: None,
            updates: None,
            worker: None,
        }
    }

    async fn request(
        &mut self,
        make: impl FnOnce(oneshot::Sender<Result<Vec<Frame>>>) -> WorkerCommand,
    ) -> Result<Vec<Frame>> {
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("Nemotron STT is not started".into()))?;
        let (tx, rx) = oneshot::channel();
        commands
            .send(make(tx))
            .map_err(|_| FlowcatError::Other("Nemotron STT worker stopped".into()))?;
        rx.await
            .map_err(|_| FlowcatError::Other("Nemotron STT worker dropped its reply".into()))?
    }

    /// Newest interim only (coalesced), like the Moonshine service.
    fn drain_updates(&mut self) -> Result<Vec<Frame>> {
        let updates = self
            .updates
            .as_mut()
            .ok_or_else(|| FlowcatError::Other("Nemotron STT is not started".into()))?;
        let mut latest = None;
        let mut error = None;
        loop {
            match updates.try_recv() {
                Ok(Ok(frames)) => latest = Some(frames),
                Ok(Err(e)) => {
                    error.get_or_insert(e);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    error.get_or_insert(FlowcatError::Other(
                        "Nemotron STT update channel disconnected".into(),
                    ));
                    break;
                }
            }
        }
        match error {
            Some(e) => Err(e),
            None => Ok(latest.unwrap_or_default()),
        }
    }
}

#[async_trait]
impl SttService for NemotronNativeStt {
    fn name(&self) -> &str {
        "babel-nemotron-native"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.commands.is_some() {
            return Ok(());
        }
        let (commands_tx, commands_rx) = mpsc::channel();
        let (updates_tx, updates_rx) = mpsc::channel();
        let (started_tx, started_rx) = oneshot::channel();
        let engine = Arc::clone(&self.engine);
        let contexts = self.speech_contexts.clone();
        let worker = std::thread::Builder::new()
            .name("nemotron-stt".into())
            .spawn(move || {
                let mut stream = match Stream::create(engine, &contexts) {
                    Ok(s) => {
                        let _ = started_tx.send(Ok(()));
                        s
                    }
                    Err(e) => {
                        let _ = started_tx.send(Err(e));
                        return;
                    }
                };
                let mut pending = None;
                'worker: loop {
                    let command = match pending.take() {
                        Some(c) => c,
                        None => match commands_rx.recv() {
                            Ok(c) => c,
                            Err(_) => break,
                        },
                    };
                    match command {
                        WorkerCommand::Audio {
                            samples,
                            sample_rate,
                        } => {
                            let mut failed = false;
                            if let Err(e) = stream.push(&samples, sample_rate) {
                                failed = true;
                                let _ = updates_tx.send(Err(e));
                            }
                            // Batch everything already queued before decoding once.
                            loop {
                                match commands_rx.try_recv() {
                                    Ok(WorkerCommand::Audio {
                                        samples,
                                        sample_rate,
                                    }) => {
                                        if let Err(e) = stream.push(&samples, sample_rate) {
                                            failed = true;
                                            let _ = updates_tx.send(Err(e));
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
                            if pending.is_none() && !failed {
                                match stream.interims() {
                                    Ok(frames) if !frames.is_empty() => {
                                        let _ = updates_tx.send(Ok(frames));
                                    }
                                    Ok(_) => {}
                                    Err(e) => {
                                        let _ = updates_tx.send(Err(e));
                                    }
                                }
                            }
                        }
                        WorkerCommand::Flush { reply } => {
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
                tracing::info!("call-local Nemotron stream initialized (in-process)");
                Ok(())
            }
            Ok(Err(e)) => {
                let _ = worker.join();
                Err(e)
            }
            Err(_) => {
                let _ = worker.join();
                Err(FlowcatError::Other(
                    "Nemotron STT worker exited during startup".into(),
                ))
            }
        }
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if audio.is_empty() {
            return self.drain_updates();
        }
        let sample_rate = i32::try_from(audio.sample_rate)
            .map_err(|_| FlowcatError::Other("Nemotron input sample rate is too large".into()))?;
        let samples = mono_f32(&audio);
        let commands = self
            .commands
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("Nemotron STT is not started".into()))?;
        commands
            .send(WorkerCommand::Audio {
                samples,
                sample_rate,
            })
            .map_err(|_| FlowcatError::Other("Nemotron STT worker stopped".into()))?;
        self.drain_updates()
    }

    async fn flush(&mut self) -> Result<Vec<Frame>> {
        let final_result = self.request(|reply| WorkerCommand::Flush { reply }).await;
        let interim_result = self.drain_updates();
        match final_result {
            Err(e) => Err(e),
            Ok(final_frames) => {
                let mut frames = match interim_result {
                    Ok(frames) => frames,
                    Err(e) => {
                        tracing::warn!(error = %e, "discarding stale Nemotron interim error after successful final");
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
        let reset = self
            .request(|reply| WorkerCommand::SetMuted { muted, reply })
            .await;
        let drained = self.drain_updates().err();
        if let Some(e) = reset.err().or(drained) {
            tracing::warn!(error = %e, muted, "Nemotron mute reset failed");
        }
    }
}

impl Drop for NemotronNativeStt {
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(WorkerCommand::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

// ---- helpers -------------------------------------------------------------------

/// Fold a decoded (non-final) result into the running hypothesis: a result
/// that extends the hypothesis replaces it (full-hypothesis shape), otherwise
/// it is appended once (delta shape).
fn absorb(hypothesis: &mut String, text: &str) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    if hypothesis.is_empty() || text.starts_with(hypothesis.as_str()) {
        *hypothesis = text.to_string();
    } else if !hypothesis.ends_with(text) {
        if !hypothesis.ends_with(' ') {
            hypothesis.push(' ');
        }
        hypothesis.push_str(text);
    }
}

fn mono_f32(audio: &AudioFrame) -> Vec<f32> {
    let channels = usize::from(audio.num_channels.max(1));
    if channels == 1 {
        return audio.pcm.iter().map(|s| *s as f32 / 32_768.0).collect();
    }
    audio
        .pcm
        .chunks(channels)
        .map(|frame| {
            let sum: i64 = frame.iter().map(|s| i64::from(*s)).sum();
            (sum as f32 / frame.len() as f32) / 32_768.0
        })
        .collect()
}

fn spoken_text(text: &str) -> Option<String> {
    let text = text.trim();
    text.chars()
        .any(char::is_alphanumeric)
        .then(|| text.to_string())
}

fn interim_frame(text: String) -> Frame {
    Frame::InterimTranscription {
        text,
        user_id: Arc::from("user"),
        language: Some(Language("en-US".into())),
    }
}

fn final_frame(text: String) -> Frame {
    Frame::Transcription {
        text,
        user_id: Arc::from("user"),
        language: Some(Language("en-US".into())),
        final_: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_parse() {
        assert_eq!(Device::parse("auto").unwrap(), Device::Gpu(0));
        assert_eq!(Device::parse("Metal").unwrap(), Device::Gpu(0));
        assert_eq!(Device::parse("cpu").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("cuda:1").unwrap(), Device::Gpu(1));
        assert!(Device::parse("tpu").is_err());
    }

    #[test]
    fn hypothesis_absorbs_both_full_and_delta_interims() {
        let mut h = String::new();
        absorb(&mut h, "What");
        absorb(&mut h, "What time"); // full hypothesis so far
        absorb(&mut h, "is it"); // delta
        assert_eq!(h, "What time is it");
        absorb(&mut h, "is it"); // repeated delta is ignored
        assert_eq!(h, "What time is it");
        absorb(&mut h, ""); // empty results are no-ops
        assert_eq!(h, "What time is it");
    }

    /// Live probe against the installed model: run with
    /// `cargo test --features nemotron-native -- --ignored native_probe --nocapture`.
    /// Prints every result the C API yields for the T1 fixture so the
    /// interim/final semantics are verified against reality, and asserts the
    /// forced-endpoint final says "what time is it".
    #[test]
    #[ignore]
    fn native_probe() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap();
        let model = root.join("models/nemotron/nvidia/nemotron-speech-streaming-en-0.6b/ebe59e5a817142986528bbbee5dba8db7b38ed50/nemotron-speech-streaming-en-0.6b.q8_0.gguf");
        let engine = load_engine(model.to_str().unwrap(), Device::Gpu(0), 6).expect("load");
        let mut stream = Stream::create(engine, &[]).expect("stream");
        let wav = std::fs::read(root.join("fixtures/t1_time.wav")).expect("fixture");
        let pcm: Vec<f32> = wav[44..]
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32_768.0)
            .collect();
        for chunk in pcm.chunks(320) {
            stream.push(chunk, 16_000).unwrap();
            for d in stream.drain().unwrap() {
                println!("result: {d:?}");
                stream.absorb_interim(&d.text);
            }
        }
        println!("hypothesis before flush: {:?}", stream.hypothesis);
        let frames = stream.flush().unwrap();
        println!("flush frames: {frames:?}");
        let text = match &frames[0] {
            Frame::Transcription { text, .. } => text.to_lowercase(),
            other => panic!("unexpected {other:?}"),
        };
        assert!(text.contains("what time is it"), "got {text:?}");
    }
}
