//! Qwen3-TTS streaming backend (`POC_TTS_BACKEND=qwen`, Cargo feature `qwen-tts`).
//!
//! poc-qwen-streaming's engine runs in-process: one Python thread owns the GIL
//! and mlx-audio's own worker thread owns Metal (see its `engine.rs`). Each
//! synthesis is a clone from the configured preset voice; chunks arrive on an
//! `mpsc` receiver as the model emits them (~0.2 s to the first one) and are
//! re-cut into 20 ms frames so the sink's pacing matches the other backends.
//! Dropping the stream drops the receiver, which stops generation after the
//! current chunk — that is the barge-in path.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::TtsService;
use flowcat_core::{FlowcatError, Result};
use futures::stream::BoxStream;
use futures::StreamExt;
use poc_qwen_streaming::engine::{Engine, StreamEvent};
use serde_json::{json, Value};
use tokio::sync::mpsc;

pub const SAMPLE_RATE: u32 = 24_000;
/// 20 ms at 24 kHz — the frame size the other backends feed the sink.
const FRAME_SAMPLES: usize = SAMPLE_RATE as usize / 50;

/// The preset voice every call clones from (resolved once at start-up).
#[derive(Clone, Debug)]
pub struct QwenVoice {
    pub name: String,
    /// Absolute path of the reference clip (from the engine's catalog).
    pub ref_audio: String,
    /// Its sidecar transcript — required for ICL cloning.
    pub ref_text: String,
    pub size: String,
    pub language: String,
    /// Seconds of audio per streamed chunk (mlx-audio `streaming_interval`).
    pub interval_s: f64,
}

impl QwenVoice {
    fn params(&self, text: &str) -> Value {
        json!({
            "text": text,
            "ref_audio": self.ref_audio,
            "ref_text": self.ref_text,
            "language": self.language,
            "size": self.size,
            "interval_s": self.interval_s,
        })
    }
}

/// Engine + voice shared by every call (the engine is a channel handle).
#[derive(Clone)]
pub struct QwenShared {
    pub engine: Engine,
    pub voice: Arc<QwenVoice>,
    /// Process-preloaded `Ready.` greeting so reconnects never synthesize.
    pub ready_pcm: Option<Arc<[i16]>>,
}

pub struct QwenTts {
    shared: QwenShared,
    ctx_counter: u64,
}

impl QwenTts {
    pub fn new(shared: QwenShared) -> Self {
        Self {
            shared,
            ctx_counter: 0,
        }
    }

    fn cached_ready_pcm(&self, text: &str) -> Option<&[i16]> {
        let normalized = text.trim();
        let is_ready =
            normalized.eq_ignore_ascii_case("ready") || normalized.eq_ignore_ascii_case("ready.");
        is_ready
            .then_some(self.shared.ready_pcm.as_deref())
            .flatten()
    }

    fn next_context(&mut self) -> Arc<str> {
        self.ctx_counter += 1;
        Arc::from(format!("qwen-{}", self.ctx_counter))
    }
}

/// Whole-utterance frames from finished PCM (cached greeting path).
pub fn frames_from_pcm(pcm: &[i16], context_id: Arc<str>) -> Vec<Frame> {
    let mut out = Vec::with_capacity(pcm.len() / FRAME_SAMPLES + 2);
    out.push(Frame::TtsStarted {
        context_id: Some(context_id.clone()),
    });
    for samples in pcm.chunks(FRAME_SAMPLES) {
        out.push(Frame::TtsAudio {
            audio: Arc::new(AudioFrame::mono(samples.to_vec(), SAMPLE_RATE)),
            context_id: Some(context_id.clone()),
        });
    }
    out.push(Frame::TtsStopped {
        context_id: Some(context_id),
    });
    out
}

/// Re-cuts engine chunks into 20 ms frames and closes the span on Done/Error.
struct Cutter {
    pending: Vec<i16>,
    queue: VecDeque<Frame>,
    context_id: Arc<str>,
    sample_rate: u32,
    finished: bool,
}

impl Cutter {
    fn new(context_id: Arc<str>) -> Self {
        Self {
            pending: Vec::new(),
            queue: VecDeque::new(),
            context_id,
            sample_rate: SAMPLE_RATE,
            finished: false,
        }
    }

    fn audio_frame(&self, samples: Vec<i16>) -> Frame {
        Frame::TtsAudio {
            audio: Arc::new(AudioFrame::mono(samples, self.sample_rate)),
            context_id: Some(self.context_id.clone()),
        }
    }

    fn on_event(&mut self, event: StreamEvent) {
        match event {
            StreamEvent::Start { sample_rate, .. } => {
                self.sample_rate = sample_rate;
                self.queue.push_back(Frame::TtsStarted {
                    context_id: Some(self.context_id.clone()),
                });
            }
            StreamEvent::Audio { samples } => {
                self.pending.extend_from_slice(&samples);
                let full = self.pending.len() / FRAME_SAMPLES * FRAME_SAMPLES;
                let rest = self.pending.split_off(full);
                let ready = std::mem::replace(&mut self.pending, rest);
                for chunk in ready.chunks(FRAME_SAMPLES) {
                    let frame = self.audio_frame(chunk.to_vec());
                    self.queue.push_back(frame);
                }
            }
            StreamEvent::Done { timings } => {
                tracing::debug!(%timings, "qwen tts done");
                self.finish();
            }
            StreamEvent::Error(message) => {
                tracing::warn!(%message, "qwen tts error mid-stream");
                self.finish();
            }
        }
    }

    /// Engine channel closed (Done, Error, or the engine went away).
    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if !self.pending.is_empty() {
            let tail = std::mem::take(&mut self.pending);
            let frame = self.audio_frame(tail);
            self.queue.push_back(frame);
        }
        self.queue.push_back(Frame::TtsStopped {
            context_id: Some(self.context_id.clone()),
        });
    }
}

/// Turn the engine's event channel into a frame stream. The receiver lives in
/// the stream, so dropping the stream cancels the generation.
pub fn event_stream(
    rx: mpsc::Receiver<StreamEvent>,
    first: StreamEvent,
    context_id: Arc<str>,
) -> BoxStream<'static, Frame> {
    let mut cutter = Cutter::new(context_id);
    cutter.on_event(first);
    futures::stream::unfold((rx, cutter), |(mut rx, mut cutter)| async move {
        loop {
            if let Some(frame) = cutter.queue.pop_front() {
                return Some((frame, (rx, cutter)));
            }
            if cutter.finished {
                return None;
            }
            match rx.recv().await {
                Some(event) => cutter.on_event(event),
                None => cutter.finish(),
            }
        }
    })
    .boxed()
}

/// Synthesize `text` to finished PCM (start-up greeting cache, tests).
pub async fn synthesize_pcm(engine: &Engine, voice: &QwenVoice, text: &str) -> Result<Vec<i16>> {
    let mut rx = engine
        .generate("clone", voice.params(text))
        .map_err(|e| FlowcatError::Other(format!("qwen tts: {e}")))?;
    let mut pcm = Vec::new();
    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Audio { samples } => pcm.extend_from_slice(&samples),
            StreamEvent::Error(message) => {
                return Err(FlowcatError::Other(format!("qwen tts: {message}")))
            }
            StreamEvent::Start { .. } | StreamEvent::Done { .. } => {}
        }
    }
    Ok(pcm)
}

#[async_trait]
impl TtsService for QwenTts {
    fn name(&self) -> &str {
        "qwen"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        let stream = self.run_tts_stream(text).await?;
        Ok(stream.collect().await)
    }

    async fn run_tts_stream<'a>(&'a mut self, text: &'a str) -> Result<BoxStream<'a, Frame>> {
        let context_id = self.next_context();
        if let Some(cached) = self.cached_ready_pcm(text) {
            tracing::info!(samples = cached.len(), "serving cached greeting audio");
            return Ok(futures::stream::iter(frames_from_pcm(cached, context_id)).boxed());
        }
        let requested = std::time::Instant::now();
        let mut rx = self
            .shared
            .engine
            .generate("clone", self.shared.voice.params(text))
            .map_err(|e| FlowcatError::Other(format!("qwen tts: {e}")))?;
        // Surface start-up failures (bad voice, engine gone) as an error rather
        // than an empty stream; the first event is Start or Error.
        match rx.recv().await {
            Some(StreamEvent::Error(message)) => {
                Err(FlowcatError::Other(format!("qwen tts: {message}")))
            }
            Some(first) => {
                // Time to the first chunk, request → Start (the engine sends
                // Start with its first audio chunk). Includes any wait behind a
                // previous synthesis on the single MLX worker.
                tracing::info!(
                    chars = text.chars().count(),
                    ttfc_ms = requested.elapsed().as_millis() as u64,
                    "qwen tts first chunk"
                );
                Ok(event_stream(rx, first, context_id))
            }
            None => Err(FlowcatError::Other("qwen tts: engine closed the stream".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_len(frames: &[Frame]) -> usize {
        frames
            .iter()
            .filter_map(|f| match f {
                Frame::TtsAudio { audio, .. } => Some(audio.pcm.len()),
                _ => None,
            })
            .sum()
    }

    #[tokio::test]
    async fn events_become_started_twenty_ms_frames_and_stopped() {
        let (tx, rx) = mpsc::channel(8);
        let ctx: Arc<str> = Arc::from("qwen-1");
        tx.send(StreamEvent::Audio { samples: vec![1; 1000] }).await.unwrap(); // 2 full frames + 40 left
        tx.send(StreamEvent::Audio { samples: vec![2; 500] }).await.unwrap(); // 540 → 1 frame + 60
        tx.send(StreamEvent::Done { timings: json!({}) }).await.unwrap();
        drop(tx);
        let frames: Vec<Frame> = event_stream(
            rx,
            StreamEvent::Start { sample_rate: 24_000, model: "m".into() },
            ctx,
        )
        .collect()
        .await;
        assert!(matches!(frames.first(), Some(Frame::TtsStarted { .. })));
        assert!(matches!(frames.last(), Some(Frame::TtsStopped { .. })));
        let audio: Vec<usize> = frames
            .iter()
            .filter_map(|f| match f {
                Frame::TtsAudio { audio, .. } => Some(audio.pcm.len()),
                _ => None,
            })
            .collect();
        assert_eq!(audio, vec![480, 480, 480, 60], "20 ms frames, remainder flushed on Done");
        assert_eq!(audio_len(&frames), 1500);
    }

    #[tokio::test]
    async fn mid_stream_error_closes_the_span() {
        let (tx, rx) = mpsc::channel(8);
        tx.send(StreamEvent::Audio { samples: vec![0; 480] }).await.unwrap();
        tx.send(StreamEvent::Error("boom".into())).await.unwrap();
        let frames: Vec<Frame> = event_stream(
            rx,
            StreamEvent::Start { sample_rate: 24_000, model: "m".into() },
            Arc::from("qwen-2"),
        )
        .collect()
        .await;
        assert_eq!(frames.len(), 3);
        assert!(matches!(frames[2], Frame::TtsStopped { .. }));
    }

    #[test]
    fn cached_greeting_frames_are_whole() {
        let frames = frames_from_pcm(&[7; 1000], Arc::from("qwen-3"));
        assert_eq!(frames.len(), 1 + 3 + 1);
        assert_eq!(audio_len(&frames), 1000);
    }
}
