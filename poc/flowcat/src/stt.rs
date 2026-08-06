//! BabelStt — whole-utterance whisper.cpp STT for the duplex pipeline.
//!
//! FlowCat's `WhisperLocalStt` emits a final transcription every fixed ~4 s
//! window, which in full-duplex mode fires turns on partial utterances and
//! hallucinates on silence. This service pairs with the vendored `SpeechGate`:
//! it accumulates the gated speech audio and transcribes the WHOLE utterance
//! once, when the gate's all-zero flush marker (length
//! `SPEECH_GATE_FLUSH_SAMPLES`) arrives at a VAD falling edge. One VAD turn →
//! one final transcription. Inference pattern cribbed from upstream
//! `whisper_local.rs` (scoped thread; greedy; annotations stripped).

use std::sync::Arc;

use async_trait::async_trait;
use flowcat_core::pipeline::SPEECH_GATE_FLUSH_SAMPLES;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;
use flowcat_core::{FlowcatError, Result};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const WHISPER_RATE: u32 = 16_000;
/// Ignore utterances shorter than this (VAD blips): 300 ms of 16 kHz samples.
const MIN_UTTERANCE_SAMPLES: usize = 4_800;
/// Safety cap: finalize anyway past 29 s of buffered speech.
const MAX_UTTERANCE_SAMPLES: usize = 29 * 16_000;

pub struct BabelStt {
    model_path: String,
    ctx: Option<Arc<WhisperContext>>,
    buf: Vec<f32>,
    muted: bool,
}

impl BabelStt {
    pub fn new(model_path: String) -> Self {
        Self {
            model_path,
            ctx: None,
            buf: Vec::new(),
            muted: false,
        }
    }

    /// i16 any-rate mono-ish → f32 16 kHz (linear resample, as upstream).
    fn append(&mut self, audio: &AudioFrame) {
        let src_rate = audio.sample_rate.max(1);
        if src_rate == WHISPER_RATE {
            self.buf
                .extend(audio.pcm.iter().map(|s| *s as f32 / 32768.0));
            return;
        }
        let ratio = WHISPER_RATE as f64 / src_rate as f64;
        let out_len = (audio.pcm.len() as f64 * ratio) as usize;
        self.buf.reserve(out_len);
        for i in 0..out_len {
            let src = (i as f64 / ratio) as usize;
            let s = audio.pcm.get(src).copied().unwrap_or(0);
            self.buf.push(s as f32 / 32768.0);
        }
    }

    fn transcribe(&mut self) -> Result<Vec<Frame>> {
        let samples = std::mem::take(&mut self.buf);
        if samples.len() < MIN_UTTERANCE_SAMPLES {
            return Ok(vec![]);
        }
        let ctx = self
            .ctx
            .clone()
            .ok_or_else(|| FlowcatError::Other("BabelStt: not started".into()))?;
        let text = std::thread::scope(|s| {
            s.spawn(move || -> Result<String> {
                let mut state = ctx
                    .create_state()
                    .map_err(|e| FlowcatError::Other(format!("whisper create_state: {e}")))?;
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_n_threads(
                    std::thread::available_parallelism()
                        .map(|n| n.get().min(4))
                        .unwrap_or(2) as i32,
                );
                params.set_translate(false);
                params.set_print_special(false);
                params.set_print_progress(false);
                params.set_print_realtime(false);
                params.set_print_timestamps(false);
                params.set_language(Some("en"));
                state
                    .full(params, &samples)
                    .map_err(|e| FlowcatError::Other(format!("whisper inference: {e}")))?;
                let mut out = String::new();
                for i in 0..state.full_n_segments() {
                    if let Some(seg) = state.get_segment(i) {
                        if let Ok(s) = seg.to_str_lossy() {
                            out.push_str(s.as_ref());
                        }
                    }
                }
                Ok(out.trim().to_string())
            })
            .join()
            .map_err(|_| FlowcatError::Other("BabelStt: inference thread panicked".into()))?
        })?;
        // Strip whisper's non-speech annotations ([BLANK_AUDIO], (silence), …).
        let spoken: String = {
            let mut s = text;
            for (open, close) in [('[', ']'), ('(', ')')] {
                while let (Some(a), Some(b)) = (s.find(open), s.find(close)) {
                    if a < b {
                        s.replace_range(a..=b, "");
                    } else {
                        break;
                    }
                }
            }
            s.trim().to_string()
        };
        if !spoken.chars().any(|c| c.is_alphanumeric()) {
            return Ok(vec![]);
        }
        tracing::info!(text = %spoken, "utterance transcribed");
        Ok(vec![Frame::Transcription {
            text: spoken,
            user_id: Arc::from("user"),
            language: None,
            final_: true,
        }])
    }
}

#[async_trait]
impl SttService for BabelStt {
    fn name(&self) -> &str {
        "babel-whisper"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        let ctx =
            WhisperContext::new_with_params(&self.model_path, WhisperContextParameters::default())
                .map_err(|e| FlowcatError::Other(format!("whisper load {}: {e}", self.model_path)))?;
        self.ctx = Some(Arc::new(ctx));
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        let is_flush =
            audio.pcm.len() == SPEECH_GATE_FLUSH_SAMPLES && audio.pcm.iter().all(|s| *s == 0);
        if is_flush || self.buf.len() >= MAX_UTTERANCE_SAMPLES {
            return self.transcribe();
        }
        self.append(&audio);
        Ok(vec![])
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if muted {
            self.buf.clear();
        }
    }
}
