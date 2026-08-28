//! BabelStt — whole-utterance whisper.cpp STT for the duplex pipeline.
//!
//! FlowCat's `WhisperLocalStt` emits a final transcription every fixed ~4 s
//! window, which in full-duplex mode fires turns on partial utterances and
//! hallucinates on silence. This service pairs with the vendored `SpeechGate`:
//! it accumulates the gated speech audio and transcribes the WHOLE utterance
//! once through `SttService::flush()` at a VAD falling edge. One VAD turn → one
//! final transcription. Inference pattern cribbed from upstream
//! `whisper_local.rs` (scoped thread; greedy; annotations stripped).

use std::sync::Arc;

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::SttService;
use flowcat_core::{FlowcatError, Result};
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

const WHISPER_RATE: u32 = 16_000;
/// Ignore utterances shorter than this (VAD blips): 300 ms of 16 kHz samples.
const MIN_UTTERANCE_SAMPLES: usize = 4_800;
/// Safety cap: finalize anyway past 29 s of buffered speech.
const MAX_UTTERANCE_SAMPLES: usize = 29 * 16_000;

/// The immutable whisper model can be shared between calls. Each call creates
/// and retains its own mutable inference state with [`WhisperContext::create_state`].
pub type SharedWhisperContext = Arc<WhisperContext>;

/// Load the whisper model once so callers can clone the returned context into
/// each call-local [`BabelStt`] instance.
pub fn load_context(model_path: &str) -> Result<SharedWhisperContext> {
    let ctx = WhisperContext::new_with_params(model_path, WhisperContextParameters::default())
        .map_err(|e| FlowcatError::Other(format!("whisper load {model_path}: {e}")))?;
    Ok(Arc::new(ctx))
}

pub struct BabelStt {
    ctx: Option<SharedWhisperContext>,
    /// Mutable decoder/KV/compute state. This is call-local and is never shared;
    /// retaining it avoids reallocating whisper.cpp's large work buffers for
    /// every utterance.
    state: Option<WhisperState>,
    threads: usize,
    buf: Vec<f32>,
    muted: bool,
}

impl BabelStt {
    /// Create a call-local STT service backed by an already-loaded model.
    ///
    /// Cloning the context shares model weights only; `buf`, `muted`, and the
    /// state created once by [`SttService::start`] remain private to this call.
    pub fn from_context(ctx: SharedWhisperContext) -> Self {
        Self {
            ctx: Some(ctx),
            state: None,
            threads: default_thread_count(),
            buf: Vec::new(),
            muted: false,
        }
    }

    /// Set the positive CPU worker count used by whisper.cpp.
    ///
    /// A zero supplied by an unvalidated environment/config value is clamped to
    /// one so it can never wrap into whisper.cpp's invalid zero-thread mode.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1).min(i32::MAX as usize);
        self
    }

    #[cfg(test)]
    fn without_context() -> Self {
        Self {
            ctx: None,
            state: None,
            threads: default_thread_count(),
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
        let state = self
            .state
            .as_mut()
            .ok_or_else(|| FlowcatError::Other("BabelStt: not started".into()))?;
        let threads = self.threads;
        let audio_ms = samples.len() as u64 * 1_000 / WHISPER_RATE as u64;
        let decode_started = std::time::Instant::now();
        let decoded = std::thread::scope(|s| {
            s.spawn(move || -> Result<String> {
                let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
                params.set_n_threads(threads as i32);
                params.set_translate(false);
                // Every VAD turn is independent. Make this explicit rather than
                // relying on whisper.cpp defaults, which have changed over time.
                params.set_no_context(true);
                // Babel uses only text; suppressing timestamp tokens avoids work
                // and timestamp-only failure modes without changing turn shape.
                params.set_no_timestamps(true);
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
        });
        let decode_ms = decode_started.elapsed().as_millis();
        tracing::info!(
            audio_ms,
            decode_ms,
            threads,
            success = decoded.is_ok(),
            "whisper utterance decode finished"
        );
        let text = decoded?;
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
        if self.state.is_some() {
            return Ok(());
        }
        let ctx = self
            .ctx
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("BabelStt: no preloaded context".into()))?;
        let state_started = std::time::Instant::now();
        self.state = Some(
            ctx.create_state()
                .map_err(|e| FlowcatError::Other(format!("whisper create_state: {e}")))?,
        );
        tracing::info!(
            state_init_ms = state_started.elapsed().as_millis(),
            threads = self.threads,
            "call-local whisper state initialized"
        );
        Ok(())
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        self.append(&audio);
        if self.buf.len() >= MAX_UTTERANCE_SAMPLES {
            return self.transcribe();
        }
        Ok(vec![])
    }

    async fn flush(&mut self) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(vec![]);
        }
        self.transcribe()
    }

    async fn set_muted(&mut self, muted: bool) {
        self.muted = muted;
        if muted {
            self.buf.clear();
        }
    }
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(4))
        .unwrap_or(2)
        .max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn flush_drains_a_short_vad_blip_without_a_marker_chunk() {
        let mut stt = BabelStt::without_context();
        let audio = Arc::new(AudioFrame::mono(
            vec![1; MIN_UTTERANCE_SAMPLES - 1],
            WHISPER_RATE,
        ));

        assert!(stt.run_stt(audio).await.expect("buffer audio").is_empty());
        assert_eq!(stt.buf.len(), MIN_UTTERANCE_SAMPLES - 1);
        assert!(stt.flush().await.expect("flush audio").is_empty());
        assert!(stt.buf.is_empty());
    }

    #[tokio::test]
    async fn call_local_audio_and_mute_state_are_isolated() {
        let mut first = BabelStt::without_context();
        let mut second = BabelStt::without_context();
        let audio = Arc::new(AudioFrame::mono(vec![1; 160], WHISPER_RATE));

        first
            .run_stt(Arc::clone(&audio))
            .await
            .expect("buffer first call audio");
        assert_eq!(first.buf.len(), 160);
        assert!(second.buf.is_empty());

        first.set_muted(true).await;
        assert!(first.buf.is_empty());
        second
            .run_stt(audio)
            .await
            .expect("buffer second call audio");
        assert_eq!(second.buf.len(), 160);
        assert!(!second.muted);
    }

    #[test]
    fn thread_builder_accepts_config_and_clamps_zero() {
        assert_eq!(BabelStt::without_context().with_threads(8).threads, 8);
        assert_eq!(BabelStt::without_context().with_threads(0).threads, 1);
    }

    #[tokio::test]
    async fn start_requires_a_preloaded_context() {
        let mut stt = BabelStt::without_context();
        let params = StartParams::default();

        let error = stt.start(&params).await.expect_err("missing context");
        assert!(error.to_string().contains("no preloaded context"));
        assert!(stt.state.is_none());
    }
}
