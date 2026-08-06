//! ChatterboxTts — cloned-voice TTS via the Chatterbox-TTS-Server's OpenAI
//! `/v1/audio/speech` endpoint (Phase 1b, T14).
//!
//! Why not FlowCat's `KokoroTts` client with a base-URL swap: Chatterbox
//! rejects `response_format: "pcm"` (422) and takes the voice as a reference
//! WAV filename. This service requests `wav` and strips the RIFF header down
//! to the `data` chunk. Pipecat needed a 139-line subclass for the same
//! backend (`chatterbox_tts.py`); this is the FlowCat-side equivalent for the
//! T14 workaround-LOC comparison.

use std::sync::Arc;

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::TtsService;
use flowcat_core::{FlowcatError, Result};
use serde_json::json;

const SAMPLE_RATE: u32 = 24_000;

pub struct ChatterboxTts {
    base_url: String,
    voice: String,
    http: reqwest::Client,
    ctx_counter: u64,
}

impl ChatterboxTts {
    pub fn new(base_url: String, voice: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            voice,
            http: reqwest::Client::new(),
            ctx_counter: 0,
        }
    }
}

/// Return the PCM payload of a RIFF/WAVE byte stream (scan for the `data`
/// chunk; fall back to the raw bytes when no RIFF magic is present).
fn strip_wav(bytes: &[u8]) -> &[u8] {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" {
        return bytes;
    }
    let mut i = 12;
    while i + 8 <= bytes.len() {
        let id = &bytes[i..i + 4];
        let len = u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]])
            as usize;
        if id == b"data" {
            let start = i + 8;
            return &bytes[start..(start + len).min(bytes.len())];
        }
        i += 8 + len + (len & 1);
    }
    bytes
}

#[async_trait]
impl TtsService for ChatterboxTts {
    fn name(&self) -> &str {
        "chatterbox"
    }

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        Ok(())
    }

    async fn run_tts(&mut self, text: &str) -> Result<Vec<Frame>> {
        self.ctx_counter += 1;
        let context_id: Arc<str> = Arc::from(format!("cb-{}", self.ctx_counter));
        let resp = self
            .http
            .post(format!("{}/v1/audio/speech", self.base_url))
            .json(&json!({
                "model": "chatterbox",
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
            }))
            .send()
            .await
            .map_err(|e| FlowcatError::Other(format!("chatterbox request: {e}")))?;
        if !resp.status().is_success() {
            return Err(FlowcatError::Other(format!(
                "chatterbox {}: {}",
                resp.status(),
                resp.text().await.unwrap_or_default()
            )));
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| FlowcatError::Other(format!("chatterbox body: {e}")))?;
        let pcm_bytes = strip_wav(&bytes);

        // Mirror flowcat's one-shot TTS framing: Started, ~20 ms chunks, Stopped.
        let pcm: Vec<i16> = pcm_bytes
            .chunks_exact(2)
            .map(|b| i16::from_le_bytes([b[0], b[1]]))
            .collect();
        let chunk = (SAMPLE_RATE as usize / 50).max(1);
        let mut out = Vec::with_capacity(pcm.len() / chunk + 2);
        out.push(Frame::TtsStarted {
            context_id: Some(context_id.clone()),
        });
        for samples in pcm.chunks(chunk) {
            out.push(Frame::TtsAudio {
                audio: Arc::new(AudioFrame::mono(samples.to_vec(), SAMPLE_RATE)),
                context_id: Some(context_id.clone()),
            });
        }
        out.push(Frame::TtsStopped {
            context_id: Some(context_id),
        });
        Ok(out)
    }
}
