//! ChatterboxTts — cloned-voice TTS via the Chatterbox-TTS-Server's OpenAI
//! `/v1/audio/speech` endpoint (Phase 1b, T14).
//!
//! Why not FlowCat's `KokoroTts` client with a base-URL swap: Chatterbox
//! rejects `response_format: "pcm"` (422) and takes the voice as a reference
//! WAV filename. This service requests `wav` and strips the RIFF header down
//! to the `data` chunk. Pipecat needed a 139-line subclass for the same
//! backend (`chatterbox_tts.py`); this is the FlowCat-side equivalent for the
//! T14 workaround-LOC comparison.

use std::io::{self, ErrorKind};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, StartParams};
use flowcat_core::service::TtsService;
use flowcat_core::{FlowcatError, Result};
use serde_json::json;

pub const SAMPLE_RATE: u32 = 24_000;

pub struct ChatterboxTts {
    base_url: String,
    voice: String,
    http: reqwest::Client,
    ctx_counter: u64,
    /// Process-preloaded audio for the deterministic connect greeting. The PCM
    /// is immutable and Arc-backed so each per-call TTS service can share it
    /// without sharing mutable service state.
    ready_pcm: Option<Arc<[i16]>>,
}

impl ChatterboxTts {
    pub fn new(base_url: String, voice: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            voice,
            http: reqwest::Client::new(),
            ctx_counter: 0,
            ready_pcm: None,
        }
    }

    /// Configure an optional process-preloaded `Ready.` greeting.
    ///
    /// Empty PCM is treated as a missing cache, preserving the HTTP fallback
    /// instead of silently emitting an empty greeting. Clone this [`Arc`] into
    /// each call's service to share the samples without sharing `ctx_counter`.
    pub fn with_ready_pcm(mut self, ready_pcm: Option<Arc<[i16]>>) -> Self {
        self.ready_pcm = ready_pcm.filter(|pcm| !pcm.is_empty());
        self
    }

    fn cached_ready_pcm(&self, text: &str) -> Option<&[i16]> {
        let normalized = text.trim();
        let is_ready =
            normalized.eq_ignore_ascii_case("ready") || normalized.eq_ignore_ascii_case("ready.");
        is_ready.then_some(self.ready_pcm.as_deref()).flatten()
    }
}

fn invalid_wav(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

/// Decode a mono, 16-bit PCM, 24 kHz RIFF/WAVE file into shareable samples.
///
/// This deliberately validates the format used by [`ChatterboxTts`] instead of
/// treating arbitrary WAV bytes as s16 audio. Chunk sizes and offsets use
/// checked arithmetic, so malformed or truncated input returns `InvalidData`
/// rather than panicking.
pub fn decode_ready_wav(bytes: &[u8]) -> io::Result<Arc<[i16]>> {
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid_wav("expected a RIFF/WAVE file"));
    }

    let mut format = None;
    let mut pcm_bytes = None;
    let mut offset = 12usize;
    while offset < bytes.len() {
        if bytes.len() - offset < 8 {
            return Err(invalid_wav("truncated WAV chunk header"));
        }
        let id = &bytes[offset..offset + 4];
        let len = u32::from_le_bytes([
            bytes[offset + 4],
            bytes[offset + 5],
            bytes[offset + 6],
            bytes[offset + 7],
        ]) as usize;
        let start = offset
            .checked_add(8)
            .ok_or_else(|| invalid_wav("WAV chunk offset overflow"))?;
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid_wav("WAV chunk length overflow"))?;
        if end > bytes.len() {
            return Err(invalid_wav("truncated WAV chunk data"));
        }

        match id {
            b"fmt " => {
                if len < 16 {
                    return Err(invalid_wav("WAV fmt chunk is too short"));
                }
                let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
                let u32_at = |at: usize| {
                    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
                };
                format = Some((
                    u16_at(start),
                    u16_at(start + 2),
                    u32_at(start + 4),
                    u16_at(start + 12),
                    u16_at(start + 14),
                ));
            }
            b"data" => pcm_bytes = Some(&bytes[start..end]),
            _ => {}
        }

        offset = end
            .checked_add(len & 1)
            .ok_or_else(|| invalid_wav("WAV padding offset overflow"))?;
        if offset > bytes.len() {
            return Err(invalid_wav("truncated WAV chunk padding"));
        }
    }

    let (encoding, channels, sample_rate, block_align, bits_per_sample) =
        format.ok_or_else(|| invalid_wav("WAV fmt chunk is missing"))?;
    if encoding != 1 {
        return Err(invalid_wav(format!(
            "unsupported WAV encoding {encoding}; expected PCM"
        )));
    }
    if channels != 1 {
        return Err(invalid_wav(format!(
            "unsupported WAV channel count {channels}; expected mono"
        )));
    }
    if sample_rate != SAMPLE_RATE {
        return Err(invalid_wav(format!(
            "unsupported WAV sample rate {sample_rate}; expected {SAMPLE_RATE}"
        )));
    }
    if bits_per_sample != 16 || block_align != 2 {
        return Err(invalid_wav(format!(
            "unsupported WAV sample format: {bits_per_sample}-bit, block align {block_align}; expected s16 mono"
        )));
    }

    let pcm_bytes = pcm_bytes.ok_or_else(|| invalid_wav("WAV data chunk is missing"))?;
    if pcm_bytes.is_empty() {
        return Err(invalid_wav("WAV data chunk is empty"));
    }
    if pcm_bytes.len() % 2 != 0 {
        return Err(invalid_wav("WAV data chunk has a partial s16 sample"));
    }
    Ok(pcm_bytes
        .chunks_exact(2)
        .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
        .collect::<Vec<_>>()
        .into())
}

/// Load and validate a cached Chatterbox greeting from disk.
pub fn load_ready_wav(path: impl AsRef<Path>) -> io::Result<Arc<[i16]>> {
    decode_ready_wav(&std::fs::read(path)?)
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
        let len =
            u32::from_le_bytes([bytes[i + 4], bytes[i + 5], bytes[i + 6], bytes[i + 7]]) as usize;
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
        let pcm = if let Some(cached) = self.cached_ready_pcm(text) {
            tracing::info!(samples = cached.len(), "serving cached greeting audio");
            cached.to_vec()
        } else {
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
            strip_wav(&bytes)
                .chunks_exact(2)
                .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
                .collect()
        };

        // Mirror flowcat's one-shot TTS framing: Started, ~20 ms chunks, Stopped.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn wav(pcm: &[i16], sample_rate: u32, channels: u16, bits_per_sample: u16) -> Vec<u8> {
        let data_len = (pcm.len() * 2) as u32;
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * u32::from(block_align);
        let mut bytes = Vec::with_capacity(44 + data_len as usize);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + data_len).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&sample_rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits_per_sample.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&data_len.to_le_bytes());
        for sample in pcm {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    fn frame_context(frame: &Frame) -> Option<&str> {
        match frame {
            Frame::TtsStarted { context_id }
            | Frame::TtsAudio { context_id, .. }
            | Frame::TtsStopped { context_id } => context_id.as_deref(),
            _ => None,
        }
    }

    #[test]
    fn ready_wav_loader_validates_and_decodes_expected_format() {
        let expected = [-32_768, -7, 0, 42, 32_767];
        let decoded = decode_ready_wav(&wav(&expected, SAMPLE_RATE, 1, 16)).unwrap();
        assert_eq!(&*decoded, expected.as_slice());

        let wrong_rate = decode_ready_wav(&wav(&expected, 16_000, 1, 16)).unwrap_err();
        assert_eq!(wrong_rate.kind(), ErrorKind::InvalidData);
        assert!(wrong_rate.to_string().contains("sample rate"));

        let truncated = decode_ready_wav(&wav(&expected, SAMPLE_RATE, 1, 16)[..42]).unwrap_err();
        assert_eq!(truncated.kind(), ErrorKind::InvalidData);
    }

    #[tokio::test]
    async fn cached_ready_skips_http_and_preserves_one_shot_framing() {
        let pcm: Arc<[i16]> = (0..1_001)
            .map(|sample| sample as i16)
            .collect::<Vec<_>>()
            .into();
        // Port zero cannot serve HTTP. Success therefore proves the cache branch
        // did not attempt a request.
        let mut tts = ChatterboxTts::new("http://127.0.0.1:0".into(), "marvin.wav".into())
            .with_ready_pcm(Some(pcm.clone()));

        let frames = tts.run_tts(" \nREADY.\t").await.unwrap();
        assert!(matches!(frames.first(), Some(Frame::TtsStarted { .. })));
        assert!(matches!(frames.last(), Some(Frame::TtsStopped { .. })));
        assert!(frames
            .iter()
            .all(|frame| frame_context(frame) == Some("cb-1")));

        let audio: Vec<_> = frames
            .iter()
            .filter_map(|frame| match frame {
                Frame::TtsAudio { audio, .. } => Some(audio),
                _ => None,
            })
            .collect();
        assert_eq!(audio.len(), 3, "1001 samples should use 480-sample chunks");
        assert!(audio
            .iter()
            .all(|audio| audio.sample_rate == SAMPLE_RATE && audio.num_channels == 1));
        let emitted: Vec<i16> = audio
            .into_iter()
            .flat_map(|audio| audio.pcm.iter().copied())
            .collect();
        assert_eq!(emitted.as_slice(), &*pcm);

        let second = tts.run_tts("Ready").await.unwrap();
        assert!(second
            .iter()
            .all(|frame| frame_context(frame) == Some("cb-2")));
    }

    #[tokio::test]
    async fn missing_cache_and_non_ready_text_keep_http_fallback() {
        let cached: Arc<[i16]> = vec![1, 2, 3].into();
        let mut without_cache =
            ChatterboxTts::new("http://127.0.0.1:0".into(), "marvin.wav".into());
        let missing = without_cache.run_tts("Ready.").await.unwrap_err();
        assert!(missing.to_string().contains("chatterbox request"));

        let mut with_cache = ChatterboxTts::new("http://127.0.0.1:0".into(), "marvin.wav".into())
            .with_ready_pcm(Some(cached));
        for text in ["Ready!", "Ready now", "not ready"] {
            let error = with_cache.run_tts(text).await.unwrap_err();
            assert!(error.to_string().contains("chatterbox request"));
        }
    }

    #[test]
    fn empty_ready_cache_is_treated_as_missing() {
        let tts = ChatterboxTts::new("http://127.0.0.1:0".into(), "marvin.wav".into())
            .with_ready_pcm(Some(Arc::from([])));
        assert!(tts.cached_ready_pcm("Ready.").is_none());
    }
}
