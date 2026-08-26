//! Qwen3-TTS as a library: the PyO3-embedded mlx-audio engine and its
//! streaming [`engine::StreamEvent`] channel, used in-process by the server's
//! `qwen-tts` feature. The Python package lives in `python/qwen_tts`.

pub mod config;
pub mod engine;
pub mod pcm;
