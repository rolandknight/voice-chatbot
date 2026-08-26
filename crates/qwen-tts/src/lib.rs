//! Qwen3-TTS as a library: the PyO3-embedded mlx-audio engine and its
//! streaming [`engine::StreamEvent`] channel, used in-process by the server's
//! `qwen-tts` feature. The Python bridge lives in `python/poc_qwen_streaming`.

pub mod config;
pub mod engine;
pub mod pcm;
