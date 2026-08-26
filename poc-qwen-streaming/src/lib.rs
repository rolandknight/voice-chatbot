//! poc-qwen-streaming as a library: the PyO3-embedded mlx-audio engine and its
//! streaming [`engine::StreamEvent`] channel, reusable in-process by other Rust
//! binaries (the FlowCat PoC's `qwen-tts` feature) without the HTTP server.

pub mod bench;
pub mod config;
pub mod engine;
pub mod pcm;
pub mod server;
