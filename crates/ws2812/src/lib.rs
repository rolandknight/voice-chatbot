//! Proof of concept: a WS2812B strip on a Raspberry Pi, driven over SPI from
//! Rust, showing a Larson scanner (docs/adr/0008-ws2812-strip-over-spi.md).
//!
//! Three small pieces, each usable on its own once this moves into the client:
//! [`color`] (an RGB triple and its command-line spelling), [`larson`] (the
//! effect, a pure state machine), and [`strip`] (the wire encoding and the
//! spidev writer).

pub mod color;
pub mod larson;
pub mod strip;
