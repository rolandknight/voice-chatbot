//! Stateful mono sample-rate conversion for device PCM streams.

use anyhow::{bail, Context, Result};
use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{
    Async, FixedAsync, Resampler as _, SincInterpolationParameters, SincInterpolationType,
    WindowFunction,
};

const BLOCK: usize = 256;

/// A streaming, mono, signed-16-bit resampler.
///
/// Device callbacks can deliver arbitrary buffer sizes. The carry buffer makes
/// those buffers a continuous stream while `rubato` receives its fixed-size
/// input blocks.
pub struct StreamingResampler {
    from_rate: u32,
    to_rate: u32,
    inner: Option<Async<f32>>,
    carry: Vec<i16>,
    input: Vec<f32>,
    output: Vec<f32>,
}

impl StreamingResampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Result<Self> {
        if from_rate == 0 || to_rate == 0 {
            bail!("sample rates must be greater than zero ({from_rate} -> {to_rate})");
        }
        if from_rate == to_rate {
            return Ok(Self {
                from_rate,
                to_rate,
                inner: None,
                carry: Vec::new(),
                input: Vec::new(),
                output: Vec::new(),
            });
        }

        let params = SincInterpolationParameters {
            sinc_len: 256,
            f_cutoff: Some(0.95),
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Cubic,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = Async::<f32>::new_sinc(
            to_rate as f64 / from_rate as f64,
            1.0,
            &params,
            BLOCK,
            1,
            FixedAsync::Input,
        )
        .with_context(|| format!("initialize resampler {from_rate} -> {to_rate}"))?;
        let output = vec![0.0; inner.output_frames_max()];

        Ok(Self {
            from_rate,
            to_rate,
            inner: Some(inner),
            carry: Vec::new(),
            input: Vec::with_capacity(BLOCK),
            output,
        })
    }

    pub fn process(&mut self, pcm: &[i16]) -> Result<Vec<i16>> {
        let Some(inner) = self.inner.as_mut() else {
            return Ok(pcm.to_vec());
        };

        self.carry.extend_from_slice(pcm);
        let mut converted = Vec::new();
        let mut consumed = 0;
        while self.carry.len() - consumed >= BLOCK {
            self.input.clear();
            self.input.extend(
                self.carry[consumed..consumed + BLOCK]
                    .iter()
                    .map(|sample| *sample as f32 / 32768.0),
            );
            let input = InterleavedSlice::new(&self.input, 1, BLOCK)
                .context("construct resampler input")?;
            let output_capacity = self.output.len();
            let mut output = InterleavedSlice::new_mut(&mut self.output, 1, output_capacity)
                .context("construct resampler output")?;
            let (_, frames) = inner
                .process_into_buffer(&input, &mut output, None)
                .with_context(|| format!("resample {} -> {} Hz", self.from_rate, self.to_rate))?;
            converted.extend(
                self.output[..frames]
                    .iter()
                    .map(|sample| (sample * 32768.0).round().clamp(-32768.0, 32767.0) as i16),
            );
            consumed += BLOCK;
        }
        if consumed != 0 {
            self.carry.drain(..consumed);
        }
        Ok(converted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_preserves_samples() {
        let input = vec![i16::MIN, -100, 0, 100, i16::MAX];
        let mut resampler = StreamingResampler::new(48_000, 48_000).unwrap();
        assert_eq!(resampler.process(&input).unwrap(), input);
    }

    #[test]
    fn arbitrary_chunks_are_buffered_and_resampled() {
        let mut resampler = StreamingResampler::new(16_000, 48_000).unwrap();
        assert!(resampler.process(&vec![100; 128]).unwrap().is_empty());
        let output = resampler.process(&vec![100; 128]).unwrap();
        assert!(
            output.len() >= 700 && output.len() <= 800,
            "{}",
            output.len()
        );
    }

    #[test]
    fn zero_rate_is_rejected() {
        assert!(StreamingResampler::new(0, 48_000).is_err());
        assert!(StreamingResampler::new(48_000, 0).is_err());
    }
}
