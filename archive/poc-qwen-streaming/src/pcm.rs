//! Sample-format helpers: float32 → int16 for the wire, int16 → WAV for reports.

use anyhow::Result;
use std::path::Path;

pub fn f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)
        .collect()
}

pub fn i16_to_le_bytes(samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

pub fn write_wav(path: &Path, samples: &[i16], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in samples {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_and_scales() {
        let v = f32_to_i16(&[0.0, 1.0, -1.0, 2.0, 0.5]);
        assert_eq!(v, vec![0, 32767, -32767, 32767, 16384]);
    }

    #[test]
    fn little_endian_bytes() {
        assert_eq!(i16_to_le_bytes(&[1, -1]), vec![1, 0, 0xff, 0xff]);
    }
}
