//! The chime a timer sounds before it speaks.
//!
//! Synthesized rather than shipped as an asset, for one reason that decides the
//! design: the call's output stage builds a fixed `Resampler::new(tts_rate,
//! carrier_rate)` and pushes every `Frame::OutputAudio` through it *without
//! consulting the frame's own `sample_rate`* (`cascaded.rs:1323,1331`). Audio
//! injected at any other rate plays at the wrong pitch and speed. Generating
//! the tone at whatever `tts_rate` the configured backend reports sidesteps
//! that entirely — and costs no asset, no HTTP route and no ffmpeg.

/// Pitch of the chime. A5 — high enough to carry over a room, low enough not to
/// be shrill on a small speaker.
const TONE_HZ: f32 = 880.0;
/// How long each of the two beeps sounds.
const BEEP_MS: u32 = 160;
/// Silence between the two beeps.
const GAP_MS: u32 = 90;
/// Raised-cosine attack and release on each beep. Without it the tone begins
/// and ends on a discontinuity, which is audible as a click.
const EDGE_MS: u32 = 8;
/// Peak amplitude as a fraction of full scale. Loud enough to cut through a
/// room, quiet enough that summing with speech cannot clip.
const PEAK: f32 = 0.35;

/// Total duration of the chime.
pub const DURATION_MS: u32 = BEEP_MS * 2 + GAP_MS;

/// Mono PCM for the chime at `sample_rate`.
///
/// Empty when `sample_rate` is 0, so a misconfigured backend degrades to "no
/// chime" rather than a panic.
pub fn alarm_pcm(sample_rate: u32) -> Vec<i16> {
    if sample_rate == 0 {
        return Vec::new();
    }
    let samples_for = |ms: u32| (sample_rate as u64 * ms as u64 / 1000) as usize;
    let beep = samples_for(BEEP_MS);
    let gap = samples_for(GAP_MS);
    // Never let the two edges overlap: at an implausibly low rate a full-length
    // attack and release would run past each other and the envelope would stop
    // reaching zero at the ends.
    let edge = samples_for(EDGE_MS).min(beep / 2);

    let mut pcm = Vec::with_capacity(beep * 2 + gap);
    for burst in 0..2 {
        if burst == 1 {
            pcm.extend(std::iter::repeat_n(0i16, gap));
        }
        for i in 0..beep {
            let envelope = if edge == 0 {
                1.0
            } else if i < edge {
                // Raised cosine: 0 at the first sample, 1 once the edge is done.
                0.5 - 0.5 * (std::f32::consts::PI * i as f32 / edge as f32).cos()
            } else if i >= beep - edge {
                let from_end = beep - 1 - i;
                0.5 - 0.5 * (std::f32::consts::PI * from_end as f32 / edge as f32).cos()
            } else {
                1.0
            };
            let phase = std::f32::consts::TAU * TONE_HZ * i as f32 / sample_rate as f32;
            let value = PEAK * envelope * phase.sin();
            pcm.push((value * i16::MAX as f32) as i16);
        }
    }
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATES: [u32; 4] = [16_000, 22_050, 24_000, 48_000];

    fn samples_for(ms: u32, rate: u32) -> usize {
        (rate as u64 * ms as u64 / 1000) as usize
    }

    #[test]
    fn is_the_expected_length_at_every_backend_rate() {
        for rate in RATES {
            let pcm = alarm_pcm(rate);
            assert_eq!(
                pcm.len(),
                samples_for(DURATION_MS, rate),
                "wrong length at {rate} Hz"
            );
        }
    }

    #[test]
    fn never_clips() {
        let ceiling = (PEAK * i16::MAX as f32).ceil() as i32;
        for rate in RATES {
            let peak = alarm_pcm(rate).iter().map(|s| (*s as i32).abs()).max();
            let peak = peak.expect("chime is not empty");
            assert!(peak <= ceiling, "peaked at {peak} > {ceiling} at {rate} Hz");
            assert!(
                peak > ceiling / 2,
                "chime is implausibly quiet at {rate} Hz"
            );
        }
    }

    #[test]
    fn begins_and_ends_in_silence_so_it_cannot_click() {
        for rate in RATES {
            let pcm = alarm_pcm(rate);
            assert_eq!(pcm[0], 0, "clicks on at {rate} Hz");
            assert_eq!(pcm[pcm.len() - 1], 0, "clicks off at {rate} Hz");
        }
    }

    #[test]
    fn is_two_beeps_separated_by_real_silence() {
        let rate = 48_000;
        let pcm = alarm_pcm(rate);
        let beep = samples_for(BEEP_MS, rate);
        let gap = samples_for(GAP_MS, rate);

        let energy = |slice: &[i16]| slice.iter().map(|s| (*s as i64).abs()).sum::<i64>();
        let first = energy(&pcm[..beep]);
        let middle = energy(&pcm[beep..beep + gap]);
        let second = energy(&pcm[beep + gap..]);

        assert!(first > 0, "first beep is silent");
        assert!(second > 0, "second beep is silent");
        assert_eq!(middle, 0, "the gap between beeps is not silent");
    }

    #[test]
    fn a_zero_sample_rate_yields_no_chime_rather_than_a_panic() {
        assert!(alarm_pcm(0).is_empty());
    }
}
