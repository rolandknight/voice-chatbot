//! The chime a timer sounds before it speaks.
//!
//! Synthesized rather than shipped as an asset, because the call's output stage
//! will not accept audio at any other rate. It builds one fixed
//! `Resampler::new(tts_rate, carrier_rate)` (`cascaded.rs:1323`) and pushes
//! every `Frame::OutputAudio` through it; `Resampler::process` compares the
//! chunk's rate against the one it was built for and returns an error on a
//! mismatch (`codec.rs:185`). The sink funnels that into `record_error`, which
//! drops the chunk *and* latches it as the call's error — so a wrong-rate chime
//! is not merely off-pitch, it is silently discarded and poisons the call
//! record. Generating the tone at whatever `tts_rate` the configured backend
//! reports avoids that, and costs no asset, no HTTP route and no ffmpeg.

/// Pitch of the chime. A5 — high enough to carry over a room, low enough not to
/// be shrill on a small speaker.
const TONE_HZ: f32 = 880.0;
/// How long each of the two beeps sounds.
const BEEP_MS: u32 = 160;
/// Silence between the two beeps.
const GAP_MS: u32 = 90;
/// Trailing silence.
///
/// Not padding for its own sake: `BotSpeakingNotifier::note_audio` sizes its
/// playout estimate from the samples actually sent, and emits
/// `BotStoppedSpeaking` when that estimate runs dry (`cascaded.rs:355-366`).
/// The words that follow the chime are still being synthesized at that point —
/// Kokoro and Chatterbox both deliver a reply as one burst — so without a tail
/// the client unducks and immediately re-ducks between the chime and the
/// speech: an audible gain flap on live radio, a decoder stop/start on a
/// recorded show, and a release of any `after_speech` clip that was waiting.
/// This spans a typical synthesis gap. Tune by ear against a real call.
const TAIL_MS: u32 = 500;
/// Raised-cosine attack and release on each beep. Without it the tone begins
/// and ends on a discontinuity, which is audible as a click.
const EDGE_MS: u32 = 8;
/// Peak amplitude as a fraction of full scale. A sine reads louder than speech
/// at equal peak, so this sits deliberately below the level TTS comes out at.
const PEAK: f32 = 0.22;

/// Total duration of the chime, trailing silence included.
pub const DURATION_MS: u32 = BEEP_MS * 2 + GAP_MS + TAIL_MS;

/// The tail is what holds the media duck open until the synthesized words
/// arrive, so it has to outlast the beeps it follows. Checked at compile time —
/// a later tweak to any of these three cannot quietly break it.
const _: () = assert!(TAIL_MS > BEEP_MS * 2 + GAP_MS);

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
    let edge = samples_for(EDGE_MS);

    let mut pcm = Vec::with_capacity(samples_for(DURATION_MS));
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
            pcm.push((PEAK * envelope * phase.sin() * i16::MAX as f32) as i16);
        }
    }
    pcm.extend(std::iter::repeat_n(0i16, samples_for(TAIL_MS)));
    pcm
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATES: [u32; 5] = [16_000, 22_050, 24_000, 44_100, 48_000];

    fn samples_for(ms: u32, rate: u32) -> usize {
        (rate as u64 * ms as u64 / 1000) as usize
    }

    /// Tone frequency from zero-crossing density. The whole reason this chime is
    /// synthesized rather than loaded from a file is that it must come out at
    /// the caller's rate; nothing else here would notice a hardcoded divisor.
    fn measured_hz(pcm: &[i16], rate: u32) -> f32 {
        let crossings = pcm.windows(2).filter(|w| (w[0] < 0) != (w[1] < 0)).count();
        crossings as f32 * rate as f32 / (2.0 * pcm.len() as f32)
    }

    #[test]
    fn sounds_at_the_right_pitch_whatever_the_backend_rate() {
        for rate in RATES {
            let pcm = alarm_pcm(rate);
            // The steady middle of the first beep, clear of both envelopes.
            let beep = samples_for(BEEP_MS, rate);
            let edge = samples_for(EDGE_MS, rate);
            let steady = &pcm[edge * 2..beep - edge * 2];
            let hz = measured_hz(steady, rate);
            assert!(
                (hz - TONE_HZ).abs() / TONE_HZ < 0.03,
                "measured {hz:.0} Hz at {rate} Hz, expected {TONE_HZ}"
            );
        }
    }

    #[test]
    fn is_the_expected_length_at_every_backend_rate() {
        for rate in RATES {
            assert_eq!(
                alarm_pcm(rate).len(),
                samples_for(DURATION_MS, rate),
                "wrong length at {rate} Hz"
            );
        }
    }

    #[test]
    fn never_clips_and_stays_below_speech_level() {
        // Pinned to an absolute ceiling, not to PEAK: deriving the bound from
        // the constant it is meant to police would pass at any amplitude.
        const CEILING: i32 = 8_000; // ~0.24 FS
        for rate in RATES {
            let peak = alarm_pcm(rate)
                .iter()
                .map(|s| (*s as i32).abs())
                .max()
                .expect("chime is not empty");
            assert!(peak <= CEILING, "peaked at {peak} > {CEILING} at {rate} Hz");
            assert!(peak > CEILING / 2, "implausibly quiet at {rate} Hz");
        }
    }

    #[test]
    fn begins_and_ends_in_silence_so_it_cannot_click() {
        for rate in RATES {
            let pcm = alarm_pcm(rate);
            assert_eq!(pcm[0], 0, "clicks on at {rate} Hz");
            assert_eq!(pcm[pcm.len() - 1], 0, "clicks off at {rate} Hz");
            // The release ramp must land on zero too, or the gap starts with a step.
            let beep = samples_for(BEEP_MS, rate);
            assert_eq!(pcm[beep - 1], 0, "clicks between beeps at {rate} Hz");
        }
    }

    #[test]
    fn is_two_beeps_a_silent_gap_and_a_silent_tail() {
        let rate = 48_000;
        let pcm = alarm_pcm(rate);
        let beep = samples_for(BEEP_MS, rate);
        let gap = samples_for(GAP_MS, rate);
        let tail = samples_for(TAIL_MS, rate);

        let energy = |slice: &[i16]| slice.iter().map(|s| (*s as i64).abs()).sum::<i64>();
        assert!(energy(&pcm[..beep]) > 0, "first beep is silent");
        assert_eq!(energy(&pcm[beep..beep + gap]), 0, "the gap is not silent");
        let second = beep + gap;
        assert!(
            energy(&pcm[second..second + beep]) > 0,
            "second beep is silent"
        );
        assert_eq!(
            energy(&pcm[second + beep..]),
            0,
            "the tail that holds the duck open is not silent"
        );
        assert_eq!(pcm.len(), second + beep + tail);
    }

    #[test]
    fn a_zero_sample_rate_yields_no_chime_rather_than_a_panic() {
        assert!(alarm_pcm(0).is_empty());
    }
}
