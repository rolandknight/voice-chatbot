#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

// Audio pipeline:
//   ES7210 mics → esp_codec_dev capture
//     → esp-sr AFE (BSS + AEC + NS + AGC)
//       → ring buffer (pre-roll)
//       → fan-out to:
//           - wake-word detector (consumes mono 16 kHz int16 frames)
//           - WebRTC uplink (only when a session is open)
//   ES8311 speaker ← decoded TTS frames from WebRTC downlink

#ifdef __cplusplus
extern "C" {
#endif

// Bring up codec, AFE, and the capture task. Idempotent.
esp_err_t audio_pipeline_init(void);

// Block-pop the next preprocessed mono 16 kHz int16 frame. Returns the number
// of samples written to `out`. Used by the wake-word task.
//
// `out` must have room for AUDIO_FRAME_SAMPLES int16_t values.
size_t audio_pipeline_read_frame(int16_t *out, size_t max_samples);

// Snapshot `preroll_ms` of the most recent audio out of the ring buffer into
// `out`. Returns samples written. Used right after wake to backfill the
// WebRTC uplink so the first phoneme of the user's query isn't clipped.
size_t audio_pipeline_snapshot_preroll(int16_t *out, size_t max_samples,
                                       uint32_t preroll_ms);

// Open / close the uplink fan-out. While open, every captured frame is also
// pushed to the WebRTC encoder task.
void audio_pipeline_uplink_open(void);
void audio_pipeline_uplink_close(void);

// Play decoded PCM out of the speaker. Sample rate is `sample_rate` (the codec
// driver resamples to whatever ES8311 is configured for). Non-blocking for
// frames up to ~40 ms.
void audio_pipeline_play(const int16_t *pcm, size_t samples, uint32_t sample_rate);

#ifdef __cplusplus
}
#endif
