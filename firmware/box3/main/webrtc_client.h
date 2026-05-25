#pragma once

#include <stdint.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Callback for incoming TTS audio. Same signature as audio_pipeline_play
// so the caller can pass the latter directly.
typedef void (*webrtc_remote_audio_cb_t)(const int16_t *pcm, size_t samples,
                                         uint32_t sample_rate);

typedef struct {
    // "ollama" or "claude" — sent over the DataChannel as soon as it opens,
    // so the backend's BackendRouter knows which LLM to route to.
    const char *backend;
    // How much pre-wake audio to send as soon as the SRTP channel comes up.
    uint32_t    preroll_ms;

    // Optional lifecycle hooks. All callbacks run on the WebRTC task.
    void (*on_connected)(void);
    webrtc_remote_audio_cb_t on_remote_audio;
    void (*on_closed)(void);
} webrtc_session_t;

// One-time setup: parse ICE config, register codecs (Opus). Idempotent.
esp_err_t webrtc_client_init(void);

// Open a new session. Returns immediately; lifecycle is reported via the
// callbacks in `sess`. If a session is already open, the new request is
// ignored (single active session at a time).
esp_err_t webrtc_client_open(const webrtc_session_t *sess);

// Force-close the current session if one is open.
void webrtc_client_close(void);

#ifdef __cplusplus
}
#endif
