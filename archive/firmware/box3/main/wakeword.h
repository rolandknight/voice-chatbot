#pragma once

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *phrase;       // "hey babel" or "hey marvin"
    const char *backend;      // "ollama" or "claude" — for the DataChannel
    float       confidence;   // 0..1
} wake_event_t;

typedef void (*wake_cb_t)(const wake_event_t *evt);

// Spin up the wake-word task. The task pulls frames from audio_pipeline and
// runs both microWakeWord models in parallel; on a high-confidence detection
// (above WAKE_THRESHOLD in config.h) `cb` is invoked from the wake task.
esp_err_t wakeword_init(wake_cb_t cb);

#ifdef __cplusplus
}
#endif
