// Wake-word detector. Loads two microWakeWord models embedded into the
// firmware binary (hey_babel.tflite, hey_marvin.tflite) and runs them in
// parallel on every 20 ms frame coming out of the audio pipeline.
//
// On the first frame whose probability crosses WAKE_THRESHOLD, the user
// callback fires. A WAKE_REFRACTORY_MS suppression window prevents
// re-triggers on the same utterance.

#include "wakeword.h"
#include "audio_pipeline.h"
#include "config.h"

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"

// TFLite Micro C++ headers — included in C via the tflite-micro C wrapper
// that esp-tflite-micro publishes. If the wrapper changes, swap to the
// MicroInterpreter C++ API in a .cpp source file.
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "wake";

// Models are embedded via EMBED_FILES in CMakeLists.txt.
extern const uint8_t hey_babel_tflite_start[]  asm("_binary_hey_babel_tflite_start");
extern const uint8_t hey_babel_tflite_end[]    asm("_binary_hey_babel_tflite_end");
extern const uint8_t hey_marvin_tflite_start[] asm("_binary_hey_marvin_tflite_start");
extern const uint8_t hey_marvin_tflite_end[]   asm("_binary_hey_marvin_tflite_end");

// Per-model interpreter state. microWakeWord's streaming DS-CNN holds
// internal state across frames via TFLite Micro persistent tensors, so each
// model gets its own arena.
typedef struct {
    const char *phrase;
    const char *backend;
    const uint8_t *model_start;
    const uint8_t *model_end;
    uint8_t *arena;
    size_t   arena_size;
    // Opaque pointer to a tflite::MicroInterpreter — set in wakeword_init.
    void    *interp;
} mww_model_t;

#define ARENA_SIZE  (32 * 1024)

static mww_model_t s_models[] = {
    {
        .phrase = "hey babel",
        .backend = "ollama",
        .model_start = hey_babel_tflite_start,
        .model_end   = hey_babel_tflite_end,
    },
    {
        .phrase = "hey marvin",
        .backend = "claude",
        .model_start = hey_marvin_tflite_start,
        .model_end   = hey_marvin_tflite_end,
    },
};
#define N_MODELS (sizeof(s_models) / sizeof(s_models[0]))

static wake_cb_t s_cb;
static int64_t   s_last_fire_us;

// Forward decls for the TFLite Micro glue implemented in wakeword_tflm.cc.
// Keeping the C++ surface in a separate translation unit lets this file
// stay C-clean.
extern void *mww_create_interpreter(const uint8_t *model, size_t len,
                                    uint8_t *arena, size_t arena_size);
extern float mww_invoke(void *interp, const int16_t *frame, size_t samples);

static void wake_task(void *arg) {
    int16_t *frame = heap_caps_malloc(AUDIO_FRAME_SAMPLES * sizeof(int16_t),
                                      MALLOC_CAP_SPIRAM);
    if (!frame) { ESP_LOGE(TAG, "frame alloc failed"); vTaskDelete(NULL); }

    while (true) {
        size_t n = audio_pipeline_read_frame(frame, AUDIO_FRAME_SAMPLES);
        if (n == 0) continue;

        int64_t now = esp_timer_get_time();
        if (now - s_last_fire_us < (int64_t)WAKE_REFRACTORY_MS * 1000) continue;

        // Run both models on the same frame. Whichever crosses the threshold
        // first wins; ties broken by higher confidence.
        float best_conf = 0.0f;
        int   best_idx  = -1;
        for (int i = 0; i < (int)N_MODELS; i++) {
            float c = mww_invoke(s_models[i].interp, frame, n);
            if (c > WAKE_THRESHOLD && c > best_conf) {
                best_conf = c;
                best_idx  = i;
            }
        }
        if (best_idx >= 0) {
            s_last_fire_us = now;
            wake_event_t evt = {
                .phrase     = s_models[best_idx].phrase,
                .backend    = s_models[best_idx].backend,
                .confidence = best_conf,
            };
            if (s_cb) s_cb(&evt);
        }
    }
}

esp_err_t wakeword_init(wake_cb_t cb) {
    s_cb = cb;
    for (size_t i = 0; i < N_MODELS; i++) {
        s_models[i].arena_size = ARENA_SIZE;
        s_models[i].arena = heap_caps_malloc(ARENA_SIZE, MALLOC_CAP_SPIRAM);
        if (!s_models[i].arena) {
            ESP_LOGE(TAG, "arena alloc failed for %s", s_models[i].phrase);
            return ESP_ERR_NO_MEM;
        }
        size_t model_len = s_models[i].model_end - s_models[i].model_start;
        s_models[i].interp = mww_create_interpreter(s_models[i].model_start,
                                                    model_len,
                                                    s_models[i].arena,
                                                    s_models[i].arena_size);
        if (!s_models[i].interp) {
            ESP_LOGE(TAG, "interpreter create failed for %s", s_models[i].phrase);
            return ESP_FAIL;
        }
        ESP_LOGI(TAG, "loaded %s (%u bytes)", s_models[i].phrase,
                 (unsigned)model_len);
    }
    xTaskCreatePinnedToCore(wake_task, "wakeword", 8 * 1024,
                            NULL, 6, NULL, 1);
    return ESP_OK;
}
