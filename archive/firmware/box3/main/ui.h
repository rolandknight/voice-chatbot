#pragma once

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UI_STATE_IDLE,        // waiting for wake
    UI_STATE_LISTENING,   // streaming mic to backend
    UI_STATE_THINKING,    // server ack, awaiting TTS
    UI_STATE_SPEAKING,    // playing TTS
    UI_STATE_ERROR,       // wifi / signaling / rtc failure
} ui_state_t;

esp_err_t ui_init(void);

// Set the displayed state. `detail` is an optional secondary line — phrase
// for LISTENING, persona for SPEAKING, message for ERROR. Pass NULL for none.
// Safe to call from any task; the LVGL update happens on the LVGL task.
void ui_set_state(ui_state_t state, const char *detail);

#ifdef __cplusplus
}
#endif
