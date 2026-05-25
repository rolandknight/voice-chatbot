// LCD status UI. Five full-screen states, color-coded so you can read the
// device state across a room. Implemented in LVGL via esp-bsp's display
// bring-up; the only widgets are two labels (state + detail) on a tinted
// background.

#include "ui.h"

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "bsp/esp-bsp.h"
#include "lvgl.h"

static const char *TAG = "ui";

static lv_obj_t  *s_bg;
static lv_obj_t  *s_label_state;
static lv_obj_t  *s_label_detail;

typedef struct {
    const char *text;
    uint32_t    color;
} state_style_t;

static const state_style_t k_styles[] = {
    [UI_STATE_IDLE]      = { "say 'hey babel'", 0x202840 },
    [UI_STATE_LISTENING] = { "listening...",    0x205040 },
    [UI_STATE_THINKING]  = { "...",             0x504020 },
    [UI_STATE_SPEAKING]  = { "speaking",        0x205070 },
    [UI_STATE_ERROR]     = { "error",           0x802020 },
};

typedef struct {
    ui_state_t state;
    char       detail[64];
} ui_msg_t;

static QueueHandle_t s_q;

static void ui_apply(ui_state_t state, const char *detail) {
    const state_style_t *s = &k_styles[state];
    lv_obj_set_style_bg_color(s_bg, lv_color_hex(s->color), 0);
    lv_label_set_text(s_label_state, s->text);
    lv_label_set_text(s_label_detail, detail ? detail : "");
}

static void ui_task(void *arg) {
    ui_msg_t msg;
    while (xQueueReceive(s_q, &msg, portMAX_DELAY) == pdTRUE) {
        bsp_display_lock(0);
        ui_apply(msg.state, msg.detail);
        bsp_display_unlock();
    }
}

esp_err_t ui_init(void) {
    bsp_display_start();
    bsp_display_backlight_on();

    bsp_display_lock(0);
    s_bg = lv_scr_act();
    lv_obj_set_style_bg_color(s_bg, lv_color_hex(0x000000), 0);

    s_label_state = lv_label_create(s_bg);
    lv_obj_set_style_text_color(s_label_state, lv_color_hex(0xFFFFFF), 0);
    lv_obj_set_style_text_font(s_label_state, &lv_font_montserrat_28, 0);
    lv_obj_align(s_label_state, LV_ALIGN_CENTER, 0, -16);

    s_label_detail = lv_label_create(s_bg);
    lv_obj_set_style_text_color(s_label_detail, lv_color_hex(0xCCCCCC), 0);
    lv_obj_set_style_text_font(s_label_detail, &lv_font_montserrat_16, 0);
    lv_obj_align(s_label_detail, LV_ALIGN_CENTER, 0, 24);
    bsp_display_unlock();

    s_q = xQueueCreate(4, sizeof(ui_msg_t));
    xTaskCreate(ui_task, "ui", 4 * 1024, NULL, 4, NULL);

    ESP_LOGI(TAG, "ui ready");
    return ESP_OK;
}

void ui_set_state(ui_state_t state, const char *detail) {
    if (state >= sizeof(k_styles) / sizeof(k_styles[0])) return;
    ui_msg_t msg = { .state = state };
    if (detail) {
        strncpy(msg.detail, detail, sizeof(msg.detail) - 1);
    }
    xQueueSend(s_q, &msg, 0);
}
