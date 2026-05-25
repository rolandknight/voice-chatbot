// ESP32-S3-BOX-3 voice client entry point.
//
// Boots WiFi, the LCD UI, the audio pipeline, and the wake-word detectors.
// When a wake fires, opens a WebRTC peer connection to the voice-chatbot
// backend and streams Opus until the server hangs up (idle timeout).

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"
#include "nvs_flash.h"

#include "config.h"
#include "ui.h"
#include "audio_pipeline.h"
#include "wakeword.h"
#include "webrtc_client.h"

static const char *TAG = "box3";

static EventGroupHandle_t s_wifi_evt;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static void on_wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data) {
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        ESP_LOGW(TAG, "WiFi disconnected, retrying");
        ui_set_state(UI_STATE_ERROR, "WiFi lost");
        esp_wifi_connect();
        xEventGroupClearBits(s_wifi_evt, WIFI_CONNECTED_BIT);
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ESP_LOGI(TAG, "WiFi up");
        xEventGroupSetBits(s_wifi_evt, WIFI_CONNECTED_BIT);
    }
}

static esp_err_t wifi_start(void) {
    s_wifi_evt = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                               &on_wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                               &on_wifi_event, NULL));

    wifi_config_t wc = { 0 };
    strncpy((char *)wc.sta.ssid, CONFIG_BOX3_WIFI_SSID, sizeof(wc.sta.ssid));
    strncpy((char *)wc.sta.password, CONFIG_BOX3_WIFI_PASSWORD, sizeof(wc.sta.password));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Block until we have an IP (or give up after 20 s and let the loop retry).
    EventBits_t bits = xEventGroupWaitBits(s_wifi_evt,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE, pdFALSE,
                                           pdMS_TO_TICKS(20000));
    return (bits & WIFI_CONNECTED_BIT) ? ESP_OK : ESP_ERR_TIMEOUT;
}

// Called from the wake-word detector when a phrase crosses the threshold.
// Runs in the wake-word task; keep it non-blocking — the actual WebRTC
// handshake is dispatched onto its own task.
static void on_wake(const wake_event_t *evt) {
    ESP_LOGI(TAG, "WAKE: %s (conf=%.2f)", evt->phrase, evt->confidence);
    ui_set_state(UI_STATE_LISTENING, evt->phrase);

    webrtc_session_t sess = {
        .backend = evt->backend,             // "ollama" or "claude"
        .preroll_ms = AUDIO_PREROLL_MS,
        .on_connected = NULL,
        .on_remote_audio = audio_pipeline_play,
        .on_closed = NULL,
    };
    webrtc_client_open(&sess);
}

void app_main(void) {
    // NVS is required for wifi calibration data.
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);

    ui_init();
    ui_set_state(UI_STATE_IDLE, NULL);

    if (strlen(CONFIG_BOX3_WIFI_SSID) > 0) {
        if (wifi_start() != ESP_OK) {
            ui_set_state(UI_STATE_ERROR, "WiFi timeout");
            ESP_LOGE(TAG, "WiFi did not come up; halting");
            return;
        }
    } else {
        ESP_LOGW(TAG, "BOX3_WIFI_SSID empty; running offline (UI/audio only)");
    }

    ESP_ERROR_CHECK(audio_pipeline_init());
    ESP_ERROR_CHECK(wakeword_init(on_wake));
    ESP_ERROR_CHECK(webrtc_client_init());

    ESP_LOGI(TAG, "ready — say 'hey babel' or 'hey marvin'");

    // Idle here. The audio pipeline + wake-word + WebRTC client all run on
    // their own FreeRTOS tasks; main can just sleep.
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
