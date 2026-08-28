// WebRTC client. Negotiates an SDP offer/answer with the voice-chatbot
// backend over HTTP, then streams Opus 16 kHz audio uplink and decodes
// incoming Opus to PCM for the speaker.
//
// Built on Espressif's esp-webrtc-solution. The peer connection lifecycle is
// modeled as a single FreeRTOS task that owns the session state — callers
// poke it via a queue.

#include "webrtc_client.h"
#include "audio_pipeline.h"
#include "config.h"
#include "ui.h"

#include <string.h>
#include <stdlib.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_log.h"
#include "esp_http_client.h"
#include "esp_heap_caps.h"
#include "cJSON.h"

// esp-webrtc-solution public API. Symbol names follow the v1.x release.
#include "esp_peer.h"
#include "esp_peer_default.h"

static const char *TAG = "webrtc";

typedef enum {
    CMD_OPEN,
    CMD_CLOSE,
    CMD_REMOTE_SDP,
} cmd_kind_t;

typedef struct {
    cmd_kind_t kind;
    webrtc_session_t sess;
    char *sdp_answer;  // owned, freed by the task
} cmd_t;

static QueueHandle_t s_cmd_q;
static esp_peer_handle s_peer;
static webrtc_session_t s_active;
static bool s_session_open;

// ---------------------------------------------------------------------------
// Signaling — POST our SDP offer to the backend, get an answer back.
// ---------------------------------------------------------------------------

static char *http_post_offer(const char *sdp_offer) {
    char url[160];
    snprintf(url, sizeof(url), "http://%s:%d%s",
             BACKEND_HOST, BACKEND_PORT, BACKEND_OFFER_PATH);

    cJSON *body = cJSON_CreateObject();
    cJSON_AddStringToObject(body, "sdp", sdp_offer);
    cJSON_AddStringToObject(body, "type", "offer");
    char *body_str = cJSON_PrintUnformatted(body);
    cJSON_Delete(body);

    esp_http_client_config_t cfg = {
        .url = url,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 5000,
    };
    esp_http_client_handle_t h = esp_http_client_init(&cfg);
    esp_http_client_set_header(h, "Content-Type", "application/json");
    esp_http_client_set_post_field(h, body_str, strlen(body_str));

    char *resp = NULL;
    esp_err_t err = esp_http_client_perform(h);
    if (err == ESP_OK && esp_http_client_get_status_code(h) == 200) {
        int len = esp_http_client_get_content_length(h);
        resp = malloc(len + 1);
        esp_http_client_read(h, resp, len);
        resp[len] = '\0';
    } else {
        ESP_LOGE(TAG, "POST /api/offer failed: %s, http=%d",
                 esp_err_to_name(err), esp_http_client_get_status_code(h));
    }
    esp_http_client_cleanup(h);
    free(body_str);
    return resp;
}

// Parse `{"sdp":"...","type":"answer"}` and return a heap copy of the SDP
// string (caller frees). Returns NULL on malformed input.
static char *parse_sdp_answer(const char *resp) {
    cJSON *j = cJSON_Parse(resp);
    if (!j) return NULL;
    cJSON *sdp = cJSON_GetObjectItem(j, "sdp");
    char *out = (sdp && cJSON_IsString(sdp)) ? strdup(sdp->valuestring) : NULL;
    cJSON_Delete(j);
    return out;
}

// ---------------------------------------------------------------------------
// Peer connection callbacks (run on esp-webrtc's internal task).
// ---------------------------------------------------------------------------

static void on_pc_connected(void) {
    ESP_LOGI(TAG, "peer connected");
    audio_pipeline_uplink_open();

    // Send a backend-selection message over the DataChannel.
    char msg[64];
    snprintf(msg, sizeof(msg), "{\"backend\":\"%s\"}", s_active.backend);
    esp_peer_send_datachannel(s_peer, "control", msg, strlen(msg));

    // Backfill the pre-roll: ~500 ms of audio captured before the wake fired.
    size_t preroll_samples = AUDIO_SAMPLE_RATE * s_active.preroll_ms / 1000;
    int16_t *preroll = heap_caps_malloc(preroll_samples * sizeof(int16_t),
                                        MALLOC_CAP_SPIRAM);
    if (preroll) {
        size_t got = audio_pipeline_snapshot_preroll(preroll, preroll_samples,
                                                     s_active.preroll_ms);
        if (got > 0) {
            esp_peer_send_audio(s_peer, preroll, got);
        }
        heap_caps_free(preroll);
    }

    if (s_active.on_connected) s_active.on_connected();
}

static void on_pc_remote_audio(const int16_t *pcm, size_t samples,
                               uint32_t sample_rate) {
    ui_set_state(UI_STATE_SPEAKING, NULL);
    if (s_active.on_remote_audio) {
        s_active.on_remote_audio(pcm, samples, sample_rate);
    }
}

static void on_pc_closed(void) {
    ESP_LOGI(TAG, "peer closed");
    audio_pipeline_uplink_close();
    s_session_open = false;
    ui_set_state(UI_STATE_IDLE, NULL);
    if (s_active.on_closed) s_active.on_closed();
}

// ---------------------------------------------------------------------------
// Uplink pump — pulls frames from the audio pipeline and pushes to the peer
// while a session is open.
// ---------------------------------------------------------------------------

static void uplink_task(void *arg) {
    int16_t frame[AUDIO_FRAME_SAMPLES];
    while (true) {
        if (!s_session_open) { vTaskDelay(pdMS_TO_TICKS(20)); continue; }
        size_t n = audio_pipeline_read_frame(frame, AUDIO_FRAME_SAMPLES);
        if (n > 0) esp_peer_send_audio(s_peer, frame, n);
    }
}

// ---------------------------------------------------------------------------
// Session task — owns peer state.
// ---------------------------------------------------------------------------

static void session_task(void *arg) {
    cmd_t cmd;
    while (xQueueReceive(s_cmd_q, &cmd, portMAX_DELAY) == pdTRUE) {
        if (cmd.kind == CMD_OPEN) {
            if (s_session_open) { ESP_LOGW(TAG, "already in session"); continue; }

            esp_peer_default_cfg_t pc_cfg = ESP_PEER_DEFAULT_CFG();
            pc_cfg.ice_stun_url     = ICE_STUN_URL;
            pc_cfg.audio_codec      = ESP_PEER_AUDIO_OPUS;
            pc_cfg.audio_sample_rate = AUDIO_SAMPLE_RATE;
            pc_cfg.on_connected     = on_pc_connected;
            pc_cfg.on_remote_audio  = on_pc_remote_audio;
            pc_cfg.on_closed        = on_pc_closed;

            if (esp_peer_create(&pc_cfg, &s_peer) != ESP_OK) {
                ESP_LOGE(TAG, "peer create failed");
                ui_set_state(UI_STATE_ERROR, "rtc create");
                continue;
            }

            memcpy(&s_active, &cmd.sess, sizeof(s_active));

            char *offer_sdp = NULL;
            if (esp_peer_create_offer(s_peer, &offer_sdp) != ESP_OK || !offer_sdp) {
                ESP_LOGE(TAG, "create offer failed");
                esp_peer_destroy(s_peer); s_peer = NULL;
                ui_set_state(UI_STATE_ERROR, "rtc offer");
                continue;
            }

            char *resp = http_post_offer(offer_sdp);
            free(offer_sdp);
            if (!resp) {
                esp_peer_destroy(s_peer); s_peer = NULL;
                ui_set_state(UI_STATE_ERROR, "no answer");
                continue;
            }
            char *answer = parse_sdp_answer(resp);
            free(resp);
            if (!answer) {
                esp_peer_destroy(s_peer); s_peer = NULL;
                ui_set_state(UI_STATE_ERROR, "bad answer");
                continue;
            }

            esp_peer_set_remote_description(s_peer, answer);
            free(answer);
            s_session_open = true;
            ui_set_state(UI_STATE_THINKING, NULL);
        } else if (cmd.kind == CMD_CLOSE) {
            if (s_peer) { esp_peer_destroy(s_peer); s_peer = NULL; }
            s_session_open = false;
            ui_set_state(UI_STATE_IDLE, NULL);
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

esp_err_t webrtc_client_init(void) {
    static bool inited = false;
    if (inited) return ESP_OK;
    s_cmd_q = xQueueCreate(4, sizeof(cmd_t));
    xTaskCreatePinnedToCore(session_task, "rtc_sess", 8 * 1024, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(uplink_task,  "rtc_up",   4 * 1024, NULL, 5, NULL, 0);
    inited = true;
    return ESP_OK;
}

esp_err_t webrtc_client_open(const webrtc_session_t *sess) {
    cmd_t cmd = { .kind = CMD_OPEN, .sess = *sess };
    return (xQueueSend(s_cmd_q, &cmd, 0) == pdTRUE) ? ESP_OK : ESP_ERR_NO_MEM;
}

void webrtc_client_close(void) {
    cmd_t cmd = { .kind = CMD_CLOSE };
    xQueueSend(s_cmd_q, &cmd, 0);
}
