// Audio pipeline. Wraps esp-bsp's codec init, esp-sr's AFE, and a ring
// buffer for pre-roll. See audio_pipeline.h for the API contract and
// docs/web-rtc.md for the architectural picture.

#include "audio_pipeline.h"
#include "config.h"

#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/ringbuf.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

// esp-bsp gives us the codec dev handles for the Box-3.
#include "bsp/esp-bsp.h"
#include "esp_codec_dev.h"

// esp-sr AFE. The v1 API is the one most stable on the S3-Box-3 reference
// boards as of esp-sr 1.4.x; if the SDK bumps to v2, the handle/init names
// change but the data flow is the same.
#include "esp_afe_sr_iface.h"
#include "esp_afe_sr_models.h"

static const char *TAG = "audio";

#define AFE_FEED_TASK_STACK   (4 * 1024)
#define AFE_FETCH_TASK_STACK  (4 * 1024)
#define UPLINK_QUEUE_LEN      8

// AFE handles + state.
static const esp_afe_sr_iface_t *s_afe;
static esp_afe_sr_data_t        *s_afe_data;

// Codec handles (capture + playback).
static esp_codec_dev_handle_t   s_mic;
static esp_codec_dev_handle_t   s_spk;

// Ring buffer of preprocessed mono int16 16 kHz samples for pre-roll.
static RingbufHandle_t          s_ring;
// Queue of int16 frames available to consumers (wake-word, uplink).
static QueueHandle_t            s_frames;
// True when WebRTC session is active and frames should also go uplink.
static volatile bool            s_uplink_open;

// ---------------------------------------------------------------------------
// Tasks: feed raw mic samples into AFE; fetch processed samples out of AFE.
// ---------------------------------------------------------------------------

static void afe_feed_task(void *arg) {
    size_t chunk_samples = s_afe->get_feed_chunksize(s_afe_data);
    size_t channels      = s_afe->get_channel_num(s_afe_data);
    size_t buf_bytes     = chunk_samples * channels * sizeof(int16_t);
    int16_t *buf = heap_caps_malloc(buf_bytes, MALLOC_CAP_SPIRAM);
    if (!buf) { ESP_LOGE(TAG, "feed alloc failed"); vTaskDelete(NULL); }

    while (true) {
        // Pull raw multi-channel samples from the mic codec. The ES7210 is
        // configured by esp-bsp for 2 mic channels + 1 playback-loopback
        // reference channel (AEC reference).
        esp_codec_dev_read(s_mic, buf, buf_bytes);
        s_afe->feed(s_afe_data, buf);
    }
}

static void afe_fetch_task(void *arg) {
    while (true) {
        afe_fetch_result_t *res = s_afe->fetch(s_afe_data);
        if (!res || res->ret_value == ESP_FAIL) {
            vTaskDelay(pdMS_TO_TICKS(1));
            continue;
        }
        // res->data is mono 16 kHz int16. Push into the ring (for pre-roll)
        // and queue (for live consumers).
        xRingbufferSend(s_ring, res->data, res->data_size, 0);

        int16_t *frame = heap_caps_malloc(res->data_size, MALLOC_CAP_SPIRAM);
        if (frame) {
            memcpy(frame, res->data, res->data_size);
            if (xQueueSend(s_frames, &frame, 0) != pdTRUE) {
                heap_caps_free(frame);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

esp_err_t audio_pipeline_init(void) {
    static bool inited = false;
    if (inited) return ESP_OK;

    // esp-bsp brings up the I2S bus + both codecs with one call.
    s_mic = bsp_audio_codec_microphone_init();
    s_spk = bsp_audio_codec_speaker_init();
    if (!s_mic || !s_spk) {
        ESP_LOGE(TAG, "codec init failed");
        return ESP_FAIL;
    }

    esp_codec_dev_sample_info_t mic_fmt = {
        .sample_rate = AUDIO_SAMPLE_RATE,
        .channel = 2,
        .bits_per_sample = 16,
    };
    esp_codec_dev_open(s_mic, &mic_fmt);

    esp_codec_dev_sample_info_t spk_fmt = {
        .sample_rate = AUDIO_PLAYBACK_RATE,
        .channel = 1,
        .bits_per_sample = 16,
    };
    esp_codec_dev_open(s_spk, &spk_fmt);

    // AFE init. AFE_MODE_LOW_COST keeps RAM use modest while still giving us
    // AEC; SR_HIGH_PERF eats more PSRAM if we ever care about VAD too.
    afe_config_t cfg = AFE_CONFIG_DEFAULT();
    cfg.wakenet_init    = false;            // we do wake-word with microWakeWord
    cfg.voice_communication_init = true;    // enables AEC for VOIP-style use
    cfg.aec_init        = true;
    cfg.se_init         = true;             // single-channel speech enhancement
    cfg.vad_init        = true;             // server still VADs, but device too helps gating
    cfg.afe_mode        = SR_MODE_LOW_COST;
    cfg.pcm_config.total_ch_num  = 3;       // 2 mic + 1 reference
    cfg.pcm_config.mic_num       = 2;
    cfg.pcm_config.ref_num       = 1;
    cfg.pcm_config.sample_rate   = AUDIO_SAMPLE_RATE;

    s_afe = &ESP_AFE_SR_HANDLE;
    s_afe_data = s_afe->create_from_config(&cfg);
    if (!s_afe_data) {
        ESP_LOGE(TAG, "AFE create failed");
        return ESP_FAIL;
    }

    // ~1 s of mono int16 audio = 16000 * 2 = 32 KB. Stays in PSRAM.
    s_ring = xRingbufferCreateWithCaps(AUDIO_RING_SECONDS * AUDIO_SAMPLE_RATE * sizeof(int16_t),
                                       RINGBUF_TYPE_BYTEBUF, MALLOC_CAP_SPIRAM);
    s_frames = xQueueCreate(UPLINK_QUEUE_LEN, sizeof(int16_t *));

    xTaskCreatePinnedToCore(afe_feed_task,  "afe_feed",  AFE_FEED_TASK_STACK,
                            NULL, 5, NULL, 1);
    xTaskCreatePinnedToCore(afe_fetch_task, "afe_fetch", AFE_FETCH_TASK_STACK,
                            NULL, 5, NULL, 0);

    inited = true;
    ESP_LOGI(TAG, "audio pipeline up — %d Hz, AEC=on", AUDIO_SAMPLE_RATE);
    return ESP_OK;
}

size_t audio_pipeline_read_frame(int16_t *out, size_t max_samples) {
    int16_t *frame = NULL;
    if (xQueueReceive(s_frames, &frame, portMAX_DELAY) != pdTRUE || !frame) {
        return 0;
    }
    size_t chunk_samples = s_afe->get_fetch_chunksize(s_afe_data) / sizeof(int16_t);
    size_t n = chunk_samples < max_samples ? chunk_samples : max_samples;
    memcpy(out, frame, n * sizeof(int16_t));
    heap_caps_free(frame);
    return n;
}

size_t audio_pipeline_snapshot_preroll(int16_t *out, size_t max_samples,
                                       uint32_t preroll_ms) {
    size_t want_bytes = (AUDIO_SAMPLE_RATE * preroll_ms / 1000) * sizeof(int16_t);
    size_t out_bytes  = max_samples * sizeof(int16_t);
    if (want_bytes > out_bytes) want_bytes = out_bytes;

    size_t got = 0;
    while (got < want_bytes) {
        size_t item_size = 0;
        void *item = xRingbufferReceiveUpTo(s_ring, &item_size, 0,
                                            want_bytes - got);
        if (!item) break;
        memcpy((uint8_t *)out + got, item, item_size);
        vRingbufferReturnItem(s_ring, item);
        got += item_size;
    }
    return got / sizeof(int16_t);
}

void audio_pipeline_uplink_open(void)  { s_uplink_open = true;  }
void audio_pipeline_uplink_close(void) { s_uplink_open = false; }

void audio_pipeline_play(const int16_t *pcm, size_t samples, uint32_t sample_rate) {
    if (sample_rate != AUDIO_PLAYBACK_RATE) {
        // The Box-3 ES8311 driver in esp-bsp supports per-write rate changes
        // by reopening with a new sample_info. Cheap on the hardware mixer.
        esp_codec_dev_sample_info_t fmt = {
            .sample_rate = sample_rate,
            .channel = 1,
            .bits_per_sample = 16,
        };
        esp_codec_dev_close(s_spk);
        esp_codec_dev_open(s_spk, &fmt);
    }
    esp_codec_dev_write(s_spk, (void *)pcm, samples * sizeof(int16_t));
}
