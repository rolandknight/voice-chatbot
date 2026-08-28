# Component definition for the application code.

idf_component_register(
    SRCS
        "main.c"
        "audio_pipeline.c"
        "wakeword.c"
        "wakeword_tflm.cc"
        "webrtc_client.c"
        "ui.c"
    INCLUDE_DIRS
        "."
    EMBED_FILES
        "models/hey_babel.tflite"
        "models/hey_marvin.tflite"
    REQUIRES
        esp_event
        esp_netif
        esp_wifi
        nvs_flash
        json
        esp_http_client
)
