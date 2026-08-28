#pragma once

// ---------------------------------------------------------------------------
// Build-time configuration for the ESP32-S3-BOX-3 voice client.
//
// Override via menuconfig or environment-injected defines if you want secrets
// out of source control. For local development, editing this file is fine.
// ---------------------------------------------------------------------------

// WiFi credentials. Empty SSID skips WiFi init (useful for offline UI work).
#ifndef CONFIG_BOX3_WIFI_SSID
#define CONFIG_BOX3_WIFI_SSID     ""
#endif
#ifndef CONFIG_BOX3_WIFI_PASSWORD
#define CONFIG_BOX3_WIFI_PASSWORD ""
#endif

// voice-chatbot backend. Same LAN as the Box-3; the Mac's IP, port from
// .env (WEBRTC_PORT, default 8080). HTTPS is not required on a trusted LAN.
#define BACKEND_HOST              "192.168.1.10"
#define BACKEND_PORT              8080
#define BACKEND_OFFER_PATH        "/api/offer"

// ICE servers. STUN-only is enough on a flat LAN; add TURN if you ever route
// the device through a NAT.
#define ICE_STUN_URL              "stun:stun.l.google.com:19302"

// Audio pipeline.
#define AUDIO_SAMPLE_RATE         16000   // mics + AFE output
#define AUDIO_FRAME_MS            20      // Opus frame size for uplink
#define AUDIO_FRAME_SAMPLES       (AUDIO_SAMPLE_RATE * AUDIO_FRAME_MS / 1000)
#define AUDIO_RING_SECONDS        1       // pre-roll buffer length
#define AUDIO_PREROLL_MS          500     // how much pre-wake audio to send

// Speaker output sample rate. Kokoro emits 24 kHz; Chatterbox 16 kHz.
// The codec driver resamples to ES8311's actual rate.
#define AUDIO_PLAYBACK_RATE       24000

// Wake-word detector. Probabilities above this threshold trigger a wake.
#define WAKE_THRESHOLD            0.7f
#define WAKE_REFRACTORY_MS        2000    // suppress repeat fires for this long

// Idle timeout. If no audio activity for this long after a session, close.
#define IDLE_TIMEOUT_SEC          20
