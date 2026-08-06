//! The `/webrtc/offer` handler: accept the browser/harness SDP offer, build the
//! cascaded pipeline with directly-constructed services, and run the call
//! detached (mirrors flowcat-server's `webrtc::offer`, minus the factory).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use flowcat_core::audio::{SileroVad, VadProcessor};
use flowcat_core::observer::{FrameObserver, RtviObserver, RtviSink};
use flowcat_core::pipeline::{build_cascaded_call_duplex, CascadedConfig};
use flowcat_server::events::RtfSink;
use flowcat_services::llm::OpenRouterLlm;
use flowcat_services::tts::KokoroTts;
use flowcat_transports::webrtc::WebRtcTransport;

use crate::PocState;

/// str0m carrier rate — matches flowcat-server's webrtc playground.
const CARRIER_RATE: u32 = 16_000;

#[derive(Deserialize)]
pub struct OfferRequest {
    pub sdp: String,
}

#[derive(Serialize)]
pub struct OfferResponse {
    pub sdp: String,
    pub pc_id: String,
}

pub async fn offer(
    State(state): State<Arc<PocState>>,
    Json(body): Json<OfferRequest>,
) -> Response {
    let run_id = state.next_run.fetch_add(1, Ordering::Relaxed);
    let pc_id = format!("pc-{run_id}");

    // Register the event stream BEFORE answering so pre-subscribe events buffer.
    let (events, guard) = state.registry.register(&pc_id);

    // Bind wildcard, advertise loopback: aiortc sends its ICE checks from
    // per-interface sockets (docker bridges, LAN IP), and a 127.0.0.1-bound
    // socket cannot reply to those sources (EINVAL os error 22, observed live).
    let socket = match tokio::net::UdpSocket::bind((std::net::Ipv4Addr::UNSPECIFIED, 0)).await {
        Ok(s) => s,
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("bind media socket: {e}"),
            )
                .into_response()
        }
    };
    let advertise = Some(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
    let (transport, answer) =
        match WebRtcTransport::accept_offer(&body.sdp, socket, advertise, CARRIER_RATE) {
            Ok(t) => t,
            Err(e) => {
                return (
                    axum::http::StatusCode::UNPROCESSABLE_ENTITY,
                    format!("accept offer: {e}"),
                )
                    .into_response()
            }
        };

    let cfg = &state.cfg;
    // Phase 1c: full duplex. Silero VAD (512-sample windows @16 kHz) ahead of
    // STT; its barge-in broadcast + interrupt flag drive the duplex builder.
    // min_volume: the default 0.6 (pipecat parity) gates out moderate-volume
    // speech — observed live: only the loudest tail of an utterance passed
    // (the same failure class as the production Jabra min_volume issue).
    // stop_secs 0.5 keeps comma-pauses from splitting utterances.
    let vad_params = flowcat_core::VadParams {
        confidence: 0.7,
        start_secs: 0.2,
        stop_secs: 0.5,
        min_volume: 0.2,
    };
    let vad = match SileroVad::with_params(&cfg.vad_model, CARRIER_RATE, vad_params) {
        Ok(v) => VadProcessor::new(v, 512),
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                format!("load silero vad: {e}"),
            )
                .into_response()
        }
    };
    // Whole-utterance STT paired with the SpeechGate (one VAD turn → one final
    // transcription), replacing WhisperLocalStt's fixed 4 s windows which fire
    // turns on partials and hallucinate on silence in duplex mode.
    let stt = crate::stt::BabelStt::new(cfg.whisper_model.clone());
    let llm = OpenRouterLlm::with_model(cfg.openrouter_key.clone(), cfg.llm_model.clone());
    let tts = KokoroTts::new("", cfg.kokoro_voice.clone()).with_base_url(cfg.kokoro_url.clone());
    let brain = crate::brain::BabelBrain::new(cfg.system_prompt.clone());
    let session = state.session.clone();

    let sink: Arc<dyn RtviSink> = Arc::new(RtfSink::new(events));
    let observers: Vec<Arc<dyn FrameObserver>> = vec![Arc::new(RtviObserver::new(sink))];

    tokio::spawn(async move {
        let _guard = guard; // deregister the event stream when the call ends
        let built = build_cascaded_call_duplex(
            transport,
            vad,
            stt,
            llm,
            tts,
            brain,
            session,
            run_id,
            "poc".to_string(),
            CascadedConfig::default(),
            observers,
        )
        .await;
        match built {
            Ok(task) => {
                if let Err(e) = task.run().await {
                    tracing::warn!(run_id, error = %e, "call ended with error");
                } else {
                    tracing::info!(run_id, "call ended");
                }
            }
            Err(e) => tracing::error!(run_id, error = %e, "failed to build call"),
        }
    });

    Json(json!(OfferResponse { sdp: answer, pc_id })).into_response()
}
