//! The `/webrtc/offer` handler: accept the browser/harness SDP offer, build the
//! cascaded pipeline with directly-constructed services, and run the call
//! detached (mirrors flowcat-server's `webrtc::offer`, minus the factory).

use std::sync::atomic::Ordering;
use std::sync::Arc;

use axum::extract::{ConnectInfo, State};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use serde_json::json;

use flowcat_core::audio::{SileroVad, VadProcessor};
use flowcat_core::observer::{FrameObserver, RtviObserver, RtviSink};
use flowcat_core::pipeline::{build_cascaded_call_duplex, CascadedConfig};
use flowcat_server::events::RtfSink;
use flowcat_services::llm::{OpenAiLlm, OpenAiLlmBuilder};
use flowcat_services::tts::KokoroTts;
use flowcat_transports::webrtc::WebRtcTransport;

use crate::{LoadedStt, PocState};

/// str0m carrier rate — matches flowcat-server's webrtc playground.
const CARRIER_RATE: u32 = 16_000;

/// The local interface address that routes to `peer` (connected-UDP probe, no
/// packet sent); loopback peers get 127.0.0.1. Falls back to loopback when
/// there is no route, which keeps the same-machine case working.
pub fn advertise_ip_toward(peer: std::net::IpAddr) -> std::net::IpAddr {
    if peer.is_loopback() {
        return std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST);
    }
    let bind: std::net::SocketAddr = match peer {
        std::net::IpAddr::V4(_) => (std::net::Ipv4Addr::UNSPECIFIED, 0).into(),
        std::net::IpAddr::V6(_) => (std::net::Ipv6Addr::UNSPECIFIED, 0).into(),
    };
    std::net::UdpSocket::bind(bind)
        .and_then(|s| s.connect((peer, 9)).and_then(|_| s.local_addr()))
        .map(|a| a.ip())
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST))
}

/// Static TTS backend selection (the duplex builder is generic over the
/// concrete `TtsService`; this avoids duplicating the whole build call).
enum PocTts {
    Kokoro(KokoroTts),
    Chatterbox(crate::tts_chatterbox::ChatterboxTts),
    #[cfg(feature = "qwen-tts")]
    Qwen(crate::tts_qwen::QwenTts),
}

#[async_trait::async_trait]
impl flowcat_core::service::TtsService for PocTts {
    fn name(&self) -> &str {
        match self {
            PocTts::Kokoro(t) => t.name(),
            PocTts::Chatterbox(t) => t.name(),
            #[cfg(feature = "qwen-tts")]
            PocTts::Qwen(t) => t.name(),
        }
    }
    fn sample_rate(&self) -> u32 {
        match self {
            PocTts::Kokoro(t) => t.sample_rate(),
            PocTts::Chatterbox(t) => t.sample_rate(),
            #[cfg(feature = "qwen-tts")]
            PocTts::Qwen(t) => t.sample_rate(),
        }
    }
    async fn start(
        &mut self,
        params: &flowcat_core::processor::frame::StartParams,
    ) -> flowcat_core::Result<()> {
        match self {
            PocTts::Kokoro(t) => t.start(params).await,
            PocTts::Chatterbox(t) => t.start(params).await,
            #[cfg(feature = "qwen-tts")]
            PocTts::Qwen(t) => t.start(params).await,
        }
    }
    async fn run_tts(
        &mut self,
        text: &str,
    ) -> flowcat_core::Result<Vec<flowcat_core::processor::frame::Frame>> {
        match self {
            PocTts::Kokoro(t) => t.run_tts(text).await,
            PocTts::Chatterbox(t) => t.run_tts(text).await,
            #[cfg(feature = "qwen-tts")]
            PocTts::Qwen(t) => t.run_tts(text).await,
        }
    }
    async fn run_tts_stream<'a>(
        &'a mut self,
        text: &'a str,
    ) -> flowcat_core::Result<futures::stream::BoxStream<'a, flowcat_core::processor::frame::Frame>>
    {
        match self {
            PocTts::Kokoro(t) => t.run_tts_stream(text).await,
            PocTts::Chatterbox(t) => t.run_tts_stream(text).await,
            #[cfg(feature = "qwen-tts")]
            PocTts::Qwen(t) => t.run_tts_stream(text).await,
        }
    }
}

/// Static LLM provider selection (same reason as `PocTts`).
enum PocLlm {
    Ollama(crate::llm_ollama::OllamaLlm),
    OpenAi(OpenAiLlm),
}

#[async_trait::async_trait]
impl flowcat_core::service::LlmService for PocLlm {
    fn name(&self) -> &str {
        match self {
            PocLlm::Ollama(l) => l.name(),
            PocLlm::OpenAi(l) => l.name(),
        }
    }
    async fn start(
        &mut self,
        params: &flowcat_core::processor::frame::StartParams,
    ) -> flowcat_core::Result<()> {
        match self {
            PocLlm::Ollama(l) => l.start(params).await,
            PocLlm::OpenAi(l) => l.start(params).await,
        }
    }
    async fn run_llm<'a>(
        &'a mut self,
        ctx: &'a flowcat_core::processor::frame::LlmContext,
    ) -> flowcat_core::Result<futures::stream::BoxStream<'a, flowcat_core::processor::frame::Frame>>
    {
        match self {
            PocLlm::Ollama(l) => l.run_llm(ctx).await,
            PocLlm::OpenAi(l) => l.run_llm(ctx).await,
        }
    }
    fn set_tools(&mut self, tools: Vec<flowcat_core::service::Tool>) {
        match self {
            PocLlm::Ollama(l) => l.set_tools(tools),
            PocLlm::OpenAi(l) => l.set_tools(tools),
        }
    }
}

/// The local model, or Claude once the call's `ask_claude` flag is set. Both
/// share the rolling context, so the switch is seamless mid-conversation.
struct SwitchingLlm {
    local: PocLlm,
    claude: Option<crate::llm_claude::ClaudeLlm>,
    state: Arc<crate::skills::CallState>,
}

#[async_trait::async_trait]
impl flowcat_core::service::LlmService for SwitchingLlm {
    fn name(&self) -> &str {
        "switching-llm"
    }
    async fn start(
        &mut self,
        params: &flowcat_core::processor::frame::StartParams,
    ) -> flowcat_core::Result<()> {
        self.local.start(params).await?;
        if let Some(c) = &mut self.claude {
            c.start(params).await?;
        }
        Ok(())
    }
    async fn run_llm<'a>(
        &'a mut self,
        ctx: &'a flowcat_core::processor::frame::LlmContext,
    ) -> flowcat_core::Result<futures::stream::BoxStream<'a, flowcat_core::processor::frame::Frame>>
    {
        match (&mut self.claude, self.state.backend()) {
            (Some(c), crate::skills::LlmBackend::Claude) => c.run_llm(ctx).await,
            _ => self.local.run_llm(ctx).await,
        }
    }
    fn set_tools(&mut self, tools: Vec<flowcat_core::service::Tool>) {
        if let Some(c) = &mut self.claude {
            c.set_tools(tools.clone());
        }
        self.local.set_tools(tools);
    }
}

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
    ConnectInfo(remote): ConnectInfo<std::net::SocketAddr>,
    Json(body): Json<OfferRequest>,
) -> Response {
    let offer_started = std::time::Instant::now();
    let run_id = state.next_run.fetch_add(1, Ordering::Relaxed);
    let pc_id = format!("pc-{run_id}");

    // Register the event stream BEFORE answering so pre-subscribe events buffer.
    let (events, guard) = state.registry.register(&pc_id);

    // Bind wildcard (aiortc sends its ICE checks from per-interface sockets —
    // docker bridges, LAN IP — and a 127.0.0.1-bound socket cannot reply to
    // those sources: EINVAL os error 22, observed live) and advertise below.
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
    // Advertise an address the caller can reach: an explicit POC_ADVERTISE_IP,
    // else the local interface that routes back to the caller (loopback for a
    // same-machine peer, the LAN interface for a remote one). No STUN/TURN: the
    // PoC serves a LAN, and host candidates pair directly there.
    let advertise_ip = state
        .cfg
        .advertise_ip
        .unwrap_or_else(|| advertise_ip_toward(remote.ip()));
    let advertise = Some(advertise_ip);
    tracing::info!(%pc_id, %remote, advertise = %advertise_ip, "webrtc offer");
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
    // The 0.2 s default matches the Python chatbot's wake.vad_stop_secs.
    // Silero evaluates 512-sample windows, so this becomes about 192 ms at
    // 16 kHz instead of the former ~512 ms endpointing delay.
    let vad_params = flowcat_core::VadParams {
        confidence: 0.7,
        start_secs: 0.2,
        stop_secs: cfg.vad_stop_secs,
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
    // Per-call flags the skills and the wake gate set (voice, backend); dropped
    // with the call.
    let call_state = Arc::new(crate::skills::CallState::default());
    // Listen mode: wake gate between VAD and SpeechGate when wake heads are
    // configured (POC_WAKE_DIR / POC_WAKE_MODEL); push mode otherwise. A fire
    // selects the head's persona voice on `call_state` and publishes `wake`
    // events on the call's channel.
    let mut input_processors: Vec<Box<dyn flowcat_core::processor::FrameProcessor>> = Vec::new();
    if !cfg.wake_heads.is_empty() {
        let bank = match voice_chatbot_wake::WakeBank::load(&cfg.wake_heads, cfg.wake_threshold) {
            Ok(b) => b,
            Err(e) => {
                return (
                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                    format!("load wake heads: {e}"),
                )
                    .into_response()
            }
        };
        input_processors.push(Box::new(
            crate::wake::WakeGate::new(bank, cfg.wake_session_secs)
                .with_state(call_state.clone())
                .with_events(events.clone()),
        ));
    }
    // The selected local recognizer is loaded once per process (in PocState or
    // the local Nemotron sidecar). Each call gets isolated mutable state.
    // Streaming backends publish display-only interims; every backend publishes
    // exactly one authoritative final at the SpeechGate's external VAD
    // boundary, so Haiku/tools never run on a partial hypothesis.
    let stt: Box<dyn flowcat_core::service::SttService> = match &state.stt {
        LoadedStt::Whisper(context) => Box::new(
            crate::stt::BabelStt::from_context(context.clone()).with_threads(cfg.whisper_threads),
        ),
        #[cfg(feature = "moonshine")]
        LoadedStt::Moonshine(engine) => Box::new(
            crate::moonshine::MoonshineStt::from_engine(engine.clone())
                .with_update_interval_ms(cfg.moonshine_update_interval_ms),
        ),
        LoadedStt::Nemotron => Box::new(crate::nemotron::NemotronStt::new(
            cfg.nemotron_url.clone(),
            cfg.nemotron_speech_contexts.clone(),
        )),
        #[cfg(feature = "nemotron-native")]
        LoadedStt::NemotronNative(engine) => {
            Box::new(crate::nemotron_native::NemotronNativeStt::from_engine(
                engine.clone(),
                cfg.nemotron_speech_contexts.clone(),
            ))
        }
    };
    let inner = match cfg.llm_provider.as_str() {
        // ADR-0007 Layer 1: native /api/chat — keep_alive, num_ctx and
        // think=false on every request; prompt-cache evidence in the metrics.
        "ollama" => PocLlm::Ollama(
            crate::llm_ollama::OllamaLlm::new(cfg.llm_base_url.clone(), cfg.llm_model.clone())
                .num_ctx(cfg.llm_num_ctx),
        ),
        // OpenAI-compatible chat completions (the OpenRouter cloud profile).
        _ => PocLlm::OpenAi(
            OpenAiLlmBuilder::new(cfg.openrouter_key.clone())
                .base_url(cfg.llm_base_url.clone())
                .model(cfg.llm_model.clone())
                .build(),
        ),
    };
    let claude = (!cfg.anthropic_key.trim().is_empty()).then(|| {
        crate::llm_claude::ClaudeLlm::new(cfg.anthropic_key.clone(), cfg.claude_model.clone())
    });
    let inner = SwitchingLlm {
        local: inner,
        claude,
        state: call_state.clone(),
    };
    let llm = crate::llm::StaticGreetingLlm::new(inner, "Ready.");
    let tts = match cfg.tts_backend.as_str() {
        "chatterbox" => PocTts::Chatterbox(
            crate::tts_chatterbox::ChatterboxTts::new(
                cfg.chatterbox_url.clone(),
                cfg.chatterbox_voice.clone(),
            )
            .with_ready_pcm(state.ready_pcm.clone()),
        ),
        "kokoro" => PocTts::Kokoro(
            KokoroTts::new("", cfg.kokoro_voice.clone()).with_base_url(cfg.kokoro_url.clone()),
        ),
        #[cfg(feature = "qwen-tts")]
        "qwen" => PocTts::Qwen(
            crate::tts_qwen::QwenTts::new(
                state.qwen.clone().expect("qwen engine started at startup"),
            )
            .with_state(call_state.clone()),
        ),
        _ => unreachable!("validated at startup"),
    };
    let brain = crate::brain::BabelBrain::new(cfg.system_prompt.clone());
    let session = state.session.clone();

    // Skills drive client-side playback (radio, shows, sound effects) over the
    // same events channel the RTVI observer publishes on.
    let media = Arc::new(crate::media::MediaController::new(events.clone()));
    let sink: Arc<dyn RtviSink> = Arc::new(RtfSink::new(events));
    let observers: Vec<Arc<dyn FrameObserver>> = vec![Arc::new(RtviObserver::new(sink))];

    // Hang-up hook: the events WebSocket closing cancels this call.
    let hangup = Arc::new(tokio::sync::Notify::new());
    state
        .hangups
        .lock()
        .unwrap()
        .insert(pc_id.clone(), hangup.clone());
    // Release bot audio at realtime (+ a small lead) instead of as fast as
    // TTS produces it; see paced_transport.rs for why bursts lose audio.
    let transport = crate::paced_transport::PacedTransport::new(transport);
    let hangups = state.clone();
    let hook_pc_id = pc_id.clone();

    tokio::spawn(async move {
        let _guard = guard; // deregister the event stream when the call ends
        let built = build_cascaded_call_duplex(
            transport,
            vad,
            input_processors,
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
                // Skills that act later (timers) find this call's pipeline here.
                hangups.session.calls().register(
                    run_id,
                    crate::skills::CallHandle {
                        frames: task.task.queue_sender(),
                        media,
                        state: call_state,
                    },
                );
                let token = task.task.cancel_token();
                let watcher = tokio::spawn(async move {
                    hangup.notified().await;
                    token.cancel();
                });
                if let Err(e) = task.run().await {
                    tracing::warn!(run_id, error = %e, "call ended with error");
                } else {
                    tracing::info!(run_id, "call ended");
                }
                watcher.abort();
                hangups.session.calls().unregister(run_id);
            }
            Err(e) => tracing::error!(run_id, error = %e, "failed to build call"),
        }
        hangups.hangups.lock().unwrap().remove(&hook_pc_id);
    });

    tracing::info!(
        run_id,
        elapsed_ms = offer_started.elapsed().as_millis(),
        "webrtc offer accepted"
    );

    Json(json!(OfferResponse { sdp: answer, pc_id })).into_response()
}
