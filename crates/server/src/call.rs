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
    /// The context with the persona's system prompt swapped in; owned here so
    /// the returned stream can borrow it for the run.
    scratch: flowcat_core::processor::frame::LlmContext,
}

/// `ctx` with `prompt` as its system message (replacing a leading system
/// message, else prepended).
fn with_system_prompt(
    ctx: &flowcat_core::processor::frame::LlmContext,
    prompt: &str,
) -> flowcat_core::processor::frame::LlmContext {
    let mut out = ctx.clone();
    let system = serde_json::json!({ "role": "system", "content": prompt });
    match out.messages.first_mut() {
        Some(first) if first.get("role").and_then(|r| r.as_str()) == Some("system") => {
            *first = system;
        }
        _ => out.messages.insert(0, system),
    }
    out
}

/// Appended to the system prompt on the Claude branch.
///
/// This used to arrive as `ask_claude`'s tool *result*, which meant Claude's
/// history carried a call to a tool Claude is no longer shown — and meant the
/// brevity rule applied only to the turn immediately after the flip. As a
/// system suffix it holds for every Claude turn, and `strip_ask_claude` can run
/// unconditionally.
pub(crate) const CLAUDE_SYSTEM_SUFFIX: &str =
    "\n\nYou are now answering as Claude, on a live voice call. \
Answer the caller directly — never say you are handing over, and never ask them to repeat \
themselves. Keep replies to one or two short spoken sentences, and offer to go deeper if they \
want more. If you are going to search the web, say a short line first — \"let me check\" — so the \
caller is not sitting in silence while the search runs.";

/// Append `suffix` to `ctx`'s system message, creating one if there is none.
fn append_system_suffix(ctx: &mut flowcat_core::processor::frame::LlmContext, suffix: &str) {
    match ctx.messages.first_mut() {
        Some(first) if first.get("role").and_then(|r| r.as_str()) == Some("system") => {
            // Not `as_str()`: a system message whose content is an array of
            // blocks would then read as empty and be replaced wholesale by the
            // bare suffix, losing Babel's entire persona prompt.
            // `content_string` flattens the array the same way the Claude
            // translation downstream does, so nothing Claude would have seen is
            // dropped.
            let base = first
                .get("content")
                .map(crate::llm_claude::content_string)
                .unwrap_or_default();
            *first = serde_json::json!({"role": "system", "content": format!("{base}{suffix}")});
        }
        _ => ctx.messages.insert(
            0,
            serde_json::json!({"role": "system", "content": suffix.trim_start()}),
        ),
    }
}

/// The tool whose exchanges the local model must not see (see [`strip_ask_claude`]).
const ASK_CLAUDE: &str = "ask_claude";

fn is_empty_content(v: &serde_json::Value) -> bool {
    match v {
        serde_json::Value::Null => true,
        serde_json::Value::String(s) => s.trim().is_empty(),
        serde_json::Value::Array(a) => a.is_empty(),
        _ => false,
    }
}

fn has_ask_claude(messages: &[serde_json::Value]) -> bool {
    messages.iter().any(|m| {
        m["tool_calls"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|c| c["function"]["name"] == ASK_CLAUDE)
    })
}

/// Drop the `ask_claude` call/result pairs from a copy of the rolling context.
///
/// The tool flips the backend, and the flip reverts when the wake session goes
/// to sleep — but the exchange stays in the context for the rest of the call.
/// Reading it back, the local model concludes it *is* Claude and answers the
/// next "Claude, …" itself instead of calling the tool again: measured at half
/// of the phrasings that fire reliably on a clean context ("I am already
/// Claude, though I suppose the distinction is largely…"). Hiding just this
/// exchange restored every one of them. Claude's spoken answers stay — only the
/// call and its result go. It runs on both branches: Claude receives the
/// handover as a system-prompt suffix (`CLAUDE_SYSTEM_SUFFIX`) instead, so it
/// never sees a call to a tool that is not in its own tool list.
fn strip_ask_claude(messages: &mut Vec<serde_json::Value>) {
    let mut dropped: Vec<String> = Vec::new();
    messages.retain_mut(|m| match m["role"].as_str().unwrap_or("") {
        "assistant" => {
            let Some(calls) = m["tool_calls"].as_array_mut() else {
                return true;
            };
            calls.retain(|c| {
                let hit = c["function"]["name"] == ASK_CLAUDE;
                if hit {
                    if let Some(id) = c["id"].as_str() {
                        dropped.push(id.to_string());
                    }
                }
                !hit
            });
            if !calls.is_empty() {
                return true;
            }
            if let Some(o) = m.as_object_mut() {
                o.remove("tool_calls");
            }
            // The pipeline records a call as its own `content: null` message, so
            // that one goes; a turn that both spoke and called keeps its text.
            !is_empty_content(&m["content"])
        }
        // Its call is gone, so this would be an orphan the adapters reject.
        "tool" => !m["tool_call_id"]
            .as_str()
            .is_some_and(|id| dropped.iter().any(|d| d == id)),
        _ => true,
    });
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
        // Up to three rewrites, all landing in `scratch` so the returned stream
        // can borrow it: the persona prompt (prompt.<persona>.txt, selected by
        // a wake word or switch_persona) replacing the default system message,
        // the Claude handover suffix appended on the Claude branch, and the
        // ask_claude exchange hidden from both backends. Split borrows: the
        // stream borrows `scratch` while `local`/`claude` are borrowed mutably.
        let Self {
            local,
            claude,
            state,
            scratch,
        } = self;
        let backend = state.backend();
        let on_claude = claude.is_some() && backend == crate::skills::LlmBackend::Claude;
        let prompt = state.prompt();
        // The ask_claude exchange is hidden from *both* backends now: from the
        // local model because reading it back convinces it that it already is
        // Claude, and from Claude because it names a tool Claude is not shown.
        let rewrite = prompt.is_some() || on_claude || has_ask_claude(&ctx.messages);
        let ctx: &'a flowcat_core::processor::frame::LlmContext = if rewrite {
            *scratch = match &prompt {
                Some(p) => with_system_prompt(ctx, p),
                None => ctx.clone(),
            };
            if on_claude {
                append_system_suffix(scratch, CLAUDE_SYSTEM_SUFFIX);
            }
            strip_ask_claude(&mut scratch.messages);
            scratch
        } else {
            ctx
        };
        match (claude, backend) {
            (Some(c), crate::skills::LlmBackend::Claude) => c.run_llm(ctx).await,
            _ => local.run_llm(ctx).await,
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
    // Advertise an address the caller can reach: an explicit ADVERTISE_IP,
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
    let call_state = Arc::new(crate::skills::CallState::with_prompts(
        cfg.persona_prompts.clone(),
    ));
    // Listen mode: wake gate between VAD and SpeechGate when wake heads are
    // configured (WAKE_DIR / WAKE_MODEL); push mode otherwise. A fire
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
    // Both wake paths (server gate above, native client via apply_client_wake)
    // arm this: the wake phrase and a command after a short pause become one
    // turn instead of two racing ones.
    if cfg.wake_grace_secs > 0.0 {
        input_processors.push(Box::new(crate::wake::WakeGrace::new(
            call_state.clone(),
            std::time::Duration::from_secs_f32(cfg.wake_grace_secs),
        )));
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
        let search = cfg
            .claude_web_search
            .then(|| crate::llm_claude::SearchConfig {
                tool: cfg.claude_search_tool.clone(),
                max_uses: cfg.claude_search_max_uses,
                user_location: cfg
                    .search_location
                    .as_ref()
                    .map(|l| l.user_location())
                    .unwrap_or(serde_json::Value::Null),
            });
        crate::llm_claude::ClaudeLlm::new(
            cfg.anthropic_key.clone(),
            cfg.claude_model.clone(),
            cfg.claude_effort.clone(),
            search,
        )
    });
    let inner = SwitchingLlm {
        local: inner,
        claude,
        state: call_state.clone(),
        scratch: Default::default(),
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

#[cfg(test)]
mod prompt_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn persona_prompt_replaces_or_prepends_the_system_message() {
        let ctx = flowcat_core::processor::frame::LlmContext {
            messages: vec![
                json!({"role": "system", "content": "default"}),
                json!({"role": "user", "content": "hi"}),
            ],
            tools: vec![json!({"name": "t"})],
        };
        let out = with_system_prompt(&ctx, "marvin");
        assert_eq!(
            out.messages[0],
            json!({"role": "system", "content": "marvin"})
        );
        assert_eq!(out.messages[1], ctx.messages[1]);
        assert_eq!(out.tools, ctx.tools);

        let no_system = flowcat_core::processor::frame::LlmContext {
            messages: vec![json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        let out = with_system_prompt(&no_system, "marvin");
        assert_eq!(out.messages.len(), 2);
        assert_eq!(out.messages[0]["role"], "system");
    }

    fn call_msg(id: &str, name: &str) -> serde_json::Value {
        json!({"role": "assistant", "content": null, "tool_calls": [{
            "id": id, "type": "function",
            "function": {"name": name, "arguments": "{}"}}]})
    }

    #[test]
    fn the_local_model_never_sees_the_ask_claude_exchange() {
        let mut msgs = vec![
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "what time is it"}),
            call_msg("call_1_0", "get_current_time"),
            json!({"role": "tool", "tool_call_id": "call_1_0", "content": "8:39 PM"}),
            json!({"role": "assistant", "content": "Eight thirty-nine."}),
            json!({"role": "user", "content": "ask claude about Rome"}),
            call_msg("call_2_0", "ask_claude"),
            json!({"role": "tool", "tool_call_id": "call_2_0", "content": "You are now Claude…"}),
            json!({"role": "assistant", "content": "The Republic outgrew its institutions."}),
        ];
        assert!(has_ask_claude(&msgs));
        strip_ask_claude(&mut msgs);

        assert!(!has_ask_claude(&msgs));
        // The other tool exchange is untouched, and Claude's spoken answer stays.
        let roles: Vec<&str> = msgs.iter().map(|m| m["role"].as_str().unwrap()).collect();
        assert_eq!(
            roles,
            [
                "system",
                "user",
                "assistant",
                "tool",
                "assistant",
                "user",
                "assistant"
            ]
        );
        assert_eq!(msgs[3]["tool_call_id"], "call_1_0");
        assert_eq!(msgs[6]["content"], "The Republic outgrew its institutions.");
        // No orphan `tool` message: every result still answers a live call.
        let ids: Vec<&str> = msgs
            .iter()
            .filter_map(|m| m["tool_call_id"].as_str())
            .collect();
        assert_eq!(ids, ["call_1_0"]);
    }

    #[test]
    fn a_turn_that_spoke_and_called_ask_claude_keeps_its_words() {
        let mut msgs = vec![json!({
            "role": "assistant", "content": "One moment.",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "ask_claude", "arguments": "{}"}}]})];
        strip_ask_claude(&mut msgs);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0]["content"], "One moment.");
        assert!(msgs[0].get("tool_calls").is_none(), "{:?}", msgs[0]);
    }

    #[test]
    fn a_context_without_ask_claude_is_left_alone() {
        let msgs = vec![
            json!({"role": "user", "content": "weather"}),
            call_msg("call_1_0", "get_weather"),
            json!({"role": "tool", "tool_call_id": "call_1_0", "content": "clear, 20"}),
        ];
        assert!(!has_ask_claude(&msgs));
        let mut stripped = msgs.clone();
        strip_ask_claude(&mut stripped);
        assert_eq!(stripped, msgs);
    }

    #[test]
    fn selecting_a_persona_picks_its_prompt_or_the_default() {
        let prompts = [("marvin", "I am Marvin."), ("one-one", "I am One One.")]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let state = crate::skills::CallState::with_prompts(prompts);
        assert_eq!(
            state.prompt(),
            None,
            "default prompt until a persona is chosen"
        );
        state.set_voice("marvin");
        assert_eq!(state.prompt().as_deref(), Some("I am Marvin."));
        state.set_voice("one_one");
        assert_eq!(
            state.prompt().as_deref(),
            Some("I am One One."),
            "`_`/`-` insensitive"
        );
        state.set_voice("babel");
        assert_eq!(
            state.prompt(),
            None,
            "no prompt.babel.txt in this map → default"
        );
    }

    fn ask_claude_ctx() -> flowcat_core::processor::frame::LlmContext {
        flowcat_core::processor::frame::LlmContext {
            messages: vec![
                serde_json::json!({"role": "system", "content": "Be Babel."}),
                serde_json::json!({"role": "user", "content": "Use Claude to find showtimes"}),
                serde_json::json!({"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "ask_claude", "arguments": "{}"}}
                ]}),
                serde_json::json!({"role": "tool", "tool_call_id": "c1", "content": "handover"}),
            ],
            tools: vec![],
        }
    }

    #[test]
    fn the_ask_claude_exchange_is_stripped_for_both_backends() {
        let mut messages = ask_claude_ctx().messages;
        strip_ask_claude(&mut messages);
        let roles: Vec<&str> = messages
            .iter()
            .map(|m| m["role"].as_str().unwrap())
            .collect();
        assert_eq!(roles, vec!["system", "user"]);
    }

    #[test]
    fn the_claude_suffix_appends_to_the_existing_system_prompt() {
        let mut ctx = ask_claude_ctx();
        append_system_suffix(&mut ctx, CLAUDE_SYSTEM_SUFFIX);
        let system = ctx.messages[0]["content"].as_str().unwrap();
        assert!(system.starts_with("Be Babel."), "{system}");
        assert!(system.contains("answering as Claude"), "{system}");
        assert!(system.contains("let me check"), "{system}");
    }

    /// A system message whose content is an array of blocks used to read as
    /// empty through `as_str()`, so the suffix replaced Babel's entire persona
    /// prompt instead of appending to it.
    #[test]
    fn the_claude_suffix_keeps_a_block_shaped_system_prompt() {
        let mut ctx = flowcat_core::processor::frame::LlmContext {
            messages: vec![serde_json::json!({
                "role": "system",
                "content": [
                    {"type": "text", "text": "Be Babel."},
                    {"type": "text", "text": " Be brief."},
                ]
            })],
            tools: vec![],
        };
        append_system_suffix(&mut ctx, CLAUDE_SYSTEM_SUFFIX);
        let system = ctx.messages[0]["content"].as_str().unwrap();
        assert!(system.starts_with("Be Babel. Be brief."), "{system}");
        assert!(system.contains("answering as Claude"), "{system}");
    }

    #[test]
    fn the_claude_suffix_creates_a_system_message_when_there_is_none() {
        let mut ctx = flowcat_core::processor::frame::LlmContext {
            messages: vec![serde_json::json!({"role": "user", "content": "hi"})],
            tools: vec![],
        };
        append_system_suffix(&mut ctx, CLAUDE_SYSTEM_SUFFIX);
        assert_eq!(ctx.messages[0]["role"], "system");
        assert_eq!(ctx.messages[1]["role"], "user");
    }
}
