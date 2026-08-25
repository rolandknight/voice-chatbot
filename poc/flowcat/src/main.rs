//! flowcat-poc — the Phase 1 embedder server (docs/poc/flowcat-poc-plan.md).
//!
//! Serves the same surface as flowcat-server's webrtc mode (`GET /` playground,
//! `POST /webrtc/offer`, `GET /webrtc/events/{pc_id}`, `GET /healthz`) but with
//! Babel's shape: a no-graph brain, skills relayed to the local stub server, and
//! directly-constructed local services (selectable local STT, OpenRouter LLM,
//! Kokoro-shim TTS) — bypassing `factory::cascaded`, which can't set Kokoro's
//! base_url and demands dummy API keys for keyless local providers.

mod brain;
mod call;
mod llm;
mod llm_ollama;
mod ollama_serve;
#[cfg(feature = "moonshine")]
mod moonshine;
mod nemotron;
#[cfg(feature = "qwen-tts")]
mod tts_qwen;
mod playground;
mod session;
mod stt;
mod tts_chatterbox;
mod wake;

use std::sync::atomic::AtomicI64;
use std::sync::Arc;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use flowcat_server::events::{stream_events, EventRegistry};

use crate::session::StubSession;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SttBackend {
    Whisper,
    Moonshine,
    Nemotron,
}

impl SttBackend {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "whisper" => Ok(Self::Whisper),
            "moonshine" => Ok(Self::Moonshine),
            "nemotron" | "nvidia" => Ok(Self::Nemotron),
            _ => Err(format!(
                "unsupported POC_STT_BACKEND {value:?} (expected \"whisper\", \"moonshine\", or \"nemotron\")"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Moonshine => "moonshine",
            Self::Nemotron => "nemotron",
        }
    }
}

pub enum LoadedStt {
    Whisper(stt::SharedWhisperContext),
    #[cfg(feature = "moonshine")]
    Moonshine(moonshine::SharedMoonshineEngine),
    /// NVIDIA's model is held by a local NeMo-Speech.cpp sidecar. Calls create
    /// cheap, isolated realtime WebSocket sessions against that resident model.
    Nemotron,
}

pub struct PocConfig {
    pub openrouter_key: String,
    /// OpenAI-compatible chat-completions base (OpenRouter, or a local /v1).
    pub llm_base_url: String,
    pub llm_model: String,
    /// `ollama` (native /api/chat; ADR-0007 Layer 1) or `openrouter` (OpenAI-compatible).
    pub llm_provider: String,
    /// Context size sent on every native request (and to a spawned serve).
    pub llm_num_ctx: u32,
    /// ADR-0007 Layer 2: `auto` | `never` | `always`.
    pub ollama_supervise: ollama_serve::Supervise,
    pub ollama_bin: String,
    /// Release the model on exit (`keep_alive: 0`) so its memory returns.
    pub ollama_unload_on_exit: bool,
    pub stt_backend: SttBackend,
    pub whisper_model: String,
    /// CPU workers used by whisper.cpp for each utterance.
    pub whisper_threads: usize,
    pub moonshine_model: String,
    /// How often the streaming decoder publishes a display-only hypothesis.
    pub moonshine_update_interval_ms: u64,
    /// Optional comma-separated contextual terms for Moonshine's decoder.
    pub moonshine_keyterms: String,
    /// Base URL of the local NeMo-Speech.cpp server (no cloud STT).
    pub nemotron_url: String,
    /// Comma-separated phrases to bias the RNNT decoder toward local entities.
    pub nemotron_speech_contexts: Vec<String>,
    pub kokoro_url: String,
    pub kokoro_voice: String,
    pub system_prompt: String,
    pub vad_model: String,
    /// Silence needed to close a speech turn. The default matches the Python
    /// chatbot's `wake.vad_stop_secs` setting.
    pub vad_stop_secs: f32,
    /// Listen mode (Phase 1a): path to a wake-word head model (e.g.
    /// models/wakeword/hey_babel.onnx). Empty → push mode (no server wake).
    pub wake_model: String,
    pub wake_threshold: f32,
    /// TTS backend: "kokoro" (default) or "chatterbox" (Phase 1b cloned voice).
    pub tts_backend: String,
    pub chatterbox_url: String,
    pub chatterbox_voice: String,
    /// Qwen3-TTS (`qwen`): engine profile (poc-qwen-streaming config), preset
    /// voice name, model size, streamed-chunk interval.
    pub qwen_config: String,
    pub qwen_voice: String,
    pub qwen_size: String,
    pub qwen_interval_s: f64,
    /// Host ICE candidate address to advertise (POC_ADVERTISE_IP). None →
    /// the interface that routes back to each caller.
    pub advertise_ip: Option<std::net::IpAddr>,
}

pub struct PocState {
    pub cfg: PocConfig,
    pub registry: Arc<EventRegistry>,
    pub session: Arc<StubSession>,
    pub stt: LoadedStt,
    pub ready_pcm: Option<Arc<[i16]>>,
    pub next_run: AtomicI64,
    /// Shared Qwen engine + voice when `POC_TTS_BACKEND=qwen`.
    #[cfg(feature = "qwen-tts")]
    pub qwen: Option<tts_qwen::QwenShared>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn require_nonempty(value: &str, key: &str) -> Result<(), Box<dyn std::error::Error>> {
    if value.trim().is_empty() {
        return Err(format!("{key} must not be empty").into());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // str0m needs aws-lc-rs as the process rustls provider; install before any TLS.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,str0m=warn".into()),
        )
        .init();
    // poc/.env when run from repo root or poc/; fall back silently otherwise.
    let _ = dotenvy::from_filename("poc/.env").or_else(|_| dotenvy::from_filename(".env"));

    let manifest_dir = env!("CARGO_MANIFEST_DIR"); // poc/flowcat
    let poc_dir = std::path::Path::new(manifest_dir).parent().unwrap();
    let stt_backend = SttBackend::parse(&env_or("POC_STT_BACKEND", "whisper"))?;

    // Use the physical-core count as a practical default on SMT machines,
    // capped for predictable thermals. This laptop is 8C/16T, so it selects 8.
    let default_whisper_threads = std::thread::available_parallelism()
        .map(|parallelism| parallelism.get().div_ceil(2).min(8))
        .unwrap_or(4);
    let whisper_threads = env_or("POC_WHISPER_THREADS", &default_whisper_threads.to_string())
        .parse::<usize>()
        .map_err(|error| format!("invalid POC_WHISPER_THREADS: {error}"))?;
    if whisper_threads == 0 {
        return Err("POC_WHISPER_THREADS must be greater than zero".into());
    }
    let moonshine_update_interval_ms = env_or("POC_MOONSHINE_UPDATE_INTERVAL_MS", "250")
        .parse::<u64>()
        .map_err(|error| format!("invalid POC_MOONSHINE_UPDATE_INTERVAL_MS: {error}"))?;
    if !(200..=2_000).contains(&moonshine_update_interval_ms) {
        return Err("POC_MOONSHINE_UPDATE_INTERVAL_MS must be in [200, 2000]".into());
    }
    let vad_stop_secs = env_or("POC_VAD_STOP_SECS", "0.2")
        .parse::<f32>()
        .map_err(|error| format!("invalid POC_VAD_STOP_SECS: {error}"))?;
    if !vad_stop_secs.is_finite() || vad_stop_secs <= 0.0 || vad_stop_secs > 2.0 {
        return Err("POC_VAD_STOP_SECS must be finite and in (0, 2]".into());
    }

    let cfg = PocConfig {
        openrouter_key: env_or("OPENROUTER_API_KEY", ""),
        llm_base_url: env_or(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
        llm_model: env_or("POC_LLM_MODEL", "anthropic/claude-haiku-4.5"),
        llm_provider: {
            let base = env_or("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1");
            env_or(
                "POC_LLM_PROVIDER",
                if base.contains(":11434") { "ollama" } else { "openrouter" },
            )
        },
        llm_num_ctx: env_or("POC_LLM_NUM_CTX", "8192")
            .parse::<u32>()
            .map_err(|error| format!("invalid POC_LLM_NUM_CTX: {error}"))?,
        ollama_supervise: ollama_serve::Supervise::parse(&env_or("POC_OLLAMA_SUPERVISE", "auto"))?,
        ollama_bin: env_or("POC_OLLAMA_BIN", "ollama"),
        ollama_unload_on_exit: env_or("POC_OLLAMA_UNLOAD_ON_EXIT", "true")
            .parse::<bool>()
            .map_err(|error| format!("invalid POC_OLLAMA_UNLOAD_ON_EXIT: {error}"))?,
        stt_backend,
        whisper_model: env_or(
            "POC_WHISPER_MODEL",
            &poc_dir.join("models/ggml-base.en.bin").to_string_lossy(),
        ),
        whisper_threads,
        moonshine_model: env_or(
            "POC_MOONSHINE_MODEL",
            &poc_dir
                .join(
                    "models/moonshine/download.moonshine.ai/model/medium-streaming-en/quantized_26_07_30",
                )
                .to_string_lossy(),
        ),
        moonshine_update_interval_ms,
        moonshine_keyterms: env_or("POC_MOONSHINE_KEYTERMS", ""),
        nemotron_url: env_or("POC_NEMOTRON_URL", "http://127.0.0.1:8178"),
        nemotron_speech_contexts: env_or("POC_NEMOTRON_SPEECH_CONTEXTS", "")
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .collect(),
        kokoro_url: env_or("POC_KOKORO_URL", "http://127.0.0.1:8880"),
        kokoro_voice: env_or("POC_KOKORO_VOICE", "af_heart"),
        system_prompt: std::fs::read_to_string(env_or(
            "POC_PROMPT",
            &poc_dir.join("flowcat/prompt.txt").to_string_lossy(),
        ))?,
        vad_model: env_or(
            "POC_VAD_MODEL",
            &poc_dir.join("models/silero_vad.onnx").to_string_lossy(),
        ),
        vad_stop_secs,
        wake_model: env_or("POC_WAKE_MODEL", ""),
        wake_threshold: env_or("POC_WAKE_THRESHOLD", "0.5").parse().unwrap_or(0.5),
        tts_backend: env_or("POC_TTS_BACKEND", "kokoro"),
        chatterbox_url: env_or("POC_CHATTERBOX_URL", "http://127.0.0.1:8004"),
        chatterbox_voice: env_or("POC_CHATTERBOX_VOICE", "marvin.wav"),
        qwen_config: env_or(
            "POC_QWEN_CONFIG",
            &poc_dir
                .join("../poc-qwen-streaming/config.flowcat.yaml")
                .to_string_lossy(),
        ),
        qwen_voice: env_or("POC_QWEN_VOICE", "marvin"),
        qwen_size: env_or("POC_QWEN_SIZE", "1.7B"),
        qwen_interval_s: env_or("POC_QWEN_INTERVAL_S", "0.32")
            .parse::<f64>()
            .map_err(|error| format!("invalid POC_QWEN_INTERVAL_S: {error}"))?,
        advertise_ip: {
            let raw = env_or("POC_ADVERTISE_IP", "");
            if raw.trim().is_empty() {
                None
            } else {
                Some(raw.trim().parse::<std::net::IpAddr>().map_err(|error| format!("invalid POC_ADVERTISE_IP: {error}"))?)
            }
        },
    };
    if cfg.stt_backend == SttBackend::Whisper && !std::path::Path::new(&cfg.whisper_model).exists()
    {
        return Err(format!("whisper model missing: {}", cfg.whisper_model).into());
    }
    if cfg.stt_backend == SttBackend::Moonshine
        && !std::path::Path::new(&cfg.moonshine_model).exists()
    {
        return Err(format!(
            "moonshine model missing: {} (run ./scripts/setup_moonshine.sh)",
            cfg.moonshine_model
        )
        .into());
    }
    if cfg.stt_backend == SttBackend::Nemotron {
        require_nonempty(&cfg.nemotron_url, "POC_NEMOTRON_URL")?;
    }
    if !std::path::Path::new(&cfg.vad_model).exists() {
        return Err(format!("silero vad model missing: {}", cfg.vad_model).into());
    }
    require_nonempty(&cfg.llm_model, "POC_LLM_MODEL")?;
    require_nonempty(&cfg.llm_base_url, "OPENROUTER_BASE_URL")?;
    match cfg.llm_provider.as_str() {
        "openrouter" => require_nonempty(&cfg.openrouter_key, "OPENROUTER_API_KEY")?,
        "ollama" => {
            if cfg.llm_num_ctx < 2048 {
                return Err("POC_LLM_NUM_CTX must be >= 2048".into());
            }
        }
        other => {
            return Err(format!(
                "unsupported POC_LLM_PROVIDER {other:?} (expected \"ollama\" or \"openrouter\")"
            )
            .into())
        }
    }
    match cfg.tts_backend.as_str() {
        "kokoro" => {}
        "chatterbox" => {
            require_nonempty(&cfg.chatterbox_url, "POC_CHATTERBOX_URL")?;
            require_nonempty(&cfg.chatterbox_voice, "POC_CHATTERBOX_VOICE")?;
        }
        "qwen" => {
            if !cfg!(feature = "qwen-tts") {
                return Err("POC_TTS_BACKEND=qwen needs the qwen-tts build: POC_TTS_BACKEND=qwen make build".into());
            }
            require_nonempty(&cfg.qwen_config, "POC_QWEN_CONFIG")?;
            require_nonempty(&cfg.qwen_voice, "POC_QWEN_VOICE")?;
            if !std::path::Path::new(&cfg.qwen_config).exists() {
                return Err(format!("qwen engine config missing: {}", cfg.qwen_config).into());
            }
        }
        other => {
            return Err(format!(
                "unsupported POC_TTS_BACKEND {other:?} (expected \"kokoro\", \"chatterbox\", or \"qwen\")"
            )
            .into())
        }
    }

    let skills_path = env_or(
        "POC_SKILLS",
        &poc_dir.join("stubs/skills.json").to_string_lossy(),
    );
    let session = StubSession::new(
        &std::fs::read_to_string(&skills_path)?,
        env_or("POC_STUBS_URL", "http://127.0.0.1:8790"),
        poc_dir.join("logs/artifacts"),
    )?;

    // ADR-0007: the chatbot owns its LLM's lifecycle. Ensure a serve, pull the
    // model if missing, warm the exact prefix, verify residency — before the
    // audio engines load, so `--warm-only` (make ollama) is quick.
    let warm_only = std::env::args().any(|a| a == "--warm-only");
    let ollama_serve = if cfg.llm_provider == "ollama" {
        Some(start_ollama(&cfg, &session, poc_dir).await?)
    } else {
        None
    };
    if warm_only {
        tracing::info!("--warm-only: LLM warm and resident; exiting");
        if let Some(serve) = ollama_serve {
            // A serve we spawned must outlive this short-lived process so the
            // stack that follows finds it: detach it instead of killing it.
            serve.detach();
        }
        return Ok(());
    }

    // `run_poc.sh up` creates this with a real synthesis after Chatterbox is
    // warm. Reuse it for the fixed connect greeting; direct binary launches
    // without the file still work, but fall back to normal TTS synthesis.
    let ready_pcm = if cfg.tts_backend == "chatterbox" {
        let path = env_or(
            "POC_GREETING_WAV",
            &poc_dir.join("logs/chatterbox-health.wav").to_string_lossy(),
        );
        match tts_chatterbox::load_ready_wav(&path) {
            Ok(pcm) => {
                tracing::info!(samples = pcm.len(), %path, "cached greeting loaded");
                Some(pcm)
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(%path, "cached greeting missing; connect greeting will use TTS");
                None
            }
            Err(error) => {
                return Err(format!("invalid cached greeting {path}: {error}").into());
            }
        }
    } else {
        None
    };

    // Health only becomes reachable after the selected heavyweight STT model is
    // resident. Calls share model weights and keep stream/decoder state local.
    let preload_started = std::time::Instant::now();
    let loaded_stt = match cfg.stt_backend {
        SttBackend::Whisper => LoadedStt::Whisper(stt::load_context(&cfg.whisper_model)?),
        SttBackend::Moonshine => {
            #[cfg(feature = "moonshine")]
            {
                let keyterms = (!cfg.moonshine_keyterms.trim().is_empty())
                    .then_some(cfg.moonshine_keyterms.as_str());
                LoadedStt::Moonshine(moonshine::load_engine(&cfg.moonshine_model, keyterms)?)
            }
            #[cfg(not(feature = "moonshine"))]
            {
                return Err(
                    "POC_STT_BACKEND=moonshine requires a Moonshine build; run POC_STT_BACKEND=moonshine make poc-build"
                        .into(),
                );
            }
        }
        SttBackend::Nemotron => {
            let ready_url = format!("{}/ready", cfg.nemotron_url.trim_end_matches('/'));
            let response = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()?
                .get(&ready_url)
                .send()
                .await
                .map_err(|error| {
                    format!(
                        "Nemotron sidecar is unavailable at {ready_url}: {error} (run ./scripts/start_nemotron.sh)"
                    )
                })?
                .error_for_status()
                .map_err(|error| format!("Nemotron sidecar is not ready at {ready_url}: {error}"))?;
            let readiness: serde_json::Value = response.json().await.map_err(|error| {
                format!("invalid Nemotron readiness response from {ready_url}: {error}")
            })?;
            tracing::info!(%ready_url, %readiness, "Nemotron sidecar ready");
            LoadedStt::Nemotron
        }
    };
    tracing::info!(
        backend = cfg.stt_backend.as_str(),
        elapsed_ms = preload_started.elapsed().as_millis(),
        "STT model preloaded"
    );

    // Qwen3-TTS: start the in-process engine, preload the clone model + voice,
    // and synthesize the fixed greeting once so reconnects never wait on TTS.
    #[cfg(feature = "qwen-tts")]
    let (qwen, ready_pcm) = if cfg.tts_backend == "qwen" {
        let shared = start_qwen(&cfg).await?;
        let pcm = shared.ready_pcm.clone();
        (Some(shared), pcm)
    } else {
        (None, ready_pcm)
    };

    let state = Arc::new(PocState {
        cfg,
        registry: Arc::new(EventRegistry::new()),
        session: Arc::new(session),
        stt: loaded_stt,
        ready_pcm,
        next_run: AtomicI64::new(1),
        #[cfg(feature = "qwen-tts")]
        qwen,
    });

    let bind = env_or("POC_BIND", "127.0.0.1:6210");
    let app = Router::new()
        .route("/", get(playground::page))
        .route("/healthz", get(healthz))
        .route("/webrtc/offer", post(call::offer))
        .route("/webrtc/events/{pc_id}", get(events_ws))
        .with_state(state.clone());

    tracing::info!(%bind, "flowcat-poc listening");
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    // ConnectInfo gives the offer handler the caller's address so it can
    // advertise a reachable ICE candidate (POC_BIND=0.0.0.0:6210 for the LAN).
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    // Shutdown: release the model (its ~17 GB returns), then stop the serve
    // we spawned. A hard exit backstop covers a wedged unload or connection.
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        tracing::warn!("shutdown backstop: exiting");
        std::process::exit(0);
    });
    if state.cfg.llm_provider == "ollama" && state.cfg.ollama_unload_on_exit {
        let llm = llm_ollama::OllamaLlm::new(state.cfg.llm_base_url.clone(), state.cfg.llm_model.clone());
        match llm.unload().await {
            Ok(()) => tracing::info!(model = %state.cfg.llm_model, "ollama: model unloaded"),
            Err(error) => tracing::warn!(%error, "ollama: unload failed"),
        }
    }
    if let Some(serve) = ollama_serve {
        serve.shutdown().await;
    }
    Ok(())
}

/// SIGTERM (`run_poc.sh down`) or ctrl-c.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutdown signal received");
}

/// ADR-0007 Layers 1+2 at start-up: a serve (ours or existing), the model
/// pulled, the exact prefix warmed, residency verified. Returns the supervisor
/// handle so shutdown can stop what we started.
async fn start_ollama(
    cfg: &PocConfig,
    session: &StubSession,
    poc_dir: &std::path::Path,
) -> Result<ollama_serve::OllamaServe, Box<dyn std::error::Error>> {
    use flowcat_core::SessionSource;

    let started = std::time::Instant::now();
    let serve = ollama_serve::OllamaServe::ensure(
        &cfg.llm_base_url,
        cfg.ollama_supervise,
        &cfg.ollama_bin,
        cfg.llm_num_ctx,
        &poc_dir.join("logs/ollama.log"),
    )
    .await?;
    let llm = llm_ollama::OllamaLlm::new(serve.base_url(), cfg.llm_model.clone())
        .num_ctx(cfg.llm_num_ctx);
    match llm.version().await {
        Ok(v) => {
            let minor = v.split('.').nth(1).and_then(|m| m.parse::<u32>().ok());
            if minor.map(|m| m < 32).unwrap_or(false) {
                tracing::warn!(version = %v, "ollama older than 0.32: prompt-cache accounting differs");
            } else {
                tracing::info!(version = %v, "ollama serve");
            }
        }
        Err(error) => tracing::warn!(%error, "ollama: version check failed"),
    }
    if !llm.has_model().await? {
        tracing::info!(model = %cfg.llm_model, "ollama: pulling model");
        let status = tokio::process::Command::new(&cfg.ollama_bin)
            .args(["pull", &cfg.llm_model])
            .env("OLLAMA_HOST", serve.base_url())
            .status()
            .await?;
        if !status.success() {
            return Err(format!("`ollama pull {}` failed: {status}", cfg.llm_model).into());
        }
    }
    // The prefix every call sends: the brain's system prompt + the session's
    // advertised tools (name-sorted inside the service).
    let tools = session.node_tools(0, "poc", "babel").await?;
    let report = llm.warm(&cfg.system_prompt, &tools).await?;
    match llm.residency().await? {
        Some(r) if r.pinned && r.context_length == u64::from(cfg.llm_num_ctx) => {
            tracing::info!(
                model = %cfg.llm_model, load_ms = report.load_ms, prompt_eval_ms = report.prompt_eval_ms,
                prompt_tokens = report.prompt_tokens, total_ms = report.total_ms,
                size_vram_mb = r.size_vram / (1 << 20), elapsed_s = started.elapsed().as_secs_f32(),
                "ollama: model warm and resident (pinned, ctx verified)"
            );
        }
        Some(r) => {
            return Err(format!(
                "ollama: model resident but not as requested (pinned={}, context_length={}, want {}): \
                 the serve overrides native request options; check its OLLAMA_* env and version",
                r.pinned, r.context_length, cfg.llm_num_ctx
            )
            .into())
        }
        None => return Err(format!("ollama: {} not resident after warm", cfg.llm_model).into()),
    }
    Ok(serve)
}

async fn healthz(State(state): State<Arc<PocState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "stt": {"backend": state.cfg.stt_backend.as_str(), "status": "ready"}
    }))
}

async fn events_ws(
    State(state): State<Arc<PocState>>,
    Path(pc_id): Path<String>,
    ws: WebSocketUpgrade,
) -> Response {
    match state.registry.take_receiver(&pc_id) {
        Some(rx) => ws.on_upgrade(move |socket| stream_events(socket, rx)),
        None => (axum::http::StatusCode::NOT_FOUND, "no such call").into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stt_backend_parser_accepts_only_supported_local_engines() {
        assert_eq!(SttBackend::parse("whisper").unwrap(), SttBackend::Whisper);
        assert_eq!(
            SttBackend::parse(" MOONSHINE ").unwrap(),
            SttBackend::Moonshine
        );
        assert_eq!(SttBackend::parse("nemotron").unwrap(), SttBackend::Nemotron);
        assert_eq!(SttBackend::parse("NVIDIA").unwrap(), SttBackend::Nemotron);
        let error = SttBackend::parse("cloud").expect_err("cloud STT is unsupported");
        assert!(error.contains("POC_STT_BACKEND"));
    }
}

/// Start poc-qwen-streaming's engine in-process, wait for its preload (model
/// load + warm-up + preset voice priming, ~11 s), resolve the configured voice
/// from the engine catalog, and cache the `Ready.` greeting.
#[cfg(feature = "qwen-tts")]
async fn start_qwen(cfg: &PocConfig) -> Result<tts_qwen::QwenShared, Box<dyn std::error::Error>> {
    use poc_qwen_streaming::config::Config as QwenConfig;
    use poc_qwen_streaming::engine::Engine;

    let started = std::time::Instant::now();
    let qcfg = QwenConfig::load(std::path::Path::new(&cfg.qwen_config))
        .map_err(|e| format!("qwen config: {e}"))?;
    // Engine::start blocks until the Python bridge is constructed.
    let engine = tokio::task::spawn_blocking(move || Engine::start(&qcfg))
        .await?
        .map_err(|e| format!("qwen engine: {e}"))?;
    engine.preload().await.map_err(|e| format!("qwen preload: {e}"))?;
    loop {
        let info = engine.info().await.map_err(|e| format!("qwen info: {e}"))?;
        let state = info["preload"]["state"].as_str().unwrap_or("");
        if state == "done" {
            let errors = &info["preload"]["errors"];
            if errors.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                tracing::warn!(%errors, "qwen preload reported errors");
            }
            tracing::info!(
                active_gb = %info["active_gb"], peak_gb = %info["peak_gb"],
                elapsed_s = started.elapsed().as_secs_f32(), "qwen engine preloaded"
            );
            break;
        }
        if started.elapsed() > std::time::Duration::from_secs(900) {
            return Err("qwen preload did not finish within 900 s".into());
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
    let catalog = engine.catalog().await.map_err(|e| format!("qwen catalog: {e}"))?;
    let entry = catalog["voices"]
        .as_array()
        .and_then(|vs| vs.iter().find(|v| v["name"] == cfg.qwen_voice.as_str()))
        .ok_or_else(|| format!("POC_QWEN_VOICE {:?} not in the engine's voices/ catalog", cfg.qwen_voice))?;
    let ref_text = entry["transcript"].as_str().unwrap_or("").trim().to_string();
    if ref_text.is_empty() {
        return Err(format!("voice {:?} has no sidecar transcript (voices/<name>.txt)", cfg.qwen_voice).into());
    }
    let voice = tts_qwen::QwenVoice {
        name: cfg.qwen_voice.clone(),
        ref_audio: entry["path"].as_str().unwrap_or("").to_string(),
        ref_text,
        size: cfg.qwen_size.clone(),
        language: "English".to_string(),
        interval_s: cfg.qwen_interval_s,
    };
    let greeting = tts_qwen::synthesize_pcm(&engine, &voice, "Ready.").await?;
    tracing::info!(samples = greeting.len(), voice = %voice.name, "qwen greeting cached");
    Ok(tts_qwen::QwenShared {
        engine,
        voice: Arc::new(voice),
        ready_pcm: (!greeting.is_empty()).then(|| Arc::from(greeting.into_boxed_slice())),
    })
}
