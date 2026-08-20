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
#[cfg(feature = "moonshine")]
mod moonshine;
mod nemotron;
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
    pub llm_model: String,
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
}

pub struct PocState {
    pub cfg: PocConfig,
    pub registry: Arc<EventRegistry>,
    pub session: Arc<StubSession>,
    pub stt: LoadedStt,
    pub ready_pcm: Option<Arc<[i16]>>,
    pub next_run: AtomicI64,
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
        openrouter_key: std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| "OPENROUTER_API_KEY not set (see poc/.env)")?,
        llm_model: env_or("POC_LLM_MODEL", "anthropic/claude-haiku-4.5"),
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
    require_nonempty(&cfg.openrouter_key, "OPENROUTER_API_KEY")?;
    require_nonempty(&cfg.llm_model, "POC_LLM_MODEL")?;
    match cfg.tts_backend.as_str() {
        "kokoro" => {}
        "chatterbox" => {
            require_nonempty(&cfg.chatterbox_url, "POC_CHATTERBOX_URL")?;
            require_nonempty(&cfg.chatterbox_voice, "POC_CHATTERBOX_VOICE")?;
        }
        other => {
            return Err(format!(
                "unsupported POC_TTS_BACKEND {other:?} (expected \"kokoro\" or \"chatterbox\")"
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

    let state = Arc::new(PocState {
        cfg,
        registry: Arc::new(EventRegistry::new()),
        session: Arc::new(session),
        stt: loaded_stt,
        ready_pcm,
        next_run: AtomicI64::new(1),
    });

    let bind = env_or("POC_BIND", "127.0.0.1:6210");
    let app = Router::new()
        .route("/", get(playground::page))
        .route("/healthz", get(healthz))
        .route("/webrtc/offer", post(call::offer))
        .route("/webrtc/events/{pc_id}", get(events_ws))
        .with_state(state);

    tracing::info!(%bind, "flowcat-poc listening");
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
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
