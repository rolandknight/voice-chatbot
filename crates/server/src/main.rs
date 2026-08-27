//! voice-chatbot-server — the production embedder server (from the FlowCat PoC, docs/poc/flowcat-poc-plan.md).
//!
//! Serves the same surface as flowcat-server's webrtc mode (`GET /` playground,
//! `POST /webrtc/offer`, `GET /webrtc/events/{pc_id}`, `GET /healthz`) but with
//! Babel's shape: a no-graph brain, in-process skills (`skills/`), and
//! directly-constructed local services (selectable local STT, OpenRouter LLM,
//! Kokoro-shim TTS) — bypassing `factory::cascaded`, which can't set Kokoro's
//! base_url and demands dummy API keys for keyless local providers.

mod brain;
mod call;
mod env_file;
mod llm;
mod llm_claude;
mod llm_ollama;
mod media;
#[cfg(feature = "moonshine")]
mod moonshine;
mod nemotron;
#[cfg(feature = "nemotron-native")]
mod nemotron_native;
mod ollama_serve;
mod paced_transport;
mod playground;
mod session;
mod skills;
mod spotify_login;
mod stt;
mod tts_chatterbox;
#[cfg(feature = "qwen-tts")]
mod tts_qwen;
mod wake;

use std::sync::atomic::AtomicI64;
use std::sync::Arc;

use axum::extract::ws::WebSocketUpgrade;
use axum::extract::{Path, State};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::json;

use flowcat_server::events::EventRegistry;

use crate::session::SkillSession;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SttBackend {
    Whisper,
    Moonshine,
    /// In-process NeMo-Speech.cpp (feature `nemotron-native`); falls back to
    /// the sidecar when the binary was built without it.
    Nemotron,
    /// The localhost NeMo-Speech.cpp WebSocket sidecar (`scripts/start_nemotron.sh`).
    NemotronSidecar,
}

impl SttBackend {
    fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "whisper" => Ok(Self::Whisper),
            "moonshine" => Ok(Self::Moonshine),
            "nemotron" | "nvidia" => Ok(if cfg!(feature = "nemotron-native") {
                Self::Nemotron
            } else {
                Self::NemotronSidecar
            }),
            "nemotron-sidecar" => Ok(Self::NemotronSidecar),
            _ => Err(format!(
                "unsupported POC_STT_BACKEND {value:?} (expected \"whisper\", \"moonshine\", \"nemotron\", or \"nemotron-sidecar\")"
            )),
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Whisper => "whisper",
            Self::Moonshine => "moonshine",
            Self::Nemotron => "nemotron",
            Self::NemotronSidecar => "nemotron-sidecar",
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
    /// The same model loaded in-process; calls create native streams.
    #[cfg(feature = "nemotron-native")]
    NemotronNative(nemotron_native::SharedNemotronEngine),
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
    /// Bind address for a spawned serve: `127.0.0.1` (default) or `0.0.0.0` to
    /// share the model with the rest of the LAN. Ignored for an existing serve.
    pub ollama_host: String,
    /// Release the model on exit (`keep_alive: 0`) so its memory returns.
    pub ollama_unload_on_exit: bool,
    /// Keep-warm period: a one-token prefix request every N seconds while idle
    /// (0 = off). Ollama's scheduler re-evaluates fit on the first request after
    /// a long idle and may evict + reload the resident model when system memory
    /// has shifted (16 s, observed on a user's turn); pinging keeps that on the
    /// background path and the prompt cache hot.
    pub ollama_keepwarm_secs: u64,
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
    /// In-process Nemotron: GGUF path, device (`auto|metal|cpu|cuda:N`), RNNT
    /// right-context frames (6 = 560 ms window, the sidecar's default).
    pub nemotron_model: String,
    pub nemotron_device: String,
    pub nemotron_right_context: i32,
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
    /// Qwen3-TTS (`qwen`): engine profile (crates/qwen-tts/config/server.yaml), preset
    /// voice name, model size, streamed-chunk interval.
    pub qwen_config: String,
    pub qwen_voice: String,
    /// Extra preset voices to load for `switch_persona` (comma-separated;
    /// `qwen_voice` is always first).
    pub qwen_voices: Vec<String>,
    pub qwen_size: String,
    /// `ask_claude`: Anthropic API key (empty → the tool is not advertised) and model.
    pub anthropic_key: String,
    pub claude_model: String,
    pub qwen_interval_s: f64,
    /// Host ICE candidate address to advertise (POC_ADVERTISE_IP). None →
    /// the interface that routes back to each caller.
    pub advertise_ip: Option<std::net::IpAddr>,
    /// Optional HTTPS listener (POC_TLS_BIND + POC_TLS_CERT/KEY): browsers only
    /// expose getUserMedia on secure origins, so a LAN browser needs this; the
    /// plain listener stays for the harness and the native client.
    pub tls_bind: String,
    pub tls_cert: String,
    pub tls_key: String,
}

pub struct PocState {
    pub cfg: PocConfig,
    pub registry: Arc<EventRegistry>,
    pub session: Arc<SkillSession>,
    /// Generated sound-effect clips, served at `/sfx/{file}` for the client.
    pub sfx_dir: std::path::PathBuf,
    pub stt: LoadedStt,
    pub ready_pcm: Option<Arc<[i16]>>,
    pub next_run: AtomicI64,
    /// Hang-up hooks by pc_id: notified when the call's events WebSocket
    /// closes, which cancels the pipeline at once instead of waiting for the
    /// ICE timeout (10–30 s) — long enough to exhaust the Nemotron sidecar's
    /// two realtime sessions on quick reconnects.
    pub hangups: std::sync::Mutex<std::collections::HashMap<String, Arc<tokio::sync::Notify>>>,
    /// Shared Qwen engine + voice when `POC_TTS_BACKEND=qwen`.
    #[cfg(feature = "qwen-tts")]
    pub qwen: Option<tts_qwen::QwenShared>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Boolean env switch: `off|false|0|no` → false, `on|true|1|yes` → true.
fn env_flag(key: &str, default: bool) -> bool {
    match env_or(key, "").trim().to_ascii_lowercase().as_str() {
        "" => default,
        "off" | "false" | "0" | "no" => false,
        _ => true,
    }
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
    // poc/.env holds the server profile; the repo-root .env holds the shared
    // secrets (Spotify, search keys). Neither overrides variables already set.
    env_file::load_if_unset(std::path::Path::new("poc/.env"));
    env_file::load_if_unset(std::path::Path::new(".env"));
    if let Some(pos) = std::env::args().position(|a| a == "spotify-login") {
        let headless = std::env::args().skip(pos + 1).any(|a| a == "--headless");
        return spotify_login::run(
            &env_or("SPOTIPY_CLIENT_ID", ""),
            &env_or("SPOTIPY_REDIRECT_URI", "http://127.0.0.1:8765/callback"),
            headless,
        )
        .await
        .map_err(Into::into);
    }

    // crates/server. Runtime artifacts (downloaded models, stubs, logs) still
    // live under poc/ until the production layout for them is decided.
    let manifest_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let poc_dir = manifest_dir.parent().unwrap().parent().unwrap().join("poc");
    let poc_dir = poc_dir.as_path();
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
        ollama_host: env_or("POC_OLLAMA_HOST", "127.0.0.1"),
        ollama_unload_on_exit: env_or("POC_OLLAMA_UNLOAD_ON_EXIT", "true")
            .parse::<bool>()
            .map_err(|error| format!("invalid POC_OLLAMA_UNLOAD_ON_EXIT: {error}"))?,
        ollama_keepwarm_secs: env_or("POC_OLLAMA_KEEPWARM_SECS", "60")
            .parse::<u64>()
            .map_err(|error| format!("invalid POC_OLLAMA_KEEPWARM_SECS: {error}"))?,
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
        nemotron_model: env_or(
            "POC_NEMOTRON_MODEL",
            &poc_dir
                .join("models/nemotron/nvidia/nemotron-speech-streaming-en-0.6b/ebe59e5a817142986528bbbee5dba8db7b38ed50/nemotron-speech-streaming-en-0.6b.q8_0.gguf")
                .to_string_lossy(),
        ),
        nemotron_device: env_or("POC_NEMOTRON_DEVICE", "auto"),
        nemotron_right_context: env_or("POC_NEMOTRON_RIGHT_CONTEXT", "6")
            .parse::<i32>()
            .map_err(|error| format!("invalid POC_NEMOTRON_RIGHT_CONTEXT: {error}"))?,
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
            &manifest_dir.join("prompt.txt").to_string_lossy(),
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
            &manifest_dir
                .join("../qwen-tts/config/server.yaml")
                .to_string_lossy(),
        ),
        qwen_voice: env_or("POC_QWEN_VOICE", "babel"),
        qwen_voices: env_or("POC_QWEN_VOICES", "")
            .split(',')
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .collect(),
        anthropic_key: env_or("ANTHROPIC_API_KEY", ""),
        claude_model: env_or("POC_CLAUDE_MODEL", "claude-opus-5"),
        qwen_size: env_or("POC_QWEN_SIZE", "1.7B"),
        qwen_interval_s: env_or("POC_QWEN_INTERVAL_S", "0.32")
            .parse::<f64>()
            .map_err(|error| format!("invalid POC_QWEN_INTERVAL_S: {error}"))?,
        tls_bind: env_or("POC_TLS_BIND", ""),
        tls_cert: env_or("POC_TLS_CERT", ""),
        tls_key: env_or("POC_TLS_KEY", ""),
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
    if cfg.stt_backend == SttBackend::NemotronSidecar {
        require_nonempty(&cfg.nemotron_url, "POC_NEMOTRON_URL")?;
    }
    if cfg.stt_backend == SttBackend::Nemotron
        && !std::path::Path::new(&cfg.nemotron_model).exists()
    {
        return Err(format!(
            "nemotron model missing: {} (run ./scripts/setup_nemotron.sh)",
            cfg.nemotron_model
        )
        .into());
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

    let sfx_dir = poc_dir.join("logs/sfx");
    let (registry, calls) = build_skills(&cfg, sfx_dir.clone())?;
    let session = SkillSession::new(registry, calls, poc_dir.join("logs/artifacts"));

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
            #[cfg(feature = "nemotron-native")]
            {
                let device = nemotron_native::Device::parse(&cfg.nemotron_device)?;
                LoadedStt::NemotronNative(nemotron_native::load_engine(
                    &cfg.nemotron_model,
                    device,
                    cfg.nemotron_right_context,
                )?)
            }
            #[cfg(not(feature = "nemotron-native"))]
            {
                unreachable!("SttBackend::parse maps nemotron to the sidecar without the feature")
            }
        }
        SttBackend::NemotronSidecar => {
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
        sfx_dir,
        stt: loaded_stt,
        ready_pcm,
        next_run: AtomicI64::new(1),
        hangups: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "qwen-tts")]
        qwen,
    });

    // Keep-warm: while no call is active, re-send the prefix every N seconds.
    if state.cfg.llm_provider == "ollama" && state.cfg.ollama_keepwarm_secs > 0 {
        let st = state.clone();
        tokio::spawn(async move {
            let period = std::time::Duration::from_secs(st.cfg.ollama_keepwarm_secs);
            let llm =
                llm_ollama::OllamaLlm::new(st.cfg.llm_base_url.clone(), st.cfg.llm_model.clone())
                    .num_ctx(st.cfg.llm_num_ctx);
            let tools = match flowcat_core::SessionSource::node_tools(
                &*st.session,
                0,
                "poc",
                "babel",
            )
            .await
            {
                Ok(t) => t,
                Err(error) => {
                    tracing::warn!(%error, "keep-warm: no tools; disabled");
                    return;
                }
            };
            loop {
                tokio::time::sleep(period).await;
                if !st.hangups.lock().unwrap().is_empty() {
                    continue; // a call is active; never queue behind a real turn
                }
                match llm.warm(&st.cfg.system_prompt, &tools).await {
                    Ok(r) if r.load_ms > 0 || r.prompt_eval_ms > 500 => tracing::info!(
                        load_ms = r.load_ms,
                        prompt_eval_ms = r.prompt_eval_ms,
                        "keep-warm: model was cold (reloaded/re-prefilled in the background)"
                    ),
                    Ok(r) => tracing::debug!(prompt_eval_ms = r.prompt_eval_ms, "keep-warm ok"),
                    Err(error) => tracing::warn!(%error, "keep-warm request failed"),
                }
            }
        });
    }

    let bind = env_or("POC_BIND", "127.0.0.1:6210");
    let app = Router::new()
        .route("/", get(playground::page))
        .route("/healthz", get(healthz))
        .route("/webrtc/offer", post(call::offer))
        .route("/webrtc/events/{pc_id}", get(events_ws))
        .route("/sfx/{file}", get(sfx_file))
        .with_state(state.clone());

    tracing::info!(%bind, "voice-chatbot-server listening");
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    // ConnectInfo gives the offer handler the caller's address so it can
    // advertise a reachable ICE candidate (POC_BIND=0.0.0.0:6210 for the LAN).
    // Optional HTTPS twin of the same router for LAN browsers (secure context).
    let tls_handle = axum_server::Handle::new();
    let tls_task = if state.cfg.tls_bind.trim().is_empty() {
        None
    } else {
        let cfg = &state.cfg;
        if cfg.tls_cert.is_empty() || cfg.tls_key.is_empty() {
            return Err("POC_TLS_BIND needs POC_TLS_CERT and POC_TLS_KEY (make tls-cert)".into());
        }
        let rustls_cfg =
            axum_server::tls_rustls::RustlsConfig::from_pem_file(&cfg.tls_cert, &cfg.tls_key)
                .await
                .map_err(|e| {
                    format!("load TLS cert/key ({}, {}): {e}", cfg.tls_cert, cfg.tls_key)
                })?;
        let addr: std::net::SocketAddr = cfg
            .tls_bind
            .parse()
            .map_err(|e| format!("invalid POC_TLS_BIND: {e}"))?;
        tracing::info!(%addr, cert = %cfg.tls_cert, "voice-chatbot-server listening (https)");
        let app = app.clone();
        let handle = tls_handle.clone();
        Some(tokio::spawn(async move {
            axum_server::bind_rustls(addr, rustls_cfg)
                .handle(handle)
                .serve(app.into_make_service_with_connect_info::<std::net::SocketAddr>())
                .await
        }))
    };
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    if let Some(task) = tls_task {
        tls_handle.graceful_shutdown(Some(std::time::Duration::from_secs(2)));
        let _ = task.await;
    }

    // Shutdown: release the model (its ~17 GB returns), then stop the serve
    // we spawned. A hard exit backstop covers a wedged unload or connection.
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_secs(15)).await;
        tracing::warn!("shutdown backstop: exiting");
        std::process::exit(0);
    });
    if state.cfg.llm_provider == "ollama" && state.cfg.ollama_unload_on_exit {
        let llm =
            llm_ollama::OllamaLlm::new(state.cfg.llm_base_url.clone(), state.cfg.llm_model.clone());
        match llm.unload().await {
            Ok(()) => tracing::info!(model = %state.cfg.llm_model, "ollama: model unloaded"),
            Err(error) => tracing::warn!(%error, "ollama: unload failed"),
        }
    }
    if let Some(serve) = ollama_serve {
        serve.shutdown().await;
    }
    // The embedded interpreter is never finalized, so run its exit handlers
    // ourselves; otherwise multiprocessing's resource_tracker reports the
    // semaphores libraries created at import as leaked.
    #[cfg(feature = "qwen-tts")]
    if let Some(qwen) = &state.qwen {
        match qwen.engine.shutdown().await {
            Ok(()) => tracing::info!("qwen engine: python exit handlers run"),
            Err(error) => tracing::warn!(%error, "qwen engine: shutdown failed"),
        }
    }
    Ok(())
}

/// Forward call events to the subscriber and return when either side ends.
/// Unlike `stream_events`, this also *reads* the socket, so a client that
/// closes it (browser/native-client hang-up) is noticed immediately rather
/// than at the next published event.
async fn stream_events_until_closed(
    mut socket: axum::extract::ws::WebSocket,
    mut rx: tokio::sync::mpsc::UnboundedReceiver<String>,
) {
    use axum::extract::ws::Message;
    loop {
        tokio::select! {
            frame = rx.recv() => match frame {
                Some(text) => {
                    if socket.send(Message::Text(text.into())).await.is_err() {
                        break;
                    }
                }
                None => break, // call over
            },
            incoming = socket.recv() => match incoming {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                Some(Ok(_)) => {} // pings/pongs/client chatter
            },
        }
    }
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
    session: &SkillSession,
    poc_dir: &std::path::Path,
) -> Result<ollama_serve::OllamaServe, Box<dyn std::error::Error>> {
    use flowcat_core::SessionSource;

    let started = std::time::Instant::now();
    let serve = ollama_serve::OllamaServe::ensure(
        &cfg.llm_base_url,
        cfg.ollama_supervise,
        &cfg.ollama_bin,
        cfg.llm_num_ctx,
        &cfg.ollama_host,
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

/// `POC_QWEN_VOICE` followed by `POC_QWEN_VOICES` (deduplicated), when the
/// TTS backend is Qwen; otherwise just the one configured voice.
fn qwen_persona_names(cfg: &PocConfig) -> Vec<String> {
    let mut names = vec![cfg.qwen_voice.clone()];
    if cfg.tts_backend == "qwen" {
        for v in &cfg.qwen_voices {
            if !names.contains(v) {
                names.push(v.clone());
            }
        }
    }
    names
}

/// Construct the skills this process runs. Gating happens here, once: a skill
/// that is not built is not advertised (docs/plans/skills-in-server.md).
fn build_skills(
    cfg: &PocConfig,
    sfx_dir: std::path::PathBuf,
) -> Result<(skills::Registry, skills::CallRegistry), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    let mut list: Vec<Arc<dyn skills::Skill>> = vec![
        Arc::new(skills::time::GetCurrentTime),
        Arc::new(skills::time::GetCurrentDate),
        Arc::new(skills::timer::SetTimer),
        Arc::new(skills::weather::GetWeather::new(env_or(
            "POC_WEATHER_DEFAULT_LOCATION",
            "",
        ))),
    ];
    let provider = skills::web_search::Provider::parse(&env_or("POC_WEB_SEARCH_PROVIDER", ""))?;
    list.push(Arc::new(skills::web_search::WebSearch::new(
        provider,
        env_or("BRAVE_API_KEY", ""),
        env_or("TAVILY_API_KEY", ""),
    )));
    // Playback happens on the native client (mpv); the browser playground has
    // no media, so these can be switched off for browser-only setups.
    if env_flag("POC_SKILLS_RADIO", true) {
        list.push(Arc::new(skills::radio::PlayBbcRadio::new()));
        list.push(Arc::new(skills::radio::StopBbcRadio));
    }
    if env_flag("POC_SKILLS_SHOWS", true) {
        list.push(Arc::new(skills::shows::PlayBbcShow::new()));
    }
    // Spotify needs a client id and a cached PKCE token (`spotify-login`);
    // without them the seven tools are simply not advertised.
    let spotify = if env_flag("POC_SKILLS_SPOTIFY", true) {
        match skills::spotify_client::SpotifyClient::new(env_or("SPOTIPY_CLIENT_ID", "")) {
            Ok(c) => Some(Arc::new(c)),
            Err(reason) => {
                tracing::warn!(%reason, "spotify skills disabled");
                None
            }
        }
    } else {
        None
    };
    if let Some(c) = &spotify {
        list.extend(skills::spotify::all(c.clone()));
    }
    // The generators are separate model servers (`make sfx-up`); the tool is
    // advertised regardless and reports "not running" per call.
    if env_flag("POC_SKILLS_SFX", true) {
        list.push(Arc::new(skills::sfx::GenerateSoundEffect::new(
            env_or("POC_SFX_WOOSH_URL", "http://127.0.0.1:8005"),
            env_or("POC_SFX_SAO_URL", "http://127.0.0.1:8006"),
            skills::sfx::Routing::parse(&env_or("POC_SFX_BACKEND", "auto"))?,
            sfx_dir,
        )));
    }
    // Personas are Qwen presets; with one voice there is nothing to switch to.
    let personas = qwen_persona_names(cfg);
    if personas.len() > 1 {
        list.push(Arc::new(skills::persona::SwitchPersona::new(personas)));
    }
    if !cfg.anthropic_key.trim().is_empty() {
        list.push(Arc::new(skills::claude::AskClaude));
    } else {
        tracing::info!("ask_claude disabled: ANTHROPIC_API_KEY not set");
    }
    Ok((
        skills::Registry::new(include_str!("../skills.json"), list)?,
        skills::CallRegistry::new(spotify),
    ))
}

/// Serve one generated clip to the client (`media play_file` sends `/sfx/<file>`).
async fn sfx_file(State(state): State<Arc<PocState>>, Path(file): Path<String>) -> Response {
    // One path component only: no traversal out of the clips directory.
    if file.is_empty() || file.contains(['/', '\\']) || file.starts_with('.') {
        return axum::http::StatusCode::NOT_FOUND.into_response();
    }
    match tokio::fs::read(state.sfx_dir.join(&file)).await {
        Ok(bytes) => ([(axum::http::header::CONTENT_TYPE, "audio/flac")], bytes).into_response(),
        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
    }
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
        Some(rx) => ws.on_upgrade(move |socket| async move {
            stream_events_until_closed(socket, rx).await;
            // Subscriber gone (hang-up) or call over: cancel the call now.
            let hook = state.hangups.lock().unwrap().remove(&pc_id);
            if let Some(hook) = hook {
                tracing::info!(%pc_id, "events socket closed; ending the call");
                hook.notify_one();
            }
        }),
        None => (axum::http::StatusCode::NOT_FOUND, "no such call").into_response(),
    }
}

/// Start the qwen-tts engine in-process, wait for its preload (model
/// load + warm-up + preset voice priming, ~11 s), resolve the configured voice
/// from the engine catalog, and cache the `Ready.` greeting.
#[cfg(feature = "qwen-tts")]
async fn start_qwen(cfg: &PocConfig) -> Result<tts_qwen::QwenShared, Box<dyn std::error::Error>> {
    use qwen_tts::config::Config as QwenConfig;
    use qwen_tts::engine::Engine;

    let started = std::time::Instant::now();
    let qcfg = QwenConfig::load(std::path::Path::new(&cfg.qwen_config))
        .map_err(|e| format!("qwen config: {e}"))?;
    // Engine::start blocks until the Python bridge is constructed.
    let engine = tokio::task::spawn_blocking(move || Engine::start(&qcfg))
        .await?
        .map_err(|e| format!("qwen engine: {e}"))?;
    engine
        .preload()
        .await
        .map_err(|e| format!("qwen preload: {e}"))?;
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
    let catalog = engine
        .catalog()
        .await
        .map_err(|e| format!("qwen catalog: {e}"))?;
    let mut voices: Vec<Arc<tts_qwen::QwenVoice>> = Vec::new();
    for name in qwen_persona_names(cfg) {
        let entry = catalog["voices"]
            .as_array()
            .and_then(|vs| vs.iter().find(|v| v["name"] == name.as_str()))
            .ok_or_else(|| format!("voice {name:?} (POC_QWEN_VOICE/POC_QWEN_VOICES) not in the engine's voices/ catalog"))?;
        let ref_text = entry["transcript"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();
        if ref_text.is_empty() {
            return Err(
                format!("voice {name:?} has no sidecar transcript (voices/<name>.txt)").into(),
            );
        }
        voices.push(Arc::new(tts_qwen::QwenVoice {
            name,
            ref_audio: entry["path"].as_str().unwrap_or("").to_string(),
            ref_text,
            size: cfg.qwen_size.clone(),
            language: "English".to_string(),
            interval_s: cfg.qwen_interval_s,
        }));
    }
    let voice = voices[0].clone();
    let greeting = tts_qwen::synthesize_pcm(&engine, &voice, "Ready.").await?;
    tracing::info!(samples = greeting.len(), voice = %voice.name, personas = voices.len(), "qwen greeting cached");
    Ok(tts_qwen::QwenShared {
        engine,
        voice,
        voices,
        ready_pcm: (!greeting.is_empty()).then(|| Arc::from(greeting.into_boxed_slice())),
    })
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
