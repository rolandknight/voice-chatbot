//! flowcat-poc — the Phase 1 embedder server (docs/poc/flowcat-poc-plan.md).
//!
//! Serves the same surface as flowcat-server's webrtc mode (`GET /` playground,
//! `POST /webrtc/offer`, `GET /webrtc/events/{pc_id}`, `GET /healthz`) but with
//! Babel's shape: a no-graph brain, skills relayed to the local stub server, and
//! directly-constructed local services (whisper.cpp STT, OpenRouter LLM,
//! Kokoro-shim TTS) — bypassing `factory::cascaded`, which can't set Kokoro's
//! base_url and demands dummy API keys for keyless local providers.

mod brain;
mod call;
mod session;

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

pub struct PocConfig {
    pub openrouter_key: String,
    pub llm_model: String,
    pub whisper_model: String,
    pub kokoro_url: String,
    pub kokoro_voice: String,
    pub system_prompt: String,
}

pub struct PocState {
    pub cfg: PocConfig,
    pub registry: Arc<EventRegistry>,
    pub session: Arc<StubSession>,
    pub next_run: AtomicI64,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
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

    let cfg = PocConfig {
        openrouter_key: std::env::var("OPENROUTER_API_KEY")
            .map_err(|_| "OPENROUTER_API_KEY not set (see poc/.env)")?,
        llm_model: env_or("POC_LLM_MODEL", "google/gemma-4-26b-a4b-it:free"),
        whisper_model: env_or(
            "POC_WHISPER_MODEL",
            &poc_dir.join("models/ggml-tiny.en.bin").to_string_lossy(),
        ),
        kokoro_url: env_or("POC_KOKORO_URL", "http://127.0.0.1:8880"),
        kokoro_voice: env_or("POC_KOKORO_VOICE", "af_heart"),
        system_prompt: std::fs::read_to_string(
            env_or("POC_PROMPT", &poc_dir.join("flowcat/prompt.txt").to_string_lossy()),
        )?,
    };
    if !std::path::Path::new(&cfg.whisper_model).exists() {
        return Err(format!("whisper model missing: {}", cfg.whisper_model).into());
    }

    let skills_path = env_or("POC_SKILLS", &poc_dir.join("stubs/skills.json").to_string_lossy());
    let session = StubSession::new(
        &std::fs::read_to_string(&skills_path)?,
        env_or("POC_STUBS_URL", "http://127.0.0.1:8790"),
        poc_dir.join("logs/artifacts"),
    )?;

    let state = Arc::new(PocState {
        cfg,
        registry: Arc::new(EventRegistry::new()),
        session: Arc::new(session),
        next_run: AtomicI64::new(1),
    });

    let bind = env_or("POC_BIND", "127.0.0.1:6210");
    let app = Router::new()
        .route("/", get(flowcat_server::webrtc::playground_page))
        .route("/healthz", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/webrtc/offer", post(call::offer))
        .route("/webrtc/events/{pc_id}", get(events_ws))
        .with_state(state);

    tracing::info!(%bind, "flowcat-poc listening");
    let listener = tokio::net::TcpListener::bind(&bind).await?;
    axum::serve(listener, app).await?;
    Ok(())
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
