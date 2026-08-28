//! axum server: static UI, a few JSON endpoints, and the /ws streaming socket.
//!
//! WebSocket protocol (one generation at a time per socket):
//!   client → {"type":"generate","tab":"clone"|"custom"|"design", ...params}
//!   client → {"type":"cancel"}
//!   server → {"type":"start","sample_rate":24000,"model":"..."}
//!   server → <binary> int16 LE mono PCM frames, as the model emits them
//!   server → {"type":"done","timings":{ttfa_s,gen_s,audio_s,rtf,chunks,...}}
//!   server → {"type":"error","message":"..."}

use anyhow::Result;
use axum::{
    body::Body,
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        Multipart, Path as AxPath, State,
    },
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{path::PathBuf, sync::Arc};
use tower_http::services::ServeDir;

use crate::config::Config;
use crate::engine::{Engine, StreamEvent};
use crate::pcm::i16_to_le_bytes;

#[derive(Clone)]
pub struct AppState {
    pub engine: Engine,
    pub uploads: PathBuf,
    pub runs: PathBuf,
}

pub fn router(cfg: &Config, engine: Engine) -> Router {
    let state = Arc::new(AppState {
        engine,
        uploads: cfg.uploads_dir(),
        runs: cfg.reports_dir().join("rs_runs.jsonl"),
    });
    Router::new()
        .route("/api/info", get(info))
        .route("/api/catalog", get(catalog))
        .route("/api/upload", post(upload))
        .route("/api/transcribe", post(transcribe))
        .route("/api/unload", post(unload))
        .route("/voice/{name}", get(voice_clip))
        .route("/ws", get(ws_upgrade))
        .fallback_service(ServeDir::new(cfg.ui_dir()).append_index_html_on_directories(true))
        .with_state(state)
}

type Shared = State<Arc<AppState>>;

fn err(e: anyhow::Error) -> Response {
    (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))).into_response()
}

async fn info(State(s): Shared) -> Response {
    match s.engine.info().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn catalog(State(s): Shared) -> Response {
    match s.engine.catalog().await {
        Ok(v) => Json(v).into_response(),
        Err(e) => err(e),
    }
}

async fn unload(State(s): Shared) -> Response {
    match s.engine.unload().await {
        Ok(()) => Json(json!({"ok": true})).into_response(),
        Err(e) => err(e),
    }
}

/// Reference clip from the browser (upload or mic, already a WAV). Saved under
/// uploads/ and referred to by path from then on, like Gradio's filepath mode.
async fn upload(State(s): Shared, mut mp: Multipart) -> Response {
    while let Ok(Some(field)) = mp.next_field().await {
        if field.name() != Some("file") {
            continue;
        }
        let ext = field
            .file_name()
            .and_then(|n| std::path::Path::new(n).extension().map(|e| e.to_string_lossy().to_string()))
            .unwrap_or_else(|| "wav".into());
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        };
        if let Err(e) = tokio::fs::create_dir_all(&s.uploads).await {
            return err(e.into());
        }
        let path = s.uploads.join(format!("{}.{ext}", uuid::Uuid::new_v4()));
        if let Err(e) = tokio::fs::write(&path, &bytes).await {
            return err(e.into());
        }
        return Json(json!({"path": path.to_string_lossy(), "bytes": bytes.len()})).into_response();
    }
    (StatusCode::BAD_REQUEST, "no 'file' field").into_response()
}

#[derive(Deserialize)]
struct TranscribeReq {
    path: String,
}

async fn transcribe(State(s): Shared, Json(req): Json<TranscribeReq>) -> Response {
    match s.engine.transcribe(&req.path).await {
        Ok(text) => Json(json!({"text": text})).into_response(),
        Err(e) => err(e),
    }
}

/// Serve a preset clip so the browser can play the reference (paths live outside ui/).
async fn voice_clip(State(s): Shared, AxPath(name): AxPath<String>) -> Response {
    let Ok(Some(path)) = s.engine.voice_path(&name).await else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let mime = match std::path::Path::new(&path).extension().and_then(|e| e.to_str()) {
        Some("mp3") => "audio/mpeg",
        Some("flac") => "audio/flac",
        Some("ogg") => "audio/ogg",
        Some("m4a") => "audio/mp4",
        _ => "audio/wav",
    };
    match tokio::fs::read(&path).await {
        Ok(bytes) => ([(header::CONTENT_TYPE, mime)], Body::from(bytes)).into_response(),
        Err(_) => StatusCode::NOT_FOUND.into_response(),
    }
}

async fn ws_upgrade(State(s): Shared, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |socket| async move {
        if let Err(e) = ws_session(socket, s).await {
            tracing::debug!("ws session ended: {e}");
        }
    })
}

async fn ws_session(mut ws: WebSocket, s: Arc<AppState>) -> Result<()> {
    let mut rx: Option<tokio::sync::mpsc::Receiver<StreamEvent>> = None;
    loop {
        tokio::select! {
            msg = ws.recv() => {
                let Some(Ok(msg)) = msg else { break };
                match msg {
                    Message::Text(text) => {
                        let v: Value = match serde_json::from_str(&text) {
                            Ok(v) => v,
                            Err(e) => { send_json(&mut ws, json!({"type":"error","message": format!("bad json: {e}")})).await?; continue; }
                        };
                        match v.get("type").and_then(Value::as_str) {
                            Some("generate") => {
                                if rx.is_some() {
                                    send_json(&mut ws, json!({"type":"error","message":"a generation is already running on this socket"})).await?;
                                    continue;
                                }
                                let tab = v.get("tab").and_then(Value::as_str).unwrap_or("custom").to_string();
                                // Resolve a preset name to its path so the browser never sees filesystem paths it did not upload.
                                let mut params = v.clone();
                                if let Some(preset) = v.get("preset").and_then(Value::as_str).filter(|p| !p.is_empty()) {
                                    if let Ok(Some(path)) = s.engine.voice_path(preset).await {
                                        params["ref_audio"] = Value::String(path);
                                    }
                                }
                                match s.engine.generate(&tab, params) {
                                    Ok(r) => rx = Some(r),
                                    Err(e) => send_json(&mut ws, json!({"type":"error","message": e.to_string()})).await?,
                                }
                            }
                            Some("cancel") => { rx = None; }
                            _ => {}
                        }
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            }
            ev = async { match rx.as_mut() { Some(r) => r.recv().await, None => std::future::pending().await } } => {
                match ev {
                    Some(StreamEvent::Start { sample_rate, model }) => {
                        send_json(&mut ws, json!({"type":"start","sample_rate":sample_rate,"model":model})).await?;
                    }
                    Some(StreamEvent::Audio { samples, .. }) => {
                        ws.send(Message::Binary(i16_to_le_bytes(&samples).into())).await?;
                    }
                    Some(StreamEvent::Done { timings }) => {
                        record_run(&s.runs, &timings).await;
                        send_json(&mut ws, json!({"type":"done","timings":timings})).await?;
                        rx = None;
                    }
                    Some(StreamEvent::Error(message)) => {
                        send_json(&mut ws, json!({"type":"error","message":message})).await?;
                        rx = None;
                    }
                    None => { rx = None; }
                }
            }
        }
    }
    Ok(())
}

async fn send_json(ws: &mut WebSocket, v: Value) -> Result<()> {
    ws.send(Message::Text(v.to_string().into())).await?;
    Ok(())
}

pub async fn record_run(path: &PathBuf, timings: &Value) {
    let mut row = timings.clone();
    row["ts"] = json!(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0));
    if let Some(dir) = path.parent() {
        let _ = tokio::fs::create_dir_all(dir).await;
    }
    if let Ok(mut f) = tokio::fs::OpenOptions::new().create(true).append(true).open(path).await {
        use tokio::io::AsyncWriteExt;
        let _ = f.write_all(format!("{row}\n").as_bytes()).await;
    }
}
