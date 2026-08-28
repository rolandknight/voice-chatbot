//! Local NVIDIA Nemotron streaming STT through NeMo-Speech.cpp's WebSocket API.
//!
//! NeMo-Speech.cpp owns the resident model. Each FlowCat call opens one local
//! realtime socket and reuses it across VAD turns. Audio writes are queued to a
//! worker task, so decoder work and socket backpressure never stall the input
//! processor. Server hypotheses are display-only interims; only [`flush`], at
//! FlowCat's external VAD boundary, returns an authoritative final transcript.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use flowcat_core::processor::frame::{AudioFrame, Frame, Language, StartParams};
use flowcat_core::service::SttService;
use flowcat_core::{FlowcatError, Result};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

const NEMOTRON_RATE: u32 = 16_000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const CONTEXT_BOOST: f32 = 3.0;

type RealtimeSocket = WebSocketStream<MaybeTlsStream<TcpStream>>;

/// Derive NeMo-Speech.cpp's realtime socket from its local HTTP base URL.
///
/// `http://127.0.0.1:8178` becomes
/// `ws://127.0.0.1:8178/v1/realtime`. An already-complete realtime URL is
/// accepted too, which is useful for tests and advanced local deployments.
pub fn realtime_url(base_url: &str) -> Result<String> {
    let mut url = reqwest::Url::parse(base_url.trim()).map_err(|error| {
        FlowcatError::Other(format!("invalid Nemotron base URL {base_url:?}: {error}"))
    })?;
    let websocket_scheme = match url.scheme() {
        "http" => "ws",
        "https" => "wss",
        "ws" => "ws",
        "wss" => "wss",
        other => {
            return Err(FlowcatError::Other(format!(
                "unsupported Nemotron URL scheme {other:?} (expected http, https, ws, or wss)"
            )))
        }
    };
    url.set_scheme(websocket_scheme)
        .map_err(|_| FlowcatError::Other("could not set Nemotron WebSocket scheme".into()))?;

    let path = url.path().trim_end_matches('/');
    let realtime_path = match path {
        "" => "/v1/realtime".to_string(),
        "/v1" => "/v1/realtime".to_string(),
        "/v1/realtime" | "/v1/audio/transcriptions/realtime" => path.to_string(),
        _ => format!("{path}/v1/realtime"),
    };
    url.set_path(&realtime_path);
    Ok(url.to_string())
}

/// One call-local NeMo-Speech.cpp stream.
pub struct NemotronStt {
    base_url: String,
    speech_contexts: Vec<String>,
    commands: Option<mpsc::UnboundedSender<WorkerCommand>>,
    updates: Option<mpsc::UnboundedReceiver<Result<Frame>>>,
    worker: Option<JoinHandle<()>>,
    muted: bool,
}

impl NemotronStt {
    pub fn new(base_url: String, speech_contexts: Vec<String>) -> Self {
        let speech_contexts = speech_contexts
            .into_iter()
            .map(|phrase| phrase.trim().to_string())
            .filter(|phrase| !phrase.is_empty())
            .collect();
        Self {
            base_url,
            speech_contexts,
            commands: None,
            updates: None,
            worker: None,
            muted: false,
        }
    }

    fn commands(&self) -> Result<&mpsc::UnboundedSender<WorkerCommand>> {
        self.commands
            .as_ref()
            .ok_or_else(|| FlowcatError::Other("Nemotron STT is not started".into()))
    }

    /// Coalesce pending display hypotheses so a busy pipeline never replays a
    /// backlog of stale text.
    fn drain_updates(&mut self) -> Result<Vec<Frame>> {
        let updates = self
            .updates
            .as_mut()
            .ok_or_else(|| FlowcatError::Other("Nemotron STT is not started".into()))?;
        let mut latest = None;
        let mut first_error = None;
        loop {
            match updates.try_recv() {
                Ok(Ok(frame)) => latest = Some(frame),
                Ok(Err(error)) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    if first_error.is_none() {
                        first_error = Some(FlowcatError::Other(
                            "Nemotron STT worker disconnected".into(),
                        ));
                    }
                    break;
                }
            }
        }
        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(latest.into_iter().collect())
        }
    }

    async fn request_flush(&self) -> Result<Option<String>> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands()?
            .send(WorkerCommand::Flush { reply: reply_tx })
            .map_err(|_| FlowcatError::Other("Nemotron STT worker stopped".into()))?;
        tokio::time::timeout(REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| FlowcatError::Other("Nemotron STT flush timed out".into()))?
            .map_err(|_| FlowcatError::Other("Nemotron STT dropped its flush reply".into()))?
    }

    async fn request_clear(&self) -> Result<()> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.commands()?
            .send(WorkerCommand::Clear { reply: reply_tx })
            .map_err(|_| FlowcatError::Other("Nemotron STT worker stopped".into()))?;
        tokio::time::timeout(REQUEST_TIMEOUT, reply_rx)
            .await
            .map_err(|_| FlowcatError::Other("Nemotron STT clear timed out".into()))?
            .map_err(|_| FlowcatError::Other("Nemotron STT dropped its clear reply".into()))?
    }
}

#[async_trait]
impl SttService for NemotronStt {
    fn name(&self) -> &str {
        "babel-nemotron"
    }

    async fn start(&mut self, _params: &StartParams) -> Result<()> {
        if self.commands.is_some() {
            return Ok(());
        }

        let url = realtime_url(&self.base_url)?;
        let contexts = self.speech_contexts.clone();
        let (commands_tx, commands_rx) = mpsc::unbounded_channel();
        let (updates_tx, updates_rx) = mpsc::unbounded_channel();
        let (started_tx, started_rx) = oneshot::channel();
        let worker = tokio::spawn(run_worker(
            url.clone(),
            contexts,
            commands_rx,
            updates_tx,
            started_tx,
        ));

        match started_rx.await {
            Ok(Ok(())) => {
                self.commands = Some(commands_tx);
                self.updates = Some(updates_rx);
                self.worker = Some(worker);
                tracing::info!(%url, "call-local Nemotron stream initialized");
                Ok(())
            }
            Ok(Err(error)) => {
                worker.abort();
                Err(error)
            }
            Err(_) => {
                worker.abort();
                Err(FlowcatError::Other(
                    "Nemotron STT worker exited during startup".into(),
                ))
            }
        }
    }

    async fn run_stt(&mut self, audio: Arc<AudioFrame>) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(Vec::new());
        }
        if !audio.is_empty() {
            let pcm = pcm16_le(&audio)?;
            self.commands()?
                .send(WorkerCommand::Audio(pcm))
                .map_err(|_| FlowcatError::Other("Nemotron STT worker stopped".into()))?;
        }
        self.drain_updates()
    }

    async fn flush(&mut self) -> Result<Vec<Frame>> {
        if self.muted {
            return Ok(Vec::new());
        }
        let final_result = self.request_flush().await;
        let interim_result = self.drain_updates();
        match final_result {
            Err(error) => Err(error),
            Ok(text) => {
                let mut frames = match interim_result {
                    Ok(frames) => frames,
                    Err(error) => {
                        tracing::warn!(%error, "discarding stale Nemotron interim error after successful final");
                        Vec::new()
                    }
                };
                if let Some(text) = text {
                    tracing::info!(text = %text, "Nemotron utterance finalized");
                    frames.push(final_frame(text));
                }
                Ok(frames)
            }
        }
    }

    async fn set_muted(&mut self, muted: bool) {
        if self.muted == muted {
            return;
        }
        if muted {
            // Stop accepting audio before the ordered clear barrier is queued.
            self.muted = true;
        }
        let clear_result = if self.commands.is_some() {
            self.request_clear().await
        } else {
            Ok(())
        };
        if let Err(error) = clear_result {
            tracing::warn!(%error, muted, "Nemotron mute reset failed");
            return;
        }
        if !muted {
            self.muted = false;
        }
        // Discard any hypothesis delivered immediately before the clear ack.
        if self.updates.is_some() {
            let _ = self.drain_updates();
        }
    }
}

impl Drop for NemotronStt {
    fn drop(&mut self) {
        if let Some(commands) = self.commands.take() {
            let _ = commands.send(WorkerCommand::Shutdown);
        }
        if let Some(worker) = self.worker.take() {
            worker.abort();
        }
    }
}

enum WorkerCommand {
    Audio(Vec<u8>),
    Flush {
        reply: oneshot::Sender<Result<Option<String>>>,
    },
    Clear {
        reply: oneshot::Sender<Result<()>>,
    },
    Shutdown,
}

enum PendingRequest {
    Flush {
        reply: oneshot::Sender<Result<Option<String>>>,
        completed: Option<String>,
        completed_seen: bool,
        committed: bool,
    },
    Clear {
        reply: oneshot::Sender<Result<()>>,
    },
}

async fn run_worker(
    url: String,
    speech_contexts: Vec<String>,
    mut commands: mpsc::UnboundedReceiver<WorkerCommand>,
    updates: mpsc::UnboundedSender<Result<Frame>>,
    started: oneshot::Sender<Result<()>>,
) {
    let mut socket = match connect_session(&url, &speech_contexts).await {
        Ok(socket) => {
            let _ = started.send(Ok(()));
            socket
        }
        Err(error) => {
            let _ = started.send(Err(error));
            return;
        }
    };

    let mut transcript = TranscriptState::default();
    let mut pending: Option<PendingRequest> = None;
    let mut sent_samples: usize = 0;

    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                match command {
                    WorkerCommand::Audio(pcm) => {
                        sent_samples += pcm.len() / 2;
                        if pending.is_some() {
                            let _ = updates.send(Err(FlowcatError::Other(
                                "Nemotron received audio while a stream barrier was pending".into(),
                            )));
                            continue;
                        }
                        if let Err(error) = socket.send(Message::Binary(pcm.into())).await {
                            let _ = updates.send(Err(socket_error("send audio", error)));
                            break;
                        }
                    }
                    WorkerCommand::Flush { reply } => {
                        if pending.is_some() {
                            let _ = reply.send(Err(FlowcatError::Other(
                                "Nemotron flush requested while another barrier was pending".into(),
                            )));
                            continue;
                        }
                        tracing::debug!(
                            seconds = sent_samples as f32 / 16_000.0,
                            "nemotron: committing utterance audio"
                        );
                        sent_samples = 0;
                        let event = json!({"type": "input_audio_buffer.commit"});
                        if let Err(error) = socket.send(Message::Text(event.to_string().into())).await {
                            let _ = reply.send(Err(socket_error("commit audio", error)));
                            break;
                        }
                        pending = Some(PendingRequest::Flush {
                            reply,
                            completed: None,
                            completed_seen: false,
                            committed: false,
                        });
                    }
                    WorkerCommand::Clear { reply } => {
                        if pending.is_some() {
                            let _ = reply.send(Err(FlowcatError::Other(
                                "Nemotron clear requested while another barrier was pending".into(),
                            )));
                            continue;
                        }
                        let event = json!({"type": "input_audio_buffer.clear"});
                        if let Err(error) = socket.send(Message::Text(event.to_string().into())).await {
                            let _ = reply.send(Err(socket_error("clear audio", error)));
                            break;
                        }
                        pending = Some(PendingRequest::Clear { reply });
                    }
                    WorkerCommand::Shutdown => {
                        let _ = socket.close(None).await;
                        break;
                    }
                }
            }
            message = socket.next() => {
                let message = match message {
                    Some(Ok(message)) => message,
                    Some(Err(error)) => {
                        fail_pending(&mut pending, format!("Nemotron realtime receive: {error}"));
                        let _ = updates.send(Err(socket_error("receive event", error)));
                        break;
                    }
                    None => {
                        fail_pending(&mut pending, "Nemotron realtime socket closed".into());
                        let _ = updates.send(Err(FlowcatError::Other(
                            "Nemotron realtime socket closed".into(),
                        )));
                        break;
                    }
                };

                match message {
                    Message::Text(text) => match parse_server_event(text.as_str()) {
                        Ok(ServerEvent::Delta(delta)) => {
                            if let Some(text) = transcript.append_delta(&delta) {
                                let _ = updates.send(Ok(interim_frame(text)));
                            }
                        }
                        Ok(ServerEvent::Completed(text)) => {
                            let flush_ready = match pending.as_mut() {
                                Some(PendingRequest::Flush {
                                    completed,
                                    completed_seen,
                                    committed,
                                    ..
                                }) => {
                                    transcript.replace(&text);
                                    *completed = spoken_text(&text);
                                    *completed_seen = true;
                                    *committed
                                }
                                Some(PendingRequest::Clear { .. }) => {
                                    // The clear is an ordered discard barrier. Never
                                    // publish a hypothesis that raced with it.
                                    false
                                }
                                None => {
                                    // Even if a sidecar was launched with internal
                                    // endpointing, FlowCat's VAD remains authoritative.
                                    if let Some(text) = transcript.replace(&text) {
                                        let _ = updates.send(Ok(interim_frame(text)));
                                    }
                                    false
                                }
                            };
                            if flush_ready {
                                finish_flush(&mut pending, &mut transcript);
                            }
                        }
                        Ok(ServerEvent::Committed) => {
                            let flush_ready = match pending.as_mut() {
                                Some(PendingRequest::Flush {
                                    completed_seen,
                                    committed,
                                    ..
                                }) => {
                                    *committed = true;
                                    *completed_seen
                                }
                                Some(PendingRequest::Clear { .. }) | None => false,
                            };
                            if flush_ready {
                                finish_flush(&mut pending, &mut transcript);
                            } else if matches!(pending, Some(PendingRequest::Clear { .. })) {
                                let Some(PendingRequest::Clear { reply }) = pending.take() else {
                                    unreachable!()
                                };
                                transcript.reset();
                                let _ = reply.send(Err(FlowcatError::Protocol(
                                    "Nemotron committed while clear was pending".into(),
                                )));
                            } else if pending.is_none() {
                                transcript.reset();
                            }
                        }
                        Ok(ServerEvent::Cleared) => {
                            transcript.reset();
                            match pending.take() {
                                Some(PendingRequest::Clear { reply }) => {
                                    let _ = reply.send(Ok(()));
                                }
                                Some(PendingRequest::Flush { reply, .. }) => {
                                    let _ = reply.send(Err(FlowcatError::Protocol(
                                        "Nemotron cleared while flush was pending".into(),
                                    )));
                                }
                                None => {}
                            }
                        }
                        Ok(ServerEvent::Error(message)) => {
                            let message = format!("Nemotron realtime error: {message}");
                            if pending.is_some() {
                                fail_pending(&mut pending, message);
                            } else {
                                let _ = updates.send(Err(FlowcatError::Protocol(message)));
                            }
                        }
                        Ok(ServerEvent::SessionCreated | ServerEvent::SessionUpdated | ServerEvent::Other) => {}
                        Err(error) => {
                            let _ = updates.send(Err(error));
                        }
                    },
                    Message::Ping(payload) => {
                        if let Err(error) = socket.send(Message::Pong(payload)).await {
                            let _ = updates.send(Err(socket_error("send pong", error)));
                            break;
                        }
                    }
                    Message::Close(_) => {
                        fail_pending(&mut pending, "Nemotron realtime socket closed".into());
                        let _ = updates.send(Err(FlowcatError::Other(
                            "Nemotron realtime socket closed".into(),
                        )));
                        break;
                    }
                    Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
                }
            }
        }
    }
}

async fn connect_session(url: &str, speech_contexts: &[String]) -> Result<RealtimeSocket> {
    let (mut socket, _) =
        tokio::time::timeout(CONNECT_TIMEOUT, tokio_tungstenite::connect_async(url))
            .await
            .map_err(|_| FlowcatError::Other(format!("Nemotron connection timed out: {url}")))?
            .map_err(|error| socket_error("connect", error))?;

    wait_for_event(&mut socket, |event| {
        matches!(event, ServerEvent::SessionCreated)
    })
    .await?;
    let update = session_update(speech_contexts);
    socket
        .send(Message::Text(update.to_string().into()))
        .await
        .map_err(|error| socket_error("configure session", error))?;
    wait_for_event(&mut socket, |event| {
        matches!(event, ServerEvent::SessionUpdated)
    })
    .await?;
    Ok(socket)
}

async fn wait_for_event(
    socket: &mut RealtimeSocket,
    wanted: impl Fn(&ServerEvent) -> bool,
) -> Result<()> {
    tokio::time::timeout(CONNECT_TIMEOUT, async {
        loop {
            let message = socket
                .next()
                .await
                .ok_or_else(|| FlowcatError::Other("Nemotron socket closed during setup".into()))?
                .map_err(|error| socket_error("receive setup event", error))?;
            match message {
                Message::Text(text) => {
                    let event = parse_server_event(text.as_str())?;
                    if let ServerEvent::Error(message) = event {
                        return Err(FlowcatError::Protocol(format!(
                            "Nemotron setup error: {message}"
                        )));
                    }
                    if wanted(&event) {
                        return Ok(());
                    }
                }
                Message::Ping(payload) => socket
                    .send(Message::Pong(payload))
                    .await
                    .map_err(|error| socket_error("send setup pong", error))?,
                Message::Close(_) => {
                    return Err(FlowcatError::Other(
                        "Nemotron socket closed during setup".into(),
                    ))
                }
                Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
            }
        }
    })
    .await
    .map_err(|_| FlowcatError::Other("Nemotron session setup timed out".into()))?
}

fn session_update(speech_contexts: &[String]) -> Value {
    let mut session = json!({
        "sample_rate": NEMOTRON_RATE,
        "language": "en-US",
        "automatic_punctuation": true,
        "word_timestamps": false,
        "speaker_diarization": false,
        // The sidecar is launched with endpointing disabled. Zero leaves its
        // configured threshold untouched and documents that this client never
        // requests an earlier per-session endpoint.
        "endpointing_ms": 0,
    });
    if !speech_contexts.is_empty() {
        session["speech_contexts"] = json!([{
            "phrases": speech_contexts,
            "boost": CONTEXT_BOOST,
        }]);
    }
    json!({"type": "session.update", "session": session})
}

#[derive(Debug, PartialEq)]
enum ServerEvent {
    SessionCreated,
    SessionUpdated,
    Delta(String),
    Completed(String),
    Committed,
    Cleared,
    Error(String),
    Other,
}

fn parse_server_event(text: &str) -> Result<ServerEvent> {
    let value: Value = serde_json::from_str(text)?;
    let event_type = value.get("type").and_then(Value::as_str).unwrap_or("");
    Ok(match event_type {
        "session.created" => ServerEvent::SessionCreated,
        "session.updated" => ServerEvent::SessionUpdated,
        "conversation.item.input_audio_transcription.delta" => ServerEvent::Delta(
            value
                .get("delta")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
        "conversation.item.input_audio_transcription.completed" => ServerEvent::Completed(
            value
                .get("transcript")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
        ),
        "input_audio_buffer.committed" => ServerEvent::Committed,
        "input_audio_buffer.cleared" => ServerEvent::Cleared,
        "error" => ServerEvent::Error(
            value
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("unknown server error")
                .to_string(),
        ),
        _ => ServerEvent::Other,
    })
}

#[derive(Default)]
struct TranscriptState {
    text: String,
    last_interim: String,
}

impl TranscriptState {
    fn append_delta(&mut self, delta: &str) -> Option<String> {
        self.text.push_str(delta);
        self.changed_interim()
    }

    fn replace(&mut self, text: &str) -> Option<String> {
        self.text.clear();
        self.text.push_str(text);
        self.changed_interim()
    }

    fn changed_interim(&mut self) -> Option<String> {
        let text = spoken_text(&self.text)?;
        if text == self.last_interim {
            return None;
        }
        self.last_interim.clone_from(&text);
        Some(text)
    }

    fn current(&self) -> Option<String> {
        spoken_text(&self.text)
    }

    fn reset(&mut self) {
        self.text.clear();
        self.last_interim.clear();
    }
}

fn pcm16_le(audio: &AudioFrame) -> Result<Vec<u8>> {
    if audio.sample_rate != NEMOTRON_RATE {
        return Err(FlowcatError::Other(format!(
            "Nemotron requires 16 kHz PCM, received {} Hz",
            audio.sample_rate
        )));
    }
    if audio.num_channels != 1 {
        return Err(FlowcatError::Other(format!(
            "Nemotron requires mono PCM, received {} channels",
            audio.num_channels
        )));
    }
    let mut bytes = Vec::with_capacity(audio.pcm.len() * 2);
    for sample in &audio.pcm {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    Ok(bytes)
}

fn spoken_text(text: &str) -> Option<String> {
    let text = text.trim();
    text.chars()
        .any(char::is_alphanumeric)
        .then(|| text.to_string())
}

fn interim_frame(text: String) -> Frame {
    Frame::InterimTranscription {
        text,
        user_id: Arc::from("user"),
        language: Some(Language("en-US".into())),
    }
}

fn final_frame(text: String) -> Frame {
    Frame::Transcription {
        text,
        user_id: Arc::from("user"),
        language: Some(Language("en-US".into())),
        final_: true,
    }
}

fn socket_error(operation: &str, error: tokio_tungstenite::tungstenite::Error) -> FlowcatError {
    FlowcatError::Network(format!("Nemotron {operation}: {error}"))
}

fn finish_flush(pending: &mut Option<PendingRequest>, transcript: &mut TranscriptState) {
    let Some(PendingRequest::Flush {
        reply, completed, ..
    }) = pending.take()
    else {
        return;
    };
    let text = completed.or_else(|| transcript.current());
    transcript.reset();
    let _ = reply.send(Ok(text));
}

fn fail_pending(pending: &mut Option<PendingRequest>, message: String) {
    match pending.take() {
        Some(PendingRequest::Flush { reply, .. }) => {
            let _ = reply.send(Err(FlowcatError::Other(message)));
        }
        Some(PendingRequest::Clear { reply }) => {
            let _ = reply.send(Err(FlowcatError::Other(message)));
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_the_local_realtime_url() {
        assert_eq!(
            realtime_url("http://127.0.0.1:8178").unwrap(),
            "ws://127.0.0.1:8178/v1/realtime"
        );
        assert_eq!(
            realtime_url("https://speech.local/v1/").unwrap(),
            "wss://speech.local/v1/realtime"
        );
        assert_eq!(
            realtime_url("ws://127.0.0.1:8178/v1/realtime").unwrap(),
            "ws://127.0.0.1:8178/v1/realtime"
        );
        assert!(realtime_url("file:///tmp/speech").is_err());
    }

    #[test]
    fn session_uses_external_endpointing_and_forwards_optional_contexts() {
        let update = session_update(&["Purple Rain".into(), "Radio 4".into()]);
        assert_eq!(update["type"], "session.update");
        assert_eq!(update["session"]["sample_rate"], 16_000);
        assert_eq!(update["session"]["language"], "en-US");
        assert_eq!(update["session"]["endpointing_ms"], 0);
        assert_eq!(
            update["session"]["speech_contexts"][0]["phrases"],
            json!(["Purple Rain", "Radio 4"])
        );
        assert_eq!(
            update["session"]["speech_contexts"][0]["boost"],
            CONTEXT_BOOST
        );
    }

    #[test]
    fn append_only_deltas_form_deduplicated_full_interims() {
        let mut state = TranscriptState::default();
        assert_eq!(state.append_delta("Play ").as_deref(), Some("Play"));
        assert_eq!(
            state.append_delta("Purple Rain").as_deref(),
            Some("Play Purple Rain")
        );
        assert!(state.append_delta("").is_none());
        assert_eq!(state.current().as_deref(), Some("Play Purple Rain"));
    }

    #[test]
    fn completed_event_is_parsed_without_becoming_a_final_frame() {
        let event = parse_server_event(
            r#"{"type":"conversation.item.input_audio_transcription.completed","transcript":"one two three four"}"#,
        )
        .unwrap();
        let ServerEvent::Completed(text) = event else {
            panic!("expected completed event");
        };
        // A live server completion is still display-only. The worker calls
        // this interim constructor unless an external flush is pending.
        assert!(matches!(
            interim_frame(text),
            Frame::InterimTranscription { .. }
        ));
    }

    #[test]
    fn reset_prevents_text_from_leaking_into_the_next_vad_turn() {
        let mut state = TranscriptState::default();
        state.append_delta("first turn");
        state.reset();
        assert!(state.current().is_none());
        assert_eq!(
            state.append_delta("second turn").as_deref(),
            Some("second turn")
        );
    }

    #[test]
    fn audio_wire_format_is_mono_16khz_little_endian_pcm16() {
        let audio = AudioFrame::mono(vec![0x1234, -2], NEMOTRON_RATE);
        assert_eq!(pcm16_le(&audio).unwrap(), vec![0x34, 0x12, 0xfe, 0xff]);

        let wrong_rate = AudioFrame::mono(vec![0], 48_000);
        assert!(pcm16_le(&wrong_rate).is_err());
        let stereo = AudioFrame {
            pcm: vec![0, 0],
            sample_rate: NEMOTRON_RATE,
            num_channels: 2,
        };
        assert!(pcm16_le(&stereo).is_err());
    }
}
