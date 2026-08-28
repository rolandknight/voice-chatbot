//! Human-readable rendering of FlowCat's server-to-client event WebSocket,
//! plus dispatch of the server's media commands to the local player.

use std::time::Duration;

use futures::{SinkExt, StreamExt};
use reqwest::Url;
use serde_json::Value;
use tokio::sync::watch;
use tokio_tungstenite::tungstenite::Message;

use crate::media::MediaPlayer;

pub async fn run(
    url: Url,
    mut shutdown: watch::Receiver<bool>,
    mut media: Option<MediaPlayer>,
    mut outbound: tokio::sync::mpsc::UnboundedReceiver<String>,
    activity: crate::wake::Activity,
) {
    let (mut socket, _) = match tokio_tungstenite::connect_async(url.as_str()).await {
        Ok(connection) => connection,
        Err(error) => {
            tracing::warn!(%error, "event stream unavailable; audio remains connected");
            return;
        }
    };
    let mut housekeeping = tokio::time::interval(Duration::from_secs(1));
    // Client → server frames (on-device wake reports). Once every sender is
    // gone the branch is disabled instead of spinning on `None`.
    let mut outbound_open = true;

    loop {
        tokio::select! {
            changed = shutdown.changed() => {
                if changed.is_err() || *shutdown.borrow() {
                    let _ = socket.close(None).await;
                    return;
                }
            }
            _ = housekeeping.tick() => {
                if let Some(line) = media.as_mut().and_then(MediaPlayer::tick) {
                    println!("{line}");
                }
            }
            frame = outbound.recv(), if outbound_open => match frame {
                Some(text) => {
                    if let Err(error) = socket.send(Message::Text(text.into())).await {
                        tracing::warn!(%error, "failed to send a client frame");
                    }
                }
                None => outbound_open = false,
            },
            message = socket.next() => {
                match message {
                    Some(Ok(Message::Text(text))) => {
                        note_activity(&activity, &text);
                        match render(&text) {
                            Ok(Some(line)) => println!("{line}"),
                            Ok(None) => {}
                            Err(error) => tracing::warn!(%error, "ignoring malformed FlowCat event"),
                        }
                        if let Some(player) = media.as_mut() {
                            if let Some(line) = dispatch_media(player, &text) {
                                println!("{line}");
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        tracing::warn!(%error, "FlowCat event stream closed");
                        return;
                    }
                }
            }
        }
    }
}

/// Conversation activity the wake session window re-arms on (and the speech
/// boundaries that suspend it while someone is talking).
fn note_activity(activity: &crate::wake::Activity, input: &str) {
    let Ok(message) = serde_json::from_str::<Value>(input) else {
        return;
    };
    let Some(kind) = message.get("type").and_then(Value::as_str) else {
        return;
    };
    if let Some(signal) = crate::wake::Activity::signal_for(kind) {
        activity.note(signal);
    }
}

/// Hand `{type, payload}` to the player (media commands and the speaking
/// boundaries it ducks on).
fn dispatch_media(player: &mut MediaPlayer, input: &str) -> Option<String> {
    let message: Value = serde_json::from_str(input).ok()?;
    let kind = message.get("type").and_then(Value::as_str)?;
    let payload = message.get("payload").cloned().unwrap_or(Value::Null);
    player.on_event(kind, &payload)
}

pub fn render(input: &str) -> anyhow::Result<Option<String>> {
    let message: Value = serde_json::from_str(input)?;
    let Some(kind) = message.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let payload = message.get("payload").unwrap_or(&Value::Null);
    let text = || payload.get("text").and_then(Value::as_str).unwrap_or("");
    let rendered = match kind {
        "rtf-user-transcription" => {
            let prefix = if payload
                .get("final")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                "you"
            } else {
                "you (partial)"
            };
            Some(format!("{prefix}: {}", text()))
        }
        "rtf-bot-text" => Some(format!("assistant: {}", text())),
        "rtf-bot-started-speaking" => Some("[assistant speaking]".to_string()),
        "rtf-bot-stopped-speaking" => Some("[assistant finished]".to_string()),
        "rtf-function-call-start" => Some(format!(
            "[tool: {}]",
            payload
                .get("function_name")
                .or_else(|| payload.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        )),
        "rtf-function-call-end" => Some(format!(
            "[tool finished: {}]",
            payload
                .get("function_name")
                .or_else(|| payload.get("tool_name"))
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        )),
        voice_chatbot_protocol::WAKE_EVENT => {
            match voice_chatbot_protocol::WakeState::from_payload(payload) {
                Ok(voice_chatbot_protocol::WakeState::Awake {
                    model,
                    score,
                    persona,
                }) => Some(format!("[awake: {} {score:.2}]", persona.unwrap_or(model))),
                Ok(voice_chatbot_protocol::WakeState::Asleep) => Some("[asleep]".to_string()),
                Err(_) => None,
            }
        }
        _ => None,
    };
    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_known_events() {
        assert_eq!(
            render(r#"{"type":"rtf-user-transcription","payload":{"text":"hello","final":false}}"#)
                .unwrap()
                .as_deref(),
            Some("you (partial): hello")
        );
        assert_eq!(
            render(r#"{"type":"rtf-user-transcription","payload":{"text":"hello","final":true}}"#)
                .unwrap()
                .as_deref(),
            Some("you: hello")
        );
        assert_eq!(
            render(r#"{"type":"rtf-bot-text","payload":{"text":"Hi."}}"#)
                .unwrap()
                .as_deref(),
            Some("assistant: Hi.")
        );
        assert_eq!(
            render(r#"{"type":"wake","payload":{"state":"awake","model":"hey_marvin","score":0.874,"persona":"marvin"}}"#)
                .unwrap()
                .as_deref(),
            Some("[awake: marvin 0.87]")
        );
        assert_eq!(
            render(r#"{"type":"wake","payload":{"state":"asleep"}}"#)
                .unwrap()
                .as_deref(),
            Some("[asleep]")
        );
    }

    #[test]
    fn ignores_unknown_events_and_rejects_bad_json() {
        assert_eq!(
            render(r#"{"type":"future-event","payload":{}}"#).unwrap(),
            None
        );
        assert!(render("not json").is_err());
    }
}
