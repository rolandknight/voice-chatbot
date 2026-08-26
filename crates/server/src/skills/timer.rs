//! `set_timer` — countdown that speaks when it fires (port of skills/core/set_timer).
//!
//! The alert is a [`Frame::TtsSpeak`] queued at the head of the call's
//! pipeline (`PipelineTask::queue_sender`), so it plays even mid-conversation.
//! Timers are per-call and in-memory: if the call has ended by the time the
//! timer fires, the send fails and the alert is logged and dropped — the same
//! as the Python behaviour when its pipeline was gone.

use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use flowcat_core::processor::frame::Frame;

use super::{arg_str, CallCtx, Skill};

/// "45 seconds", "1 minute", "5 minutes", "2.5 minutes".
pub fn format_duration(minutes: f64) -> String {
    if minutes < 1.0 {
        return format!("{} seconds", (minutes * 60.0).round() as i64);
    }
    if (minutes - minutes.round()).abs() < 0.05 {
        let m = minutes.round() as i64;
        return if m == 1 {
            "1 minute".to_string()
        } else {
            format!("{m} minutes")
        };
    }
    // Python's `{minutes:g}`: shortest repr without trailing zeros.
    format!("{minutes} minutes")
}

/// The LLM sometimes sends numbers as strings; accept both like Python's `float()`.
fn parse_minutes(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse().ok(),
        _ => None,
    }
}

pub fn alert_text(label: &str) -> String {
    if label.is_empty() {
        "Your timer is up.".to_string()
    } else {
        format!("Your {label} timer is up.")
    }
}

pub struct SetTimer;

#[async_trait]
impl Skill for SetTimer {
    fn name(&self) -> &str {
        "set_timer"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let Some(minutes) = parse_minutes(args.get("minutes")) else {
            return "I couldn't understand the timer duration.".to_string();
        };
        if minutes <= 0.0 {
            return "The timer duration needs to be greater than zero.".to_string();
        }
        let label = arg_str(args, "label").to_string();
        let Some(frames) = ctx.frames.clone() else {
            // No live pipeline registered for this run (shouldn't happen in a call).
            tracing::warn!(run_id = ctx.run_id, "set_timer: no pipeline for this call");
            return "I can't set a timer right now.".to_string();
        };
        let delay = Duration::from_secs_f64(minutes * 60.0);
        let run_id = ctx.run_id;
        let fire_label = label.clone();
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let text = alert_text(&fire_label);
            match frames.send(Frame::TtsSpeak {
                text: text.clone(),
                append_to_context: None,
            }) {
                Ok(()) => tracing::info!(run_id, %text, "timer fired"),
                Err(_) => {
                    tracing::info!(run_id, %text, "timer fired after the call ended; dropped")
                }
            }
        });
        let pretty = format_duration(minutes);
        let tail = if label.is_empty() {
            String::new()
        } else {
            format!(" for {label}")
        };
        format!("Timer set for {pretty}{tail}.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tokio::sync::mpsc;

    #[test]
    fn durations_match_python() {
        assert_eq!(format_duration(0.5), "30 seconds");
        assert_eq!(format_duration(1.0), "1 minute");
        assert_eq!(format_duration(5.0), "5 minutes");
        assert_eq!(format_duration(2.5), "2.5 minutes");
        assert_eq!(format_duration(1.02), "1 minute");
    }

    #[test]
    fn minutes_accepts_numbers_and_strings() {
        assert_eq!(parse_minutes(Some(&json!(2))), Some(2.0));
        assert_eq!(parse_minutes(Some(&json!("1.5"))), Some(1.5));
        assert_eq!(parse_minutes(Some(&json!("soon"))), None);
        assert_eq!(parse_minutes(None), None);
    }

    #[tokio::test(start_paused = true)]
    async fn timer_speaks_into_the_call_after_the_delay() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let ctx = CallCtx {
            run_id: 7,
            frames: Some(tx),
        };
        let reply = SetTimer
            .call(&json!({"minutes": 0.5, "label": "tea"}), &ctx)
            .await;
        assert_eq!(reply, "Timer set for 30 seconds for tea.");
        assert!(rx.try_recv().is_err(), "nothing spoken before the delay");
        tokio::time::advance(Duration::from_secs(31)).await;
        match rx.recv().await {
            Some(Frame::TtsSpeak { text, .. }) => assert_eq!(text, "Your tea timer is up."),
            other => panic!("unexpected {other:?}"),
        }
    }

    #[tokio::test]
    async fn rejects_bad_durations_and_missing_pipeline() {
        let ctx = CallCtx {
            run_id: 1,
            frames: None,
        };
        assert_eq!(
            SetTimer.call(&json!({"minutes": -1}), &ctx).await,
            "The timer duration needs to be greater than zero."
        );
        assert_eq!(
            SetTimer.call(&json!({}), &ctx).await,
            "I couldn't understand the timer duration."
        );
        assert_eq!(
            SetTimer.call(&json!({"minutes": 1}), &ctx).await,
            "I can't set a timer right now."
        );
    }
}
