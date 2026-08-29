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

/// A timer name as the user's words reach us, reduced to something matchable.
///
/// This is the load-bearing piece for voice: the model passes `label: "pasta"`
/// when the timer is set, but on "cancel the pasta timer" it will happily pass
/// `"the pasta timer"`. Returns `None` when nothing identifying is left, so
/// "cancel the timer" falls through to the "only one running" rule.
pub fn normalize_name(raw: &str) -> Option<String> {
    let lowered = raw.trim().to_ascii_lowercase();
    let mut s = lowered.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(rest) = s.strip_prefix("the ") {
        s = rest.to_string();
    }
    if let Some(rest) = s.strip_suffix(" timer") {
        s = rest.to_string();
    }
    if s.is_empty() || s == "timer" {
        return None;
    }
    Some(s)
}

/// How long is left, as a countdown reads aloud. Deliberately separate from
/// `format_duration`, whose "2.5 minutes" form is right for "timer set for …"
/// and wrong for "… left".
pub fn format_remaining(left: Duration) -> String {
    let secs = left.as_secs();
    if secs == 0 {
        return "no time".to_string();
    }
    if secs < 60 {
        return if secs == 1 {
            "1 second".to_string()
        } else {
            format!("{secs} seconds")
        };
    }
    let mins = (left.as_secs_f64() / 60.0).round() as u64;
    if mins == 1 {
        "about a minute".to_string()
    } else {
        format!("about {mins} minutes")
    }
}

/// Adjectival form of a requested duration: "the **5 minute** timer".
pub fn duration_adjective(minutes: f64) -> String {
    let d = format_duration(minutes);
    // Written as a match, not `.map(..).unwrap_or(d)`: the latter is a borrow
    // of `d` in the same expression that moves it.
    match d.strip_suffix('s') {
        Some(trimmed) => trimmed.to_string(),
        None => d,
    }
}

/// "a", "a and b", "a, b, and c" — a list that reads aloud.
pub fn join_and(parts: &[String]) -> String {
    match parts {
        [] => String::new(),
        [one] => one.clone(),
        [a, b] => format!("{a} and {b}"),
        [rest @ .., last] => format!("{}, and {last}", rest.join(", ")),
    }
}

/// Longest timer we accept. A hallucinated `1e12` should get a spoken error,
/// not a task that sleeps for two million years.
pub const MAX_MINUTES: f64 = 24.0 * 60.0;

/// The LLM sometimes sends numbers as strings; accept both like Python's `float()`.
///
/// Rejects anything `Duration::from_secs_f64` would panic on. This matters
/// because the caller's `minutes <= 0.0` guard is *false* for NaN, so without
/// the `is_finite` check a `{"minutes": "NaN"}` tool call panics the task.
fn parse_minutes(v: Option<&Value>) -> Option<f64> {
    let n = match v? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse().ok()?,
        _ => return None,
    };
    (n.is_finite() && n <= MAX_MINUTES).then_some(n)
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
            media: None,
            spotify: None,
            state: None,
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
        let ctx = CallCtx::detached(1);
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

    #[test]
    fn minutes_rejects_non_finite_and_absurd_values() {
        // `Duration::from_secs_f64` panics on these, and `n <= 0.0` is false
        // for NaN, so the caller's guard would not catch them.
        assert_eq!(parse_minutes(Some(&json!("NaN"))), None);
        assert_eq!(parse_minutes(Some(&json!("inf"))), None);
        assert_eq!(parse_minutes(Some(&json!("-inf"))), None);
        assert_eq!(parse_minutes(Some(&json!(1e12))), None);
        // Still accepts everything reasonable.
        assert_eq!(parse_minutes(Some(&json!(0.5))), Some(0.5));
        assert_eq!(parse_minutes(Some(&json!(1440))), Some(1440.0));
    }

    #[tokio::test]
    async fn non_finite_duration_answers_instead_of_panicking() {
        let ctx = CallCtx::detached(1);
        assert_eq!(
            SetTimer.call(&json!({"minutes": "NaN"}), &ctx).await,
            "I couldn't understand the timer duration."
        );
        assert_eq!(
            SetTimer.call(&json!({"minutes": "inf"}), &ctx).await,
            "I couldn't understand the timer duration."
        );
    }

    #[test]
    fn normalize_name_strips_voice_wrapping() {
        assert_eq!(normalize_name("pasta"), Some("pasta".into()));
        assert_eq!(normalize_name("  Pasta  "), Some("pasta".into()));
        assert_eq!(normalize_name("The Pasta Timer"), Some("pasta".into()));
        assert_eq!(normalize_name("the pasta"), Some("pasta".into()));
        assert_eq!(normalize_name("pasta   sauce"), Some("pasta sauce".into()));
        // Nothing that identifies a specific timer.
        assert_eq!(normalize_name(""), None);
        assert_eq!(normalize_name("   "), None);
        assert_eq!(normalize_name("timer"), None);
        assert_eq!(normalize_name("the timer"), None);
    }

    #[test]
    fn remaining_reads_naturally() {
        assert_eq!(format_remaining(Duration::from_secs(0)), "no time");
        assert_eq!(format_remaining(Duration::from_secs(1)), "1 second");
        assert_eq!(format_remaining(Duration::from_secs(30)), "30 seconds");
        assert_eq!(format_remaining(Duration::from_secs(59)), "59 seconds");
        assert_eq!(format_remaining(Duration::from_secs(60)), "about a minute");
        assert_eq!(format_remaining(Duration::from_secs(200)), "about 3 minutes");
        assert_eq!(format_remaining(Duration::from_secs(600)), "about 10 minutes");
    }

    #[test]
    fn duration_adjective_drops_the_plural() {
        // "the 5 minute timer", not "the 5 minutes timer".
        assert_eq!(duration_adjective(5.0), "5 minute");
        assert_eq!(duration_adjective(1.0), "1 minute");
        assert_eq!(duration_adjective(0.5), "30 second");
        assert_eq!(duration_adjective(2.5), "2.5 minute");
    }

    #[test]
    fn join_and_speaks_a_list() {
        assert_eq!(join_and(&[]), "");
        assert_eq!(join_and(&["a".to_string()]), "a");
        assert_eq!(join_and(&["a".to_string(), "b".to_string()]), "a and b");
        assert_eq!(
            join_and(&["a".to_string(), "b".to_string(), "c".to_string()]),
            "a, b, and c"
        );
    }
}
