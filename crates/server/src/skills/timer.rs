//! `set_timer` — countdown that speaks when it fires (port of skills/core/set_timer).
//!
//! The alert is a [`Frame::TtsSpeak`] queued at the head of the call's
//! pipeline (`PipelineTask::queue_sender`), so it plays even mid-conversation.
//! Timers are per-call and in-memory: if the call has ended by the time the
//! timer fires, the send fails and the alert is logged and dropped — the same
//! as the Python behaviour when its pipeline was gone.

use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

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

/// One live timer on one call.
#[derive(Clone)]
pub struct TimerEntry {
    /// Monotonic per call. The unambiguous internal handle — names are not unique.
    pub id: u64,
    /// Normalized name for matching (`normalize_name`). `None` when unnamed.
    pub name: Option<String>,
    /// The label as the user said it, for the spoken alert.
    pub spoken_name: Option<String>,
    /// The duration as requested, so "the 5 minute timer" can address it.
    pub minutes: f64,
    pub deadline: Instant,
    /// Fired, and still announcing. Ringing timers stay in the book so they
    /// can be silenced.
    pub ringing: bool,
    pub cancel: CancellationToken,
}

/// Every timer on one call.
///
/// Each timer's token is a child of `call_token`, so cancelling the parent
/// cancels all of them at once. Lives inside `CallState`, so it is per call and
/// nothing is persisted.
pub struct TimerBook {
    call_token: CancellationToken,
    next_id: u64,
    timers: Vec<TimerEntry>,
}

impl Default for TimerBook {
    fn default() -> Self {
        Self {
            call_token: CancellationToken::new(),
            next_id: 1,
            timers: Vec::new(),
        }
    }
}

impl Drop for TimerBook {
    /// The call ended. Wake every sleeping timer task so it exits now instead
    /// of holding a thread-pool slot until a deadline that no longer matters.
    /// (A plain `CancellationToken` drop does *not* cancel — this does.)
    fn drop(&mut self) {
        self.call_token.cancel();
    }
}

impl TimerBook {
    /// Register a timer. Returns its id and its own cancellation token; the
    /// caller spawns the task that waits on them.
    pub fn insert(
        &mut self,
        name: Option<String>,
        spoken_name: Option<String>,
        minutes: f64,
        deadline: Instant,
    ) -> (u64, CancellationToken) {
        let id = self.next_id;
        self.next_id += 1;
        let cancel = self.call_token.child_token();
        self.timers.push(TimerEntry {
            id,
            name,
            spoken_name,
            minutes,
            deadline,
            ringing: false,
            cancel: cancel.clone(),
        });
        (id, cancel)
    }

    /// Forget a timer without cancelling it — the task calls this on its way out.
    pub fn remove(&mut self, id: u64) {
        self.timers.retain(|t| t.id != id);
    }

    pub fn mark_ringing(&mut self, id: u64) {
        if let Some(t) = self.timers.iter_mut().find(|t| t.id == id) {
            t.ringing = true;
        }
    }

    /// Cancel one timer and forget it. False when the id is unknown.
    pub fn cancel(&mut self, id: u64) -> bool {
        match self.timers.iter().position(|t| t.id == id) {
            Some(i) => {
                self.timers.remove(i).cancel.cancel();
                true
            }
            None => false,
        }
    }

    /// Cancel every timer; returns how many there were.
    ///
    /// A cancelled token stays cancelled forever, so the parent is *replaced*
    /// rather than reused — otherwise every timer set later on this call would
    /// be born already-cancelled and never fire.
    pub fn cancel_all(&mut self) -> usize {
        let n = self.timers.len();
        self.call_token.cancel();
        self.call_token = CancellationToken::new();
        self.timers.clear();
        n
    }

    pub fn entries(&self) -> Vec<TimerEntry> {
        self.timers.clone()
    }

    pub fn len(&self) -> usize {
        self.timers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.timers.is_empty()
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

    fn at(secs: u64) -> Instant {
        Instant::now() + Duration::from_secs(secs)
    }

    #[test]
    fn book_tracks_and_removes_timers() {
        let mut book = TimerBook::default();
        assert!(book.is_empty());
        let (a, _) = book.insert(Some("pasta".into()), Some("pasta".into()), 5.0, at(300));
        let (b, _) = book.insert(None, None, 10.0, at(600));
        assert_eq!(book.len(), 2);
        assert_ne!(a, b, "ids are unique");

        book.mark_ringing(a);
        assert!(book.entries().iter().find(|t| t.id == a).unwrap().ringing);
        assert!(!book.entries().iter().find(|t| t.id == b).unwrap().ringing);

        book.remove(a);
        assert_eq!(book.len(), 1);
        book.remove(a); // removing twice is harmless
        assert_eq!(book.len(), 1);
    }

    #[test]
    fn cancel_fires_the_token_and_drops_the_entry() {
        let mut book = TimerBook::default();
        let (id, token) = book.insert(None, None, 1.0, at(60));
        assert!(!token.is_cancelled());
        assert!(book.cancel(id));
        assert!(token.is_cancelled(), "cancel() is synchronous");
        assert!(book.is_empty());
        assert!(!book.cancel(id), "cancelling an unknown id reports false");
    }

    #[test]
    fn cancel_all_clears_everything_and_leaves_the_book_usable() {
        let mut book = TimerBook::default();
        let (_, t1) = book.insert(None, None, 1.0, at(60));
        let (_, t2) = book.insert(None, None, 2.0, at(120));
        assert_eq!(book.cancel_all(), 2);
        assert!(t1.is_cancelled() && t2.is_cancelled());
        assert!(book.is_empty());

        // A cancelled CancellationToken stays cancelled forever, so cancel_all
        // must install a *fresh* parent or every later timer is born dead.
        let (_, t3) = book.insert(None, None, 3.0, at(180));
        assert!(!t3.is_cancelled(), "timers set after a cancel-all must still fire");
        assert_eq!(book.cancel_all(), 1);
    }

    #[test]
    fn dropping_the_book_cancels_live_timers() {
        let mut book = TimerBook::default();
        let (_, token) = book.insert(None, None, 30.0, at(1800));
        drop(book);
        assert!(
            token.is_cancelled(),
            "the call ended, so a sleeping task must wake and exit"
        );
    }
}
