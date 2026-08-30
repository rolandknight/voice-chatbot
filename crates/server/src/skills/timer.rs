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
use tokio::time::Instant;
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

/// Whether two normalized names refer to the same timer by word-boundary
/// containment: "pasta" finds "pasta sauce" and vice versa, but "steak" does
/// not find "tea" just because the letters happen to occur inside it (plain
/// `"steak".contains("tea")` is true on raw substrings — a real bug, since a
/// `{"name":"steak"}` call would cancel a running "tea" timer).
fn names_overlap(a: &str, b: &str) -> bool {
    let a_words: Vec<&str> = a.split_whitespace().collect();
    let b_words: Vec<&str> = b.split_whitespace().collect();
    contains_word_run(&a_words, &b_words) || contains_word_run(&b_words, &a_words)
}

/// True if `needle` occurs as a contiguous run inside `haystack`, comparing
/// whole words rather than raw characters.
fn contains_word_run(haystack: &[&str], needle: &[&str]) -> bool {
    !needle.is_empty() && haystack.windows(needle.len()).any(|w| w == needle)
}

/// How long is left, as a countdown reads aloud. Deliberately separate from
/// `format_duration`, whose "2.5 minutes" form is right for "timer set for …"
/// and wrong for "… left".
pub fn format_remaining(left: Duration) -> String {
    let secs = left.as_secs();
    if secs == 0 {
        return "less than a second".to_string();
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
    // Written as a match, not `.map(..).unwrap_or(d)`, for readability.
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
    /// Not `pub`: `entries()` clones `TimerEntry` for read-only display, and
    /// a caller that could reach this token directly could cancel it without
    /// going through `TimerBook::cancel`/`cancel_all`, which remove the book
    /// entry *before* cancelling the token. Cancelling the token alone would
    /// leave a dead entry in the book forever, since the announce task's
    /// "cancelled before firing" branch returns without removing anything —
    /// it relies on `cancel()` having already done so. Keeping the field
    /// private keeps that invariant a compile-time guarantee instead of a
    /// convention every future call site has to remember.
    cancel: CancellationToken,
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

    /// True if a timer with this id is still registered.
    ///
    /// Checking this under the same lock as the send that follows it is what
    /// actually closes the cancel-vs-announce race: `cancel()` removes the
    /// entry *before* cancelling its token, so a check that only reads
    /// `token.is_cancelled()` can still be false for an id `cancel()` has
    /// already forgotten. `contains` gives the announce loop something to
    /// check while holding the same lock `cancel()` uses to remove the entry,
    /// so "entry gone" and "frame sent" can never both be true.
    pub fn contains(&self, id: u64) -> bool {
        self.timers.iter().any(|t| t.id == id)
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

/// Gap between repeats of the expiry alert.
pub const REPEAT_EVERY: Duration = Duration::from_secs(10);
/// Total announcements before a timer gives up: the initial alert plus four
/// repeats, a ringing window of about forty seconds.
pub const MAX_ANNOUNCEMENTS: usize = 5;

/// The LLM sometimes sends numbers as strings; accept both like Python's `float()`.
fn extract_finite_number(v: Option<&Value>) -> Option<f64> {
    let n = match v? {
        Value::Number(n) => n.as_f64()?,
        Value::String(s) => s.trim().parse().ok()?,
        _ => return None,
    };
    n.is_finite().then_some(n)
}

/// A duration for `set_timer` to create.
///
/// Rejects anything `Duration::from_secs_f64` would panic on. This matters
/// because the caller's `minutes <= 0.0` guard is *false* for NaN, so without
/// the `is_finite` check (inside `extract_finite_number`) a `{"minutes":
/// "NaN"}` tool call panics the task. Also rejects anything over
/// `MAX_MINUTES`: a hallucinated `1e12` should get a spoken error, not a task
/// that sleeps for two million years.
fn parse_minutes(v: Option<&Value>) -> Option<f64> {
    extract_finite_number(v).filter(|n| *n <= MAX_MINUTES)
}

/// A duration for `cancel_timer` to match an existing timer against.
///
/// Deliberately has no upper cap: `MAX_MINUTES` is a `set_timer` policy about
/// what may be *created*, not a definition of "argument absent". Reusing
/// `parse_minutes` here folded "over the cap" into `None`, indistinguishable
/// from no `minutes` argument at all, so `{"minutes": 2000}` fell through to
/// the sole-timer rule and cancelled whatever was running.
fn parse_cancel_minutes(v: Option<&Value>) -> Option<f64> {
    extract_finite_number(v)
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
        // A live call gives us both, or neither (`CallRegistry::ctx`).
        let (Some(frames), Some(state)) = (ctx.frames.clone(), ctx.state.clone()) else {
            tracing::warn!(run_id = ctx.run_id, "set_timer: no live call");
            return "I can't set a timer right now.".to_string();
        };

        let label = arg_str(args, "label").to_string();
        // Both the spoken name and the match key come from the *normalized*
        // label. Storing the raw label as `spoken_name` while only the match
        // key was normalized doubled "timer" in every surface that speaks it
        // back: "set the oven timer for 10 minutes" routinely gives a
        // wrapped label, and "the oven timer" spoken verbatim by TTS reads
        // as "Your the oven timer timer is up." The `tail` below still uses
        // the raw `label`, so "Timer set for 30 seconds for tea." is
        // unaffected.
        let spoken_name = normalize_name(&label);
        let name = spoken_name.clone();

        let delay = Duration::from_secs_f64(minutes * 60.0);
        // A single absolute deadline, fixed now and shared by the book entry
        // (for a future "about N minutes left") and the sleep below — not a
        // relative `sleep(delay)` constructed lazily on the task's first
        // poll. Under a paused test clock, that first poll can happen
        // *after* the test has already advanced time past this point (e.g.
        // inside `tokio::time::advance`'s own internal yield), which would
        // silently push the fire time back by however much the clock moved
        // before the task ever ran. Pinning the deadline here, before
        // `tokio::spawn`, makes firing independent of when the scheduler
        // gets around to polling the new task. It also has to be a
        // `tokio::time::Instant`, not `std::time::Instant`: only tokio's own
        // clock moves under `start_paused`, so a `std` deadline stored in the
        // book would read as "the full delay left" no matter how far a test
        // advances.
        let deadline = Instant::now() + delay;
        let (id, token) =
            state.with_timers(|b| b.insert(name, spoken_name.clone(), minutes, deadline));

        let run_id = ctx.run_id;
        let text = alert_text(spoken_name.as_deref().unwrap_or(""));
        // Weak, never Arc: a strong reference would keep `CallState` — and so
        // the whole `TimerBook` — alive past the end of the call, and the
        // book's `Drop` would never cancel anything.
        let state = std::sync::Arc::downgrade(&state);
        tokio::spawn(async move {
            tokio::select! {
                _ = token.cancelled() => {
                    // cancel()/cancel_all()/end-of-call already removed us.
                    tracing::info!(run_id, id, "timer cancelled before firing");
                    return;
                }
                _ = tokio::time::sleep_until(deadline) => {}
            }
            let Some(s) = state.upgrade() else {
                // The call ended between the deadline firing and this point;
                // there is no more book — and likely no more call — to speak
                // into.
                return;
            };
            s.with_timers(|b| b.mark_ringing(id));
            for i in 0..MAX_ANNOUNCEMENTS {
                // `cancel()` sets this synchronously, so checking here is a
                // cheap early-out. It is *not* what makes the "no announcement
                // after cancel" guarantee true on its own: `cancel()` removes
                // the book entry before it cancels the token, so a window
                // exists where this check would still pass for an id
                // `cancel()` has already forgotten. The `contains` check
                // below, taken under the same lock `cancel()` uses, is what
                // actually closes that window.
                if token.is_cancelled() {
                    break;
                }
                let frame = Frame::TtsSpeak {
                    text: text.clone(),
                    // Intent: only the first announcement should register as
                    // part of the conversation, not every repeat — though no
                    // consumer of this field currently acts on it either way.
                    append_to_context: (i > 0).then_some(false),
                };
                let sent = s.with_timers(|b| {
                    if b.contains(id) {
                        Some(frames.send(frame))
                    } else {
                        None
                    }
                });
                match sent {
                    None => {
                        tracing::info!(run_id, id, "timer cancelled before this announcement");
                        break;
                    }
                    Some(Err(_)) => {
                        tracing::info!(run_id, %text, "timer fired after the call ended; dropped");
                        break;
                    }
                    Some(Ok(())) => {}
                }
                tracing::info!(run_id, id, %text, announcement = i + 1, "timer fired");
                if i + 1 == MAX_ANNOUNCEMENTS {
                    break;
                }
                tokio::select! {
                    _ = token.cancelled() => break,
                    _ = tokio::time::sleep(REPEAT_EVERY) => {}
                }
            }
            s.with_timers(|b| b.remove(id));
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

/// "a pasta timer" / "a 10 minute timer" — for enumerating.
fn a_timer(t: &TimerEntry) -> String {
    match &t.spoken_name {
        Some(n) => format!("a {n} timer"),
        None => format!("a {} timer", duration_adjective(t.minutes)),
    }
}

/// "the pasta timer" / "the 10 minute timer" — for confirming one.
fn the_timer(t: &TimerEntry) -> String {
    match &t.spoken_name {
        Some(n) => format!("the {n} timer"),
        None => format!("the {} timer", duration_adjective(t.minutes)),
    }
}

/// "a pasta timer with about 5 minutes left", or "… going off now".
fn timer_with_remaining(t: &TimerEntry, now: Instant) -> String {
    if t.ringing {
        return format!("{} going off now", a_timer(t));
    }
    let left = t.deadline.saturating_duration_since(now);
    format!("{} with {} left", a_timer(t), format_remaining(left))
}

/// The spoken "which one?" question. Never cancels anything.
fn ask_which(candidates: &[TimerEntry], now: Instant) -> String {
    let parts: Vec<String> = candidates
        .iter()
        .map(|t| timer_with_remaining(t, now))
        .collect();
    format!("You have {}. Which should I cancel?", join_and(&parts))
}

pub struct CancelTimer;

#[async_trait]
impl Skill for CancelTimer {
    fn name(&self) -> &str {
        "cancel_timer"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        // Outside a live call there is nothing to cancel — same answer as an
        // empty board, because that is what the user hears either way.
        let Some(state) = ctx.state.clone() else {
            return "You don't have any timers running.".to_string();
        };

        if args.get("all").and_then(Value::as_bool).unwrap_or(false) {
            return match state.with_timers(|b| b.cancel_all()) {
                0 => "You don't have any timers running.".to_string(),
                1 => "Cancelled your timer.".to_string(),
                _ => "Cancelled all your timers.".to_string(),
            };
        }

        let entries = state.with_timers(|b| b.entries());
        if entries.is_empty() {
            return "You don't have any timers running.".to_string();
        }

        let raw = arg_str(args, "name");
        let wanted = normalize_name(raw);
        let minutes = parse_cancel_minutes(args.get("minutes"));

        let candidates: Vec<TimerEntry> = if let Some(w) = &wanted {
            let exact: Vec<TimerEntry> = entries
                .iter()
                .filter(|t| t.name.as_deref() == Some(w.as_str()))
                .cloned()
                .collect();
            let name_matched = if exact.is_empty() {
                // "pasta" should still find "pasta sauce", and vice versa —
                // on word boundaries, so "steak" does not match a "tea"
                // timer just because the letters occur inside it.
                entries
                    .iter()
                    .filter(|t| t.name.as_deref().is_some_and(|n| names_overlap(n, w)))
                    .cloned()
                    .collect()
            } else {
                exact
            };
            // A duplicate name ("two timers named pasta") produces a
            // "which one?" question naming both. If the answer also supplies
            // a duration, that is the natural way to resolve it — narrow to
            // the duration match rather than asking the identical question
            // again. An empty narrowing (the duration matches none of the
            // name-matched timers) keeps the full name-matched set, so the
            // question still lists the right candidates instead of emptying
            // out.
            if name_matched.len() > 1 {
                if let Some(m) = minutes {
                    let narrowed: Vec<TimerEntry> = name_matched
                        .iter()
                        .filter(|t| (t.minutes - m).abs() < 0.01)
                        .cloned()
                        .collect();
                    if narrowed.is_empty() {
                        name_matched
                    } else {
                        narrowed
                    }
                } else {
                    name_matched
                }
            } else {
                name_matched
            }
        } else if let Some(m) = minutes {
            // Never `==` on f64.
            entries
                .iter()
                .filter(|t| (t.minutes - m).abs() < 0.01)
                .cloned()
                .collect()
        } else {
            entries.clone()
        };

        let now = Instant::now();
        match candidates.as_slice() {
            [] => {
                let running: Vec<String> = entries.iter().map(a_timer).collect();
                // A name that reached this arm is always `Some`: a name that
                // normalizes to `None` (like "the timer") can't produce an
                // empty candidate list — it leaves `candidates` as the whole
                // book instead. Speaking the *normalized* name (not `raw`)
                // is what keeps "the rice timer" from echoing back as "a the
                // rice timer timer".
                let what = match &wanted {
                    Some(w) => w.clone(),
                    None => duration_adjective(minutes.unwrap_or_default()),
                };
                format!(
                    "You don't have a {what} timer. You have {}.",
                    join_and(&running)
                )
            }
            [only] => {
                state.with_timers(|b| b.cancel(only.id));
                format!("Cancelled {}.", the_timer(only))
            }
            many => ask_which(many, now),
        }
    }
}

pub struct ListTimers;

#[async_trait]
impl Skill for ListTimers {
    fn name(&self) -> &str {
        "list_timers"
    }

    async fn call(&self, _args: &Value, ctx: &CallCtx) -> String {
        let entries = match &ctx.state {
            Some(s) => s.with_timers(|b| b.entries()),
            None => Vec::new(),
        };
        if entries.is_empty() {
            return "You don't have any timers running.".to_string();
        }
        let now = Instant::now();
        let parts: Vec<String> = entries
            .iter()
            .map(|t| timer_with_remaining(t, now))
            .collect();
        format!("You have {}.", join_and(&parts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skills::CallState;
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
    fn constants_are_pinned() {
        // The tests below are written in terms of these constants, so a
        // change to either would still compile and pass without this pin —
        // but "five announcements, ten seconds apart" is the behaviour the
        // brief specifies, not an implementation detail free to drift.
        assert_eq!(MAX_ANNOUNCEMENTS, 5);
        assert_eq!(REPEAT_EVERY, Duration::from_secs(10));
    }

    #[test]
    fn minutes_accepts_numbers_and_strings() {
        assert_eq!(parse_minutes(Some(&json!(2))), Some(2.0));
        assert_eq!(parse_minutes(Some(&json!("1.5"))), Some(1.5));
        assert_eq!(parse_minutes(Some(&json!("soon"))), None);
        assert_eq!(parse_minutes(None), None);
    }

    /// Stand-in for a TTS backend's rate. The chime is generated at whatever
    /// the live backend reports, so the tests pin a plausible one.
    const TEST_TTS_RATE: u32 = 24_000;

    /// A `CallCtx` with a live pipeline and live per-call state.
    fn live_ctx() -> (
        CallCtx,
        mpsc::UnboundedReceiver<Frame>,
        std::sync::Arc<CallState>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = std::sync::Arc::new(CallState::default());
        let ctx = CallCtx {
            run_id: 7,
            frames: Some(tx),
            media: None,
            spotify: None,
            state: Some(state.clone()),
            tts_rate: Some(TEST_TTS_RATE),
        };
        (ctx, rx, state)
    }

    fn spoken(rx: &mut mpsc::UnboundedReceiver<Frame>) -> Vec<String> {
        let mut out = Vec::new();
        while let Ok(Frame::TtsSpeak { text, .. }) = rx.try_recv() {
            out.push(text);
        }
        out
    }

    #[tokio::test(start_paused = true)]
    async fn timer_speaks_into_the_call_after_the_delay() {
        let (ctx, mut rx, _state) = live_ctx();
        let reply = SetTimer
            .call(&json!({"minutes": 0.5, "label": "tea"}), &ctx)
            .await;
        assert_eq!(reply, "Timer set for 30 seconds for tea.");
        assert!(
            spoken(&mut rx).is_empty(),
            "nothing spoken before the delay"
        );
        tokio::time::advance(Duration::from_secs(31)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx), vec!["Your tea timer is up."]);
    }

    #[tokio::test(start_paused = true)]
    async fn alert_repeats_a_bounded_number_of_times() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx).len(), 1, "one announcement on firing");

        // Four repeats, ten seconds apart.
        for n in 2..=MAX_ANNOUNCEMENTS {
            tokio::time::advance(REPEAT_EVERY).await;
            tokio::task::yield_now().await;
            assert_eq!(
                spoken(&mut rx),
                vec!["Your tea timer is up."],
                "announcement {n}"
            );
        }

        // Then it gives up and forgets itself.
        tokio::time::advance(REPEAT_EVERY * 10).await;
        tokio::task::yield_now().await;
        assert!(spoken(&mut rx).is_empty(), "stops after the cap");
        assert!(state.with_timers(|b| b.is_empty()), "book is cleaned up");
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_before_it_fires_speaks_nothing() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 5, "label": "tea"}), &ctx)
            .await;
        let id = state.with_timers(|b| b.entries()[0].id);
        assert!(state.with_timers(|b| b.cancel(id)));
        tokio::time::advance(Duration::from_secs(600)).await;
        tokio::task::yield_now().await;
        assert!(spoken(&mut rx).is_empty());
        assert!(state.with_timers(|b| b.is_empty()));
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_while_ringing_stops_the_announcements() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx).len(), 1);
        assert!(
            state.with_timers(|b| b.entries()[0].ringing),
            "a ringing timer stays in the book so it can be silenced"
        );

        let id = state.with_timers(|b| b.entries()[0].id);
        state.with_timers(|b| b.cancel(id));
        tokio::time::advance(REPEAT_EVERY * 10).await;
        tokio::task::yield_now().await;
        assert!(spoken(&mut rx).is_empty(), "no announcement after cancel()");
        assert!(state.with_timers(|b| b.is_empty()), "book is cleaned up");
    }

    /// The exact window `cancel()` passes through internally: the entry is
    /// gone from the book, but its token has not (yet, or ever, in this
    /// test) been cancelled. A guard that only checks `token.is_cancelled()`
    /// would still send the next announcement here; only a guard serialized
    /// on the book (`TimerBook::contains`) catches it. This is what actually
    /// closes the race the reviewer flagged: `cancel()` removes the entry
    /// *before* cancelling the token, so a concurrent announce loop that only
    /// checked the token could observe "not cancelled" for an id `cancel()`
    /// has already forgotten.
    #[tokio::test(start_paused = true)]
    async fn entry_removed_without_cancelling_the_token_still_silences_the_timer() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx).len(), 1, "one announcement on firing");

        let id = state.with_timers(|b| b.entries()[0].id);
        let token = state.with_timers(|b| b.entries()[0].cancel.clone());
        state.with_timers(|b| b.remove(id));
        assert!(
            !token.is_cancelled(),
            "remove() alone must not cancel the token"
        );
        assert!(state.with_timers(|b| !b.contains(id)));

        tokio::time::advance(REPEAT_EVERY * 10).await;
        tokio::task::yield_now().await;
        assert!(
            spoken(&mut rx).is_empty(),
            "no announcement once the entry is gone, even with a live token"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn several_timers_run_at_once_and_each_says_its_own_name() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        SetTimer
            .call(&json!({"minutes": 2, "label": "pasta"}), &ctx)
            .await;
        SetTimer.call(&json!({"minutes": 3}), &ctx).await;
        assert_eq!(state.with_timers(|b| b.len()), 3);

        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx), vec!["Your tea timer is up."]);

        tokio::time::advance(Duration::from_secs(60)).await;
        tokio::task::yield_now().await;
        let heard = spoken(&mut rx);
        assert!(heard.contains(&"Your pasta timer is up.".to_string()));
        assert!(
            heard.contains(&"Your tea timer is up.".to_string()),
            "tea repeats"
        );

        // By t=181s the tea (last announcement at t=100) and pasta (last at
        // t=160) timers have both finished their five announcements each;
        // only the unlabelled 3-minute timer (deadline t=180) is still due.
        tokio::time::advance(Duration::from_secs(60)).await;
        tokio::task::yield_now().await;
        assert!(
            spoken(&mut rx).contains(&"Your timer is up.".to_string()),
            "the unlabelled timer speaks the generic alert"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_timer_whose_call_ended_is_dropped() {
        let (ctx, rx, state) = live_ctx();
        let weak = std::sync::Arc::downgrade(&state);
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        drop(rx);
        drop(ctx);
        // Only this test's own handle should be left: if `SetTimer::call`
        // captured a strong `Arc<CallState>` into the spawned task instead of
        // a `Weak`, that task — still sleeping until the deadline — would be
        // holding a second one here.
        assert_eq!(
            std::sync::Arc::strong_count(&state),
            1,
            "the spawned task must not be holding a strong Arc<CallState>"
        );
        drop(state); // the call ended: last strong Arc<CallState> released
        assert!(
            weak.upgrade().is_none(),
            "CallState must actually be dropped, not kept alive by the timer task"
        );
        tokio::time::advance(Duration::from_secs(300)).await;
        tokio::task::yield_now().await;
        // No panic, no leak. The task woke on the cancelled parent and exited.
    }

    /// FIX 8 (final review): the entire safety argument behind holding one
    /// upgraded `Arc<CallState>` for a ringing timer's whole announcement
    /// window (rather than re-upgrading the `Weak` on every iteration) is
    /// that it cannot outlive the call: once every external
    /// `Arc<CallState>` and the frame receiver are gone, the next
    /// announcement's send fails, the task removes itself and exits, and
    /// only then does the held `Arc` drop — releasing `CallState` and,
    /// through its `TimerBook`'s own `Drop`, cancelling every sibling
    /// timer's token. Nothing pinned this before.
    #[tokio::test(start_paused = true)]
    async fn call_ending_mid_ring_still_drops_call_state_and_cancels_siblings() {
        let (ctx, mut rx, state) = live_ctx();
        let weak = std::sync::Arc::downgrade(&state);

        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        SetTimer
            .call(&json!({"minutes": 30, "label": "bread"}), &ctx)
            .await;

        let sibling_token = state.with_timers(|b| {
            b.entries()
                .into_iter()
                .find(|t| t.spoken_name.as_deref() == Some("bread"))
                .expect("the bread timer is registered")
                .cancel
        });

        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            spoken(&mut rx),
            vec!["Your tea timer is up."],
            "tea is ringing"
        );
        assert!(
            state.with_timers(|b| b
                .entries()
                .into_iter()
                .find(|t| t.spoken_name.as_deref() == Some("tea"))
                .unwrap()
                .ringing),
            "tea is mid-ring"
        );

        // End of call: the frame receiver and every Arc<CallState> this test
        // holds are gone. Only the ringing task's own held reference could
        // remain.
        drop(rx);
        drop(ctx);
        drop(state);
        assert!(
            weak.upgrade().is_some(),
            "the ringing task's held Arc keeps CallState alive through its window"
        );
        assert!(
            !sibling_token.is_cancelled(),
            "the sibling has not been cancelled yet"
        );

        tokio::time::advance(REPEAT_EVERY).await;
        tokio::task::yield_now().await;

        assert!(
            weak.upgrade().is_none(),
            "the next failed send must let the ringing task's held Arc drop"
        );
        assert!(
            sibling_token.is_cancelled(),
            "dropping CallState cancels every sibling timer through call_token"
        );
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
        assert_eq!(
            format_remaining(Duration::from_secs(0)),
            "less than a second"
        );
        assert_eq!(format_remaining(Duration::from_secs(1)), "1 second");
        assert_eq!(format_remaining(Duration::from_secs(30)), "30 seconds");
        assert_eq!(format_remaining(Duration::from_secs(59)), "59 seconds");
        assert_eq!(format_remaining(Duration::from_secs(60)), "about a minute");
        assert_eq!(
            format_remaining(Duration::from_secs(200)),
            "about 3 minutes"
        );
        assert_eq!(
            format_remaining(Duration::from_secs(600)),
            "about 10 minutes"
        );
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
        assert!(
            !t3.is_cancelled(),
            "timers set after a cancel-all must still fire"
        );
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

    #[test]
    fn call_state_owns_a_timer_book() {
        let state = CallState::default();
        assert!(state.with_timers(|b| b.is_empty()));
        let token = state.with_timers(|b| b.insert(None, None, 1.0, at(60)).1);
        assert_eq!(state.with_timers(|b| b.len()), 1);
        drop(state);
        assert!(token.is_cancelled(), "dropping the call cancels its timers");
    }

    /// Set `n` timers and return the ctx/receiver/state, so a cancel test can
    /// start from a known board.
    async fn board(
        specs: &[(f64, Option<&str>)],
    ) -> (
        CallCtx,
        mpsc::UnboundedReceiver<Frame>,
        std::sync::Arc<CallState>,
    ) {
        let (ctx, rx, state) = live_ctx();
        for (minutes, label) in specs {
            let args = match label {
                Some(l) => json!({"minutes": minutes, "label": l}),
                None => json!({"minutes": minutes}),
            };
            SetTimer.call(&args, &ctx).await;
        }
        (ctx, rx, state)
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_by_name_however_the_user_phrases_it() {
        for phrasing in ["pasta", "the pasta timer", "Pasta"] {
            let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
            assert_eq!(
                CancelTimer.call(&json!({"name": phrasing}), &ctx).await,
                "Cancelled the pasta timer."
            );
            assert_eq!(state.with_timers(|b| b.len()), 1);
        }
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_by_partial_name() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta sauce"))]).await;
        assert_eq!(
            CancelTimer.call(&json!({"name": "pasta"}), &ctx).await,
            "Cancelled the pasta sauce timer."
        );
        assert!(state.with_timers(|b| b.is_empty()));
    }

    /// FIX 4 (final review): matching was raw substring containment
    /// (`n.contains(w) || w.contains(n)`), so with only a "tea" timer
    /// running, `{"name":"steak"}` matched because `"steak".contains("tea")`.
    /// Matching must respect word boundaries: "pasta" still finds "pasta
    /// sauce" (covered by `cancels_by_partial_name` above), but "steak" must
    /// not find "tea".
    #[tokio::test(start_paused = true)]
    async fn an_unrelated_word_does_not_match_on_raw_substring() {
        let (ctx, _rx, state) = board(&[(5.0, Some("tea"))]).await;
        assert_eq!(
            CancelTimer.call(&json!({"name": "steak"}), &ctx).await,
            "You don't have a steak timer. You have a tea timer."
        );
        assert_eq!(
            state.with_timers(|b| b.len()),
            1,
            "the tea timer must survive"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn cancels_by_duration_including_fractions() {
        let (ctx, _rx, state) = board(&[(5.0, None), (0.5, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"minutes": 0.5}), &ctx).await,
            "Cancelled the 30 second timer."
        );
        assert_eq!(state.with_timers(|b| b.len()), 1);
        assert_eq!(
            CancelTimer.call(&json!({"minutes": 5}), &ctx).await,
            "Cancelled the 5 minute timer."
        );
        assert!(state.with_timers(|b| b.is_empty()));
    }

    /// FIX 5 (final review): `cancel_timer` reused `set_timer`'s
    /// `parse_minutes`, which folds "non-finite or over the 24 hour cap"
    /// into `None` — indistinguishable from "argument absent". With one
    /// 2-minute timer running, `{"minutes":2000}` must not fall through to
    /// the sole-timer rule and cancel it.
    #[tokio::test(start_paused = true)]
    async fn an_over_cap_minutes_argument_does_not_match_the_sole_timer() {
        let (ctx, _rx, state) = board(&[(2.0, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"minutes": 2000}), &ctx).await,
            "You don't have a 2000 minute timer. You have a 2 minute timer."
        );
        assert_eq!(
            state.with_timers(|b| b.len()),
            1,
            "the only timer must survive"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_bare_cancel_works_when_there_is_exactly_one_timer() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta"))]).await;
        assert_eq!(
            CancelTimer.call(&json!({"name": "the timer"}), &ctx).await,
            "Cancelled the pasta timer."
        );
        assert!(state.with_timers(|b| b.is_empty()));
    }

    #[tokio::test(start_paused = true)]
    async fn a_bare_cancel_with_several_timers_asks_instead_of_guessing() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
        let reply = CancelTimer.call(&json!({}), &ctx).await;
        assert_eq!(
            reply,
            "You have a pasta timer with about 5 minutes left and a 10 minute \
             timer with about 10 minutes left. Which should I cancel?"
        );
        assert_eq!(state.with_timers(|b| b.len()), 2, "nothing was cancelled");
    }

    #[tokio::test(start_paused = true)]
    async fn duplicate_names_ask_instead_of_guessing() {
        let (ctx, _rx, state) = board(&[(3.0, Some("pasta")), (8.0, Some("pasta"))]).await;
        let reply = CancelTimer.call(&json!({"name": "pasta"}), &ctx).await;
        assert_eq!(
            reply,
            "You have a pasta timer with about 3 minutes left and a pasta timer \
             with about 8 minutes left. Which should I cancel?"
        );
        assert_eq!(state.with_timers(|b| b.len()), 2);
    }

    /// FIX 2 (final review): `name` and `minutes` were or-ed, never and-ed,
    /// so answering the "which pasta timer?" question with the duration
    /// ({"name":"pasta","minutes":3}) asked the identical question forever
    /// — only dropping the name worked. A name match with more than one
    /// candidate must be narrowed by duration when one is also supplied.
    #[tokio::test(start_paused = true)]
    async fn duplicate_names_narrowed_by_duration_resolve_instead_of_looping() {
        let (ctx, _rx, state) = board(&[(3.0, Some("pasta")), (8.0, Some("pasta"))]).await;
        let reply = CancelTimer.call(&json!({"name": "pasta"}), &ctx).await;
        assert_eq!(
            reply,
            "You have a pasta timer with about 3 minutes left and a pasta timer \
             with about 8 minutes left. Which should I cancel?"
        );
        assert_eq!(
            state.with_timers(|b| b.len()),
            2,
            "nothing was cancelled yet"
        );

        assert_eq!(
            CancelTimer
                .call(&json!({"name": "pasta", "minutes": 3}), &ctx)
                .await,
            "Cancelled the pasta timer."
        );
        let remaining = state.with_timers(|b| b.entries());
        assert_eq!(remaining.len(), 1, "only the 3 minute pasta timer is gone");
        assert!(
            (remaining[0].minutes - 8.0).abs() < 0.01,
            "the 8 minute pasta timer is still running"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn an_unknown_name_says_what_is_running() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"name": "rice"}), &ctx).await,
            "You don't have a rice timer. You have a pasta timer and a 10 minute timer."
        );
        assert_eq!(state.with_timers(|b| b.len()), 2);
    }

    /// The model routinely wraps a name the way `SetTimer`'s own callers do
    /// ("the rice timer", not bare "rice"). The no-match description must be
    /// built from the *normalized* name, not the raw phrase — otherwise this
    /// speaks the doubled, oddly-capitalized "You don't have a the rice timer
    /// timer."
    #[tokio::test(start_paused = true)]
    async fn an_unknown_wrapped_name_speaks_the_normalized_name() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta"))]).await;
        assert_eq!(
            CancelTimer
                .call(&json!({"name": "the rice timer"}), &ctx)
                .await,
            "You don't have a rice timer. You have a pasta timer."
        );
        assert_eq!(state.with_timers(|b| b.len()), 1);
    }

    /// FIX 1 (final review, blocks merge): `spoken_name` was stored raw
    /// while only the match key was normalized, so a routine wrapped label
    /// ("set the oven timer for 10 minutes") doubled "timer" in every
    /// spoken surface — the alert (spoken verbatim by TTS, up to five
    /// times), `list_timers`, and the `cancel_timer` confirmation alike.
    /// This is the same bug class commit 65f5276 already fixed for
    /// `cancel_timer`'s no-match reply; the set side was missed.
    #[tokio::test(start_paused = true)]
    async fn a_wrapped_label_does_not_double_timer_in_any_spoken_surface() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 5, "label": "the oven timer"}), &ctx)
            .await;

        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You have a oven timer with about 5 minutes left."
        );

        tokio::time::advance(Duration::from_secs(301)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx), vec!["Your oven timer is up."]);

        assert_eq!(
            CancelTimer.call(&json!({"name": "oven"}), &ctx).await,
            "Cancelled the oven timer."
        );
        assert!(state.with_timers(|b| b.is_empty()));
    }

    /// Same bug, the other realistic phrasing ("oven timer" with no leading
    /// "the") — both must normalize to the same spoken name.
    #[tokio::test(start_paused = true)]
    async fn a_wrapped_label_speaks_the_normalized_alert_however_phrased() {
        for label in ["the oven timer", "oven timer"] {
            let (ctx, mut rx, _state) = live_ctx();
            SetTimer
                .call(&json!({"minutes": 1, "label": label}), &ctx)
                .await;
            tokio::time::advance(Duration::from_secs(61)).await;
            tokio::task::yield_now().await;
            assert_eq!(
                spoken(&mut rx),
                vec!["Your oven timer is up."],
                "label {label:?}"
            );
        }
    }

    #[tokio::test(start_paused = true)]
    async fn an_unknown_duration_says_what_is_running() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"minutes": 7}), &ctx).await,
            "You don't have a 7 minute timer. You have a pasta timer and a 10 minute timer."
        );
        assert_eq!(state.with_timers(|b| b.len()), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_all_clears_the_board() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"all": true}), &ctx).await,
            "Cancelled all your timers."
        );
        assert!(state.with_timers(|b| b.is_empty()));
        // And the call is still usable afterwards.
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        assert_eq!(state.with_timers(|b| b.len()), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn cancel_all_with_one_timer_uses_the_singular() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta"))]).await;
        assert_eq!(
            CancelTimer.call(&json!({"all": true}), &ctx).await,
            "Cancelled your timer."
        );
        assert!(state.with_timers(|b| b.is_empty()));
    }

    /// The end-to-end proof of the send/cancel race fix `SetTimer` relies on:
    /// a timer that is actively ringing (already announced once, sleeping
    /// until its next repeat) gets silenced by `CancelTimer::call` itself —
    /// not just by calling `TimerBook::cancel` directly — and stays silent
    /// through every remaining repeat.
    #[tokio::test(start_paused = true)]
    async fn cancelling_a_ringing_timer_silences_it_end_to_end() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "tea"}), &ctx)
            .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            spoken(&mut rx),
            vec!["Your tea timer is up."],
            "first announcement"
        );
        assert!(
            state.with_timers(|b| b.entries()[0].ringing),
            "the timer is mid-ring when we cancel it"
        );

        assert_eq!(
            CancelTimer.call(&json!({"name": "tea"}), &ctx).await,
            "Cancelled the tea timer."
        );

        tokio::time::advance(REPEAT_EVERY * 10).await;
        tokio::task::yield_now().await;
        assert!(
            spoken(&mut rx).is_empty(),
            "no announcement after cancelling a ringing timer"
        );
        assert!(state.with_timers(|b| b.is_empty()), "book is cleaned up");
    }

    #[tokio::test(start_paused = true)]
    async fn cancelling_with_nothing_running_says_so() {
        let (ctx, _rx, _state) = live_ctx();
        assert_eq!(
            CancelTimer.call(&json!({"name": "pasta"}), &ctx).await,
            "You don't have any timers running."
        );
        assert_eq!(
            CancelTimer.call(&json!({"all": true}), &ctx).await,
            "You don't have any timers running."
        );
    }

    #[tokio::test]
    async fn cancelling_outside_a_call_answers_instead_of_failing() {
        let ctx = CallCtx::detached(1);
        assert_eq!(
            CancelTimer.call(&json!({"name": "pasta"}), &ctx).await,
            "You don't have any timers running."
        );
    }

    #[tokio::test(start_paused = true)]
    async fn lists_nothing_one_and_several() {
        let (ctx, _rx, _state) = live_ctx();
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You don't have any timers running."
        );

        SetTimer
            .call(&json!({"minutes": 5, "label": "pasta"}), &ctx)
            .await;
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You have a pasta timer with about 5 minutes left."
        );

        SetTimer.call(&json!({"minutes": 10}), &ctx).await;
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You have a pasta timer with about 5 minutes left and a 10 minute \
             timer with about 10 minutes left."
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_ringing_timer_is_reported_as_going_off() {
        let (ctx, _rx, _state) = live_ctx();
        SetTimer
            .call(&json!({"minutes": 1, "label": "pasta"}), &ctx)
            .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You have a pasta timer going off now."
        );
    }

    #[tokio::test]
    async fn listing_outside_a_call_answers_instead_of_failing() {
        let ctx = CallCtx::detached(1);
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You don't have any timers running."
        );
    }
}
