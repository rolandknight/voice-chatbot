# Named Timers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the fire-and-forget `set_timer` skill into a small per-call timer service that supports named timers, several at once, cancel-by-name, and a repeating spoken alert.

**Architecture:** A `TimerBook` (a `Vec<TimerEntry>` behind a `Mutex`) hangs off the existing per-call `CallState`. Each timer owns a `CancellationToken` that is a child of a call-level parent, and a spawned task that sleeps to the deadline, then announces up to five times ten seconds apart, checking the token before each announcement. Two new skills, `cancel_timer` and `list_timers`, read and mutate the same book. Nothing outside `crates/server/src/skills/` changes behaviour.

**Tech Stack:** Rust 2021, tokio 1 (`test-util` for the paused clock), `tokio-util` 0.7 (`rt` feature) for `CancellationToken`, `async-trait`, `serde_json`, `flowcat-core` for `Frame::TtsSpeak`.

**Spec:** `docs/superpowers/specs/2026-08-29-named-timers-design.md`

## Global Constraints

- Rust edition 2021; the workspace builds with `cargo` from `.hermit` (see `Makefile:13`).
- `tokio-util = { version = "0.7", features = ["rt"] }` — must match the version vendored flowcat-core already uses (`third_party/flowcat-core/Cargo.toml:62`). Do not bump it.
- `Skill::call` must **never** panic and never return `Err`. Every failure folds into spoken text (`crates/server/src/skills/mod.rs:236`).
- These existing reply strings are **byte-identical contracts** — tests assert them and the LLM prompt depends on them. Do not reword:
  - `"I couldn't understand the timer duration."`
  - `"The timer duration needs to be greater than zero."`
  - `"I can't set a timer right now."`
  - `"Timer set for {pretty}{tail}."` (e.g. `"Timer set for 30 seconds for tea."`)
  - `"Your {label} timer is up."` / `"Your timer is up."`
- All new spoken output is **read aloud by TTS**. No digits-as-symbols, no punctuation that does not speak, no markdown.
- Every timing test runs on the paused clock: `#[tokio::test(start_paused = true)]` with `tokio::time::advance`. Never `std::thread::sleep`, never a real wall-clock wait.
- Timers remain **per call**. Nothing is persisted to disk and nothing survives the call ending.

## Tight-loop test command

```bash
cargo test -p voice-chatbot-server skills::timer
```

Default features only — no `PYO3_PYTHON` or `NEMO_SPEECH_LIB_DIR` needed (`crates/server/build.rs` returns early without the `moonshine` feature). The full gate before the final commit is `make check` (fmt, clippy `-D warnings`, workspace tests).

## File Structure

| File | Responsibility | Change |
| --- | --- | --- |
| `crates/server/src/skills/timer.rs` | All timer behaviour: the book, the three skills, parsing, matching, spoken-phrase helpers, tests | Heavily extended (currently 160 lines) |
| `crates/server/src/skills/mod.rs` | Per-call state container | One field + one accessor on `CallState` |
| `crates/server/skills.json` | LLM-facing tool schemas | `set_timer` description updated; two schemas added |
| `crates/server/src/main.rs` | Startup skill registration | Two lines beside `SetTimer` (line 1015) |
| `crates/server/Cargo.toml` | Dependencies | `tokio-util` |
| `README.md`, `docs/prd/prd.md` | User-facing docs | Skill list (`README.md:166`), SKILL-3 row |

Everything stays in `timer.rs` deliberately: the book, the matching rules and the three skills change together, and splitting them across files would put `TimerEntry`'s private fields behind accessors for no gain. At the end of this plan `timer.rs` is roughly 700 lines including tests, which is in line with `shows.rs` (17 K) and `sfx.rs` (11 K).

---

### Task 1: Stop `set_timer` panicking on a non-finite duration

A latent bug found while reviewing. `parse_minutes` accepts `"NaN"` and `"inf"` from a JSON string; the guard `minutes <= 0.0` is **false** for NaN, so it slips through; `Duration::from_secs_f64(NaN * 60.0)` then panics, taking down the tool-call task instead of returning spoken text. Verified against rustc 1.97.1: `cannot convert float seconds to Duration: value is either too big or NaN`.

This task is independent of everything else and ships on its own.

**Files:**
- Modify: `crates/server/src/skills/timer.rs:36-42` (`parse_minutes`)
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `const MAX_MINUTES: f64` and a `parse_minutes` that returns `None` for any value that is not finite or exceeds `MAX_MINUTES`. Tasks 5 and 6 both call `parse_minutes`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `crates/server/src/skills/timer.rs`:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: `minutes_rejects_non_finite_and_absurd_values` FAILS with `assertion \`left == right\` failed: left: Some(NaN)`.

- [ ] **Step 3: Harden `parse_minutes`**

Replace `parse_minutes` (`timer.rs:35-42`) with:

```rust
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
```

Negative values still return `Some(-1.0)` so the existing "greater than zero" reply is preserved — do not fold that case in here.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS, including the four pre-existing tests.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/timer.rs
git commit -m "fix(timer): reject non-finite durations instead of panicking

parse_minutes accepted \"NaN\"/\"inf\" from a JSON string, the minutes <= 0.0
guard is false for NaN, and Duration::from_secs_f64 then panicked the
tool-call task. Skill::call must never panic."
```

---

### Task 2: Spoken-phrase helpers — normalization, remaining time, adjectives, list joining

Pure functions with table tests. No state, no async. Everything later depends on these, so they land first and separately.

**Files:**
- Modify: `crates/server/src/skills/timer.rs`
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces, all used by Tasks 5–7:
  - `pub fn normalize_name(raw: &str) -> Option<String>`
  - `pub fn format_remaining(left: Duration) -> String`
  - `pub fn duration_adjective(minutes: f64) -> String`
  - `pub fn join_and(parts: &[String]) -> String`

- [ ] **Step 1: Write the failing tests**

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL to compile — `cannot find function \`normalize_name\` in this scope`.

- [ ] **Step 3: Implement the helpers**

Add near the top of `timer.rs`, after `format_duration`:

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/timer.rs
git commit -m "feat(timer): spoken-phrase helpers for names, countdowns and lists"
```

---

### Task 3: The `TimerBook`

The registry itself, with no skill wired to it yet. Fully unit-testable without a pipeline or a call.

**Files:**
- Modify: `crates/server/Cargo.toml` (dependencies)
- Modify: `crates/server/src/skills/timer.rs`
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces, used by Tasks 4–7:
  - `pub struct TimerEntry { pub id: u64, pub name: Option<String>, pub spoken_name: Option<String>, pub minutes: f64, pub deadline: Instant, pub ringing: bool, pub cancel: CancellationToken }` — `Clone`
  - `pub struct TimerBook` — `Default`
  - `TimerBook::insert(&mut self, name: Option<String>, spoken_name: Option<String>, minutes: f64, deadline: Instant) -> (u64, CancellationToken)`
  - `TimerBook::remove(&mut self, id: u64)`
  - `TimerBook::mark_ringing(&mut self, id: u64)`
  - `TimerBook::cancel(&mut self, id: u64) -> bool`
  - `TimerBook::cancel_all(&mut self) -> usize`
  - `TimerBook::entries(&self) -> Vec<TimerEntry>`
  - `TimerBook::len(&self) -> usize`
  - `TimerBook::is_empty(&self) -> bool`

- [ ] **Step 1: Add the dependency**

In `crates/server/Cargo.toml`, after the `tokio` line:

```toml
# CancellationToken for per-timer cancellation (skills/timer.rs). Already in
# the tree: vendored flowcat-core depends on the same version and feature.
tokio-util = { version = "0.7", features = ["rt"] }
```

- [ ] **Step 2: Write the failing tests**

```rust
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL to compile — `cannot find type \`TimerBook\` in this scope`.

- [ ] **Step 4: Implement `TimerEntry` and `TimerBook`**

In `timer.rs`, **replace** the existing `use std::time::Duration;` (line 9) —
do not add a second `use` for the same module — and add the token import:

```rust
use std::time::{Duration, Instant};

use tokio_util::sync::CancellationToken;
```

Then, after the helpers from Task 2:

```rust
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/server/Cargo.toml Cargo.lock crates/server/src/skills/timer.rs
git commit -m "feat(timer): per-call TimerBook with child cancellation tokens"
```

---

### Task 4: Hang the book off `CallState`

**Files:**
- Modify: `crates/server/src/skills/mod.rs:95-108` (the `CallState` struct) and its `impl` block
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: `timer::TimerBook` from Task 3.
- Produces: `CallState::with_timers<R>(&self, f: impl FnOnce(&mut timer::TimerBook) -> R) -> R` — the single accessor Tasks 5–7 use. The `Mutex` never escapes, so no lock can be held across an `.await`.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn call_state_owns_a_timer_book() {
        let state = CallState::default();
        assert!(state.with_timers(|b| b.is_empty()));
        let token = state.with_timers(|b| b.insert(None, None, 1.0, at(60)).1);
        assert_eq!(state.with_timers(|b| b.len()), 1);
        drop(state);
        assert!(token.is_cancelled(), "dropping the call cancels its timers");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL to compile — `no method named \`with_timers\` found for struct \`CallState\``.

- [ ] **Step 3: Bring `CallState` into scope in `timer.rs`**

`timer.rs` currently imports only `use super::{arg_str, CallCtx, Skill};`, and
`mod tests` picks up its scope through `use super::*`. Every test from here on
names `CallState`, so widen it:

```rust
use super::{arg_str, CallCtx, CallState, Skill};
```

- [ ] **Step 4: Add the field and the accessor**

In `crates/server/src/skills/mod.rs`, add to the `CallState` struct (after `wake_armed_at`):

```rust
    /// Live countdown timers for this call (`skills/timer.rs`). Dropping
    /// `CallState` cancels them all, so nothing outlives the call.
    timers: Mutex<timer::TimerBook>,
```

And to `impl CallState`:

```rust
    /// Operate on this call's timers. The lock never escapes, so it cannot be
    /// held across an `.await`.
    pub fn with_timers<R>(&self, f: impl FnOnce(&mut timer::TimerBook) -> R) -> R {
        f(&mut self.timers.lock().unwrap())
    }
```

`#[derive(Default)]` on `CallState` keeps working because `TimerBook` implements `Default`.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/server/src/skills/mod.rs crates/server/src/skills/timer.rs
git commit -m "feat(timer): hang the TimerBook off CallState"
```

---

### Task 5: `set_timer` registers its timer and repeats the alert

The core behaviour change. Where the old skill spawned a detached sleep-then-speak, the new one registers an entry and runs a cancellable announce loop.

**Files:**
- Modify: `crates/server/src/skills/timer.rs` (`SetTimer::call`)
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: `parse_minutes` (Task 1), `normalize_name` (Task 2), `TimerBook` (Task 3), `CallState::with_timers` (Task 4).
- Produces: `pub const REPEAT_EVERY: Duration`, `pub const MAX_ANNOUNCEMENTS: usize`, and the invariant Tasks 6–7 rely on: **a timer is in the book from the moment `set_timer` returns until its task exits, ringing included.**

**Critical detail — the task must hold a `Weak<CallState>`, never an `Arc`.** `Arc<CallState>` is held by `CallHandle` (`call.rs:545`), the wake gate (`call.rs:410`) and the Qwen TTS stage (`call.rs:494`); all are released when the call ends. A timer task holding a strong `Arc` would keep `CallState` — and therefore `TimerBook` — alive, so `Drop for TimerBook` would never run and a 30-minute timer would keep a task parked long after the call was over. With a `Weak`, the last strong reference dropping cancels the token and the task wakes immediately.

- [ ] **Step 1: Write the failing tests**

Replace the existing `timer_speaks_into_the_call_after_the_delay` test (it constructs `CallCtx { state: None }`, which no longer sets a timer) and add the rest:

```rust
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
        assert!(spoken(&mut rx).is_empty(), "nothing spoken before the delay");
        tokio::time::advance(Duration::from_secs(31)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx), vec!["Your tea timer is up."]);
    }

    #[tokio::test(start_paused = true)]
    async fn alert_repeats_a_bounded_number_of_times() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer.call(&json!({"minutes": 1, "label": "tea"}), &ctx).await;
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
        SetTimer.call(&json!({"minutes": 5, "label": "tea"}), &ctx).await;
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
        SetTimer.call(&json!({"minutes": 1, "label": "tea"}), &ctx).await;
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
    }

    #[tokio::test(start_paused = true)]
    async fn several_timers_run_at_once_and_each_says_its_own_name() {
        let (ctx, mut rx, state) = live_ctx();
        SetTimer.call(&json!({"minutes": 1, "label": "tea"}), &ctx).await;
        SetTimer.call(&json!({"minutes": 2, "label": "pasta"}), &ctx).await;
        SetTimer.call(&json!({"minutes": 3}), &ctx).await;
        assert_eq!(state.with_timers(|b| b.len()), 3);

        tokio::time::advance(Duration::from_secs(61)).await;
        tokio::task::yield_now().await;
        assert_eq!(spoken(&mut rx), vec!["Your tea timer is up."]);

        tokio::time::advance(Duration::from_secs(60)).await;
        tokio::task::yield_now().await;
        let heard = spoken(&mut rx);
        assert!(heard.contains(&"Your pasta timer is up.".to_string()));
        assert!(heard.contains(&"Your tea timer is up.".to_string()), "tea repeats");
    }

    #[tokio::test(start_paused = true)]
    async fn a_timer_whose_call_ended_is_dropped() {
        let (ctx, rx, state) = live_ctx();
        SetTimer.call(&json!({"minutes": 1, "label": "tea"}), &ctx).await;
        drop(rx);
        drop(ctx);
        drop(state); // the call ended: last strong Arc<CallState> released
        tokio::time::advance(Duration::from_secs(300)).await;
        tokio::task::yield_now().await;
        // No panic, no leak. The task woke on the cancelled parent and exited.
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL — `alert_repeats_a_bounded_number_of_times` fails at `announcement 2` (the current code speaks once), and `cannot find value \`MAX_ANNOUNCEMENTS\``.

- [ ] **Step 3: Rewrite `SetTimer::call`**

Add the constants near `MAX_MINUTES`:

```rust
/// Gap between repeats of the expiry alert.
pub const REPEAT_EVERY: Duration = Duration::from_secs(10);
/// Total announcements before a timer gives up: the initial alert plus four
/// repeats, a ringing window of about forty seconds.
pub const MAX_ANNOUNCEMENTS: usize = 5;
```

Replace the body of `SetTimer::call`:

```rust
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
        let spoken_name = (!label.is_empty()).then(|| label.clone());
        let name = spoken_name.as_deref().and_then(normalize_name);

        let delay = Duration::from_secs_f64(minutes * 60.0);
        let (id, token) = state.with_timers(|b| {
            b.insert(name, spoken_name.clone(), minutes, Instant::now() + delay)
        });

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
                _ = tokio::time::sleep(delay) => {}
            }
            if let Some(s) = state.upgrade() {
                s.with_timers(|b| b.mark_ringing(id));
            }
            for i in 0..MAX_ANNOUNCEMENTS {
                // `cancel()` sets this synchronously, so checking here means a
                // cancelled timer can never get one last word in.
                if token.is_cancelled() {
                    break;
                }
                let frame = Frame::TtsSpeak {
                    text: text.clone(),
                    // Record the timer going off once, not five times.
                    append_to_context: (i > 0).then_some(false),
                };
                if frames.send(frame).is_err() {
                    tracing::info!(run_id, %text, "timer fired after the call ended; dropped");
                    break;
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
            if let Some(s) = state.upgrade() {
                s.with_timers(|b| b.remove(id));
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS, including the pre-existing `rejects_bad_durations_and_missing_pipeline`.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/timer.rs
git commit -m "feat(timer): register timers in the book and repeat the alert

The expiry alert now repeats up to MAX_ANNOUNCEMENTS times REPEAT_EVERY
apart, checking its CancellationToken before each one so a cancel is
never followed by one more announcement. The task holds a Weak<CallState>
so it never outlives its call."
```

---

### Task 6: `cancel_timer`

**Files:**
- Modify: `crates/server/src/skills/timer.rs`
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `pub struct CancelTimer` implementing `Skill` with `name() == "cancel_timer"`, plus two private helpers `a_timer(&TimerEntry) -> String` and `the_timer(&TimerEntry) -> String`. Task 7 uses `a_timer`.

- [ ] **Step 1: Write the failing tests**

```rust
    /// Set `n` timers and return the ctx/receiver/state, so a cancel test can
    /// start from a known board.
    async fn board(specs: &[(f64, Option<&str>)]) -> (
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

    #[tokio::test(start_paused = true)]
    async fn an_unknown_name_says_what_is_running() {
        let (ctx, _rx, state) = board(&[(5.0, Some("pasta")), (10.0, None)]).await;
        assert_eq!(
            CancelTimer.call(&json!({"name": "rice"}), &ctx).await,
            "You don't have a rice timer. You have a pasta timer and a 10 minute timer."
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
        SetTimer.call(&json!({"minutes": 1, "label": "tea"}), &ctx).await;
        assert_eq!(state.with_timers(|b| b.len()), 1);
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL to compile — `cannot find value \`CancelTimer\` in this scope`.

- [ ] **Step 3: Implement the describing helpers and `CancelTimer`**

```rust
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
        let minutes = parse_minutes(args.get("minutes"));

        let candidates: Vec<TimerEntry> = if let Some(w) = &wanted {
            let exact: Vec<TimerEntry> = entries
                .iter()
                .filter(|t| t.name.as_deref() == Some(w.as_str()))
                .cloned()
                .collect();
            if exact.is_empty() {
                // "pasta" should still find "pasta sauce", and vice versa.
                entries
                    .iter()
                    .filter(|t| {
                        t.name
                            .as_deref()
                            .is_some_and(|n| n.contains(w.as_str()) || w.contains(n))
                    })
                    .cloned()
                    .collect()
            } else {
                exact
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
                let what = if raw.is_empty() {
                    format_duration(minutes.unwrap_or_default())
                } else {
                    raw.to_string()
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/timer.rs
git commit -m "feat(timer): cancel_timer with name, duration and sole-timer matching

Ambiguity is answered with a spoken question rather than a guess."
```

---

### Task 7: `list_timers`

**Files:**
- Modify: `crates/server/src/skills/timer.rs`
- Test: `crates/server/src/skills/timer.rs` (`mod tests`)

**Interfaces:**
- Consumes: `a_timer`, `timer_with_remaining` (Task 6), `TimerBook` (Task 3).
- Produces: `pub struct ListTimers` implementing `Skill` with `name() == "list_timers"`.

- [ ] **Step 1: Write the failing tests**

```rust
    #[tokio::test(start_paused = true)]
    async fn lists_nothing_one_and_several() {
        let (ctx, _rx, _state) = live_ctx();
        assert_eq!(
            ListTimers.call(&json!({}), &ctx).await,
            "You don't have any timers running."
        );

        SetTimer.call(&json!({"minutes": 5, "label": "pasta"}), &ctx).await;
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
        SetTimer.call(&json!({"minutes": 1, "label": "pasta"}), &ctx).await;
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: FAIL to compile — `cannot find value \`ListTimers\` in this scope`.

- [ ] **Step 3: Implement `ListTimers`**

```rust
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p voice-chatbot-server skills::timer`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/server/src/skills/timer.rs
git commit -m "feat(timer): list_timers reports what is running and how long is left"
```

---

### Task 8: Advertise the tools and update the docs

Until this task the two new skills exist but the LLM cannot call them — `Registry::new` only advertises a skill that has a schema, and only constructs what `main.rs` passes it. Note that `Registry::new` **fails at startup** if a constructed skill has no schema (`skills/mod.rs:289`), so the JSON and the registration must land in the same commit.

**Files:**
- Modify: `crates/server/skills.json` (the `set_timer` entry; two new entries)
- Modify: `crates/server/src/main.rs:1015`
- Modify: `README.md:166`
- Modify: `docs/prd/prd.md` (SKILL-3 row)

**Interfaces:**
- Consumes: `SetTimer` (Task 5), `CancelTimer` (Task 6), `ListTimers` (Task 7).
- Produces: nothing further.

- [ ] **Step 1: Update the `set_timer` description in `crates/server/skills.json`**

Replace its `description` (keep `parameters` exactly as they are):

```
"Start a countdown timer. When it expires the assistant says 'Your <label> timer is up' out loud and repeats it a few times. Use for any request like 'set a timer', 'remind me in N minutes', 'wake me in N minutes'. Always pass the label the user gave it ('pasta', 'tea', 'laundry') — that label is how the user cancels or asks about this timer later. Several timers can run at once."
```

- [ ] **Step 2: Add the two schemas to `crates/server/skills.json`**

Insert immediately after the `set_timer` entry:

```json
  {
    "name": "cancel_timer",
    "description": "Cancel a running timer, or silence one that is going off right now. Use for 'cancel the pasta timer', 'stop the timer', 'turn it off', 'never mind the timer'. Pass the name the user gave the timer if they named one; pass minutes instead when they identify it by its length ('cancel the five minute timer'); pass all=true for 'cancel all my timers'. Pass no arguments when the user says only 'cancel the timer' without saying which one.",
    "parameters": {
      "type": "object",
      "properties": {
        "name": {
          "type": "string",
          "description": "Name of the timer to cancel, as the user said it (e.g. 'pasta', 'the tea timer')."
        },
        "minutes": {
          "type": "number",
          "description": "Length the timer was originally set for, when the user identifies it that way."
        },
        "all": {
          "type": "boolean",
          "description": "Cancel every running timer."
        }
      },
      "required": []
    }
  },
  {
    "name": "list_timers",
    "description": "Say which timers are running and how long is left on each. Use for 'what timers do I have?', 'how long left on the pasta timer?', 'is my timer still going?'.",
    "parameters": {
      "type": "object",
      "properties": {},
      "required": []
    }
  },
```

- [ ] **Step 3: Register the skills in `crates/server/src/main.rs`**

At line 1015, beside `Arc::new(skills::timer::SetTimer),`:

```rust
        Arc::new(skills::timer::SetTimer),
        Arc::new(skills::timer::CancelTimer),
        Arc::new(skills::timer::ListTimers),
```

- [ ] **Step 4: Verify the registry accepts them**

Run: `cargo test -p voice-chatbot-server skills::`
Expected: PASS, in particular `shipped_skills_json_parses` (`skills/mod.rs:373`), which parses the real `skills.json`.

- [ ] **Step 5: Update the docs**

`README.md:166` — replace the single bullet with:

```markdown
- `set_timer(minutes, label?)` — counts down and speaks the alert out loud,
  repeating it up to five times ten seconds apart. Several timers can run at
  once; timers are per call and do not survive a disconnect.
- `cancel_timer(name?, minutes?, all?)` — cancels a timer by name ("the pasta
  timer"), by length ("the five minute timer"), or the only one running, and
  silences one that is going off. Ambiguity is answered with a question.
- `list_timers()` — says what is running and how long is left on each.
```

`docs/prd/prd.md`, the SKILL-3 row — replace `set_timer(minutes, label?)` with a spoken alert` with:

```
`set_timer(minutes, label?)` with a repeating spoken alert, `cancel_timer(name?, minutes?, all?)`, `list_timers()`
```

- [ ] **Step 6: Run the full gate**

Run: `make check`
Expected: `cargo fmt --check` clean, `clippy -D warnings` clean, whole workspace test suite passing.

- [ ] **Step 7: Commit**

```bash
git add crates/server/skills.json crates/server/src/main.rs README.md docs/prd/prd.md
git commit -m "feat(timer): advertise cancel_timer and list_timers to the LLM"
```

---

## Deviations from the spec

Recorded so a reviewer can see them without diffing:

1. **`Drop for TimerBook` instead of a `DropGuard` held by `CallState`.** The spec proposed `CallState` holding `call_token.clone().drop_guard()`. That breaks when `cancel_all` replaces the parent token — the guard would still be watching the dead one. Implementing `Drop` on the book itself always cancels the *current* parent and needs no re-arming.
2. **The task holds `Weak<CallState>`.** Not in the spec, and load-bearing: an `Arc` would keep the book alive and stop `Drop` from ever running. See the note in Task 5.
3. **Disambiguation wording is generalized.** The spec's example was `"You have two pasta timers — one with 3 minutes left and one with 8. Which should I cancel?"`. The implementation produces `"You have a pasta timer with about 3 minutes left and a pasta timer with about 8 minutes left. Which should I cancel?"` — slightly more repetitive, but one code path for every arity and no counting words.
4. **Cancel-all replies avoid number words** (`"Cancelled all your timers."` rather than `"Cancelled all three timers."`), which removes a spell-the-number helper for no loss of clarity.
5. **`cancel_timer` and `list_timers` outside a live call** return `"You don't have any timers running."` rather than a distinct error. It is true, and it is what the user should hear.

## Out of scope

Per the spec's non-goals: persistence across a disconnect or restart; rescheduling ("add five minutes"); wall-clock alarms ("wake me at seven"); a distinct alert sound; and auto-stopping the alert on any user speech.
