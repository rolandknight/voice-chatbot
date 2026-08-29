# Named timers — cancellable, repeating, multi-timer support for the `set_timer` skill

**Date:** 2026-08-29 · **Branch:** `named-timers` · **Crates:** `server`

## Goal

Turn the one-shot `set_timer` skill into a small timer service that supports the
five capabilities a household assistant needs:

1. set a timer
2. set a *named* timer
3. run several timers at once
4. cancel a timer by name
5. on expiry, repeat "Your `<name>` timer is up" rather than saying it once

Everything stays inside the skills layer. No pipeline, transport or `call.rs`
changes.

## What exists today

One skill, `SetTimer`, in `crates/server/src/skills/timer.rs:52`, with its
schema in `crates/server/skills.json`. It parses `minutes` (number or string),
takes an optional `label`, then `tokio::spawn`s a `sleep` and sends a single
`Frame::TtsSpeak` into the call's head-injection queue (`timer.rs:76-88`). That
queue is `PipelineTask::queue_sender()`, registered per call at
`crates/server/src/call.rs:543`.

Skills are startup singletons (`main.rs:1015`); everything per-call reaches them
through `CallCtx` (`skills/mod.rs:38`), whose `state: Option<Arc<CallState>>`
field is the existing home for per-call mutable data — voice, persona prompt,
LLM backend and wake-grace all live there behind `Mutex`es (`skills/mod.rs:96`).

## The gaps

| Capability | Status | Gap |
| --- | --- | --- |
| Set timer | works | — |
| Named timer | half | `label` is spoken at fire time but never stored or normalized, so it is not an addressable name |
| Multiple timers | accidental | N spawns do run concurrently, but nothing tracks them; two unnamed timers are indistinguishable and duplicate labels pass silently |
| Cancel by name | missing | the `JoinHandle` is dropped at `timer.rs:76`; no registry, no tool, no schema |
| Repeating alert | missing | `timer.rs:77-88` sends once and the task exits |

Two knock-on gaps: there is no way to enumerate timers, so the assistant cannot
answer "what timers do I have?" nor disambiguate an ambiguous cancel; and a
timer whose call has ended is logged and dropped (`timer.rs:85`).

## Decisions

| Question | Decision |
| --- | --- |
| What stops the repeating alert? | A cap (5 announcements, 10 s apart) plus an explicit `cancel_timer`. No frame observer, no pipeline changes. |
| How is a timer addressed? | Name first, then requested duration, then "the only one running". Real ambiguity gets a spoken question, never a guess. |
| How long does a timer live? | Per call, as today. It dies with `CallState`. |
| Where does the registry live? | On `CallState`, with the type and all logic in `timer.rs`. |

Rejected: a process-wide registry keyed by `run_id` (re-implements the per-call
lifetime `CallState` already gives, and leaks without explicit teardown in
`CallRegistry::unregister`); a per-call timer actor task with a command channel
(lock-free and tidy, but needs a task per call, a channel on `CallHandle`,
`call.rs` changes and reply plumbing — far more machinery than one `Mutex<Vec>`).

## Cancellation primitive: `tokio_util::sync::CancellationToken`

Add `tokio-util = { version = "0.7", features = ["rt"] }` to
`crates/server/Cargo.toml`. This is not new surface: tokio-util 0.7.19 is
already in `Cargo.lock`, and vendored flowcat-core declares it directly with the
same `rt` feature (`third_party/flowcat-core/Cargo.toml:62`), which is the
feature `CancellationToken` sits behind. Declaring it in the server adds zero
compilation.

Chosen over `tokio::task::AbortHandle` on the merits:

1. **No stray announcement after "stop".** `abort()` is asynchronous — it stops
   the task at its next await point, so a ringing task that has already passed
   its check can still push one more `TtsSpeak`. `cancel()` sets the flag
   synchronously, so `is_cancelled()` immediately before each send is
   authoritative. This is the difference between "stop!" being obeyed and
   "stop!" … "Your pasta timer is up."
2. **One cleanup path.** An aborted task cannot run cleanup, so removing the
   entry would have to be duplicated in the canceller and raced against the
   task's own completion. With a token the task always exits through its own
   tail.
3. **Cancel-all and end-of-call fall out of the hierarchy.** A call-level parent
   token with `child_token()` per timer makes "cancel all timers" one `cancel()`.
   `CallState` also holds `call_token.clone().drop_guard()`, so dropping it when
   the call ends cancels every pending timer immediately instead of leaving a
   long-running task asleep until its deadline. (A plain `CancellationToken` drop
   does *not* cancel — the `DropGuard` is what makes this true.)
4. **It is already the codebase idiom.** `PipelineTask::cancel_token()`
   (`third_party/flowcat-core/src/pipeline/task.rs:167`) returns one, and
   `call.rs:548` already drives hang-up through it.

## Data model

In `timer.rs`:

```rust
pub struct TimerEntry {
    id: u64,                     // monotonic per call; unambiguous internal reference
    name: Option<String>,        // normalized; None for an unnamed timer
    spoken_name: Option<String>, // as the user said it, for the alert text
    minutes: f64,                // as requested, for "the 5 minute timer" matching
    deadline: Instant,           // for "how long left"
    ringing: bool,               // fired, still announcing
    cancel: CancellationToken,   // child of the call token
}

pub struct TimerBook {
    call_token: CancellationToken,   // parent; each timer holds a child
    next_id: u64,
    timers: Vec<TimerEntry>,
}
```

**`call_token` must be replaced, not reused, after a cancel-all.** A cancelled
`CancellationToken` stays cancelled forever, so a timer created afterwards from
`child_token()` would be born already-cancelled and never fire. `cancel_all()`
therefore cancels the current parent and immediately installs a fresh one, with
`CallState` re-arming its `DropGuard` against the replacement. This gets a test
of its own: set a timer after a cancel-all and assert it still fires.

`CallState` gains one field, `timers: Mutex<TimerBook>`, and one accessor,
matching the `Mutex` pattern already used there. All behaviour stays in
`timer.rs` so `mod.rs` does not grow a second responsibility.

An entry stays in the book *while ringing*, so a ringing alert is cancellable,
and removes itself when its task exits by any route.

## Firing and repeating

Two constants in `timer.rs`, so the cadence is one edit and directly testable:

```rust
const REPEAT_EVERY: Duration = Duration::from_secs(10);
const MAX_ANNOUNCEMENTS: usize = 5;   // initial + 4 repeats ≈ a 40 s ringing window
```

The spawned task:

```rust
tokio::select! {                                  // pending phase
    _ = token.cancelled() => { book.remove(id); return; }
    _ = tokio::time::sleep(delay) => {}
}
book.mark_ringing(id);
for i in 0..MAX_ANNOUNCEMENTS {
    if token.is_cancelled() { break; }            // cancel() is synchronous: authoritative
    if frames.send(Frame::TtsSpeak { .. }).is_err() { break; }   // call ended
    if i + 1 == MAX_ANNOUNCEMENTS { break; }
    tokio::select! {
        _ = token.cancelled() => break,
        _ = tokio::time::sleep(REPEAT_EVERY) => {}
    }
}
book.remove(id);                                  // single exit path, every route
```

Notes:

- **Alert text is unchanged.** `alert_text()` (`timer.rs:44`) already produces
  "Your pasta timer is up." / "Your timer is up.", and every repeat says the
  identical sentence.
- **Barge-in quiets it for free.** User speech makes the pipeline broadcast
  `Interruption` and flush queued speech, so the current announcement stops with
  no timer-side work. The repeat loop continues unless a `cancel_timer` lands —
  correct, since a passing remark should not kill the alert but "stop the pasta
  timer" should.
- **`append_to_context`.** Repeats 2–5 will pass `Some(false)` so history records
  the timer firing once, not five times. Grepping flowcat-core finds only
  producers of that field and no consumer, so today it appears advisory; set it
  as intent-documenting and confirm the consumer during implementation. It is
  not load-bearing.

## Tool surface

### `set_timer(minutes, label?)`

Schema shape unchanged, so existing behaviour and reply strings are preserved
verbatim ("Timer set for 30 seconds for tea."). Only the description changes, to
tell the model that a label is how a timer is cancelled later and that the alert
repeats. Duplicate labels are allowed and resolved by the disambiguation path at
cancel time, not by a rule at set time.

### `cancel_timer(name?, minutes?, all?)`

Resolution order:

1. `all: true` → cancel everything (one `cancel()` on the parent token).
2. `name` → normalize, then exact match, then substring (so "pasta" finds
   "pasta sauce").
3. `minutes` → match the *requested* duration: "the 5 minute timer". Compared
   with a tolerance (`(a - b).abs() < 0.01`), never `==` on `f64`.
4. No argument and exactly one timer → cancel it.
5. Anything else → a spoken question, and nothing is cancelled.

Replies: `"Cancelled the pasta timer."` · `"Cancelled the 5 minute timer."` ·
`"You don't have a pasta timer. You have a 5 minute timer and a rice timer."` ·
`"You have two pasta timers — one with 3 minutes left and one with 8. Which
should I cancel?"` Cancelling a ringing timer uses the same text and stops the
announcements.

### `list_timers()`

No arguments. `"You don't have any timers running."` · `"You have one timer:
pasta, 3 minutes left."` · `"You have two timers: pasta with 3 minutes left, and
a 10 minute timer with 7 minutes left."` A ringing timer reports as "your pasta
timer is going off now".

### Schemas to add to `crates/server/skills.json`

```json
{
  "name": "cancel_timer",
  "description": "Cancel a running timer, or silence one that is going off right now. Use for 'cancel the pasta timer', 'stop the timer', 'turn it off', 'never mind the timer'. Pass the name the user gave the timer if they named one; pass minutes instead when they identify it by its length ('cancel the five minute timer'); pass all=true for 'cancel all my timers'. Pass no arguments when the user says only 'cancel the timer' without saying which one.",
  "parameters": {
    "type": "object",
    "properties": {
      "name": { "type": "string", "description": "Name of the timer to cancel, as the user said it (e.g. 'pasta', 'the tea timer')." },
      "minutes": { "type": "number", "description": "Length the timer was set for, when the user identifies it that way." },
      "all": { "type": "boolean", "description": "Cancel every running timer." }
    },
    "required": []
  }
}
```

```json
{
  "name": "list_timers",
  "description": "Say which timers are running and how long is left on each. Use for 'what timers do I have?', 'how long left on the pasta timer?', 'is my timer still going?'.",
  "parameters": { "type": "object", "properties": {}, "required": [] }
}
```

## Name normalization and matching

`normalize(name)`: trim, lowercase, collapse internal whitespace, strip a
leading `"the "`, strip a trailing `" timer"`. So `"The Pasta Timer"` → `"pasta"`.

This is the load-bearing piece for voice. The model passes `label: "pasta"` at
set time, but on "cancel the pasta timer" it will frequently pass
`"the pasta timer"`. Without normalization, cancel-by-name misses constantly.

`format_remaining(Duration)` is new and separate from the existing
`format_duration`, which produces things like `"2.5 minutes"` that read badly as
a countdown. Under a minute → `"30 seconds"`; otherwise rounded →
`"about 3 minutes"`. Leaving `format_duration` untouched keeps its Python-parity
tests green.

## Error handling

Every path folds into spoken text. The `Skill` trait contract
(`skills/mod.rs:236`) is that `call` never fails.

**Latent bug fixed here.** `parse_minutes` (`timer.rs:36`) accepts `"NaN"` and
`"inf"` from a JSON string, the guard `minutes <= 0.0` (`timer.rs:65`) is *false*
for NaN so it slips through, and `Duration::from_secs_f64(NaN * 60.0)`
(`timer.rs:73`) panics — verified against rustc 1.97.1:
`cannot convert float seconds to Duration: value is either too big or NaN`. An
LLM emitting `{"minutes": "NaN"}` therefore panics the tool-call task instead of
getting a spoken reply. (`-inf` is caught by the existing guard; only NaN and
`+inf` get through.) Fix: require `is_finite()` and cap at 24 hours before
constructing a `Duration`, reusing the existing string "I couldn't understand the
timer duration." All other duration strings stay byte-identical.

Other cases:

- `set_timer` now needs both `frames` and `state`. In production
  `CallRegistry::ctx` (`skills/mod.rs:223`) populates both from the same handle,
  so it is both-or-neither; missing → the existing "I can't set a timer right
  now."
- Cancelling or listing with nothing running is a normal spoken answer.
- A timer whose call has ended: `frames.send` fails, the task logs and removes
  itself, as today.

## Files touched

| File | Change |
| --- | --- |
| `crates/server/src/skills/timer.rs` | `TimerBook`, `TimerEntry`, rewritten `SetTimer`, new `CancelTimer` and `ListTimers`, normalization, matching, `format_remaining`, tests |
| `crates/server/src/skills/mod.rs` | one `timers` field on `CallState` plus an accessor |
| `crates/server/skills.json` | updated `set_timer` description; `cancel_timer` and `list_timers` schemas |
| `crates/server/src/main.rs` | register the two new skills beside `SetTimer` (line 1015) |
| `crates/server/Cargo.toml` | `tokio-util` with the `rt` feature |
| `README.md` | skill list at line 166 |
| `docs/prd/prd.md` | SKILL-3 row |

## Test plan

All on the paused clock the file already uses
(`#[tokio::test(start_paused = true)]` with `tokio::time::advance`).

The four existing tests stay. One needs an edit:
`timer_speaks_into_the_call_after_the_delay` builds `CallCtx { state: None }`
and must now supply a `CallState`.

New tests:

- exactly five announcements at 10 s spacing, and none after
- cancel while *pending* → no announcement ever
- cancel while *ringing* → announcements stop, and specifically no send occurs
  after `cancel()` returns
- cancel by exact name; by normalized name ("the pasta timer"); by substring
- cancel by duration, including a fractional one (0.5) matched by tolerance
- a timer set *after* a cancel-all still fires (the parent token was replaced)
- cancel with no argument against exactly one timer
- cancel with no argument against three timers → asks, cancels nothing
- two timers named "pasta" → asks with remaining times, cancels nothing
- unknown name → reports what *is* running
- cancel all
- `list_timers` across empty / one / several / ringing
- normalization table test
- book is empty after each of: natural completion, cancel, call-token cancel
- regression: `{"minutes": "NaN"}` and `{"minutes": "inf"}` return the spoken
  error instead of panicking

`shipped_skills_json_parses` (`skills/mod.rs:373`) picks up the two new schemas
for free, and `Registry::new` already fails the build if a skill lacks a schema.

## Non-goals

- **Persistence.** Timers still die with the call — a client disconnect, network
  drop or server restart loses them. Making them survive reconnects needs a
  stable device identity and a rule for which live call an orphaned timer speaks
  into; surviving restarts needs storage and a policy for timers that expired
  while the server was down. Both are their own piece of work.
- **Rescheduling** ("add five minutes to the pasta timer").
- **Alarms at a wall-clock time** ("wake me at seven") as distinct from
  countdowns.
- **A distinct alert sound** before or instead of the spoken alert.
- **Auto-stopping the alert on any user speech.** Considered and rejected: it
  needs a new `FrameObserver` watching `UserStartedSpeaking` wired into
  `call.rs`, and a cough or a television would silence the alert.
