//! In-process skills: the tools Babel advertises to the LLM and runs itself.
//!
//! Ported from the legacy Python `skills/` package (docs/plans/skills-in-server.md).
//! Schemas are data (`crates/server/skills.json`) so the rendered tool list —
//! part of the Ollama prompt prefix (ADR-0003) — is reviewable and stable; the
//! implementations live in the submodules. Gating is decided once at startup
//! (config/credentials): a skill that is not constructed is not advertised.
//! The tool list never changes per turn.

pub mod alias;
pub mod claude;
pub mod persona;
pub mod radio;
pub mod sfx;
pub mod shows;
pub mod spotify;
pub mod spotify_client;
pub mod time;
pub mod timer;
pub mod weather;
pub mod web_search;

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::mpsc;

use flowcat_core::processor::frame::Frame;
use flowcat_core::session::ToolDecl;

use crate::media::MediaController;

/// What a skill gets per invocation. Grows with the tiers in the plan (media
/// controller, per-call state).
pub struct CallCtx {
    pub run_id: i64,
    /// Head-injection queue of this call's pipeline (`PipelineTask::queue_sender`),
    /// for skills that speak later (timers). `None` outside a live call.
    pub frames: Option<mpsc::UnboundedSender<Frame>>,
    /// Media playback on this call's client. `None` outside a live call.
    pub media: Option<Arc<MediaController>>,
    /// Process-wide Spotify control, when configured.
    pub spotify: Option<Arc<spotify_client::SpotifyClient>>,
    /// This call's voice/backend flags. `None` outside a live call.
    pub state: Option<Arc<CallState>>,
    /// Sample rate of this call's TTS backend. Audio injected as
    /// `Frame::OutputAudio` must be generated at exactly this rate: the output
    /// stage resamples with a fixed `tts_rate -> carrier_rate` converter and
    /// ignores the frame's own `sample_rate`. `None` outside a live call.
    pub tts_rate: Option<u32>,
}

impl CallCtx {
    /// Outside a live call (tests, warm-up).
    pub fn detached(run_id: i64) -> Self {
        Self {
            run_id,
            frames: None,
            media: None,
            spotify: None,
            state: None,
            tts_rate: None,
        }
    }

    /// Before starting one audio source, silence the others: the Python
    /// handlers cross-stopped radio/shows and Spotify both ways.
    pub async fn stop_other_audio(&self) {
        if let Some(m) = &self.media {
            m.stop();
        }
        self.stop_spotify().await;
    }

    /// Stop Spotify if it is playing; true when it was.
    pub async fn stop_spotify(&self) -> bool {
        match &self.spotify {
            Some(s) if s.is_playing() => {
                s.stop().await;
                true
            }
            _ => false,
        }
    }
}

/// Which model answers the call's turns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmBackend {
    Local,
    Claude,
}

/// Per-call mutable flags that skills set and the pipeline stages read on
/// every turn: the TTS voice (`switch_persona`) and the LLM backend
/// (`ask_claude`). Created per call, so both revert when the call ends —
/// the Python "resets when the assistant goes back to sleep" semantics.
#[derive(Default)]
pub struct CallState {
    voice: Mutex<Option<String>>,
    /// System prompt for the selected persona (`prompt.<persona>.txt`), None =
    /// the process default. Read by the LLM adapter on every run.
    prompt: Mutex<Option<String>>,
    /// persona → prompt text, loaded once at startup (`crates/server/prompt.*.txt`).
    prompts: HashMap<String, String>,
    claude: std::sync::atomic::AtomicBool,
    /// When a wake word last fired (server gate or the native client's report).
    /// Consumed once by `wake::WakeGrace` to hold the first end-of-speech edge
    /// after the wake, so "Hey Marvin … what time is it" is one turn.
    wake_armed_at: Mutex<Option<std::time::Instant>>,
    /// Live countdown timers for this call (`skills/timer.rs`). Dropping
    /// `CallState` cancels them all, so nothing outlives the call.
    timers: Mutex<timer::TimerBook>,
}

/// `prompt.<persona>.txt` lookup key: lowercase, `_` as `-` (the persona
/// convention wake heads and Qwen presets share).
pub fn prompt_key(persona: &str) -> String {
    persona.trim().to_ascii_lowercase().replace('_', "-")
}

impl CallState {
    pub fn with_prompts(prompts: HashMap<String, String>) -> Self {
        Self {
            prompts,
            ..Self::default()
        }
    }

    pub fn voice(&self) -> Option<String> {
        self.voice.lock().unwrap().clone()
    }

    /// Select a persona: its voice, and its prompt when `prompt.<persona>.txt`
    /// exists (otherwise the process default prompt applies again).
    pub fn set_voice(&self, name: &str) {
        *self.voice.lock().unwrap() = Some(name.to_string());
        let prompt = self.prompts.get(&prompt_key(name)).cloned();
        tracing::info!(
            persona = name,
            prompt = if prompt.is_some() {
                "persona"
            } else {
                "default"
            },
            "persona selected"
        );
        *self.prompt.lock().unwrap() = prompt;
    }

    /// The persona's system prompt, if one is selected and has a file.
    pub fn prompt(&self) -> Option<String> {
        self.prompt.lock().unwrap().clone()
    }

    pub fn backend(&self) -> LlmBackend {
        if self.claude.load(std::sync::atomic::Ordering::Relaxed) {
            LlmBackend::Claude
        } else {
            LlmBackend::Local
        }
    }

    pub fn set_backend(&self, backend: LlmBackend) {
        self.claude.store(
            backend == LlmBackend::Claude,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// A wake word just fired: the next end-of-speech edge is the wake phrase's
    /// own, and may be followed by the command after a short pause.
    pub fn arm_wake_grace(&self) {
        *self.wake_armed_at.lock().unwrap() = Some(std::time::Instant::now());
    }

    /// Consume the arm if it fired within `max_age` (a stale arm — a wake with
    /// no speech edge behind it — must not hold a later, unrelated turn).
    pub fn take_wake_grace(&self, max_age: std::time::Duration) -> bool {
        match self.wake_armed_at.lock().unwrap().take() {
            Some(at) => at.elapsed() <= max_age,
            None => false,
        }
    }

    /// Operate on this call's timers. The lock never escapes, so it cannot be
    /// held across an `.await`.
    pub fn with_timers<R>(&self, f: impl FnOnce(&mut timer::TimerBook) -> R) -> R {
        f(&mut self.timers.lock().unwrap())
    }
}

/// Handles `call.rs` registers for the life of a call.
#[derive(Clone)]
pub struct CallHandle {
    pub frames: mpsc::UnboundedSender<Frame>,
    pub media: Arc<MediaController>,
    pub state: Arc<CallState>,
    /// The TTS backend's sample rate; see [`CallCtx::tts_rate`].
    pub tts_rate: u32,
}

/// Per-call handles a skill may need. `SessionSource::tool_call` only
/// receives the `run_id`, so this is how a tool call finds its own call.
#[derive(Default)]
pub struct CallRegistry {
    calls: Mutex<HashMap<i64, CallHandle>>,
    spotify: Option<Arc<spotify_client::SpotifyClient>>,
}

impl CallRegistry {
    pub fn new(spotify: Option<Arc<spotify_client::SpotifyClient>>) -> Self {
        Self {
            calls: Mutex::new(HashMap::new()),
            spotify,
        }
    }

    pub fn register(&self, run_id: i64, handle: CallHandle) {
        self.calls.lock().unwrap().insert(run_id, handle);
    }

    pub fn unregister(&self, run_id: i64) {
        self.calls.lock().unwrap().remove(&run_id);
    }

    /// The per-call flags of a live call (None once it has ended / before it
    /// registered).
    pub fn state(&self, run_id: i64) -> Option<Arc<CallState>> {
        self.calls
            .lock()
            .unwrap()
            .get(&run_id)
            .map(|h| h.state.clone())
    }

    pub fn ctx(&self, run_id: i64) -> CallCtx {
        let handle = self.calls.lock().unwrap().get(&run_id).cloned();
        CallCtx {
            run_id,
            frames: handle.as_ref().map(|h| h.frames.clone()),
            media: handle.as_ref().map(|h| h.media.clone()),
            spotify: self.spotify.clone(),
            tts_rate: handle.as_ref().map(|h| h.tts_rate),
            state: handle.map(|h| h.state),
        }
    }
}

/// One tool. `call` returns the spoken-friendly text fed back to the LLM and
/// never fails: errors fold into the text ("I couldn't reach …"), exactly the
/// `SessionSource::tool_call` contract.
#[async_trait]
pub trait Skill: Send + Sync {
    fn name(&self) -> &str;
    async fn call(&self, args: &Value, ctx: &CallCtx) -> String;
}

/// One entry in `skills.json` (`parameters` is the JSON-Schema).
#[derive(Deserialize)]
struct SkillDef {
    name: String,
    description: String,
    parameters: Value,
}

/// Advertised schemas + dispatch table.
pub struct Registry {
    decls: Vec<ToolDecl>,
    skills: BTreeMap<String, Arc<dyn Skill>>,
}

impl Registry {
    /// Pair every constructed skill with its schema. A skill without a schema
    /// is a build error (the LLM could never call it); a schema without a
    /// skill is a gated-off tool and is simply not advertised.
    pub fn new(skills_json: &str, skills: Vec<Arc<dyn Skill>>) -> Result<Self, String> {
        let defs: Vec<SkillDef> =
            serde_json::from_str(skills_json).map_err(|e| format!("parse skills.json: {e}"))?;
        let mut by_name: BTreeMap<String, Arc<dyn Skill>> = BTreeMap::new();
        for s in skills {
            if by_name.insert(s.name().to_string(), s).is_some() {
                return Err(format!(
                    "duplicate skill {:?}",
                    by_name.keys().last().unwrap()
                ));
            }
        }
        let mut decls = Vec::new();
        let mut advertised = Vec::new();
        let mut gated = Vec::new();
        for d in defs {
            if by_name.contains_key(&d.name) {
                advertised.push(d.name.clone());
                decls.push(ToolDecl {
                    name: d.name,
                    description: d.description,
                    params: d.parameters,
                });
            } else {
                gated.push(d.name);
            }
        }
        let missing: Vec<&String> = by_name.keys().filter(|n| !advertised.contains(n)).collect();
        if !missing.is_empty() {
            return Err(format!(
                "skills without a schema in skills.json: {missing:?}"
            ));
        }
        tracing::info!(?advertised, ?gated, "skills loaded");
        Ok(Self {
            decls,
            skills: by_name,
        })
    }

    pub fn decls(&self) -> Vec<ToolDecl> {
        self.decls.clone()
    }

    /// Run a tool. Unknown names (hallucinated by the model) get a spoken
    /// answer rather than an error, like every other failure.
    pub async fn call(&self, name: &str, args: &Value, ctx: &CallCtx) -> String {
        let Some(skill) = self.skills.get(name) else {
            tracing::warn!(tool = name, "unknown tool");
            return format!("There is no tool called {name}.");
        };
        tracing::info!(tool = name, %args, run_id = ctx.run_id, "tool invoke");
        let started = std::time::Instant::now();
        let out = skill.call(args, ctx).await;
        tracing::info!(
            tool = name,
            elapsed_ms = started.elapsed().as_millis(),
            result = %out,
            "tool return"
        );
        out
    }
}

/// `args[key]` as a trimmed string ("" when absent/null/not a string).
pub fn arg_str<'a>(args: &'a Value, key: &str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or("").trim()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(&'static str);
    #[async_trait]
    impl Skill for Fake {
        fn name(&self) -> &str {
            self.0
        }
        async fn call(&self, _: &Value, _: &CallCtx) -> String {
            format!("{} ran", self.0)
        }
    }

    const JSON: &str = r#"[
      {"name":"a","description":"A","parameters":{"type":"object","properties":{}}},
      {"name":"b","description":"B","parameters":{"type":"object","properties":{}}}
    ]"#;

    #[tokio::test]
    async fn advertises_only_constructed_skills_and_dispatches() {
        let r = Registry::new(JSON, vec![Arc::new(Fake("a"))]).unwrap();
        let names: Vec<_> = r.decls().into_iter().map(|d| d.name).collect();
        assert_eq!(names, vec!["a"]);
        let ctx = CallCtx::detached(1);
        assert_eq!(r.call("a", &Value::Null, &ctx).await, "a ran");
        assert_eq!(
            r.call("b", &Value::Null, &ctx).await,
            "There is no tool called b."
        );
    }

    #[test]
    fn skill_without_schema_is_an_error() {
        match Registry::new(JSON, vec![Arc::new(Fake("zzz"))]) {
            Err(err) => assert!(err.contains("zzz"), "{err}"),
            Ok(_) => panic!("expected an error"),
        }
    }

    #[test]
    fn shipped_skills_json_parses() {
        assert!(Registry::new(include_str!("../../skills.json"), vec![]).is_ok());
    }
}
