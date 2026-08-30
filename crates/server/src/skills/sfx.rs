//! `generate_sound_effect` — text-to-foley via the Woosh (Sony) or Stable
//! Audio Open model servers (port of skills/sfx/generate_sound_effect).
//!
//! The generators stay separate Python model servers (`make sfx-up`); this
//! skill only POSTs to them. The tool is always advertised; each call probes
//! the routed backend first and, if it is down, says so instead of promising
//! a sound. Generation runs in the background after the reply — the clip is
//! written under the server's `/sfx/` route and the client plays it once the
//! assistant has finished speaking.

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{json, Value};

use super::{arg_str, CallCtx, Skill};

const PROBE_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Woosh,
    StableAudio,
}

impl Backend {
    fn label(self) -> &'static str {
        match self {
            Backend::Woosh => "woosh",
            Backend::StableAudio => "stable_audio",
        }
    }
}

/// `auto` (keyword route) | `woosh` | `stable_audio`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Routing {
    Auto,
    Only(Backend),
}

impl Routing {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Routing::Auto),
            "woosh" => Ok(Routing::Only(Backend::Woosh)),
            "stable_audio" | "sao" => Ok(Routing::Only(Backend::StableAudio)),
            other => Err(format!(
                "unsupported SFX_BACKEND {other:?} (expected auto, woosh, or stable_audio)"
            )),
        }
    }
}

/// Prompts mentioning these route to Stable Audio Open: it produces 44.1 kHz
/// stereo and handles bodily/comedic sounds far better than Woosh's foley training.
const SAO_KEYWORDS: &[&str] = &[
    "fart",
    "farts",
    "farting",
    "burp",
    "burps",
    "burping",
    "belch",
    "belches",
    "belching",
    "laugh",
    "laughs",
    "laughing",
    "laughter",
    "giggle",
    "giggles",
    "giggling",
    "chuckle",
    "chuckles",
    "cough",
    "coughs",
    "coughing",
    "sneeze",
    "sneezes",
    "sneezing",
    "hiccup",
    "hiccups",
    "hiccuping",
    "raspberry",
    "raspberries",
    "snore",
    "snores",
    "snoring",
    "yawn",
    "yawns",
    "yawning",
    "gulp",
    "gulps",
    "gulping",
    "slurp",
    "slurps",
    "slurping",
];

fn mentions_sao_keyword(description: &str) -> bool {
    super::alias::normalise(description)
        .split_whitespace()
        .any(|w| SAO_KEYWORDS.contains(&w))
}

/// Which backend a description goes to, given what is configured.
pub fn route(description: &str, woosh: bool, sao: bool, routing: Routing) -> Option<Backend> {
    match (woosh, sao) {
        (false, false) => None,
        (true, false) => Some(Backend::Woosh),
        (false, true) => Some(Backend::StableAudio),
        (true, true) => Some(match routing {
            Routing::Only(b) => b,
            Routing::Auto if mentions_sao_keyword(description) => Backend::StableAudio,
            Routing::Auto => Backend::Woosh,
        }),
    }
}

fn request_body(backend: Backend, description: &str, seed: u32) -> Value {
    match backend {
        Backend::Woosh => json!({
            "version": "0.1",
            "token": "string",
            "args": {
                "model": "Woosh-DFlow",
                "prompt": description,
                "cfg": 3.0,
                "sampler": "heun",
                "num_steps": 5,
                "sigma_min": 0.0001,
                "sigma_max": 80,
                "rho": 7,
                "S_churn": 1,
                "S_min": 0,
                "S_noise": 1,
                "guidance_scale": 7.5,
                "noise_scheduler": "karras",
                "seed": seed,
            },
        }),
        // 40 steps × 3 s lands around ~30 s on MPS (vs ~110 s at 100×5 s);
        // the quality difference for short comedic SFX is hard to notice.
        Backend::StableAudio => json!({
            "prompt": description,
            "seconds": 3.0,
            "steps": 40,
            "cfg_scale": 7.0,
            "seed": seed,
        }),
    }
}

pub struct GenerateSoundEffect {
    http: reqwest::Client,
    woosh_url: Option<String>,
    sao_url: Option<String>,
    routing: Routing,
    /// Where clips are written; served by the server at `/sfx/{file}`.
    out_dir: PathBuf,
}

impl GenerateSoundEffect {
    pub fn new(woosh_url: String, sao_url: String, routing: Routing, out_dir: PathBuf) -> Self {
        let opt =
            |s: String| Some(s.trim().trim_end_matches('/').to_string()).filter(|s| !s.is_empty());
        Self {
            http: reqwest::Client::new(),
            woosh_url: opt(woosh_url),
            sao_url: opt(sao_url),
            routing,
            out_dir,
        }
    }

    fn url_for(&self, backend: Backend) -> &str {
        match backend {
            Backend::Woosh => self.woosh_url.as_deref().unwrap_or(""),
            Backend::StableAudio => self.sao_url.as_deref().unwrap_or(""),
        }
    }

    /// `GET /docs` (the same readiness probe run.sh uses).
    async fn is_up(&self, base: &str) -> bool {
        self.http
            .get(format!("{base}/docs"))
            .timeout(PROBE_TIMEOUT)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    async fn generate(&self, backend: Backend, description: &str) -> Result<PathBuf, String> {
        let timeout = Duration::from_secs(if backend == Backend::StableAudio {
            120
        } else {
            60
        });
        let body = request_body(backend, description, rand::random::<u32>());
        let bytes = self
            .http
            .post(format!("{}/generate", self.url_for(backend)))
            .timeout(timeout)
            .json(&body)
            .send()
            .await
            .map_err(|e| e.to_string())?
            .error_for_status()
            .map_err(|e| e.to_string())?
            .bytes()
            .await
            .map_err(|e| e.to_string())?;
        tokio::fs::create_dir_all(&self.out_dir)
            .await
            .map_err(|e| e.to_string())?;
        let path = self.out_dir.join(format!(
            "sfx_{}.flac",
            chrono::Utc::now().timestamp_millis()
        ));
        tokio::fs::write(&path, &bytes)
            .await
            .map_err(|e| e.to_string())?;
        Ok(path)
    }
}

#[async_trait]
impl Skill for GenerateSoundEffect {
    fn name(&self) -> &str {
        "generate_sound_effect"
    }

    async fn call(&self, args: &Value, ctx: &CallCtx) -> String {
        let desc = arg_str(args, "description").to_string();
        if desc.is_empty() {
            return "Tell me what sound to make.".to_string();
        }
        let Some(backend) = route(
            &desc,
            self.woosh_url.is_some(),
            self.sao_url.is_some(),
            self.routing,
        ) else {
            return "Sound effects aren't configured on this server.".to_string();
        };
        if !self.is_up(self.url_for(backend)).await {
            return "The sound effect server isn't running — start it with make sfx-up."
                .to_string();
        }
        let Some(media) = ctx.media.clone() else {
            return "I can't play sounds on this connection.".to_string();
        };
        tracing::info!(backend = backend.label(), %desc, "sfx: routing");
        let spoken = desc.trim_end_matches(['.', '!', '?']).to_string();
        // Reply now; generate and hand the clip to the client in the background.
        let this = Self {
            http: self.http.clone(),
            woosh_url: self.woosh_url.clone(),
            sao_url: self.sao_url.clone(),
            routing: self.routing,
            out_dir: self.out_dir.clone(),
        };
        tokio::spawn(async move {
            match this.generate(backend, &desc).await {
                Ok(path) => {
                    let name = path
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    media.play_file(&format!("/sfx/{name}"), true);
                }
                Err(error) => {
                    tracing::warn!(backend = backend.label(), %desc, %error, "sfx: generate failed")
                }
            }
        });
        format!("Playing a {spoken}.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_like_python() {
        assert_eq!(
            route("thunder", true, true, Routing::Auto),
            Some(Backend::Woosh)
        );
        assert_eq!(
            route("a loud fart", true, true, Routing::Auto),
            Some(Backend::StableAudio)
        );
        assert_eq!(
            route("Farts!", true, true, Routing::Auto),
            Some(Backend::StableAudio)
        );
        assert_eq!(
            route("a loud fart", true, true, Routing::Only(Backend::Woosh)),
            Some(Backend::Woosh)
        );
        assert_eq!(
            route("a loud fart", true, false, Routing::Auto),
            Some(Backend::Woosh)
        );
        assert_eq!(
            route("rain", false, true, Routing::Auto),
            Some(Backend::StableAudio)
        );
        assert_eq!(route("rain", false, false, Routing::Auto), None);
        assert_eq!(
            Routing::parse("SAO").unwrap(),
            Routing::Only(Backend::StableAudio)
        );
        assert!(Routing::parse("meow").is_err());
    }

    #[test]
    fn bodies_match_the_python_requests() {
        let w = request_body(Backend::Woosh, "rain", 7);
        assert_eq!(w["args"]["model"], "Woosh-DFlow");
        assert_eq!(w["args"]["num_steps"], 5);
        assert_eq!(w["args"]["seed"], 7);
        let s = request_body(Backend::StableAudio, "rain", 7);
        assert_eq!(s["steps"], 40);
        assert_eq!(s["seconds"], 3.0);
        assert_eq!(s["prompt"], "rain");
    }
}

#[cfg(test)]
mod offline_tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn reports_not_running_when_the_backend_is_down() {
        // Nothing listens on this port: the probe fails fast and the reply
        // tells the user how to start the server instead of promising a sound.
        let s = GenerateSoundEffect::new(
            "http://127.0.0.1:1".into(),
            String::new(),
            Routing::Auto,
            std::env::temp_dir(),
        );
        assert_eq!(
            s.call(&json!({"description": "thunder"}), &CallCtx::detached(0))
                .await,
            "The sound effect server isn't running — start it with make sfx-up."
        );
        assert_eq!(
            s.call(&json!({}), &CallCtx::detached(0)).await,
            "Tell me what sound to make."
        );
        let none = GenerateSoundEffect::new(
            String::new(),
            String::new(),
            Routing::Auto,
            std::env::temp_dir(),
        );
        assert_eq!(
            none.call(&json!({"description": "thunder"}), &CallCtx::detached(0))
                .await,
            "Sound effects aren't configured on this server."
        );
    }
}
