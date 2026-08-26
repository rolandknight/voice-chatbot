//! `ollama serve` supervisor — ADR-0007 Layer 2.
//!
//! The chatbot owns its LLM's lifetime: if nothing answers on the base URL it
//! spawns `ollama serve` as a child (env `OLLAMA_KEEP_ALIVE=-1` and
//! `OLLAMA_CONTEXT_LENGTH=<num_ctx>` as belt and braces — the native `/api/chat`
//! requests carry both anyway), waits for `/api/tags`, and on shutdown
//! terminates the child it started. A serve that is already running (brew,
//! launchd, the Ollama.app) is used as-is: with the native API its environment
//! no longer decides residency or context size.

use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Supervise {
    /// Spawn only if nothing answers (default).
    Auto,
    /// Never spawn; fail if nothing answers.
    Never,
    /// Refuse to use a serve we did not start.
    Always,
}

impl Supervise {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "never" => Ok(Self::Never),
            "always" => Ok(Self::Always),
            other => Err(format!(
                "invalid POC_OLLAMA_SUPERVISE {other:?} (expected auto, never, or always)"
            )),
        }
    }
}

/// The decision table, kept pure so it is unit-testable without a serve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    UseExisting,
    Spawn,
    Fail(&'static str),
}

pub fn plan(supervise: Supervise, port_answers: bool) -> Plan {
    match (supervise, port_answers) {
        (Supervise::Never, true) | (Supervise::Auto, true) => Plan::UseExisting,
        (Supervise::Never, false) => Plan::Fail(
            "no ollama answers and POC_OLLAMA_SUPERVISE=never; start `ollama serve` or use auto",
        ),
        (Supervise::Auto, false) | (Supervise::Always, false) => Plan::Spawn,
        (Supervise::Always, true) => Plan::Fail(
            "POC_OLLAMA_SUPERVISE=always but a serve already answers on the port (quit the Ollama.app or use auto)",
        ),
    }
}

pub struct OllamaServe {
    base_url: String,
    child: Option<tokio::process::Child>,
}

impl OllamaServe {
    pub async fn is_up(base_url: &str) -> bool {
        let client = match reqwest::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
        {
            Ok(c) => c,
            Err(_) => return false,
        };
        client
            .get(format!("{}/api/tags", base_url.trim_end_matches('/')))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// Make a serve available at `base_url` per the decision table.
    pub async fn ensure(
        base_url: &str,
        supervise: Supervise,
        bin: &str,
        num_ctx: u32,
        log_path: &Path,
    ) -> Result<Self, String> {
        let base_url = base_url
            .trim_end_matches('/')
            .trim_end_matches("/v1")
            .to_string();
        let up = Self::is_up(&base_url).await;
        match plan(supervise, up) {
            Plan::UseExisting => {
                tracing::info!(%base_url, "ollama: using the serve already running");
                Ok(Self {
                    base_url,
                    child: None,
                })
            }
            Plan::Fail(msg) => Err(msg.to_string()),
            Plan::Spawn => {
                if let Some(dir) = log_path.parent() {
                    std::fs::create_dir_all(dir)
                        .map_err(|e| format!("create {}: {e}", dir.display()))?;
                }
                let log = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                    .map_err(|e| format!("open {}: {e}", log_path.display()))?;
                let log_err = log
                    .try_clone()
                    .map_err(|e| format!("clone log handle: {e}"))?;
                tracing::info!(bin, num_ctx, log = %log_path.display(), "ollama: starting `ollama serve`");
                let child = tokio::process::Command::new(bin)
                    .arg("serve")
                    .env("OLLAMA_KEEP_ALIVE", "-1")
                    .env("OLLAMA_CONTEXT_LENGTH", num_ctx.to_string())
                    .stdout(std::process::Stdio::from(log))
                    .stderr(std::process::Stdio::from(log_err))
                    .stdin(std::process::Stdio::null())
                    // Own process group: survives the parent's terminal/session
                    // (needed for `--warm-only`, which detaches and exits).
                    .process_group(0)
                    .kill_on_drop(true)
                    .spawn()
                    .map_err(|e| format!("spawn `{bin} serve`: {e} (is ollama installed?)"))?;
                let started = Instant::now();
                while !Self::is_up(&base_url).await {
                    if started.elapsed() > Duration::from_secs(60) {
                        return Err(format!(
                            "`{bin} serve` did not answer on {base_url} within 60 s; see {}",
                            log_path.display()
                        ));
                    }
                    tokio::time::sleep(Duration::from_millis(500)).await;
                }
                tracing::info!(
                    elapsed_ms = started.elapsed().as_millis() as u64,
                    "ollama: serve is up"
                );
                Ok(Self {
                    base_url,
                    child: Some(child),
                })
            }
        }
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Leave a spawned serve running after this process exits (`--warm-only`).
    /// `kill_on_drop` would otherwise take it down with us.
    pub fn detach(mut self) {
        if let Some(child) = self.child.take() {
            tracing::info!(
                pid = child.id(),
                "ollama: leaving the spawned serve running"
            );
            std::mem::forget(child);
        }
    }

    /// Stop the child we started (SIGTERM, then SIGKILL after 10 s). A serve
    /// we did not start is left running.
    pub async fn shutdown(mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };
        let Some(pid) = child.id() else {
            return;
        };
        tracing::info!(pid, "ollama: stopping the serve we started");
        let _ = tokio::process::Command::new("kill")
            .arg(pid.to_string())
            .status()
            .await;
        match tokio::time::timeout(Duration::from_secs(10), child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                tracing::warn!(pid, "ollama: serve ignored SIGTERM; killing");
                let _ = child.kill().await;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_table() {
        assert_eq!(plan(Supervise::Auto, true), Plan::UseExisting);
        assert_eq!(plan(Supervise::Auto, false), Plan::Spawn);
        assert_eq!(plan(Supervise::Never, true), Plan::UseExisting);
        assert!(matches!(plan(Supervise::Never, false), Plan::Fail(_)));
        assert_eq!(plan(Supervise::Always, false), Plan::Spawn);
        assert!(matches!(plan(Supervise::Always, true), Plan::Fail(_)));
    }

    #[test]
    fn parse_modes() {
        assert_eq!(Supervise::parse("AUTO").unwrap(), Supervise::Auto);
        assert_eq!(Supervise::parse(" never ").unwrap(), Supervise::Never);
        assert!(Supervise::parse("sometimes").is_err());
    }

    #[tokio::test]
    async fn is_up_is_false_on_a_closed_port() {
        assert!(!OllamaServe::is_up("http://127.0.0.1:1").await);
    }
}
