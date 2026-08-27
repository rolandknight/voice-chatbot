//! SkillSession — a [`SessionSource`] whose workflow tools are the in-process
//! skills (`crate::skills`), replacing the PoC stub relay.
//!
//! Modeled on flowcat-server's `StaticSession`, but with real `node_tools` /
//! `tool_call`: the registry's schemas are advertised to the LLM and every
//! call dispatches straight to the matching [`crate::skills::Skill`].

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::{json, Value};

use flowcat_core::session::{Finalize, ResolvedCall, ToolDecl, UploadTarget};
use flowcat_core::{FlowcatError, SessionSource};

use std::sync::Arc;

use crate::skills::{CallRegistry, Registry};

pub struct SkillSession {
    skills: Registry,
    calls: Arc<CallRegistry>,
    artifact_dir: PathBuf,
}

impl SkillSession {
    pub fn new(skills: Registry, calls: CallRegistry, artifact_dir: PathBuf) -> Self {
        Self {
            skills,
            calls: Arc::new(calls),
            artifact_dir,
        }
    }

    /// Live-call handles; `call.rs` registers each call's pipeline here.
    pub fn calls(&self) -> &CallRegistry {
        &self.calls
    }
}

#[async_trait]
impl SessionSource for SkillSession {
    async fn resolve(&self, _run_id: i64, _token: &str) -> Result<ResolvedCall, FlowcatError> {
        Ok(ResolvedCall {
            provider: "poc".to_string(),
            brain_config: json!({}),
            is_completed: false,
        })
    }

    async fn complete(&self, run_id: i64, _token: &str, fin: Finalize) -> Result<(), FlowcatError> {
        tracing::info!(run_id, usage = %fin.usage, "poc session: run complete");
        Ok(())
    }

    async fn artifact_upload_url(
        &self,
        run_id: i64,
        _token: &str,
        kind: &str,
    ) -> Result<UploadTarget, FlowcatError> {
        std::fs::create_dir_all(&self.artifact_dir)
            .map_err(|e| FlowcatError::Session(format!("create artifact dir: {e}")))?;
        let key = format!("run-{run_id}-{kind}");
        let url = format!("file://{}", self.artifact_dir.join(&key).display());
        let content_type = match kind {
            "recording" => "audio/wav",
            "transcript" => "application/json",
            _ => "application/octet-stream",
        }
        .to_string();
        Ok(UploadTarget {
            url,
            key,
            content_type,
        })
    }

    async fn put_bytes(
        &self,
        url: &str,
        bytes: Vec<u8>,
        _content_type: &str,
    ) -> Result<(), FlowcatError> {
        let path = url.strip_prefix("file://").ok_or_else(|| {
            FlowcatError::Session(format!("expected file:// target, got {url:?}"))
        })?;
        std::fs::write(path, bytes)
            .map_err(|e| FlowcatError::Session(format!("write artifact {path}: {e}")))
    }

    async fn node_tools(
        &self,
        _run_id: i64,
        _token: &str,
        _node_id: &str,
    ) -> Result<Vec<ToolDecl>, FlowcatError> {
        Ok(self.skills.decls())
    }

    async fn tool_call(
        &self,
        run_id: i64,
        _token: &str,
        _node_id: &str,
        tool_name: &str,
        args: &Value,
    ) -> Result<String, FlowcatError> {
        // Per the SessionSource contract the result is always a spoken-friendly
        // string; the registry folds every failure into one.
        Ok(self
            .skills
            .call(tool_name, args, &self.calls.ctx(run_id))
            .await)
    }
}
