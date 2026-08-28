//! StubSession — a [`SessionSource`] that relays workflow tool calls to the
//! PoC stub server (`poc/stubs/stub_server.py`, see `poc/CONTRACT.md`).
//!
//! Modeled on flowcat-server's `StaticSession`, but with real `node_tools` /
//! `tool_call`: the 8 skill schemas from `poc/stubs/skills.json` are advertised
//! to the LLM, and every call is `POST {stubs}/tool/{name}`.

use std::path::PathBuf;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};

use flowcat_core::session::{Finalize, ResolvedCall, ToolDecl, UploadTarget};
use flowcat_core::{FlowcatError, SessionSource};

/// One entry in `skills.json` (`parameters` is the JSON-Schema).
#[derive(Deserialize)]
struct SkillDef {
    name: String,
    description: String,
    parameters: Value,
}

pub struct StubSession {
    skills: Vec<ToolDecl>,
    stubs_url: String,
    http: reqwest::Client,
    artifact_dir: PathBuf,
}

impl StubSession {
    pub fn new(
        skills_json: &str,
        stubs_url: String,
        artifact_dir: PathBuf,
    ) -> Result<Self, String> {
        let defs: Vec<SkillDef> =
            serde_json::from_str(skills_json).map_err(|e| format!("parse skills.json: {e}"))?;
        let skills = defs
            .into_iter()
            .map(|d| ToolDecl {
                name: d.name,
                description: d.description,
                params: d.parameters,
            })
            .collect();
        Ok(Self {
            skills,
            stubs_url: stubs_url.trim_end_matches('/').to_string(),
            http: reqwest::Client::new(),
            artifact_dir,
        })
    }
}

#[async_trait]
impl SessionSource for StubSession {
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
        Ok(self.skills.clone())
    }

    async fn tool_call(
        &self,
        _run_id: i64,
        _token: &str,
        _node_id: &str,
        tool_name: &str,
        args: &Value,
    ) -> Result<String, FlowcatError> {
        let url = format!("{}/tool/{}", self.stubs_url, tool_name);
        // Per the SessionSource contract: fold failures into a spoken-friendly
        // string so the call continues; never abort the turn on a stub error.
        match self.http.post(&url).json(args).send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: Value = resp.json().await.unwrap_or(json!({}));
                let result = body.get("result").cloned().unwrap_or(body);
                Ok(result.to_string())
            }
            Ok(resp) => {
                tracing::warn!(tool = tool_name, status = %resp.status(), "stub tool error");
                Ok(format!("The {tool_name} service returned an error."))
            }
            Err(e) => {
                tracing::warn!(tool = tool_name, error = %e, "stub tool unreachable");
                Ok(format!(
                    "The {tool_name} service is temporarily unavailable."
                ))
            }
        }
    }
}
