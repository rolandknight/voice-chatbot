//! In-process skills: the tools Babel advertises to the LLM and runs itself.
//!
//! Ported from the legacy Python `skills/` package (docs/plans/skills-in-server.md).
//! Schemas are data (`crates/server/skills.json`) so the rendered tool list —
//! part of the Ollama prompt prefix (ADR-0003) — is reviewable and stable; the
//! implementations live in the submodules. Gating is decided once at startup
//! (config/credentials): a skill that is not constructed is not advertised.
//! The tool list never changes per turn.

pub mod time;
pub mod weather;
pub mod web_search;

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use flowcat_core::session::ToolDecl;

/// What a skill gets per invocation. Grows with the tiers in the plan (frame
/// sender for timers, media controller, per-call state).
pub struct CallCtx {
    pub run_id: i64,
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
        let ctx = CallCtx { run_id: 1 };
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
