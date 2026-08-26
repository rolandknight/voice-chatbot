//! BabelBrain — a minimal single-state [`AgentBrain`].
//!
//! The stock `DeclarativeBrain` unconditionally advertises an `endCall` tool;
//! Babel has no notion of the model hanging up, so this brain exposes **no**
//! tools of its own. All skills arrive as workflow tools via
//! `SessionSource::node_tools` (see `session.rs`) and are relayed by the
//! framework's `BrainProcessor`, never reaching `on_tool_call`.

use serde_json::{json, Value};

use flowcat_core::brain::AgentBrain;
use flowcat_core::types::{BrainAction, ToolDecl};

pub struct BabelBrain {
    system_prompt: String,
}

impl BabelBrain {
    pub fn new(system_prompt: String) -> Self {
        Self { system_prompt }
    }
}

impl AgentBrain for BabelBrain {
    fn system_prompt(&self) -> String {
        self.system_prompt.clone()
    }

    fn tools(&self) -> Vec<ToolDecl> {
        Vec::new()
    }

    fn current_node_id(&self) -> String {
        "babel".to_string()
    }

    fn on_tool_call(&mut self, name: &str, _args: &Value) -> BrainAction {
        // Workflow tools are relayed by name before the brain sees them; anything
        // landing here is a hallucinated tool. Stay put and let the turn continue.
        tracing::warn!(tool = name, "brain saw an unexpected tool call");
        BrainAction::Stay
    }

    fn is_finished(&self) -> bool {
        false
    }

    fn collected_vars(&self) -> Value {
        json!({})
    }
}
