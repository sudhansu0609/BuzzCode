//! Subagents: scoped tool sets, separate context, same engine slot (serialized).

use crate::agent::Agent;
use crate::context::{ContextStore, FrozenPrefix};
use crate::events::AgentEvent;
use crate::message::Message;
use crate::prompt::{self, PromptEnv};
use crate::registry::ToolRegistry;
use crate::session::Session;
use crate::task::TaskClass;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SubagentKind { Explore, Plan, General }

pub const EXPLORE_TOOLS: &[&str] = &["read_file", "glob", "grep", "list_dir", "outline", "symbol_search", "repo_map", "git_status", "git_diff", "git_log"];

impl SubagentKind {
    pub fn allowed_tools(self) -> Option<&'static [&'static str]> {
        match self { SubagentKind::Explore | SubagentKind::Plan => Some(EXPLORE_TOOLS), SubagentKind::General => None }
    }
    pub fn task_class(self) -> TaskClass {
        match self { SubagentKind::Explore => TaskClass::Explore, SubagentKind::Plan => TaskClass::Plan, SubagentKind::General => TaskClass::Chat }
    }
    pub fn max_turns(self) -> u32 { match self { SubagentKind::Explore => 15, SubagentKind::Plan => 20, SubagentKind::General => 25 } }
}

/// Everything needed to create a subagent's frozen prefix + registry.
pub struct SubagentFactory {
    pub session: Arc<Session>,
    pub full_registry: Arc<ToolRegistry>,
    pub env: PromptEnv,
}

impl SubagentFactory {
    pub async fn build(&self, kind: SubagentKind, n_ctx: u32) -> anyhow::Result<Agent> {
        let registry = match kind.allowed_tools() {
            Some(allow) => self.full_registry.scoped(allow),
            None => self.full_registry.without(&["spawn_agent", "write_plan"]),
        };
        let registry = Arc::new(registry);
        let tools_json = registry.freeze();
        let mut env = self.env.clone();
        env.tool_names = registry.names();
        env.has_repo_map = false;
        let system = match kind {
            SubagentKind::Explore => prompt::explore_system_prompt(&env),
            SubagentKind::Plan => prompt::plan_system_prompt(&env),
            SubagentKind::General => prompt::main_system_prompt(&env),
        };
        let mut prefix = FrozenPrefix::new(Message::system(system), tools_json);
        prefix.system_tokens = self.session.counter.count(&prefix.system.content).await;
        prefix.tools_tokens = self.session.counter.count(&serde_json::to_string(&*prefix.tools_json).unwrap_or_default()).await;
        let ctx = ContextStore::new(prefix, n_ctx, self.session.counter.clone());
        let mut agent = Agent::new(self.session.clone(), ctx, registry, kind.task_class(), kind.max_turns());
        agent.quiet = true;
        Ok(agent)
    }
}

// ------------------------------------------------------------------ spawn_agent tool

#[derive(Deserialize, schemars::JsonSchema)]
pub struct SpawnArgs {
    /// explore = read-only investigation; plan = read-only planning; general = can edit and run commands.
    pub kind: SubagentKind,
    /// Self-contained task description. The subagent has no access to this conversation, so include every needed detail (paths, names, goals).
    pub task: String,
}

pub struct SpawnAgent { pub factory: Arc<SubagentFactory> }

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<SpawnArgs>(
    "spawn_agent",
    "Delegate a self-contained sub-task to a fresh agent with its own context (keeps this conversation small). Use explore for research across many files, general for an isolated implementation step. Returns the subagent's final report.",
    PermissionClass::ReadOnly,
));

#[async_trait::async_trait]
impl Tool for SpawnAgent {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: SpawnArgs = parse_args(&args)?;
        // A general subagent can edit/run: gate it like an Exec tool in ask mode.
        if a.kind == SubagentKind::General && cx.permission_mode == PermissionMode::Ask {
            // The permission broker already prompted for this call if configured; nothing extra here.
        }
        let session = self.factory.session.clone();
        let n_ctx = session.engine.n_ctx();
        let mut agent = self.factory.build(a.kind, n_ctx).await.map_err(|e| ToolError::Failed(format!("{e:#}")))?;
        agent.cancel = cx.cancel.clone();
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        session.emit_async(AgentEvent::AgentSpawned { id, kind: format!("{:?}", a.kind).to_lowercase(), task: a.task.clone() }).await;
        let result = agent.run_turn(a.task).await;
        let outcome = match result { Ok(o) => o, Err(e) => { session.emit_async(AgentEvent::AgentFinished { id, outcome: "error".into(), turns: agent.turn() }).await; return Err(ToolError::Failed(format!("{e:#}"))); } };
        let report = agent.last_assistant_text().unwrap_or("(subagent produced no final message)").to_string();
        let cap = 8000usize;
        let report = if report.len() > cap { format!("{}…\n[report truncated]", &report[..cap]) } else { report };
        session.emit_async(AgentEvent::AgentFinished { id, outcome: outcome.as_str().to_string(), turns: agent.turn() }).await;
        let mut out = ToolOutput::text(format!("[subagent {:?} · {} · {} turns]\n{report}", a.kind, outcome.as_str(), agent.turn()));
        out.is_error = !matches!(outcome, crate::agent::AgentOutcome::EndTurn);
        Ok(out)
    }
}
