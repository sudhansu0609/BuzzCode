//! Session: shared services for the main agent and its subagents.

use crate::events::AgentEvent;
use crate::permission::PermissionBroker;
use crate::registry::ToolRegistry;
use crate::tokens::TokenCounter;
use cb_config::Config;
use cb_engine::EngineManager;
use cb_tool_api::Extensions;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

pub struct Session {
    pub cfg: Arc<Config>,
    pub engine: Arc<EngineManager>,
    pub registry: Arc<ToolRegistry>,
    pub perms: Arc<PermissionBroker>,
    pub counter: Arc<TokenCounter>,
    pub events: mpsc::Sender<AgentEvent>,
    pub project_dir: PathBuf,
    pub extensions: Arc<Extensions>,
    /// Session id for spill file naming.
    pub id: String,
}

impl Session {
    pub fn emit(&self, ev: AgentEvent) {
        // Never block the agent on a slow UI; drop if the channel is full.
        let _ = self.events.try_send(ev);
    }

    pub async fn emit_async(&self, ev: AgentEvent) {
        let _ = self.events.send(ev).await;
    }

    pub fn spill_dir(&self) -> PathBuf { self.cfg.paths.tool_out_dir().join(&self.id) }
}
