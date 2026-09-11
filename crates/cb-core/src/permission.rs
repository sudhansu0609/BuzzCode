//! Permission broker: decides whether a tool call may run, asking the UI when needed.

use crate::message::ToolCall;
use cb_tool_api::{PermissionClass, PermissionMode};
use parking_lot::Mutex;
use regex::Regex;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionDecision {
    Allow,
    /// Allow this and future calls of the same tool for the session.
    AllowAlways,
    Deny,
}

#[derive(Debug)]
pub struct PermissionRequest {
    pub call: ToolCall,
    pub class: PermissionClass,
    pub preview: Option<String>,
    pub respond: oneshot::Sender<PermissionDecision>,
}

pub struct PermissionBroker {
    pub mode: Mutex<PermissionMode>,
    always_allow: Mutex<HashSet<String>>,
    always_ask_shell: Vec<Regex>,
    /// UI channel; None = headless (deny anything that would prompt unless yolo).
    ask_tx: Option<mpsc::Sender<PermissionRequest>>,
}

impl PermissionBroker {
    pub fn new(mode: PermissionMode, always_allow: &[String], always_ask_shell: &[String], ask_tx: Option<mpsc::Sender<PermissionRequest>>) -> Arc<Self> {
        Arc::new(Self {
            mode: Mutex::new(mode),
            always_allow: Mutex::new(always_allow.iter().cloned().collect()),
            always_ask_shell: always_ask_shell.iter().filter_map(|p| Regex::new(&format!("(?i){p}")).ok()).collect(),
            ask_tx,
        })
    }

    pub fn set_mode(&self, m: PermissionMode) { *self.mode.lock() = m; }
    pub fn mode(&self) -> PermissionMode { *self.mode.lock() }

    pub fn is_dangerous_shell(&self, call: &ToolCall) -> bool {
        if call.name != "shell" { return false; }
        let cmd = call.arguments.get("command").and_then(|v| v.as_str()).unwrap_or("");
        self.always_ask_shell.iter().any(|r| r.is_match(cmd))
    }

    /// Check if a prompt is needed for this call.
    pub fn needs_prompt(&self, call: &ToolCall, class: PermissionClass) -> bool {
        let mode = self.mode();
        if mode == PermissionMode::Yolo { return false; }
        if self.is_dangerous_shell(call) { return true; }
        if self.always_allow.lock().contains(&call.name) { return false; }
        mode.requires_prompt(class)
    }

    /// Resolve one call. `preview` is shown to the user if a prompt is needed.
    pub async fn resolve(&self, call: &ToolCall, class: PermissionClass, preview: Option<String>) -> PermissionDecision {
        let mode = self.mode();
        if mode == PermissionMode::Yolo { return PermissionDecision::Allow; }
        let dangerous = self.is_dangerous_shell(call);
        if !mode.requires_prompt(class) && !dangerous { return PermissionDecision::Allow; }
        if !dangerous && self.always_allow.lock().contains(&call.name) { return PermissionDecision::Allow; }

        let Some(tx) = &self.ask_tx else {
            // Headless without yolo: deny destructive actions.
            return if mode == PermissionMode::Yolo { PermissionDecision::Allow } else { PermissionDecision::Deny };
        };
        let (respond, rx) = oneshot::channel();
        if tx.send(PermissionRequest { call: call.clone(), class, preview, respond }).await.is_err() {
            return PermissionDecision::Deny;
        }
        let d = rx.await.unwrap_or(PermissionDecision::Deny);
        if d == PermissionDecision::AllowAlways { self.always_allow.lock().insert(call.name.clone()); }
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn yolo_mode_never_needs_prompt() {
        let broker = PermissionBroker::new(PermissionMode::Yolo, &[], &[], None);
        let shell_call = ToolCall {
            id: "call_1".into(),
            name: "shell".into(),
            arguments: json!({"command": "rm -rf /"}),
        };
        assert!(!broker.needs_prompt(&shell_call, PermissionClass::Exec));
    }

    #[test]
    fn ask_mode_prompts_for_exec_and_dangerous() {
        let broker = PermissionBroker::new(PermissionMode::Ask, &[], &[], None);
        let shell_call = ToolCall {
            id: "call_1".into(),
            name: "shell".into(),
            arguments: json!({"command": "rm -rf /"}),
        };
        assert!(broker.needs_prompt(&shell_call, PermissionClass::Exec));

        let read_call = ToolCall {
            id: "call_2".into(),
            name: "read_file".into(),
            arguments: json!({"path": "src/main.rs"}),
        };
        assert!(!broker.needs_prompt(&read_call, PermissionClass::ReadOnly));
    }
}
