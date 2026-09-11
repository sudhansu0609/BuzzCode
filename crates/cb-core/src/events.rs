//! Events emitted by the agent for UIs (TUI, JSON-lines headless mode).

use crate::message::ToolCall;
use crate::task::TaskClass;
use cb_engine::Timings;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    TurnStart { turn: u32, task: TaskClass, effort: String },
    ReasoningDelta { text: String },
    ContentDelta { text: String },
    /// Full assistant message finished (content without tool-call blocks).
    AssistantMessage { content: String, tool_calls: Vec<ToolCall> },
    ToolCallStart { id: String, name: String, arguments: serde_json::Value },
    ToolCallResult { id: String, name: String, output: String, is_error: bool, truncated: bool, elapsed_ms: u64 },
    PermissionRequested { id: String, name: String, preview: Option<String> },
    PermissionResolved { id: String, allowed: bool },
    Verify { command: String, ok: bool, output: String },
    Compaction { messages_before: usize, messages_after: usize, summary_tokens: u32 },
    Metrics { timings: Timings, ctx_used: u32, ctx_total: u32, cache_ratio: f64, decode_tps: f64 },
    Warning { text: String },
    Error { text: String },
    Finished { reason: String, turns: u32 },
    /// A subagent started working on `task` (events until `AgentFinished` belong to it).
    AgentSpawned { id: u64, kind: String, task: String },
    AgentFinished { id: u64, outcome: String, turns: u32 },
}
