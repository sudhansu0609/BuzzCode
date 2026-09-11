//! Chat messages with deterministic serialization (key order is fixed → stable prefix bytes).

use serde_json::{json, Map, Value};
use std::cell::Cell;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role { System, Developer, User, Assistant, Tool }

impl Role {
    pub fn as_str(self) -> &'static str {
        match self { Role::System => "system", Role::Developer => "developer", Role::User => "user", Role::Assistant => "assistant", Role::Tool => "tool" }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// Parsed arguments. Serialized compactly and deterministically.
    pub arguments: Value,
}

impl ToolCall {
    /// Stable hash of name+arguments (loop detection).
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        h.update(self.name.as_bytes());
        h.update(b"\0");
        h.update(serde_json::to_string(&self.arguments).unwrap_or_default().as_bytes());
        *h.finalize().as_bytes()
    }
}

/// Metadata that is never serialized into the request.
#[derive(Debug, Clone, Default)]
pub struct MsgMeta {
    pub turn: u32,
    pub tool_name: Option<String>,
    pub is_error: bool,
    pub truncated: bool,
    pub spill_path: Option<std::path::PathBuf>,
    /// Marks the compaction summary message.
    pub is_summary: bool,
    /// Marks the repo-map message.
    pub is_repo_map: bool,
    /// Reasoning text (kept for UI/transcript only; never re-sent).
    pub reasoning: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    /// Memoized exact token count for this message (as rendered in the template).
    pub tokens: Cell<Option<u32>>,
    pub meta: MsgMeta,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into(), tool_calls: vec![], tool_call_id: None, tokens: Cell::new(None), meta: MsgMeta::default() }
    }
    pub fn system(c: impl Into<String>) -> Self { Self::new(Role::System, c) }
    pub fn developer(c: impl Into<String>) -> Self { Self::new(Role::Developer, c) }
    pub fn user(c: impl Into<String>) -> Self { Self::new(Role::User, c) }
    pub fn assistant(c: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        let mut m = Self::new(Role::Assistant, c);
        m.tool_calls = calls;
        m
    }
    pub fn tool(call_id: impl Into<String>, name: impl Into<String>, content: impl Into<String>, is_error: bool) -> Self {
        let mut m = Self::new(Role::Tool, content);
        m.tool_call_id = Some(call_id.into());
        m.meta.tool_name = Some(name.into());
        m.meta.is_error = is_error;
        m
    }

    /// OpenAI-style JSON object with **fixed key order**: role, content, tool_calls, tool_call_id.
    pub fn to_value(&self) -> Value {
        let mut m = Map::new();
        m.insert("role".into(), Value::String(self.role.as_str().into()));
        m.insert("content".into(), Value::String(self.content.clone()));
        if !self.tool_calls.is_empty() {
            let calls: Vec<Value> = self.tool_calls.iter().map(|c| json!({
                "id": c.id,
                "type": "function",
                "function": { "name": c.name, "arguments": serde_json::to_string(&c.arguments).unwrap_or_else(|_| "{}".into()) }
            })).collect();
            m.insert("tool_calls".into(), Value::Array(calls));
        }
        if let Some(id) = &self.tool_call_id {
            m.insert("tool_call_id".into(), Value::String(id.clone()));
            if let Some(n) = &self.meta.tool_name { m.insert("name".into(), Value::String(n.clone())); }
        }
        Value::Object(m)
    }

    /// Rough character count used for heuristic token estimates.
    pub fn chars(&self) -> usize {
        self.content.len() + self.tool_calls.iter().map(|c| c.name.len() + serde_json::to_string(&c.arguments).map(|s| s.len()).unwrap_or(0) + 24).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn key_order_is_stable() {
        let m = Message::assistant("x", vec![ToolCall { id: "1".into(), name: "t".into(), arguments: json!({"b":1,"a":2}) }]);
        let s = serde_json::to_string(&m.to_value()).unwrap();
        assert!(s.starts_with(r#"{"role":"assistant","content":"x","tool_calls":[{"id":"1","type":"function","function":{"name":"t","arguments":"{\"b\":1,\"a\":2}"}}]}"#));
    }
}
