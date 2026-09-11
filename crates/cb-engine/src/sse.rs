//! Parse llama-server's OpenAI-compatible SSE chat chunks into [`StreamEvent`]s.

use crate::client::{FinishReason, StreamEvent, Timings};
use serde_json::Value;

/// Accumulates tool-call deltas by index while streaming.
#[derive(Debug, Default, Clone)]
pub struct ToolCallAcc {
    pub index: u32,
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Convert one SSE `data:` payload into zero or more events.
pub fn parse_chunk(data: &str) -> Vec<StreamEvent> {
    let mut out = Vec::new();
    if data.trim() == "[DONE]" { return out; }
    let v: Value = match serde_json::from_str(data) { Ok(v) => v, Err(e) => { out.push(StreamEvent::Error(format!("bad SSE json: {e}"))); return out; } };
    if let Some(err) = v.get("error") {
        out.push(StreamEvent::Error(err.get("message").and_then(Value::as_str).unwrap_or("server error").to_string()));
        return out;
    }
    if let Some(choices) = v.get("choices").and_then(Value::as_array) {
        for c in choices {
            if let Some(delta) = c.get("delta") {
                if let Some(r) = delta.get("reasoning_content").and_then(Value::as_str) { if !r.is_empty() { out.push(StreamEvent::Reasoning(r.to_string())); } }
                if let Some(r) = delta.get("reasoning").and_then(Value::as_str) { if !r.is_empty() { out.push(StreamEvent::Reasoning(r.to_string())); } }
                if let Some(t) = delta.get("content").and_then(Value::as_str) { if !t.is_empty() { out.push(StreamEvent::Content(t.to_string())); } }
                if let Some(tcs) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tc in tcs {
                        let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let id = tc.get("id").and_then(Value::as_str).map(str::to_string);
                        let f = tc.get("function");
                        let name = f.and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string);
                        let args = f.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("").to_string();
                        out.push(StreamEvent::ToolCallDelta { index, id, name, args_fragment: args });
                    }
                }
            }
            if let Some(fr) = c.get("finish_reason").and_then(Value::as_str) {
                out.push(StreamEvent::Done(match fr { "stop" => FinishReason::Stop, "length" => FinishReason::Length, "tool_calls" => FinishReason::ToolCalls, other => FinishReason::Other(other.to_string()) }));
            }
        }
    }
    if let Some(u) = v.get("usage") {
        if let (Some(p), Some(c)) = (u.get("prompt_tokens").and_then(Value::as_u64), u.get("completion_tokens").and_then(Value::as_u64)) {
            out.push(StreamEvent::Usage { prompt_tokens: p as u32, completion_tokens: c as u32 });
        }
    }
    if let Some(t) = v.get("timings") {
        let g = |k: &str| t.get(k).and_then(Value::as_f64).unwrap_or(0.0);
        out.push(StreamEvent::Timings(Timings {
            prompt_n: g("prompt_n") as u32,
            cache_n: g("cache_n") as u32,
            prompt_ms: g("prompt_ms"),
            predicted_n: g("predicted_n") as u32,
            predicted_ms: g("predicted_ms"),
            draft_n: g("draft_n") as u32,
            draft_n_accepted: g("draft_n_accepted") as u32,
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_content_and_tool_delta() {
        let d = r#"{"choices":[{"delta":{"content":"hi","tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"pa"}}]},"finish_reason":null}]}"#;
        let ev = parse_chunk(d);
        assert!(matches!(ev[0], StreamEvent::Content(ref s) if s == "hi"));
        assert!(matches!(ev[1], StreamEvent::ToolCallDelta { index: 0, .. }));
    }
    #[test]
    fn parses_timings() {
        let d = r#"{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":100,"cache_n":90,"prompt_ms":12.5,"predicted_n":10,"predicted_ms":250.0}}"#;
        let ev = parse_chunk(d);
        assert!(ev.iter().any(|e| matches!(e, StreamEvent::Timings(t) if t.cache_n == 90)));
    }
}
