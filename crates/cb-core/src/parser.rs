//! Layered tool-call parser for local models.
//!
//! 1. server-native `tool_calls` (llama-server `--jinja` parses Qwen's `<tool_call>` itself)
//! 2. `<tool_call>{...}</tool_call>` blocks in content
//! 3. ```json fenced objects with a known tool name
//! 4. a bare trailing JSON object with a known tool name
//! 5. JSON repair pass on any candidate that failed to parse
//! 6. fuzzy tool-name correction

use crate::message::ToolCall;
use cb_engine::sse::ToolCallAcc;
use nucleo_matcher::{pattern::{CaseMatching, Normalization, Pattern}, Config, Matcher};
use regex::Regex;
use serde_json::{Map, Value};
use std::sync::LazyLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseSource { ServerNative, XmlTag, FencedJson, BareJson }

#[derive(Debug, Clone)]
pub struct ParsedCalls {
    pub calls: Vec<ToolCall>,
    pub source: ParseSource,
    pub repaired: bool,
    /// Content with the tool-call blocks removed (what to show the user).
    pub leftover_text: String,
}

#[derive(Debug, Clone)]
pub enum ParseOutcome {
    Calls(ParsedCalls),
    NoCalls,
    Malformed { error: String, snippet: String },
}

static XML_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap());
/// Qwen3.x native format: `<function=NAME>\n<parameter=K>\nV\n</parameter>...\n</function>`
static QWEN_FN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<function=([A-Za-z0-9_.\-]+)>\s*(.*?)\s*</function>").unwrap());
static QWEN_PARAM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<parameter=([A-Za-z0-9_.\-]+)>\s*(.*?)\s*</parameter>").unwrap());

/// Convert a Qwen `<function=...>` body into a `{"name":..,"arguments":{..}}` JSON value.
/// Parameter values that look like JSON (objects/arrays/numbers/bools) are parsed; others stay strings.
fn qwen_function_to_json(name: &str, body: &str) -> Value {
    let mut args = Map::new();
    for c in QWEN_PARAM.captures_iter(body) {
        let k = c[1].to_string();
        let raw = c[2].to_string();
        let v = match raw.trim() {
            t if (t.starts_with('{') && t.ends_with('}')) || (t.starts_with('[') && t.ends_with(']')) => serde_json::from_str::<Value>(t).unwrap_or(Value::String(raw.clone())),
            "true" => Value::Bool(true), "false" => Value::Bool(false),
            t if t.parse::<i64>().is_ok() && !t.starts_with('0') || t == "0" => Value::Number(t.parse::<i64>().unwrap().into()),
            _ => Value::String(raw),
        };
        args.insert(k, v);
    }
    serde_json::json!({ "name": name, "arguments": Value::Object(args) })
}
static XML_OPEN_ONLY: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<tool_call>\s*(\{.*)$").unwrap());
static FENCE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)```(?:json|tool_call|tool)?\s*(\{.*?\})\s*```").unwrap());
static THINK_TAG: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)<(?:think|thought)>\s*(.*?)\s*</(?:think|thought)>").unwrap());

pub struct ToolCallParser {
    known: Vec<String>,
    next_id: std::sync::atomic::AtomicU32,
}

impl ToolCallParser {
    pub fn new(known: Vec<String>) -> Self { Self { known, next_id: std::sync::atomic::AtomicU32::new(1) } }

    fn new_id(&self) -> String {
        format!("call_{:04}", self.next_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }

    pub fn parse(&self, native: &[ToolCallAcc], content: &str) -> ParseOutcome {
        // 1. native
        if !native.is_empty() {
            let mut calls = Vec::new();
            let mut repaired = false;
            for n in native {
                let name = self.fix_name(&n.name);
                let args = match parse_json_lenient(&n.arguments) {
                    Some((v, r)) => { repaired |= r; v }
                    None => return ParseOutcome::Malformed { error: format!("tool call `{}` has invalid JSON arguments", n.name), snippet: n.arguments.chars().take(300).collect() },
                };
                let Some(name) = name else { return ParseOutcome::Malformed { error: format!("unknown tool `{}`", n.name), snippet: n.arguments.chars().take(200).collect() } };
                let id = if n.id.is_empty() { self.new_id() } else { n.id.clone() };
                calls.push(ToolCall { id, name, arguments: ensure_object(args) });
            }
            return ParseOutcome::Calls(ParsedCalls { calls, source: ParseSource::ServerNative, repaired, leftover_text: strip_blocks(content) });
        }

        // 2a. Qwen `<function=...>` blocks (inside or outside <tool_call>)
        let mut candidates: Vec<(String, ParseSource)> = QWEN_FN.captures_iter(content)
            .map(|c| (qwen_function_to_json(&c[1], &c[2]).to_string(), ParseSource::XmlTag)).collect();
        // 2b. <tool_call>{json}</tool_call> blocks
        if candidates.is_empty() {
            candidates = XML_TAG.captures_iter(content).map(|c| (c[1].to_string(), ParseSource::XmlTag)).collect();
        }
        if candidates.is_empty() {
            if let Some(c) = XML_OPEN_ONLY.captures(content) { candidates.push((c[1].to_string(), ParseSource::XmlTag)); }
        }
        // 3. fenced json
        if candidates.is_empty() {
            candidates = FENCE.captures_iter(content).map(|c| (c[1].to_string(), ParseSource::FencedJson)).collect();
        }
        // 4. bare trailing object
        if candidates.is_empty() {
            if let Some(obj) = trailing_json_object(content) { candidates.push((obj, ParseSource::BareJson)); }
        }
        if candidates.is_empty() { return ParseOutcome::NoCalls; }

        let source = candidates[0].1;
        let mut calls = Vec::new();
        let mut repaired = false;
        for (raw, _) in candidates {
            let Some((v, r)) = parse_json_lenient(&raw) else {
                return ParseOutcome::Malformed { error: "could not parse tool call JSON".into(), snippet: raw.chars().take(300).collect() };
            };
            repaired |= r;
            // Accept {"name","arguments"}, {"name","parameters"}, {"function":{"name","arguments"}}, {"tool":..,"args":..}
            let (name_raw, args) = extract_name_args(&v);
            let Some(name_raw) = name_raw else {
                if source == ParseSource::BareJson { return ParseOutcome::NoCalls; }
                return ParseOutcome::Malformed { error: "tool call object has no `name`".into(), snippet: raw.chars().take(300).collect() };
            };
            let Some(name) = self.fix_name(&name_raw) else {
                if source == ParseSource::BareJson { return ParseOutcome::NoCalls; }
                return ParseOutcome::Malformed { error: format!("unknown tool `{name_raw}`"), snippet: raw.chars().take(200).collect() };
            };
            calls.push(ToolCall { id: self.new_id(), name, arguments: ensure_object(args) });
        }
        ParseOutcome::Calls(ParsedCalls { calls, source, repaired, leftover_text: strip_blocks(content) })
    }

    /// Exact → case/underscore-insensitive → fuzzy (nucleo) name resolution.
    pub fn fix_name(&self, raw: &str) -> Option<String> {
        let raw = raw.trim().trim_matches('`').trim_start_matches("functions.").trim_start_matches("tools.");
        if self.known.iter().any(|k| k == raw) { return Some(raw.to_string()); }
        let norm = |s: &str| s.to_ascii_lowercase().replace(['-', ' ', '.'], "_");
        let n = norm(raw);
        if let Some(k) = self.known.iter().find(|k| norm(k) == n) { return Some(k.clone()); }
        if n.len() < 3 { return None; }
        let mut matcher = Matcher::new(Config::DEFAULT);
        let pat = Pattern::parse(&n, CaseMatching::Ignore, Normalization::Smart);
        let mut best: Option<(u32, &String)> = None;
        for k in &self.known {
            let mut buf = Vec::new();
            let hay = nucleo_matcher::Utf32Str::new(k, &mut buf);
            if let Some(score) = pat.score(hay, &mut matcher) {
                if best.map(|(s, _)| score > s).unwrap_or(true) { best = Some((score, k)); }
            }
        }
        // Require a strong match (≈ most characters matched in order).
        best.filter(|(s, k)| *s as usize >= k.len() * 12).map(|(_, k)| k.clone())
    }
}

fn extract_name_args(v: &Value) -> (Option<String>, Value) {
    let obj = match v { Value::Object(o) => o, _ => return (None, Value::Null) };
    if let Some(f) = obj.get("function").and_then(Value::as_object) {
        let name = f.get("name").and_then(Value::as_str).map(str::to_string);
        let args = f.get("arguments").or_else(|| f.get("parameters")).cloned().unwrap_or(Value::Null);
        return (name, unwrap_args(args));
    }
    let name = obj.get("name").or_else(|| obj.get("tool")).or_else(|| obj.get("tool_name")).or_else(|| obj.get("function_name")).and_then(Value::as_str).map(str::to_string);
    let args = obj.get("arguments").or_else(|| obj.get("parameters")).or_else(|| obj.get("args")).or_else(|| obj.get("input")).cloned().unwrap_or(Value::Null);
    (name, unwrap_args(args))
}

/// Arguments may arrive as a JSON string (double-encoded).
fn unwrap_args(v: Value) -> Value {
    match v {
        Value::String(s) => parse_json_lenient(&s).map(|(v, _)| v).unwrap_or(Value::Object(Map::new())),
        other => other,
    }
}

fn ensure_object(v: Value) -> Value { if v.is_object() { v } else { Value::Object(Map::new()) } }

/// Remove tool-call blocks / fences from content for display.
fn strip_blocks(content: &str) -> String {
    let s = XML_TAG.replace_all(content, "");
    let s = QWEN_FN.replace_all(&s, "");
    let s = XML_OPEN_ONLY.replace_all(&s, "");
    let s = FENCE.replace_all(&s, "");
    let s = THINK_TAG.replace_all(&s, "");
    s.trim().to_string()
}

/// Find a JSON object that ends the content (e.g. `...\n{"name": "read_file", "arguments": {...}}`).
fn trailing_json_object(content: &str) -> Option<String> {
    let t = content.trim_end();
    if !t.ends_with('}') { return None; }
    // Walk backwards to the matching '{' counting braces (ignoring braces inside strings roughly).
    let bytes = t.as_bytes();
    let mut depth = 0i32;
    let mut in_str = false;
    let mut i = bytes.len();
    while i > 0 {
        i -= 1;
        let b = bytes[i];
        if b == b'"' && (i == 0 || bytes[i - 1] != b'\\') { in_str = !in_str; continue; }
        if in_str { continue; }
        if b == b'}' { depth += 1; }
        if b == b'{' { depth -= 1; if depth == 0 { return Some(t[i..].to_string()); } }
    }
    None
}

/// Strict parse, then a repair pass. Returns (value, was_repaired).
pub fn parse_json_lenient(s: &str) -> Option<(Value, bool)> {
    let s = s.trim();
    if s.is_empty() { return Some((Value::Object(Map::new()), false)); }
    if let Ok(v) = serde_json::from_str::<Value>(s) { return Some((v, false)); }
    let fixed = repair_json(s);
    serde_json::from_str::<Value>(&fixed).ok().map(|v| (v, true))
}

/// Best-effort JSON repair for common small-model mistakes.
pub fn repair_json(s: &str) -> String {
    let mut t = s.trim().to_string();
    // Strip code fences / leading labels.
    if let Some(x) = t.strip_prefix("```json") { t = x.to_string(); }
    if let Some(x) = t.strip_prefix("```") { t = x.to_string(); }
    if let Some(x) = t.strip_suffix("```") { t = x.to_string(); }
    t = t.trim().to_string();
    // Python literals.
    t = t.replace(": True", ": true").replace(": False", ": false").replace(": None", ": null");
    // Single-quoted strings → double (only when there are no double quotes at all, to be safe).
    if !t.contains('"') && t.contains('\'') { t = t.replace('\'', "\""); }
    // Raw newlines/tabs inside strings → escaped.
    t = escape_control_chars_in_strings(&t);
    // Trailing commas.
    static TRAILING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r",\s*([}\]])").unwrap());
    t = TRAILING.replace_all(&t, "$1").to_string();
    // Unbalanced braces/brackets: append closers.
    let (mut braces, mut brackets, mut in_str, mut esc) = (0i32, 0i32, false, false);
    for c in t.chars() {
        if esc { esc = false; continue; }
        match c {
            '\\' if in_str => esc = true,
            '"' => in_str = !in_str,
            '{' if !in_str => braces += 1,
            '}' if !in_str => braces -= 1,
            '[' if !in_str => brackets += 1,
            ']' if !in_str => brackets -= 1,
            _ => {}
        }
    }
    if in_str { t.push('"'); }
    for _ in 0..brackets.max(0) { t.push(']'); }
    for _ in 0..braces.max(0) { t.push('}'); }
    t
}

fn escape_control_chars_in_strings(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 16);
    let (mut in_str, mut esc) = (false, false);
    for c in s.chars() {
        if in_str {
            if esc { out.push(c); esc = false; continue; }
            match c {
                '\\' => { out.push(c); esc = true; }
                '"' => { out.push(c); in_str = false; }
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                _ => out.push(c),
            }
        } else {
            if c == '"' { in_str = true; }
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> ToolCallParser { ToolCallParser::new(vec!["read_file".into(), "edit_file".into(), "grep".into(), "shell".into()]) }

    #[test]
    fn xml_block() {
        let out = p().parse(&[], "Let me look.\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>");
        match out { ParseOutcome::Calls(c) => { assert_eq!(c.calls[0].name, "read_file"); assert_eq!(c.leftover_text, "Let me look."); assert_eq!(c.source, ParseSource::XmlTag); } o => panic!("{o:?}") }
    }

    #[test]
    fn qwen_function_format() {
        let s = "I'll read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n<parameter=limit>\n40\n</parameter>\n</function>\n</tool_call>";
        match p().parse(&[], s) {
            ParseOutcome::Calls(c) => { assert_eq!(c.calls[0].name, "read_file"); assert_eq!(c.calls[0].arguments["path"], "src/main.rs"); assert_eq!(c.calls[0].arguments["limit"], 40); assert_eq!(c.leftover_text, "I'll read it."); }
            o => panic!("{o:?}"),
        }
    }

    #[test]
    fn fenced_and_repair() {
        let out = p().parse(&[], "```json\n{\"name\": \"grep\", \"arguments\": {\"pattern\": \"fn main\", \"path\": \"src\",}}\n```");
        match out { ParseOutcome::Calls(c) => { assert_eq!(c.calls[0].name, "grep"); assert!(c.repaired); } o => panic!("{o:?}") }
    }

    #[test]
    fn bare_trailing_object_with_parameters_key() {
        let out = p().parse(&[], "I'll read it now:\n{\"name\": \"read_file\", \"parameters\": {\"path\": \"x\"}}");
        match out { ParseOutcome::Calls(c) => assert_eq!(c.calls[0].arguments["path"], "x"), o => panic!("{o:?}") }
    }

    #[test]
    fn fuzzy_name() {
        assert_eq!(p().fix_name("ReadFile"), Some("read_file".into()));
        assert_eq!(p().fix_name("functions.edit_file"), Some("edit_file".into()));
        assert_eq!(p().fix_name("banana"), None);
    }

    #[test]
    fn native_double_encoded_args() {
        let n = ToolCallAcc { index: 0, id: "c1".into(), name: "shell".into(), arguments: "{\"command\": \"ls\"}".into() };
        match p().parse(&[n], "") { ParseOutcome::Calls(c) => assert_eq!(c.calls[0].arguments["command"], "ls"), o => panic!("{o:?}") }
    }

    #[test]
    fn plain_text_is_no_calls() {
        assert!(matches!(p().parse(&[], "Done. The function is in src/lib.rs."), ParseOutcome::NoCalls));
    }

    #[test]
    fn unterminated_string_repair() {
        let (v, r) = parse_json_lenient("{\"name\": \"shell\", \"arguments\": {\"command\": \"cargo test").unwrap();
        assert!(r);
        assert_eq!(v["arguments"]["command"], "cargo test");
    }

    #[test]
    fn strips_think_tags_from_leftover() {
        let text = "<think>\nThinking deeply about the task...\n</think>\n<tool_call>\n{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\"}}\n</tool_call>";
        match p().parse(&[], text) {
            ParseOutcome::Calls(c) => {
                assert_eq!(c.calls[0].name, "read_file");
                assert_eq!(c.leftover_text, "");
            }
            o => panic!("{o:?}"),
        }
    }
}
