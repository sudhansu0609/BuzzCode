//! Tool registry: sorted, freezable (the frozen `tools` JSON is part of the stable prefix).

use anyhow::{bail, Result};
use cb_tool_api::{Tool, ToolSpec};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

#[derive(Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
    frozen: OnceLock<Arc<Value>>,
}

impl ToolRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register(&mut self, t: Arc<dyn Tool>) -> Result<()> {
        if self.frozen.get().is_some() { bail!("registry is frozen; cannot register {}", t.spec().name); }
        let name = t.spec().name.clone();
        if self.tools.contains_key(&name) { bail!("duplicate tool {name}"); }
        self.tools.insert(name, t);
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> { self.tools.get(name).cloned() }
    pub fn names(&self) -> Vec<String> { self.tools.keys().cloned().collect() }
    pub fn specs(&self) -> Vec<&ToolSpec> { self.tools.values().map(|t| t.spec()).collect() }
    pub fn len(&self) -> usize { self.tools.len() }
    pub fn is_empty(&self) -> bool { self.tools.is_empty() }

    /// Subset registry (for subagents / plan mode). Not frozen.
    pub fn scoped(&self, allow: &[&str]) -> ToolRegistry {
        let mut r = ToolRegistry::new();
        for (n, t) in &self.tools { if allow.contains(&n.as_str()) { r.tools.insert(n.clone(), t.clone()); } }
        r
    }

    pub fn without(&self, deny: &[&str]) -> ToolRegistry {
        let mut r = ToolRegistry::new();
        for (n, t) in &self.tools { if !deny.contains(&n.as_str()) { r.tools.insert(n.clone(), t.clone()); } }
        r
    }

    /// OpenAI `tools` array, sorted by name, built once.
    pub fn freeze(&self) -> Arc<Value> {
        self.frozen.get_or_init(|| {
            let arr: Vec<Value> = self.tools.values().map(|t| t.spec().to_openai()).collect();
            Arc::new(Value::Array(arr))
        }).clone()
    }

    pub fn is_frozen(&self) -> bool { self.frozen.get().is_some() }

    /// Lightweight schema validation: required keys present, basic type checks, enum membership.
    pub fn validate(&self, name: &str, args: &Value) -> Result<(), ArgError> {
        let Some(t) = self.tools.get(name) else { return Err(ArgError::UnknownTool(name.into())) };
        validate_against(&t.spec().schema, args, "")
    }
}

#[derive(Debug, thiserror::Error, Clone)]
pub enum ArgError {
    #[error("unknown tool `{0}`")]
    UnknownTool(String),
    #[error("missing required argument `{0}`")]
    Missing(String),
    #[error("argument `{0}` should be {1}")]
    Type(String, String),
    #[error("argument `{0}` must be one of {1}")]
    Enum(String, String),
    #[error("arguments must be a JSON object")]
    NotObject,
}

fn validate_against(schema: &Value, args: &Value, path: &str) -> Result<(), ArgError> {
    let Some(obj) = args.as_object() else { return Err(ArgError::NotObject) };
    if let Some(req) = schema.get("required").and_then(Value::as_array) {
        for r in req.iter().filter_map(Value::as_str) {
            match obj.get(r) { Some(v) if !v.is_null() => {}, _ => return Err(ArgError::Missing(format!("{path}{r}"))) }
        }
    }
    if let Some(props) = schema.get("properties").and_then(Value::as_object) {
        for (k, v) in obj {
            let Some(ps) = props.get(k) else { continue };
            if v.is_null() { continue; }
            let types: Vec<String> = match ps.get("type") {
                Some(Value::String(s)) => vec![s.clone()],
                Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
                _ => vec![],
            };
            if !types.is_empty() {
                let ok = types.iter().any(|t| match t.as_str() {
                    "string" => v.is_string(), "integer" => v.is_i64() || v.is_u64() || v.as_f64().map(|f| f.fract() == 0.0).unwrap_or(false),
                    "number" => v.is_number(), "boolean" => v.is_boolean(), "array" => v.is_array(), "object" => v.is_object(), "null" => v.is_null(), _ => true,
                });
                if !ok { return Err(ArgError::Type(format!("{path}{k}"), types.join("|"))); }
            }
            if let Some(en) = ps.get("enum").and_then(Value::as_array) {
                if !en.contains(v) { return Err(ArgError::Enum(format!("{path}{k}"), serde_json::to_string(en).unwrap_or_default())); }
            }
        }
    }
    Ok(())
}
