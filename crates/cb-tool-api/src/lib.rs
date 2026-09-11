//! Minimal tool API shared by the agent core, builtin tools, code-intel and MCP adapters.
//! Kept dependency-light so every crate can implement [`Tool`] without pulling in the core.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

/// How dangerous a tool is; drives the permission prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionClass {
    /// Reads only; never prompts.
    ReadOnly,
    /// Writes inside the project tree.
    WriteFs,
    /// Runs arbitrary commands.
    Exec,
    /// Talks to the network (MCP remote etc.).
    Network,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    /// Prompt for WriteFs + Exec (+ Network).
    Ask,
    /// Prompt for Exec only.
    AutoAcceptEdits,
    /// Never prompt.
    Yolo,
}

impl PermissionMode {
    pub fn requires_prompt(self, class: PermissionClass) -> bool {
        match (self, class) {
            (_, PermissionClass::ReadOnly) => false,
            (PermissionMode::Yolo, _) => false,
            (PermissionMode::AutoAcceptEdits, PermissionClass::WriteFs) => false,
            _ => true,
        }
    }
}

/// Static description of a tool; `schema` is the JSON Schema of its arguments.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub schema: Value,
    pub class: PermissionClass,
}

impl ToolSpec {
    pub fn new<T: schemars::JsonSchema>(name: &str, description: &str, class: PermissionClass) -> Self {
        let mut schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or(Value::Null);
        // Strip keys the model doesn't need (keeps the frozen tools JSON compact).
        if let Value::Object(m) = &mut schema {
            m.remove("$schema");
            m.remove("title");
        }
        Self { name: name.into(), description: description.into(), schema, class }
    }

    /// OpenAI-style `{"type":"function","function":{...}}` entry.
    pub fn to_openai(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            }
        })
    }
}

/// Output-size budget handed to tools so they can self-truncate consistently.
#[derive(Debug, Clone, Copy)]
pub struct OutputBudget {
    pub max_bytes: usize,
    pub head_lines: usize,
    pub tail_lines: usize,
}

impl Default for OutputBudget {
    fn default() -> Self { Self { max_bytes: 24_000, head_lines: 150, tail_lines: 50 } }
}

/// Per-call context.
#[derive(Debug, Clone)]
pub struct ToolCtx {
    pub cwd: PathBuf,
    pub project_dir: PathBuf,
    /// Where to spill oversized outputs.
    pub spill_dir: PathBuf,
    pub cancel: CancellationToken,
    pub budget: OutputBudget,
    pub permission_mode: PermissionMode,
    pub turn: u32,
    /// Session-wide shared state (index handle etc.) — opaque to this crate.
    pub extensions: Arc<Extensions>,
}

/// Type-erased bag for cross-crate services (e.g. the code-intel index) without cycles.
#[derive(Default)]
pub struct Extensions {
    inner: parking::Map,
}

mod parking {
    use std::any::{Any, TypeId};
    use std::collections::HashMap;
    use std::sync::{Arc, RwLock};
    #[derive(Default)]
    pub struct Map(RwLock<HashMap<TypeId, Arc<dyn Any + Send + Sync>>>);
    impl Map {
        pub fn insert<T: Any + Send + Sync>(&self, v: Arc<T>) { self.0.write().unwrap().insert(TypeId::of::<T>(), v); }
        pub fn get<T: Any + Send + Sync>(&self) -> Option<Arc<T>> {
            self.0.read().unwrap().get(&TypeId::of::<T>()).cloned().and_then(|a| a.downcast::<T>().ok())
        }
    }
}

impl Extensions {
    pub fn insert<T: std::any::Any + Send + Sync>(&self, v: Arc<T>) { self.inner.insert(v); }
    pub fn get<T: std::any::Any + Send + Sync>(&self) -> Option<Arc<T>> { self.inner.get::<T>() }
}

impl std::fmt::Debug for Extensions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("Extensions") }
}

/// Something that wants to know when files were edited (e.g. the code index).
pub trait EditObserver: Send + Sync {
    fn on_edits(&self, paths: &[PathBuf]);
}

/// Registered observers (stored in `Extensions`).
#[derive(Default)]
pub struct EditObservers(pub parking_lot_lite::Mutex<Vec<Arc<dyn EditObserver>>>);

mod parking_lot_lite {
    /// Tiny std-based mutex wrapper so this crate stays dependency-light.
    pub struct Mutex<T>(std::sync::Mutex<T>);
    impl<T: Default> Default for Mutex<T> { fn default() -> Self { Self(std::sync::Mutex::new(T::default())) } }
    impl<T> Mutex<T> {
        pub fn lock(&self) -> std::sync::MutexGuard<'_, T> { self.0.lock().unwrap_or_else(|e| e.into_inner()) }
    }
}

impl EditObservers {
    pub fn notify(&self, paths: &[PathBuf]) { for o in self.0.lock().iter() { o.on_edits(paths); } }
    pub fn push(&self, o: Arc<dyn EditObserver>) { self.0.lock().push(o); }
}

/// Details about truncation so the model can page.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncInfo {
    pub total_lines: usize,
    pub shown_head: usize,
    pub shown_tail: usize,
    pub spill_path: PathBuf,
}

/// Record of a file edit (drives the edit-verify loop).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditRecord {
    pub path: PathBuf,
    /// 1-based inclusive line range touched in the *new* file.
    pub line_start: u32,
    pub line_end: u32,
    pub created: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    pub text: String,
    pub is_error: bool,
    pub truncated: Option<TruncInfo>,
    /// Shown in the permission UI before execution (unified diff etc.).
    pub diff_preview: Option<String>,
    pub edits: Vec<EditRecord>,
}

impl ToolOutput {
    pub fn text(s: impl Into<String>) -> Self { Self { text: s.into(), ..Default::default() } }
    pub fn error(s: impl Into<String>) -> Self { Self { text: s.into(), is_error: true, ..Default::default() } }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("invalid arguments: {0}")]
    InvalidArgs(String),
    #[error("denied: {0}")]
    Denied(String),
    #[error("cancelled")]
    Cancelled,
    #[error("timeout after {0}s")]
    Timeout(u64),
    #[error("{0}")]
    Failed(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<serde_json::Error> for ToolError {
    fn from(e: serde_json::Error) -> Self { ToolError::InvalidArgs(e.to_string()) }
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> &ToolSpec;

    /// Called before the permission prompt. Return a human-readable preview (e.g. diff).
    async fn preview(&self, _args: &Value, _cx: &ToolCtx) -> Result<Option<String>, ToolError> { Ok(None) }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError>;
}

/// Helper to deserialize typed args with a friendly error.
pub fn parse_args<T: for<'de> Deserialize<'de>>(args: &Value) -> Result<T, ToolError> {
    serde_json::from_value(args.clone()).map_err(|e| ToolError::InvalidArgs(e.to_string()))
}
