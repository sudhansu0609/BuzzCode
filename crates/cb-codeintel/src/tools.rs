//! Tools exposing the index: outline, symbol_search, repo_map.

use crate::index::{format_outline, Indexer};
use crate::parse::SymKind;
use crate::repomap::{MapOptions, RepoMap};
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

/// Shared handle stored in `ToolCtx.extensions`.
pub struct CodeIntel {
    pub indexer: Arc<Indexer>,
    pub map: parking_lot::Mutex<Option<Arc<RepoMap>>>,
    pub map_tokens: u32,
}

impl CodeIntel {
    pub fn new(indexer: Arc<Indexer>, map_tokens: u32) -> Self { Self { indexer, map: parking_lot::Mutex::new(None), map_tokens } }
    pub fn invalidate_map(&self) { *self.map.lock() = None; }
    pub fn repo_map(&self) -> Arc<RepoMap> {
        let mut m = self.map.lock();
        if let Some(r) = m.as_ref() { return r.clone(); }
        let r = Arc::new(RepoMap::build(&self.indexer));
        *m = Some(r.clone());
        r
    }
}

fn estimate(s: &str) -> u32 { (s.len() as f32 / 3.2).ceil() as u32 }

fn ci(cx: &ToolCtx) -> Result<Arc<CodeIntel>, ToolError> {
    cx.extensions.get::<CodeIntel>().ok_or_else(|| ToolError::Failed("code intelligence is disabled".into()))
}

// ------------------------------------------------------------------ outline

#[derive(Deserialize, schemars::JsonSchema)]
pub struct OutlineArgs {
    /// File path.
    pub path: String,
}
pub struct Outline;
static OUTLINE_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<OutlineArgs>(
    "outline", "List the symbols (functions, types, classes…) defined in a file with line numbers — much cheaper than reading the whole file.", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for Outline {
    fn spec(&self) -> &ToolSpec { &OUTLINE_SPEC }
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: OutlineArgs = parse_args(&args)?;
        let c = ci(cx)?;
        let p = { let pb = std::path::PathBuf::from(a.path.replace('/', std::path::MAIN_SEPARATOR_STR)); if pb.is_absolute() { pb } else { cx.cwd.join(pb) } };
        let syms = c.indexer.outline(&p).map_err(|e| ToolError::Failed(format!("{e:#}")))?;
        Ok(ToolOutput::text(format_outline(&c.indexer.rel(&p), &syms)))
    }
}

// ------------------------------------------------------------------ symbol_search

#[derive(Deserialize, schemars::JsonSchema)]
pub struct SearchArgs {
    /// Symbol name or fuzzy fragment, e.g. "parseConfig" or "pars cfg".
    pub query: String,
    /// Restrict to a kind: fn, method, struct, enum, trait, class, interface, const, type, mod.
    #[serde(default)]
    pub kind: Option<String>,
    /// Max results (default 20).
    #[serde(default)]
    pub limit: Option<u32>,
}
pub struct SymbolSearch;
static SEARCH_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<SearchArgs>(
    "symbol_search", "Find where a symbol (function/type/class…) is defined, fuzzy-matched across the whole repository. Returns path:line kind signature.", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for SymbolSearch {
    fn spec(&self) -> &ToolSpec { &SEARCH_SPEC }
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: SearchArgs = parse_args(&args)?;
        let c = ci(cx)?;
        let kind = a.kind.as_deref().and_then(SymKind::parse);
        let hits = c.indexer.search(&a.query, kind, a.limit.unwrap_or(20).clamp(1, 100) as usize);
        if hits.is_empty() { return Ok(ToolOutput::text(format!("No symbols match {:?}", a.query))); }
        let mut s = String::new();
        for h in hits { s.push_str(&format!("{}:{} {} {}\n", h.path, h.symbol.line_start, h.symbol.kind.label(), h.symbol.signature)); }
        Ok(ToolOutput::text(s))
    }
}

// ------------------------------------------------------------------ repo_map

#[derive(Deserialize, schemars::JsonSchema)]
pub struct MapArgs {
    /// Files to focus the map around (relative paths).
    #[serde(default)]
    pub focus_paths: Vec<String>,
    /// Identifiers of interest; files defining them are ranked higher.
    #[serde(default)]
    pub mentions: Vec<String>,
    /// Token budget (default from config).
    #[serde(default)]
    pub budget_tokens: Option<u32>,
}
pub struct RepoMapTool;
static MAP_SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<MapArgs>(
    "repo_map", "Ranked overview of the repository's most important files and symbols (PageRank over references). Use focus_paths/mentions to re-center it.", PermissionClass::ReadOnly));

#[async_trait::async_trait]
impl Tool for RepoMapTool {
    fn spec(&self) -> &ToolSpec { &MAP_SPEC }
    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: MapArgs = parse_args(&args)?;
        let c = ci(cx)?;
        let map = c.repo_map();
        let focus: Vec<String> = a.focus_paths.iter().map(|p| p.replace('\\', "/")).collect();
        let s = map.render(&MapOptions { budget_tokens: a.budget_tokens.unwrap_or(c.map_tokens).clamp(200, 8000), focus: &focus, mentions: &a.mentions, estimate: &estimate });
        Ok(ToolOutput::text(if s.is_empty() { "(empty index — no supported source files found)".into() } else { s }))
    }
}

pub fn tools() -> Vec<Arc<dyn Tool>> { vec![Arc::new(Outline), Arc::new(SymbolSearch), Arc::new(RepoMapTool)] }

/// Render the session-start repo map (message #1) with the configured budget.
pub fn initial_map(c: &CodeIntel, budget: u32) -> String {
    let map = c.repo_map();
    map.render(&MapOptions { budget_tokens: budget, focus: &[], mentions: &[], estimate: &estimate })
}
