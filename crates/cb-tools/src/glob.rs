use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Glob pattern, e.g. "**/*.rs" or "src/**/test_*.py".
    pub pattern: String,
    /// Directory to search in (default: project root).
    #[serde(default)]
    pub path: Option<String>,
}

pub struct Glob;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "glob",
    "Find files by glob pattern (respects .gitignore). Results sorted by modification time, newest first. Max 500.",
    PermissionClass::ReadOnly,
));

#[async_trait::async_trait]
impl Tool for Glob {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let root = a.path.as_deref().map(|p| fsutil::resolve(cx, p)).unwrap_or(cx.cwd.clone());
        if !root.is_dir() { return Ok(ToolOutput::error(format!("{} is not a directory", fsutil::display(cx, &root)))); }
        let pattern = a.pattern.trim().replace('\\', "/");
        let cx2 = cx.clone();
        let results = tokio::task::spawn_blocking(move || {
            let mut ob = ignore::overrides::OverrideBuilder::new(&root);
            let pat = if pattern.contains('/') || pattern.starts_with("**") { pattern.clone() } else { format!("**/{pattern}") };
            let _ = ob.add(&pat);
            let overrides = ob.build().ok();
            let mut wb = ignore::WalkBuilder::new(&root);
            wb.hidden(true).git_ignore(true).git_global(true).follow_links(false).max_depth(Some(32));
            if let Some(o) = overrides { wb.overrides(o); }
            let mut hits: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
            for e in wb.build().flatten() {
                if e.file_type().map(|t| t.is_file()).unwrap_or(false) {
                    let mtime = e.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(std::time::UNIX_EPOCH);
                    hits.push((mtime, e.into_path()));
                    if hits.len() > 5000 { break; }
                }
            }
            hits.sort_by(|a, b| b.0.cmp(&a.0));
            let total = hits.len();
            let shown: Vec<String> = hits.into_iter().take(500).map(|(_, p)| fsutil::display(&cx2, &p)).collect();
            (total, shown)
        }).await.map_err(|e| ToolError::Failed(e.to_string()))?;
        let (total, shown) = results;
        if shown.is_empty() { return Ok(ToolOutput::text(format!("No files match {:?}", a.pattern))); }
        let mut text = shown.join("\n");
        if total > shown.len() { text.push_str(&format!("\n… {} more (narrow the pattern)", total - shown.len())); }
        Ok(ToolOutput::text(text))
    }
}
