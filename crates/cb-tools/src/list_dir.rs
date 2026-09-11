use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Directory to list (default: project root).
    #[serde(default)]
    pub path: Option<String>,
    /// Recursion depth (default 2, max 6).
    #[serde(default)]
    pub depth: Option<u32>,
}

pub struct ListDir;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "list_dir",
    "List a directory as a tree (respects .gitignore). Directories end with '/'. Max 300 entries.",
    PermissionClass::ReadOnly,
));

#[async_trait::async_trait]
impl Tool for ListDir {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let root = a.path.as_deref().map(|p| fsutil::resolve(cx, p)).unwrap_or(cx.cwd.clone());
        if !root.is_dir() { return Ok(ToolOutput::error(format!("{} is not a directory", fsutil::display(cx, &root)))); }
        let depth = a.depth.unwrap_or(2).clamp(1, 6) as usize;
        let root2 = root.clone();
        let entries = tokio::task::spawn_blocking(move || {
            let mut wb = ignore::WalkBuilder::new(&root2);
            wb.hidden(true).git_ignore(true).max_depth(Some(depth)).sort_by_file_name(|a, b| a.cmp(b));
            let mut out = Vec::new();
            for e in wb.build().flatten() {
                if e.depth() == 0 { continue; }
                let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
                let rel = e.path().strip_prefix(&root2).unwrap_or(e.path()).to_string_lossy().replace('\\', "/");
                let size = if is_dir { 0 } else { e.metadata().map(|m| m.len()).unwrap_or(0) };
                out.push((rel, is_dir, size));
                if out.len() >= 300 { break; }
            }
            out
        }).await.map_err(|e| ToolError::Failed(e.to_string()))?;
        let mut text = format!("{}/\n", fsutil::display(cx, &root));
        for (rel, is_dir, size) in &entries {
            let indent = "  ".repeat(rel.matches('/').count() + 1);
            let name = rel.rsplit('/').next().unwrap_or(rel);
            if *is_dir { text.push_str(&format!("{indent}{name}/\n")); }
            else { text.push_str(&format!("{indent}{name}  ({})\n", human(*size))); }
        }
        if entries.len() >= 300 { text.push_str("… truncated at 300 entries; list a subdirectory.\n"); }
        Ok(ToolOutput::text(text))
    }
}

fn human(n: u64) -> String {
    if n < 1024 { format!("{n} B") } else if n < 1024 * 1024 { format!("{:.1} KB", n as f64 / 1024.0) } else { format!("{:.1} MB", n as f64 / 1048576.0) }
}
