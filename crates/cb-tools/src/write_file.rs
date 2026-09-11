use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;
use similar::TextDiff;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path of the file to create or overwrite.
    pub path: String,
    /// Full new content of the file.
    pub content: String,
}

pub struct WriteFile;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "write_file",
    "Create a new file or overwrite an existing one with the given content. Prefer edit_file for small changes to existing files.",
    PermissionClass::WriteFs,
));

fn diff_preview(cx: &ToolCtx, path: &std::path::Path, old: Option<&str>, new: &str) -> String {
    let name = fsutil::display(cx, path);
    match old {
        None => format!("new file {name} ({} lines)\n{}", new.lines().count(), new.lines().take(40).map(|l| format!("+{l}")).collect::<Vec<_>>().join("\n")),
        Some(o) => {
            let d = TextDiff::from_lines(o, new);
            let u = d.unified_diff().context_radius(3).header(&format!("a/{name}"), &format!("b/{name}")).to_string();
            let lines: Vec<&str> = u.lines().collect();
            if lines.len() > 200 { format!("{}\n… ({} more diff lines)", lines[..200].join("\n"), lines.len() - 200) } else { u }
        }
    }
}

#[async_trait::async_trait]
impl Tool for WriteFile {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn preview(&self, args: &Value, cx: &ToolCtx) -> Result<Option<String>, ToolError> {
        let a: Args = parse_args(args)?;
        let path = fsutil::resolve(cx, &a.path);
        let old = tokio::fs::read_to_string(&path).await.ok();
        Ok(Some(diff_preview(cx, &path, old.as_deref(), &a.content)))
    }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = fsutil::resolve(cx, &a.path);
        fsutil::check_write_allowed(cx, &path)?;
        let old = tokio::fs::read_to_string(&path).await.ok();
        if let Some(parent) = path.parent() { tokio::fs::create_dir_all(parent).await?; }
        // Preserve the existing line-ending convention.
        let content = match &old {
            Some(o) if fsutil::line_ending(o) == "\r\n" && !a.content.contains("\r\n") => a.content.replace('\n', "\r\n"),
            _ => a.content.clone(),
        };
        tokio::fs::write(&path, &content).await?;
        let lines = content.lines().count() as u32;
        let created = old.is_none();
        let mut out = ToolOutput::text(format!("{} {} ({lines} lines).", if created { "Created" } else { "Overwrote" }, fsutil::display(cx, &path)));
        out.edits.push(EditRecord { path: path.clone(), line_start: 1, line_end: lines.max(1), created });
        out.diff_preview = Some(diff_preview(cx, &path, old.as_deref(), &a.content));
        Ok(out)
    }
}
