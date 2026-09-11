use crate::fsutil;
use cb_tool_api::*;
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize, schemars::JsonSchema)]
pub struct Args {
    /// Path to the file (relative to the project root or absolute).
    pub path: String,
    /// 1-based line number to start from.
    #[serde(default)]
    pub offset: Option<u32>,
    /// Maximum number of lines to return (default 400).
    #[serde(default)]
    pub limit: Option<u32>,
}

pub struct ReadFile;

static SPEC: std::sync::LazyLock<ToolSpec> = std::sync::LazyLock::new(|| ToolSpec::new::<Args>(
    "read_file",
    "Read a text file with line numbers. Use offset/limit to page through large files. Returns `total_lines` so you know if more remains.",
    PermissionClass::ReadOnly,
));

#[async_trait::async_trait]
impl Tool for ReadFile {
    fn spec(&self) -> &ToolSpec { &SPEC }

    async fn call(&self, args: Value, cx: &ToolCtx) -> Result<ToolOutput, ToolError> {
        let a: Args = parse_args(&args)?;
        let path = fsutil::resolve(cx, &a.path);
        let bytes = tokio::fs::read(&path).await.map_err(|e| ToolError::Failed(format!("cannot read {}: {e}", fsutil::display(cx, &path))))?;
        if fsutil::is_probably_binary(&bytes) {
            return Ok(ToolOutput::error(format!("{} appears to be binary ({} bytes)", fsutil::display(cx, &path), bytes.len())));
        }
        let text = String::from_utf8_lossy(&bytes);
        let lines: Vec<&str> = text.lines().collect();
        let total = lines.len();
        let start = a.offset.unwrap_or(1).max(1) as usize;
        let limit = a.limit.unwrap_or(400).clamp(1, 2000) as usize;
        if start > total && total > 0 {
            return Ok(ToolOutput::error(format!("offset {start} is past the end ({total} lines)")));
        }
        let end = (start - 1 + limit).min(total);
        let width = end.to_string().len().max(3);
        let mut out = String::with_capacity(bytes.len().min(cx.budget.max_bytes + 256));
        let mut bytes_out = 0usize;
        let mut shown_end = start - 1;
        for (i, l) in lines.iter().enumerate().take(end).skip(start - 1) {
            let line = format!("{:>width$}\t{}\n", i + 1, l, width = width);
            if bytes_out + line.len() > cx.budget.max_bytes { break; }
            out.push_str(&line);
            bytes_out += line.len();
            shown_end = i + 1;
        }
        let header = format!("{} (lines {start}-{shown_end} of {total})\n", fsutil::display(cx, &path));
        let mut text = header + &out;
        if shown_end < total {
            text.push_str(&format!("… {} more lines. Continue with read_file(path=\"{}\", offset={}).\n", total - shown_end, a.path, shown_end + 1));
        }
        Ok(ToolOutput::text(text))
    }
}
